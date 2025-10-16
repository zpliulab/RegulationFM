from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from diffusion_model.discrete import PredefinedNoiseScheduleDiscrete, DiscreteUniformTransition, MarginalUniformTransition, network_preprocess
from diffusion_model.discrete import diffusion_utils as util
from diffusion_model.discrete.models.train_metrics import TrainLossDiscrete
from torch_geometric.data import Data, Batch
import torch_geometric.utils as utils
import copy
import pandas as pd

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class RowNormalize(nn.Module):
    def forward(self, x):
        # 计算每行的范数，添加 epsilon 防止除零
        epsilon = 1e-8
        norm = torch.norm(x, p=2, dim=1, keepdim=True) + epsilon
        # 归一化每行
        normalized_x = x / norm
        # 确保每行的两个数加起来为1
        normalized_x = normalized_x / torch.sum(normalized_x, dim=1, keepdim=True)
        return normalized_x


class Discrete_diffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        edge_percent,
        max_num_nodes=None,
        device='cpu',
        timesteps = 1000,
        sampling_timesteps = None,
        loss_type='l2',
        predflag='E_start',
        transition='marginal',  # 转移矩阵类型
        ddim_sampling_eta=0.,
        noise_type='cos',
        net_key_par
    ):
        super().__init__()
        self.Edim_output = 2       # 边的类型数量
        self.device = device
        self.train_loss = TrainLossDiscrete()
        self.val_E_kl = util.SumExceptBatchKL()
        self.val_nll = util.NLL()
        self.val_E_logp = util.SumExceptBatchMetric()
        self.model = model
        self.predflag = predflag
        self.edge_percent = edge_percent
        self.net_key_par = net_key_par
        if max_num_nodes is not None:
            self.max_num_nodes = max_num_nodes
        self.normalize_layer = RowNormalize()
        # 预定义噪音表
        self.noise_schedule = PredefinedNoiseScheduleDiscrete(timesteps=timesteps,
                                                              device=self.device,
                                                              noise=noise_type)
        # 预定义转移矩阵Q
        if transition == 'uniform':
            self.transition_model = DiscreteUniformTransition(e_classes=self.Edim_output)
            e_limit = torch.ones(self.Edim_output) / self.Edim_output
            self.limit_dist = network_preprocess.PlaceHolder(X=None, E=e_limit, y=None)

        elif transition == 'marginal':
            e_marginals = torch.tensor([1 - self.edge_percent,  self.edge_percent], dtype=torch.float32, device=self.device )
            self.transition_model = MarginalUniformTransition(e_marginals=e_marginals)
            self.limit_dist = network_preprocess.PlaceHolder(X=None, E=e_marginals, y=None)

        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

    def noisy_data2data_dense(self, noisy_data, all_edge=True):
        '''
            把加噪数据补全
            :param noisy_data: 加噪数据
            :param all_edge: 是否返回全连接图的边索引
            :return: Data格式的数据
        '''
        data_list = []
        data_list_ALL = []
        for bs_i in range(0, noisy_data['X_t'].shape[0]):
            edge_weights = noisy_data['E_t'][bs_i, :, :, :]
            node_features = noisy_data['X_t'][bs_i, :, :]
            # 生成全连接图的边索引
            num_nodes = node_features.size(0)
            edge_index = utils.dense_to_sparse(torch.ones(num_nodes, num_nodes))[0].to(self.device)
            data_ALL = Data(x=node_features, edge_attr=edge_weights.reshape(-1, edge_weights.shape[2]))
            # 设置边权重属性
            data_ALL.edge_index = edge_index
            # sparse
            edge_weights0 = edge_weights[:, :, 0]
            # 从邻接矩阵中获取边的索引
            indices_tensor = edge_weights0.nonzero(as_tuple=False).t().contiguous()
            num_edges = indices_tensor.shape[1]
            values_tensor = torch.tensor([[1, 0]], dtype=torch.float32).repeat(num_edges, 1)
            data = Data(x=node_features, edge_index=indices_tensor, edge_attr=values_tensor)
            data_list_ALL.append(data_ALL)
            data_list.append(data)

        batch_ALL = Batch.from_data_list(data_list_ALL)
        batch_ALL.edge_index.to(self.device)
        ALLedge_index = batch_ALL.edge_index
        batch = Batch.from_data_list(data_list)
        batch.edge_index.to(self.device)
        if all_edge:
            return batch_ALL, ALLedge_index
        else:
            return batch, ALLedge_index


    def pred_shape2data(self, pred, noisy_data):
        bs, n, _, c = noisy_data['E_t'].shape
        if isinstance(pred, torch.Tensor):
            pred = pred.reshape(bs, n, n, c)
            return network_preprocess.PlaceHolder(X=noisy_data['X_t'], E=pred, y=noisy_data['y_t'])
        else:
            pred.E = pred.E.reshape(bs, n, n, c)
            return network_preprocess.PlaceHolder(X=pred.X, E=pred.E, y=pred.y)


    def p_losses(self, E, noisy_data, loss_method='celoss', calAUC=True):
        """
            E: 一个干净的原始数据
            noisy_data: 加噪数据
            node_mask: 掩码矩阵
        """
        torch.set_grad_enabled(True)
        node_mask = noisy_data['node_mask']
        pred = self.model(noisy_data, self.net_key_par)
        pred = self.pred_shape2data(pred, noisy_data)
        mask_pred_probs_E = pred.mask(node_mask)
        # mask_pred_probs_ES = F.softmax(mask_pred_probs_E.E, dim=-1)
        mask_pred_probs_ES = mask_pred_probs_E.E

        if self.predflag =='E_start':
            target = E
        else:
            target = noisy_data['E_s']
        if loss_method == 'celoss':
            loss = self.train_loss(masked_pred_E=mask_pred_probs_ES, nosoft_pred_E=mask_pred_probs_E.E, true_E=target)
        else:
            loss = self.loss_xtpxt(E, pred.E, noisy_data, node_mask)

        if calAUC:
            # 预测结果：计算Xt-1
            E_pred_onehot = self.given_pred_to_edge(mask_pred_probs_E, noisy_data, node_mask, predflag2='weight')  # 根据预测的x0，计算Xt-1
            E_pred_onehot = network_preprocess.PlaceHolder(X=noisy_data['X_t'], E=E_pred_onehot, y=noisy_data['y_t'])         # (bs,n,n,2)
            E_pred_discrete = E_pred_onehot.mask(node_mask, collapse=True)

            # 真实结果： 计算Xt-1
            if self.predflag =='E_start':
                _, _, E_true_discrete_view = self.vb_terms_bpd(E, pred.E, noisy_data, node_mask)
                E_true_onehot = network_preprocess.PlaceHolder(X=noisy_data['X_t'], E=E_true_discrete_view, y=noisy_data['y_t'])  # (bs,n,n,2)
            else:
                E_true_onehot = network_preprocess.PlaceHolder(X=noisy_data['X_t'], E=noisy_data['E_s'],
                                                               y=noisy_data['y_t'])  # (bs,n,n,2)
            E_true_discrete = E_true_onehot.mask(node_mask, collapse=True)
            AUC_list = []
            for bsi in range(E.shape[0]):
                true1 = E_true_discrete.E[bsi, :, :]
                true1 = true1[:, node_mask[bsi, :]]
                true1 = true1[node_mask[bsi, :], :]
                true1 = true1.flatten()

                pred1 = E_pred_discrete.E[bsi, :, :]
                pred1 = pred1[:, node_mask[bsi, :]]
                pred1 = pred1[node_mask[bsi, :], :]
                pred1 = pred1.flatten()
                performance = util.Evaluation(y_pred=pred1, y_true=true1)
                AUC_list.append(performance['AUC'])
            AUC = np.min(AUC_list)
            return loss, pred, AUC
        else:
            return loss, pred


    def apply_noise(self, X, E, y, node_mask,t_int=None):
        """ Sample noise and apply it to the data. """
        if t_int is None:
            t_int = torch.randint(0, self.num_timesteps, size=(X.shape[0], 1), device=self.device).float()
        # 2. 得到s = t-1
        s_int = t_int - 1
        mask = (s_int == -1)
        s_int[mask] = 0

        # 3.
        t_float = t_int / self.num_timesteps
        s_float = s_int / self.num_timesteps

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_int=t_int)                         # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_int=s_int)      # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=t_int)      # (bs, 1)

        # 计算 t时刻的加噪数据
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        assert (abs(Qtb.E.sum(dim=2) - 1.) < 1e-4).all()                         # 确保概率转移矩阵的总和为1
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)        # Compute transition probabilities
        sampled_t = util.sample_discrete_features(X=X, probE=probE, node_mask=node_mask)   # 采样结果，只显示最终采样结果
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output)
        assert (E.shape == E_t.shape)
        z_t = network_preprocess.PlaceHolder(X=X, E=E_t, y=y).type_as(X).mask(node_mask)  # z_t就是预处理后带噪音的数据

        # 计算 s时刻的加噪数据
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, device=self.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        assert (abs(Qsb.E.sum(dim=2) - 1.) < 1e-4).all()                         # 确保概率转移矩阵的总和为1
        probE_s = E @ Qsb.E.unsqueeze(1)  # (bs, n, n, de_out)        # Compute transition probabilities
        sampled_s = util.sample_discrete_features(X=X, probE=probE_s, node_mask=node_mask)   # 采样结果，只显示最终采样结果
        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output)
        assert (E.shape == E_s.shape)
        z_s = network_preprocess.PlaceHolder(X=X, E=E_s, y=y).type_as(X).mask(node_mask)  # z_s就是预处理后带噪音的数据
        z_s_E = z_s.E
    #    z_s_E[mask.squeeze(), :, :, :] = E[mask.squeeze(), :, :, :]
        noisy_data = {'t_int': t_int, 't': t_float, 's_int': s_int, 's': s_float,'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'E_t': z_t.E, 'E_s': z_s_E, 'y_t': t_int, 'node_mask': node_mask}
        noisy_data['y_t'] = noisy_data['t'].float()
        return noisy_data


    def sum_except_batch(self, x):
        return x.reshape(x.size(0), -1).sum(dim=-1)


    def mask_distributions(self, true_E, pred_E, node_mask):
        # Add a small value everywhere to avoid nans
        pred_E = pred_E + 1e-7
        pred_E = pred_E / torch.sum(pred_E, dim=-1, keepdim=True)

        # Set masked rows to arbitrary distributions, so it doesn't contribute to loss
        row_E = torch.zeros(true_E.size(-1), dtype=torch.float, device=true_E.device)
        row_E[1] = 1.

        diag_mask = ~torch.eye(node_mask.size(1), device=node_mask.device, dtype=torch.bool).unsqueeze(0)
        true_E[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2) * diag_mask), :] = row_E
        pred_E[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2) * diag_mask), :] = row_E

        return true_E, pred_E

    def compute_posterior_distribution(self, M, M_t, Qt_M, Qsb_M, Qtb_M):
        ''' M: X or E
            Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T
        '''
        # Flatten feature tensors
        M = M.flatten(start_dim=1, end_dim=-2).to(torch.float32)  # (bs, N, d) with N = n * n
        M_t = M_t.flatten(start_dim=1, end_dim=-2).to(torch.float32)  # same

        Qt_M_T = torch.transpose(Qt_M, -2, -1)  # (bs, d, d)

        left_term = M_t @ Qt_M_T  # (bs, N, d)
        right_term = M @ Qsb_M  # (bs, N, d)
        product = left_term * right_term  # (bs, N, d)

        denom = M @ Qtb_M  # (bs, N, d) @ (bs, d, d) = (bs, N, d)
        denom = (denom * M_t).sum(dim=-1) +1e-7 # (bs, N, d) * (bs, N, d) + sum = (bs, N)
        prob = product / denom.unsqueeze(-1)  # (bs, N, d)

        return prob


    def posterior_distributions(self, E, E_t, Qt, Qsb, Qtb):
        prob_E = self.compute_posterior_distribution(M=E, M_t=E_t, Qt_M=Qt.E, Qsb_M=Qsb.E, Qtb_M=Qtb.E)  # (bs, n * n, de)
        return network_preprocess.PlaceHolder(X=1, E=prob_E, y=None)


    def vb_terms_bpd(self, E, pred, noisy_data, node_mask):
        bs, n, _, _ = E.shape
        pred_probs_E = F.sigmoid(pred)
        pred_probs_E = pred_probs_E.reshape(bs, n, n, -1)
        pred_probs_E = self.pred_shape2data(pred_probs_E, noisy_data)
        pred_probs_E = pred_probs_E.mask(node_mask).E
        pred_probs_E0 = pred_probs_E
        Qtb = self.transition_model.get_Qt_bar(noisy_data['alpha_t_bar'], self.device)
        Qsb = self.transition_model.get_Qt_bar(noisy_data['alpha_s_bar'], self.device)
        Qt = self.transition_model.get_Qt(noisy_data['beta_t'], self.device)

        prob_true = self.posterior_distributions(E=E,
                                                 E_t=noisy_data['E_t'],
                                                 Qt=Qt,
                                                 Qsb=Qsb,
                                                 Qtb=Qtb)
        prob_true.E = prob_true.E.reshape((bs, n, n, -1))

        prob_pred_t_1 = util.compute_batched_over0_posterior_distribution(X_t=noisy_data['E_t'],
                                                                                     Qt=Qt.E,
                                                                                     Qsb=Qsb.E,
                                                                                     Qtb=Qtb.E)

        pred_probs_E = pred_probs_E.reshape((bs, -1, pred_probs_E.shape[-1]))
        weighted_E = pred_probs_E.unsqueeze(-1) * prob_pred_t_1  # bs, N, d0, d_t-1
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E_final = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E_final = prob_E_final.reshape(bs, n, n, prob_E_final.shape[-1])

        t0mask = (noisy_data['t_int'] == 0).squeeze(1)
        # 如果t==0，那么计算p(x0)~x0的CE，否则计算p(xt-1|xt，x0)
        if torch.any(t0mask):
            prob_E_final[t0mask, ...] = pred_probs_E0[t0mask, ...]
            prob_true.E[t0mask, ...] = E[t0mask, ...]

        prob_true_E, prob_E_final = self.mask_distributions(true_E=prob_true.E, pred_E=prob_E_final, node_mask=node_mask)
        loss = self.train_loss(masked_pred_E=prob_E_final, nosoft_pred_E=prob_E_final, true_E=prob_true_E)

        return loss, prob_E_final, prob_true_E


    def loss_xtpxt(self, E, pred, noisy_data, node_mask):
        vb_losses, pred_x_start_logits,_ = self.vb_terms_bpd(E, pred, noisy_data, node_mask)   # 缺少一个t=0的判断
        ce_losses = self.train_loss(masked_pred_E=pred_x_start_logits, nosoft_pred_E=pred_x_start_logits, true_E=E)
        losses = vb_losses + 0.001 * ce_losses
        return losses


    def forward(self, Multiomics_data, *args, **kwargs):
        """
            1. 稀疏图转为完全图（每个batch的node数量一致,不存在的edge设为[1,0,0,0,0]），生成相应的node_mask
            2. 根据node_mask,再次确认dense_data中的X和E该隐藏的点特征或者边邻接是否设置为0
            3. 最终获得完全图上的 节点特征矩阵X、邻接矩阵E、node_mask  （每个batch的这仨个维度都一样）
        """
        '''将 share 表示制作成输入数据'''
        share_data, node_mask, max_num_nodes = network_preprocess.to_dense(Multiomics_data.x,
                                                                           Multiomics_data.edge_index,
                                                                           Multiomics_data.edge_attr,
                                                                           Multiomics_data.batch,
                                                                           max_num_nodes=self.max_num_nodes)
        if self.max_num_nodes is None:
            self.max_num_nodes = max_num_nodes
        '''将 mRNA 表示制作成输入数据'''
        mRNA_data, _, _ = network_preprocess.to_dense(Multiomics_data.x_1,
                                                      Multiomics_data.edge_index,
                                                      Multiomics_data.edge_attr,
                                                      Multiomics_data.batch,
                                                      max_num_nodes=self.max_num_nodes)

        '''提取 base_GRN 中的边索引和权重，并存入noisy_data以引导Transformer'''
        base_GRN_data, _, _ = network_preprocess.to_dense(Multiomics_data.x,
                                                                           Multiomics_data.base_GRN_edge_index,
                                                                           Multiomics_data.base_GRN_edge_attr,
                                                                           Multiomics_data.batch,
                                                                           max_num_nodes=self.max_num_nodes,
                                                                           discrete=False)

        """根据X、E以及Q，创建计算被污染的数据z_t(X和E)     """
        Multiomics_data.y = torch.empty(Multiomics_data.num_graphs, 0)
        noisy_data = self.apply_noise(share_data.X,
                                      share_data.E,
                                      Multiomics_data.y,
                                      node_mask)
        noisy_data['x_1'] = mRNA_data.X
        noisy_data['base_GRN'] = base_GRN_data.E
        '''将 其余组学的数据也加入 '''
        existing_x_count = sum(1 for key in Multiomics_data.keys() if key.startswith('x'))
        if existing_x_count > 2:
            for omics_i in range(2, existing_x_count):
                other_omics_data, _, _ = network_preprocess.to_dense(Multiomics_data[f'x_{omics_i}'],
                                                              Multiomics_data.edge_index,
                                                              Multiomics_data.edge_attr,
                                                              Multiomics_data.batch,
                                                              max_num_nodes=self.max_num_nodes)
                noisy_data[f'x_{omics_i}'] = other_omics_data.X

        loss, pred, AUC = self.p_losses(share_data.E, noisy_data)
        return loss, AUC


    def given_pred_to_edge(self, pred, noisy_data, node_mask, predflag2 = 'noweight'):
        X_t = noisy_data['X_t']
        bs, n, dxs = X_t.shape

        beta_t = self.noise_schedule(t_int=noisy_data['t_int'])  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_int=noisy_data['s_int'])
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=noisy_data['t_int'])
        # Retrieve transitions matrix
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)

        # Normalize predictions
        # pred_probs_E = F.softmax(pred.E, dim=-1)  # bs, n, n, d0
        pred_probs_E = pred.E
        pred_probs_E_sample = util.sample_discrete_features(X_t, pred_probs_E, node_mask=node_mask,
                                                                       test=True)
        pred_probs_E_sample.E = F.one_hot(pred_probs_E_sample.E, num_classes=self.Edim_output).float()
        pred_probs_E_sample = pred_probs_E_sample.mask(node_mask)
        pred_probs_E = pred_probs_E_sample.E

        # 去掉对角线
        diag_mask = torch.eye(n)  # 创建一个单位矩阵
        diag_mask = ~diag_mask.type_as(pred_probs_E).bool()  # 单位阵的类型=E，并且对角False，其余True
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)  # 掩码矩阵的维度与E一致
        pred_probs_E = pred_probs_E * diag_mask

        if self.predflag == 'E_start':
            if predflag2 == 'weight':
                #  p(s)<-p(t,t0)p(t0)
                prob_pred_t_1 = util.compute_batched_over0_posterior_distribution(X_t=noisy_data['E_t'],
                                                                                             Qt=Qt.E,
                                                                                             Qsb=Qsb.E,
                                                                                             Qtb=Qtb.E)
                # 对edge进行加权
                pred_probs_E_re = pred_probs_E.reshape((bs, -1, pred_probs_E.shape[-1]))  # 512*81*5
                weighted_E = pred_probs_E_re.unsqueeze(-1) * prob_pred_t_1  # bs, N, d0, d_t-1
                # 归一化处理
                unnormalized_prob_E = weighted_E.sum(dim=-2)
                unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
                prob_E_final = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
                prob_E_final = prob_E_final.reshape(bs, n, n, prob_E_final.shape[-1])
                assert ((prob_E_final.sum(dim=-1) - 1).abs() < 1e-4).all()
            else:
                #  p(s)<-p(t,t0)
                pred_probs_E = pred_probs_E.reshape(bs, n, n, -1)
                prob_true = self.posterior_distributions(E=pred_probs_E,
                                                         E_t=noisy_data['E_t'],
                                                         Qt=Qt,
                                                         Qsb=Qsb,
                                                         Qtb=Qtb)
                unnormalized_prob_E = torch.sum(prob_true.E, dim=-1, keepdim=True)
                unnormalized_prob_E[unnormalized_prob_E == 0] = 1e-5
                prob_E_final = prob_true.E / unnormalized_prob_E
                prob_E_final = prob_E_final.reshape(bs, n, n, prob_E_final.shape[-1])
            #    assert ((prob_E_final.sum(dim=-1) - 1).abs() < 1e-3).all()
        else:
            prob_E_final = pred_probs_E / torch.sum(pred_probs_E, dim=-1, keepdim=True)

        # 如果存在t=0，则不预测xt-1，而是直接输出pred
        t0mask = (noisy_data['t_int'] == 0).squeeze(1)
        if torch.any(t0mask):
            prob_E_final_x0 = pred_probs_E / torch.sum(pred_probs_E, dim=-1, keepdim=True)
            prob_E_final[t0mask, ...] = prob_E_final_x0[t0mask, ...]

        sampled_s = util.sample_discrete_features(X_t, prob_E_final, node_mask=node_mask, test=True)

        # 去掉对角线
        diag_mask = torch.eye(n)  # 创建一个单位矩阵
        diag_mask = ~diag_mask.type_as(pred_probs_E).bool()  # 单位阵的类型=E，并且对角False，其余True
        diag_mask = diag_mask.unsqueeze(0)  # 掩码矩阵的维度与E一致
        sampled_s.E = sampled_s.E * diag_mask

        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()
        return E_s

    @torch.no_grad()
    def sample_zt(self, noisy_data):

        """Samples from zs ~ p(zs | zt). Only used during sampling.
                  if last_step, return the graph prediction as well"""

        node_mask = noisy_data['node_mask']
        X_t = noisy_data['X_t']
        # trueE = noisy_data['trueE']
        y_t = noisy_data['y_t']
        # Neural net predictions
        torch.set_grad_enabled(False)
        pred = self.model(noisy_data, self.net_key_par)
        pred = self.pred_shape2data(pred, noisy_data)
        pred1 = pred.mask(node_mask)
        E_s1 = self.given_pred_to_edge(pred1, noisy_data, node_mask, predflag2='weight')

        out_one_hot = network_preprocess.PlaceHolder(X=X_t, E=E_s1, y=pred.y)
        out_discrete = network_preprocess.PlaceHolder(X=X_t, E=E_s1, y=pred.y)

        return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=False).type_as(y_t)


    @torch.no_grad()
    def test_step(self, testdata, TrueData=None, show=True, seed=None):
        """
        :param batch_id: int
        """
        self.model.eval()
        X = testdata.x
        n_nodes = torch.tensor(X.shape[0])         # 采样节点数
        batch_size = torch.tensor(1)      # 采样批次（1次）
        n_max = self.max_num_nodes         # 最大节点数
        # to dense X
        dense_X = torch.zeros([1, n_max, X.shape[1]], device=self.device)
        dense_X[0, :X.shape[0], :] = X

        # Build the masks
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes

        '''将 share 表示制作成输入数据'''
        share_data, _, _ = network_preprocess.to_dense(testdata.x,
                                                       testdata.edge_index,
                                                       testdata.edge_attr,
                                                       testdata.batch,
                                                       max_num_nodes=self.max_num_nodes)

        '''将 mRNA 表示制作成输入数据'''
        mRNA_data, _, _ = network_preprocess.to_dense(testdata.x_1,
                                                      testdata.edge_index,
                                                      testdata.edge_attr,
                                                      testdata.batch,
                                                      max_num_nodes=self.max_num_nodes)

        '''提取 base_GRN 中的边索引和权重，并存入noisy_data以引导Transformer'''
        base_GRN_data, _, _ = network_preprocess.to_dense(testdata.x,
                                                           testdata.base_GRN_edge_index,
                                                           testdata.base_GRN_edge_attr,
                                                           testdata.batch,
                                                           max_num_nodes=self.max_num_nodes,
                                                           discrete=False)
        '''将 其余组学的数据也加入 '''
        existing_x_count = sum(1 for key in testdata.keys() if key.startswith('x'))
        new_dict = {}
        if existing_x_count > 2:
            for omics_i in range(2, existing_x_count):
                other_omics_data, _, _ = network_preprocess.to_dense(testdata[f'x_{omics_i}'],
                                                                     testdata.edge_index,
                                                                     testdata.edge_attr,
                                                                     testdata.batch,
                                                                     max_num_nodes=self.max_num_nodes)
                new_dict[f'x_{omics_i}'] = other_omics_data.X

        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        pre_GRN = None
        # pre_GRN = network_preprocess.cal_GRNBOOST2(testdata.x_1.cpu().detach().numpy())
        z_T = util.sample_discrete_feature_noise(X=share_data.X, limit_dist=self.limit_dist, node_mask=node_mask, seed=seed, pre_GRN=pre_GRN)
        E, y = z_T.E, z_T.y
        y = y.to(E.device)
        all_adj = []
        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        if show:
            pbar2 = tqdm(range(self.num_timesteps - 1, -1, -1), ncols=100)
        else:
            pbar2 = range(self.num_timesteps - 1, -1, -1)
        for t_int in pbar2:
            t_array = t_int * torch.ones((batch_size, 1)).type_as(E)
            if t_int == 0:
                s_array = t_array
            else:
                s_array = t_array - 1
            s_norm = s_array / self.num_timesteps
            t_norm = t_array / self.num_timesteps

            # Sample z_s
            noisy_data = {'X_t': share_data.X, 'trueE': E, 'E_t': E, 'y_t': share_data.y, 's': s_norm, 't': t_norm, 't_int': t_array,
                          's_int': s_array, 'node_mask': node_mask, 'base_GRN': base_GRN_data.E, 'x_1': mRNA_data.X}
            noisy_data['y_t'] = noisy_data['t'].float()
            noisy_data = {**noisy_data, **new_dict}
            sampled_s, sampled_s2 = self.sample_zt(noisy_data)
            E, y = sampled_s.E, sampled_s.y
            sampled_s1 = copy.deepcopy(sampled_s).mask(node_mask, collapse=True)
            sampled_s1.E = sampled_s1.E.squeeze(0)  # 删除批次
            sampled_s1.E = sampled_s1.E[:, node_mask.squeeze(0)]
            sampled_s1.E = sampled_s1.E[node_mask.squeeze(0), :]
            if show:
                performance = util.Evaluation(y_pred=sampled_s1.E[:, :].flatten(),
                                                  y_true=TrueData.flatten())
                pbar2.set_description(f" t = {int(t_array.cpu().numpy()):3.0f} -- AUC:  {performance['AUC']:.4f} -- AUPR:  {performance['AUPR']:.4f} -- Epr:  {performance['Epr']:.4f}")
            adj = sampled_s1.E.cpu().detach().numpy().copy()
            if testdata.y is not None:
                adj = pd.DataFrame(adj, index=testdata.y, columns=testdata.y)
            all_adj.append(adj)
        return sampled_s1.E, all_adj
