import copy
import math

import torch
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn import functional as F
from torch import Tensor
from diffusion_model.discrete import network_preprocess, diffusion_utils
from .layers import Xtoy, Etoy, masked_softmax
from .GraphAttention import GraphAttention_Encode
from ..noise_predefined import PredefinedNoiseScheduleDiscrete


class TransformerLayer(nn.Module):
    def __init__(self, dx, de, dy, dim_ffX, dim_ffE, dim_ffy, dropout=0.3, layer_norm_eps=1e-5, **kwargs):
        super().__init__()
        self.dx = dx
        self.de = de
        self.dy = dy
        self.dim_ffX = dim_ffX
        self.dim_ffE = dim_ffE
        self.dim_ffy = dim_ffy
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps

        self.linX1 = nn.Linear(dx, dim_ffX, **kwargs)
        self.linX2 = nn.Linear(dim_ffX, dx, **kwargs)
        self.normX1 = nn.LayerNorm(dx, eps=layer_norm_eps, **kwargs)
        self.normX2 = nn.LayerNorm(dx, eps=layer_norm_eps, **kwargs)
        self.dropoutX1 = nn.Dropout(dropout)
        self.dropoutX2 = nn.Dropout(dropout)
        self.dropoutX3 = nn.Dropout(dropout)

        self.linE1 = nn.Linear(de, dim_ffE, **kwargs)
        self.linE2 = nn.Linear(dim_ffE, de, **kwargs)
        self.normE1 = nn.LayerNorm(de, eps=layer_norm_eps, **kwargs)
        self.normE2 = nn.LayerNorm(de, eps=layer_norm_eps, **kwargs)
        self.dropoutE1 = nn.Dropout(dropout)
        self.dropoutE2 = nn.Dropout(dropout)
        self.dropoutE3 = nn.Dropout(dropout)

        self.lin_y1 = nn.Linear(dy, dim_ffy, **kwargs)
        self.lin_y2 = nn.Linear(dim_ffy, dy, **kwargs)
        self.norm_y1 = nn.LayerNorm(dy, eps=layer_norm_eps, **kwargs)
        self.norm_y2 = nn.LayerNorm(dy, eps=layer_norm_eps, **kwargs)
        self.dropout_y1 = nn.Dropout(dropout)
        self.dropout_y2 = nn.Dropout(dropout)
        self.dropout_y3 = nn.Dropout(dropout)

        self.activation = F.gelu

    def forward(self, X, E, y, newX, newE, new_y):
        newX_d = self.dropoutX1(newX)
        X = self.normX1(X + newX_d)

        newE_d = self.dropoutE1(newE)
        E = self.normE1(E + newE_d)

        new_y_d = self.dropout_y1(new_y)
        y = self.norm_y1(y + new_y_d)

        ff_outputX = self.linX2(self.dropoutX2(self.activation(self.linX1(X))))
        ff_outputX = self.dropoutX3(ff_outputX)
        X = self.normX2(X + ff_outputX)

        ff_outputE = self.linE2(self.dropoutE2(self.activation(self.linE1(E))))
        ff_outputE = self.dropoutE3(ff_outputE)
        E = self.normE2(E + ff_outputE)

        ff_output_y = self.lin_y2(self.dropout_y2(self.activation(self.lin_y1(y))))
        y = self.norm_y2(y + ff_output_y)

        return X, E, y


class XEyTransformerLayer(nn.Module):
    """ Transformer that updates node, edge and global features
        d_x: node features
        d_e: edge features
        dz : global features
        n_head: the number of heads in the multi_head_attention
        dim_feedforward: the dimension of the feedforward network model after self-attention
        dropout: dropout probablility. 0 to disable
        layer_norm_eps: eps value in layer normalizations.
    """
    def __init__(self, dx: int, de: int, dy: int, n_head: int, dim_ffX: int = 2048,
                 dim_ffE: int = 128, dim_ffy: int = 2048, dropout: float = 0.3,
                 layer_norm_eps: float = 1e-5, device=None, dtype=None) -> None:
        kw = {'device': device, 'dtype': dtype}
        super().__init__()

        # self.self_attn1 = NodeEdgeBlock(dx, de, dy, n_head, **kw)
        self.self_attn2 = NodeEdgeBlock(dx, de, dy, n_head, **kw)
        # self.self_attn3 = NodeEdgeBlock(dx, de, dy, n_head, **kw)

        self.Transformer_layer1 = TransformerLayer(dx, de, dy, dim_ffX, dim_ffE, dim_ffy, dropout=dropout, layer_norm_eps=layer_norm_eps)
        self.Transformer_layer2 = TransformerLayer(dx, de, dy, dim_ffX, dim_ffE, dim_ffy, dropout=dropout, layer_norm_eps=layer_norm_eps)

    def forward(self, X_list, E: Tensor, y, t_init,node_mask: Tensor, base_GRN=None):
        """ Pass the input through the encoder layer.
            X: (bs, n, d)
            E: (bs, n, n, d)
            y: (bs, dy)
            node_mask: (bs, n) Mask for the src keys per batch (optional)
            Output: newX, newE, new_y with the same shape.
        """
        # newX1, newE1, new_y1 = self.self_attn1([X_list[1], X_list[1]], E, y, t_init,
        #                                    node_mask=node_mask,
        #                                    base_GRN=base_GRN)  # Encoder: 1. self-Attention

        newX2, newE2, new_y2 = self.self_attn2([X_list[1], X_list[0]], E, y, t_init,
                                           node_mask=node_mask,
                                           base_GRN=base_GRN)  # Encoder: 1. self-Attention

        # newX3, newE3, new_y3 = self.self_attn3([X_list[0], X_list[0]], E, y, t_init,
        #                                    node_mask=node_mask,
        #                                    base_GRN=base_GRN)  # Encoder: 1. self-Attention

        # g1 = self.gate1(newX1)
        # g2 = self.gate2(newX2)
        # g3 = self.gate3(newX3)
        # newX = g1 * newX1 + g2 * newX2 + g3 * newX3
        # newE = g1.unsqueeze(-2)  * newE1 + g2.unsqueeze(-2)  * newE2 + g3.unsqueeze(-2)  * newE3
        # newX = self.w1 * newX1 + self.w2 * newX2 + self.w2 * newX2
        # newE = self.w1 * newE1 + self.w2 * newE2 + self.w3 * newE3
        X_list[0], E, y = self.Transformer_layer1(X_list[0], E, y, newX2, newE2, new_y2)
        X_list[1], E, y = self.Transformer_layer2(X_list[1], E, y, newX2, newE2, new_y2)
        return X_list, E, y

class NodeEdgeBlock(nn.Module):
    """ Self attention layer that also updates the representations on the edges. """
    def __init__(self, dx, de, dy, n_head, **kwargs):
        super().__init__()
        assert dx % n_head == 0, f"dx: {dx} -- nhead: {n_head}"
        self.dx = dx
        self.de = de
        self.dy = dy
        self.df = int(dx / n_head)
        self.n_head = n_head

        # Attention
        self.q = Linear(dx, dx)
        self.k = Linear(dx, dx)
        self.v = Linear(dx, dx)

        # FiLM E to X
        self.e_add = Linear(de, n_head)
        self.e_mul = Linear(de, n_head)

        # FiLM E to X
        self.base_GRN_add = Linear(de, n_head)
        self.base_GRN_mul = Linear(de, n_head)
        self.e_llm_add = Linear(de, n_head)
        self.e_llm_mul = Linear(de, n_head)

        # FiLM y to E
        self.y_e_mul = Linear(dy, n_head)           # Warning: here it's dx and not de
        self.y_e_add = Linear(dy, n_head)

        # FiLM y to base_GRN
        self.y_base_GRN_mul = Linear(dy, de)           # Warning: here it's dx and not de
        self.y_base_GRN_add = Linear(dy, de)

        # FiLM y to X
        self.y_x_mul = Linear(dy, dx)
        self.y_x_add = Linear(dy, dx)

        # Process y
        self.y_y = Linear(dy, dy)
        self.x_y = Xtoy(dx, dy)
        self.e_y = Etoy(de, dy)

        # Output layers
        self.x_out = Linear(dx, dx)
        self.e_out = Linear(n_head, de)
        self.y_out = nn.Sequential(nn.Linear(dy, dy), nn.GELU(), nn.Linear(dy, dy))


        self.noise_schedule = PredefinedNoiseScheduleDiscrete(timesteps=1000,
                                                              device='cpu',
                                                              noise='cos')

    def forward(self, X_list, E, y, t_init,node_mask, base_GRN=None):
        """
        :param X: bs, n, d        node features
        :param E: bs, n, n, d     edge features
        :param y: bs, dz           global features
        :param node_mask: bs, n
        :return: newX, newE, new_y with the same shape.
        """
        X = X_list[0]
        X_1 = X_list[1]
        bs, n, _ = X.shape
        x_mask = node_mask.unsqueeze(-1)        # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)           # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)           # bs, 1, n, 1

        # 1. 将节点特征X转换为Q,K,V
        Q = self.q(X_1) * x_mask           # (bs, n, dx)
        K = self.k(X) * x_mask           # (bs, n, dx)
        diffusion_utils.assert_correctly_masked(Q, x_mask)
        Q = Q.reshape((Q.size(0), Q.size(1), self.n_head, self.df))  # (bs, n, n_head, df)
        K = K.reshape((K.size(0), K.size(1), self.n_head, self.df))  # (bs, n, n_head, df)

        # 2. 计算自注意力分数
        Y = torch.matmul(Q.permute(0, 2, 1, 3), K.permute(0, 2, 3, 1)) / math.sqrt(Q.size(-1))  # Y为 Q*K 的点积unnormalized结果，(bs, n_head, n, n)
        Y = Y.permute(0, 2, 3, 1)  # (bs, n, n, n_head)
        diffusion_utils.assert_correctly_masked(Y, (e_mask1 * e_mask2))

        # 使用先验边的信息增强自注意力的信息: 先验边 > 自注意力
        E1 = self.e_mul(E) * e_mask1 * e_mask2  # bs, n, n, n_head
        E2 = self.e_add(E) * e_mask1 * e_mask2  # bs, n, n, n_head
        Y = Y * (E1 + 1) + E2  # (bs, n, n, n_head) FiLM(E,Y) = Y .* E1 + Y + E2, where E1 = liner1(E), E2 = liner2(E)

        # 使用时间y的信息增强自注意力信息: 时间y > 自注意力
        ye1 = self.y_e_add(y).unsqueeze(1).unsqueeze(1)   # bs, 1, 1, n_head
        ye2 = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
        newE = ye1 + (ye2 + 1) * Y         # (bs, n, n, n_head) FiLM(y,newE) = newE .* y1 + newE + y2, where E1 = liner1(E), E2 = liner2(E)


        if base_GRN is not None:
            # 使用先验边的信息增强自注意力的信息: 先验边 > 自注意力
            zero_mask = base_GRN[:, :, :, 0] == 0

            base_GRN1 = self.base_GRN_mul(base_GRN) * e_mask1 * e_mask2  # bs, n, n, n_head
            zero_mask1 = zero_mask.unsqueeze(-1).expand(-1, -1, -1, base_GRN1.size(-1))
            base_GRN1 = torch.where(zero_mask1, torch.zeros_like(base_GRN1), base_GRN1)

            base_GRN2 = self.base_GRN_add(base_GRN) * e_mask1 * e_mask2  # bs, n, n, dx
            zero_mask2 = zero_mask.unsqueeze(-1).expand(-1, -1, -1, base_GRN2.size(-1))
            base_GRN2 = torch.where(zero_mask2, torch.zeros_like(base_GRN2), base_GRN2)

            # 刚开始base_GRN 作用比较大，随着t_init变小，其作用也开始慢慢变小，
            self.noise_schedule.alphas = self.noise_schedule.alphas.to(t_init.device)
            t_init_cos = self.noise_schedule.alphas[(t_init.squeeze(-1) * 1000).long()].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            base_GRN_newE = t_init_cos * newE * base_GRN1 + (1 - t_init_cos) * newE + t_init_cos * base_GRN2
            # base_GRN_newE = newE * base_GRN1 + newE + base_GRN1
            newE = torch.where(zero_mask1, newE, base_GRN_newE)

       # 计算 attention*V
        softmax_mask = e_mask2.expand(-1, n, -1, self.n_head)  # bs, 1, n, 1
        attn = masked_softmax(Y, softmax_mask, dim=2)  # bs, n, n, n_head
        V = self.v(X) * x_mask  # bs, n, dx
        V = V.reshape((V.size(0), V.size(1), self.n_head, self.df))  # bs, n, n_head, dx
        weighted_V = torch.matmul(attn.permute(0, 3, 1, 2), V.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)  # (bs, n, n_head, df)
        weighted_V = weighted_V.flatten(start_dim=2)  # (bs, n, dx)

        # 使用时间y的信息增强节点信息: 时间y > 节点
        yx1 = self.y_x_add(y).unsqueeze(1)
        yx2 = self.y_x_mul(y).unsqueeze(1)
        newX = yx1 + (yx2 + 1) * weighted_V

        # 使用节点和边的信息增强时间y信息: 节点+边 > 时间y  PNA
        y = self.y_y(y)
        e_y = self.e_y(E)
        x_y = self.x_y(X)
        new_y = y + x_y + e_y

        # Output X
        newX = self.x_out(newX) * x_mask
        diffusion_utils.assert_correctly_masked(newX, x_mask)

        # Output E
        newE = self.e_out(newE) * e_mask1 * e_mask2      # bs, n, n, de
        diffusion_utils.assert_correctly_masked(newE, e_mask1 * e_mask2)

        # Output y
        new_y = self.y_out(new_y)               # bs, dy

        # 查看维度是否有效
        assert newX.shape == X.shape, f"{newX.shape} != {X.shape}"
        assert newE.shape == E.shape, f"{newE.shape} != {E.shape}"
        assert new_y.shape == y.shape, f"{new_y.shape} != {y.shape}"

        return newX, newE, new_y

class Multimodel_Transformer(nn.Module):
    """
    n_layers : int -- number of layers
    dims : dict -- contains dimensions for each feature type
    """
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict, hidden_dims: dict,
                 output_dims: dict, act_fn_in: nn.GELU(), act_fn_out: nn.GELU()):
        super().__init__()
        self.n_layers = n_layers             # 层的数量
        Multi_omics_len = len(input_dims['X'])

        self.mlp_in_X_list = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dims['X'][i], hidden_mlp_dims['X']),  # 使用 input_dims['X'][i] 作为输入维度
                act_fn_in,nn.Linear(hidden_mlp_dims['X'], hidden_dims['dx']),act_fn_in)
            for i in range(Multi_omics_len)  # 根据 Multi_omics_len 进行循环
        ])
        self.GAT = GraphAttention_Encode(input_dim=input_dims['X'][0], hidden1_dim=128, hidden2_dim=64, hidden3_dim=32,
                                             num_head1=4, num_head2=4, alpha=0.2, reduction='mean')

        self.GAT_base_GRN = GraphAttention_Encode(input_dim=input_dims['X'][1], hidden1_dim=128, hidden2_dim=64, hidden3_dim=32,
                                             num_head1=4, num_head2=4, alpha=0.2, reduction='mean')

        self.mlp_in_E = nn.Sequential(nn.Linear(input_dims['E'], hidden_mlp_dims['E']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['E'], hidden_dims['de']), act_fn_in)

        self.mlp_in_base_GRN = nn.Sequential(nn.Linear(1, hidden_dims['de']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['E'], hidden_dims['de']), act_fn_in)

        self.mlp_in_y = nn.Sequential(nn.Linear(input_dims['y'], hidden_mlp_dims['y']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['y'], hidden_dims['dy']), act_fn_in)

        self.tf_layers = nn.ModuleList([XEyTransformerLayer(dx=hidden_dims['dx'],
                                                            de=hidden_dims['de'],
                                                            dy=hidden_dims['dy'],
                                                            n_head=hidden_dims['n_head'],
                                                            dim_ffX=hidden_dims['dim_ffX'],
                                                            dim_ffE=hidden_dims['dim_ffE'])
                                        for i in range(n_layers)])

        self.mlp_out_E = nn.Sequential(nn.Linear(hidden_dims['de'], hidden_mlp_dims['E']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['E'], output_dims['E']))

        self.mlp_out_y = nn.Sequential(nn.Linear(hidden_dims['dy'], hidden_mlp_dims['y']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['y'], output_dims['y']))


    def forward(self, noisy_data, key_par):
        # 1. 输入数据预处理
        X, E, y, node_mask, base_GRN = self.preprocess_inputs(noisy_data)

        # 2. 分解 GRN
        precise_base_GRN, public_precise_base_GRN, needs_correction_base_GRN = self.decompose_GRN(base_GRN)

        # 3. 修正边特征 E for GAT
        newE = self.correct_edge_features(E, needs_correction_base_GRN, noisy_data['y_t'])

        # 4. 更新图特征 X_list
        X_list = [value for key, value in noisy_data.items() if key.startswith('x')]
        X_list.insert(0, X)
        X_list = self.update_graph_features(X_list, newE, noisy_data['y_t'])

        # 5. 输入特征的多层感知机处理
        X_list = self.process_node_features(X_list, E, y, node_mask)

        # 6. 处理边特征 E 的掩码
        E = self.mask_edge_features(E, X.shape[1])
        after_in = network_preprocess.PlaceHolder(X=X, E=self.mlp_in_E(E),
                                                  y=self.mlp_in_y(y)).mask(node_mask)
        E, y = after_in.E, after_in.y

        # 7. 多层感知机处理模糊base GRN
        needs_correction_base_GRN = self.process_needs_correction_base_GRN(needs_correction_base_GRN, base_GRN)

        # 8. 经过多层 Transformer 层
        for layer in self.tf_layers:
            X_list, E, y = layer(X_list, E, y, noisy_data['y_t'], node_mask, base_GRN=needs_correction_base_GRN)

        # 9. 输出层处理
        X = X_list[0]
        E = self.mlp_out_E(E)
        y = self.mlp_out_y(y)
        E = self.mask_edge_features(E, X.shape[1])  # 再次掩码处理

        pred_probs_E = self.process_output_edges(E, precise_base_GRN, public_precise_base_GRN)

        return network_preprocess.PlaceHolder(X=X, E=pred_probs_E, y=y).mask(node_mask)

    def preprocess_inputs(self, noisy_data):
        X = noisy_data['X_t']
        E = noisy_data['E_t'][:, :, :, 1].unsqueeze(-1)  # 非原地操作，避免修改原始数据
        y = noisy_data['y_t']
        node_mask = noisy_data['node_mask']
        base_GRN = copy.deepcopy(noisy_data['base_GRN'])
        return X, E, y, node_mask, base_GRN

    def decompose_GRN(self, base_GRN):
        precise_base_GRN = torch.where((base_GRN == 1), base_GRN, torch.zeros_like(base_GRN))
        public_precise_base_GRN = torch.where((base_GRN == 3), torch.ones_like(base_GRN), torch.zeros_like(base_GRN))
        needs_correction_base_GRN = torch.where(base_GRN > 0, torch.ones_like(base_GRN), torch.zeros_like(base_GRN))
        return precise_base_GRN, public_precise_base_GRN, needs_correction_base_GRN

    def correct_edge_features(self, E, needs_correction_base_GRN, t_init):
        newE = copy.deepcopy(E)
        for idx, ti in enumerate(t_init):
            newE[idx, :, :, :] = ti.unsqueeze(-1) * needs_correction_base_GRN[idx, :, :, :] + (
                    1 - ti.unsqueeze(-1)) * E[idx, :, :, :]
        return newE

    def update_graph_features(self, X_list, newE, t_init):
        X_list[0] = self.GAT(X_list[0], newE, 1 - t_init)
        X_list[1] = self.GAT_base_GRN(X_list[1], newE, 1 - t_init)
        return X_list

    def process_node_features(self, X_list, E, y, node_mask):
        for idx, x1 in enumerate(X_list):
            after_in = network_preprocess.PlaceHolder(X=self.mlp_in_X_list[idx](X_list[idx]), E=E, y=y).mask(node_mask)
            X_list[idx] = after_in.X
        return X_list

    def mask_edge_features(self, E, node_size):
        diag_mask = torch.eye(node_size).type_as(E).bool().unsqueeze(0).unsqueeze(-1).expand(E.shape[0], -1, -1, -1)
        E = torch.where(diag_mask, torch.zeros_like(E), E)

        return E

    def process_needs_correction_base_GRN(self, needs_correction_base_GRN, base_GRN):
        if torch.sum(base_GRN) != 0:
            zero_mask = needs_correction_base_GRN == 0
            needs_correction_base_GRN = self.mlp_in_base_GRN(needs_correction_base_GRN)
            extended_mask = zero_mask.expand(-1, -1, -1, needs_correction_base_GRN.size(-1))
            needs_correction_base_GRN = torch.where(extended_mask, torch.zeros_like(needs_correction_base_GRN),
                                                    needs_correction_base_GRN)
        else:
            needs_correction_base_GRN = None
        return needs_correction_base_GRN

    def compute_attention_weights(self, E, attention_scores, precise_base_GRN):
        attention_weights = torch.sigmoid(attention_scores)  # 使用 sigmoid 限制在 [0, 1] 区间
        E = E * attention_weights + (1 - attention_weights) * precise_base_GRN
        return E

    def process_output_edges(self, E, precise_base_GRN, public_precise_base_GRN):

        pred_probs_E = F.softmax(E, dim=-1)  # 计算 softmax 输出

        # 根据置信度动态调整 precise_base_GRN 的权重影响
        base_GRN = F.one_hot(precise_base_GRN.squeeze(-1).to(torch.int64), num_classes=E.shape[-1])
        one_mask = precise_base_GRN == 1
        extended_mask = one_mask.expand(-1, -1, -1, pred_probs_E.size(-1))
        replacement = torch.tensor([0, 1]).expand_as(base_GRN).to(pred_probs_E.device)

        public_base_GRN = F.one_hot(public_precise_base_GRN.squeeze(-1).to(torch.int64), num_classes=E.shape[-1])
        public_one_mask = public_precise_base_GRN == 1
        public_extended_mask = public_one_mask.expand(-1, -1, -1, pred_probs_E.size(-1))
        public_replacement = torch.tensor([0, 1]).expand_as(public_base_GRN).to(pred_probs_E.device)

        # 将置信度低的边进行动态替换
        confidence_threshold = 0.1    # 设定置信度阈值
        low_confidence_mask = pred_probs_E < confidence_threshold

        pred_probs_E = torch.where(public_extended_mask & ~low_confidence_mask, public_replacement, pred_probs_E)
        pred_probs_E = torch.where(extended_mask, replacement, pred_probs_E)

        return pred_probs_E

