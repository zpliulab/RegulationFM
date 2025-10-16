import torch
from torch.nn import functional as F
import numpy as np
import math
from .network_preprocess import PlaceHolder
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc, f1_score, roc_curve
from torchmetrics import Metric, MeanSquaredError
from torch import Tensor
import pandas as pd
import os
import networkx as nx

class CrossEntropyMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_ce', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """ Update state with predictions and targets.
            preds: Predictions from model   (bs * n, d) or (bs * n * n, d)
            target: Ground truth values     (bs * n, d) or (bs * n * n, d). """
        target = torch.argmax(target, dim=-1)
        output = F.cross_entropy(preds, target, reduction='sum')
        self.total_ce += output
        self.total_samples += preds.size(0)

    def compute(self):
        return self.total_ce / self.total_samples


class SumExceptBatchMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_value', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, values) -> None:
        self.total_value += torch.sum(values)
        self.total_samples += values.shape[0]

    def compute(self):
        return self.total_value / self.total_samples


class SumExceptBatchKL(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_value', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, p, q) -> None:
        self.total_value += F.kl_div(q, p, reduction='sum')
        self.total_samples += p.size(0)

    def compute(self):
        return self.total_value / self.total_samples


class NLL(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total_nll', default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('total_samples', default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, batch_nll) -> None:
        self.total_nll += torch.sum(batch_nll)
        self.total_samples += batch_nll.numel()

    def compute(self):
        return self.total_nll / self.total_samples


def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)


def cosine_beta_schedule_discrete(timesteps, s=0.008):
    """ Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ. """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = 1 - alphas
    return betas.squeeze()


def get_diffusion_betas(spec):
    """Get betas from the hyperparameters."""
    if spec['type'] == 'linear':
        # Used by Ho et al. for DDPM, https://arxiv.org/abs/2006.11239.
        # To be used with Gaussian diffusion models in continuous and discrete
        # state spaces.
        # To be used with transition_mat_type = 'gaussian'
        return np.linspace(spec['start'], spec['stop'], spec['num_timesteps'])
    elif spec['type'] == 'cosine':
        # Schedule proposed by Hoogeboom et al. https://arxiv.org/abs/2102.05379
        # To be used with transition_mat_type = 'uniform'.
        steps = (
            np.arange(spec['num_timesteps'] + 1, dtype=np.float64) /
            spec['num_timesteps'])
        alpha_bar = np.cos((steps + 0.008) / 1.008 * np.pi / 2)
        betas = np.minimum(1 - alpha_bar[1:] / alpha_bar[:-1], 0.999)
        return betas
    elif spec['type'] == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        # Proposed by Sohl-Dickstein et al., https://arxiv.org/abs/1503.03585
        # To be used with absorbing state models.
        # ensures that the probability of decaying to the absorbing state
        # increases linearly over time, and is 1 for t = T-1 (the final time).
        # To be used with transition_mat_type = 'absorbing'
        return 1. / np.linspace(spec['num_timesteps'], 1., spec['num_timesteps'])
    else:
        raise NotImplementedError(spec.type)


def custom_beta_schedule_discreteDig(timesteps, average_num_nodes=200, s=0.008):
    """ Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ. """
    steps = timesteps+1
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = 1 - alphas

    # assert timesteps >= 100
    #
    # p = 1 / 2       # 1 - 1 / num_edge_classes
    # num_edges = average_num_nodes * (average_num_nodes - 1) / 2
    #
    # # First 100 steps: only a few updates per graph
    # updates_per_graph = 1.2
    # beta_first = updates_per_graph / (p * num_edges)
    #
    # betas[betas < beta_first] = beta_first
    return np.array(betas)


def sample_discrete_max(X, probE, node_mask, test=False):
    ''' Sample features from multinomial distribution with given probabilities (probX, probE, proby)
        :param probX: bs, n, dx_out        node features
        :param probE: bs, n, n, de_out     edge features
        :param proby: bs, dy_out           global features.
    '''
    bs, n, _ = X.shape
    # Noise E
    # The masked rows should define probability distributions as well
    inverse_edge_mask = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
    diag_mask = torch.eye(n).unsqueeze(0).expand(bs, -1, -1)

    probE[inverse_edge_mask] = 1 / probE.shape[-1]
    probE[diag_mask.bool()] = 1 / probE.shape[-1]
    E_t = torch.argmax(probE, dim=-1)
    E_t = delte_dig_from_batch(E_t)
    return PlaceHolder(X=X, E=E_t, y=None)


def sample_discrete_features(X, probE, node_mask, randomseed=42, test=True):
    ''' Sample features from multinomial distribution with given probabilities (probX, probE, proby)
        :param probX: bs, n, dx_out        node features
        :param probE: bs, n, n, de_out     edge features
        :param proby: bs, dy_out           global features.
    '''
    bs, n, _ = X.shape
    # Noise E
    # The masked rows should define probability distributions as well
    inverse_edge_mask = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
    diag_mask = torch.eye(n).unsqueeze(0).expand(bs, -1, -1)

    probE[inverse_edge_mask] = 1 / probE.shape[-1]
    probE[diag_mask.bool()] = 1 / probE.shape[-1]
    probE = probE.reshape(bs * n * n, -1)    # (bs * n * n, de_out)

    # Sample E
    # temperature = 0.7  # 调整温度参数，<1使分布更尖锐，>1更平滑   这个还没试  可以试试，解决1比较少的问题
    # probE_adjusted = F.softmax(probE / temperature, dim=-1)
    E_t = torch.multinomial(probE, num_samples=1, replacement=True).reshape(bs, n, n)  # (bs, n, n)
    E_t = delte_dig_from_batch(E_t)

    return PlaceHolder(X=X, E=E_t, y=None)


def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask.long())).abs().max().item() < 1e-4, \
        'Variables not masked properly.'


def top_evaluate(truth_edges, A):
    truth_edges_idx = np.where(truth_edges)[0]
    num_nodes = A.shape[0]
    num_truth_edges = len(truth_edges_idx)
    A_val = list(np.sort(abs(A), 0))
    A_val.reverse()
    cutoff_all = A_val[num_truth_edges]
    A_indicator_all = np.zeros([num_nodes])
    A_indicator_all[abs(A) > cutoff_all] = 1
    A_edges_idx = np.where(A_indicator_all)[0]
    overlap_A = set(A_edges_idx).intersection(set(truth_edges_idx))
    return len(overlap_A), 1. * len(overlap_A) / ((num_truth_edges ** 2) / num_nodes)

def Evaluation(y_true, y_pred, flag=False):
    y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().detach().numpy()
    else:
        y_pred = np.array(y_pred)

    CC = []
    EC = []
    if flag:
        G = nx.from_numpy_array(y_pred)
        CC = nx.closeness_centrality(G)
        EC = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06)

    y_p = y_pred.flatten()
    y_t = y_true.flatten().astype(int)
    # print(np.count_nonzero(y_t), ' - ', np.count_nonzero(y_p))
    AUC = roc_auc_score(y_true=y_t, y_score=y_p, average=None)
    Ep, Epr = top_evaluate(y_t, y_p)
    fpr, tpr, thresholds = roc_curve(y_true=y_t, y_score=y_p)
    best_threshold_index = np.argmax(tpr - fpr)
    best_threshold = thresholds[best_threshold_index]

    precision, recall, thresholds = precision_recall_curve(y_true=y_t, probas_pred=y_p)
    AUPR = auc(recall, precision)
    AUPR_norm = AUPR/np.mean(y_t)

    y_p[y_p > best_threshold] = 1
    y_p[y_p != 1] = 0
    f1 = f1_score(y_t, y_p)
    return {'AUC': AUC, 'AUPR': AUPR, 'AUPR_norm': AUPR_norm,
            'F1': f1, 'Ep': Ep,  'Epr': Epr, 'CC': CC, 'EC': EC}


def find_denominator_element(df, threshold=0.3):
    """
    找到第一个使 percent > threshold 的分母元素值。

    参数:
    - df: pandas DataFrame
    - column: 需要分析的列名
    - threshold: 阈值，默认为0.3

    返回:
    - 分母的元素值，如果不存在则返回 None
    """
    all_values = df.values.flatten()
    series = pd.Series(all_values)

    # Step 1: 统计每个元素出现的次数
    counts = series.value_counts()

    # Step 2: 按元素值降序排序
    counts = counts.sort_index(ascending=False)

    # Step 3: 计算累加和（累积频数）
    cum_sum = counts.cumsum()

    # Step 4: 创建一个包含元素、频数和累加和的 DataFrame
    df_counts = counts.reset_index()
    df_counts.columns = ['element', 'count']
    df_counts['cum_sum'] = cum_sum.values

    # Step 5: 计算 percent = 当前累加和 / 前一个累加和 - 1
    df_counts['percent'] = df_counts['cum_sum'] / df_counts['cum_sum'].shift(1) - 1

    # Step 6: 找到第一个 percent > threshold 的位置
    condition = df_counts['percent'] > threshold

    if condition.any():
        first_idx = df_counts[condition].index[0]
        denominator_element = df_counts.loc[first_idx - 1, 'element']
        return denominator_element
    else:
        return 10


def compute_batched_over0_posterior_distribution(X_t, Qt, Qsb, Qtb):
    """ M: X or E
        Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T for each possible value of x0
        X_t: bs, n, dt          or bs, n, n, dt
        Qt: bs, d_t-1, dt
        Qsb: bs, d0, d_t-1
        Qtb: bs, d0, dt.
    """
    # Flatten feature tensors
    # Careful with this line. It does nothing if X is a node feature. If X is an edge features it maps to
    # bs x (n ** 2) x d 这行其实是针对边特征来说的，输入边的特征为 512*9*9*5，转换后为512*81*4
    X_t = X_t.flatten(start_dim=1, end_dim=-2).to(torch.float32)            # bs x N x dt

    Qt_T = Qt.transpose(-1, -2)                 # bs, dt, d_t-1
    left_term = X_t @ Qt_T                      # bs, N, d_t-1
    left_term = left_term.unsqueeze(dim=2)      # bs, N, 1, d_t-1

    right_term = Qsb.unsqueeze(1)               # bs, 1, d0, d_t-1
    numerator = left_term * right_term          # bs, N, d0, d_t-1

    X_t_transposed = X_t.transpose(-1, -2)      # bs, dt, N

    prod = Qtb @ X_t_transposed                 # bs, d0, N
    prod = prod.transpose(-1, -2)               # bs, N, d0
    denominator = prod.unsqueeze(-1)            # bs, N, d0, 1
    denominator[denominator == 0] = 1e-6

    out = numerator / denominator
    return out


def sample_discrete_feature_noise(X, limit_dist, node_mask, seed=42, pre_GRN=None):
    """ Sample from the limit distribution of the diffusion process"""
    bs, n_max = node_mask.shape

    if pre_GRN is not None:
        pre_GRN = torch.tensor(pre_GRN, device=X.device, dtype=torch.float32) .unsqueeze(0)
        expanded_tensor = torch.zeros((1, n_max, n_max), device=X.device, dtype=torch.float32)  # 初始化为全0的 1x306x306
        expanded_tensor[:, :pre_GRN.shape[1], :pre_GRN.shape[1]] = pre_GRN   # 将原始的部分填充
        min_val = expanded_tensor.min()
        max_val = expanded_tensor.max()
        if max_val - min_val != 0:
            normalized_tensor = (expanded_tensor - min_val) / (max_val - min_val)
            normalized_tensor = normalized_tensor * 0.6 + 0.2
        else:
            normalized_tensor = torch.full_like(
                expanded_tensor, 0.2, device=expanded_tensor.device, dtype=torch.float32
            )
        e_limit = torch.zeros((1, n_max, n_max, 2), device=X.device, dtype=torch.float32)
        e_limit[:, :, :, 1] = normalized_tensor  # [:,:,:,1] 是原始Tensor的值
        e_limit[:, :, :, 0] = 1 - normalized_tensor  # [:,:,:,0] 是1-原始Tensor的值
    else:
        e_limit = limit_dist.E[None, None, None, :].expand(bs, n_max, n_max, -1)

    if seed is not None:
        torch.manual_seed(seed)
        U_E = e_limit.flatten(end_dim=-2).multinomial(num_samples=1).reshape(bs, n_max, n_max)
        #torch.seed()
    else:
        U_E = e_limit.flatten(end_dim=-2).multinomial(num_samples=1).reshape(bs, n_max, n_max)
    U_E = delte_dig_from_batch(U_E)
    long_mask = node_mask.long()
    U_E = U_E.type_as(long_mask)
    U_E = F.one_hot(U_E, num_classes=e_limit.shape[-1]).float()
    y = torch.empty(1, 0).type_as(U_E)
    return PlaceHolder(X=X, E=U_E, y=y).mask(node_mask)


def delte_dig_from_batch(batchdata):
    for i in range(0, batchdata.shape[0]):
        a = batchdata[i,:,:]
        a = a.fill_diagonal_(1)
        batchdata[i, :, :] = a
    return batchdata


def cal_identify_TF_gene(GeneName, TF_file = '/home/wcy/RegulationGPT/diffusion_model/Building_training_dataset/Gene_TF_list/TF.txt'):
    TF_list = pd.read_csv(TF_file, sep='\t')
    GeneName = pd.Series(GeneName)
    TF_list = TF_list[TF_list['Symbol'].isin(GeneName)]['Symbol']
    TF_positions = GeneName[GeneName.isin(TF_list)]
    original_list = list(range(GeneName.shape[0]))
    GENE_ID_list = [x for x in original_list if x not in TF_positions.index]
    TF_ID_list = list(TF_positions.index)
    return GENE_ID_list, TF_ID_list