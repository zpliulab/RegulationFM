from torch_geometric.utils import to_dense_adj, to_dense_batch
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import sys
from contextlib import contextmanager
@contextmanager
def suppress_output():
    """
    上下文管理器，用于临时屏蔽标准输出和标准错误。
    """
    # 保存原始的文件描述符
    original_stdout_fd = sys.stdout.fileno()
    original_stderr_fd = sys.stderr.fileno()

    # 打开空设备
    with open(os.devnull, 'w') as devnull:
        # 重定向 stdout 和 stderr
        os.dup2(devnull.fileno(), original_stdout_fd)
        os.dup2(devnull.fileno(), original_stderr_fd)
        try:
            yield
        finally:
            # 恢复原始的 stdout 和 stderr
            # 首先需要重新打开原始文件描述符
            with open(os.devnull, 'r') as f:
                os.dup2(original_stdout_fd, original_stdout_fd)
                os.dup2(original_stderr_fd, original_stderr_fd)


def normalize(X, E, y, norm_values, norm_biases, node_mask):
    E = (E - norm_biases[1]) / norm_values[1]
    y = (y - norm_biases[2]) / norm_values[2]

    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask)


def unnormalize(X, E, y, norm_values, norm_biases, node_mask, collapse=False):
    """
    X : node features
    E : edge features
    y : global features`
    norm_values : [norm value X, norm value E, norm value y]
    norm_biases : same order
    node_mask
    """
    X = (X * norm_values[0] + norm_biases[0])
    E = (E * norm_values[1] + norm_biases[1])
    y = y * norm_values[2] + norm_biases[2]

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask, collapse)


def encode_no_edge(E):
    assert len(E.shape) == 4
    if E.shape[-1] == 0:
        return E
    no_edge = torch.sum(E, dim=3) == 0   # 边特征为0，表示没有这条边，维度为E(0:2)，即batch*edge*edge
    first_elt = E[:, :, :, 1]
    first_elt[no_edge] = 1     # E本身是非0-1的邻接矩阵，这里是把不存在的边的特征修改为 [0,1]  其余的边为[1,0]这种one-hot编码的特征
    E[:, :, :, 1] = first_elt  # [1,0]的为不存在的边，其余的为存在
    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0
    return E


def get_max_node(data):
    X, _ = to_dense_batch(x=data.x, batch=data.batch)
    max_num_nodes = X.size(1)
    return max_num_nodes


def to_dense(x, edge_index, edge_attr, batch=None, training=True, max_num_nodes=None, discrete=True):
    """
     -- to_dense_batch是将每个batch中的节点数量转化为相同大小，输出大小为  batch * Max_X * X_attr = 512*9*4
         比如有[1个节点特征,3个节点特征，2个节点特征]三个batch，那么经过转换变成
         [3个节点特征,3个节点特征，3个节点特征]（这里缺失值补0），同时生成node_mask矩阵,[[True,False,False], [True,True,True], [True,True,False]]
     -- to_dense_adj是将每个batch中的稀疏边转化为具有相同相大小的邻接矩阵，输出大小为  batch * Max_Edge * Max_Edge * Edge_attr = 512*9*9*5
    """
    if training:
        X, node_mask = to_dense_batch(x=x, batch=batch, max_num_nodes=max_num_nodes)
       # max_num_nodes = X.size(1)
    else:
        X = x
        node_mask = None
    # node_mask = node_mask.float()
   # edge_index, edge_attr = torch_geometric.utils.remove_self_loops(edge_index, edge_attr)  # 移除自环边
    # TODO: carefully check if setting node_mask as a bool breaks the continuous case

    E = to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr.reshape(-1), max_num_nodes=max_num_nodes)
    E_thre = (E > 0).int()
    E_onehot = F.one_hot(E_thre.to(torch.int64), num_classes=2).float()
    if torch.any((E > 0) & (E < 1)):
        mask = E_onehot[..., 1].bool()
        E_onehot[..., 1] = torch.where(mask, E, E_onehot[..., 1])
   # E = encode_no_edge(E)
    if not discrete:
        E_onehot = E.unsqueeze(-1)
    return PlaceHolder(X=X, E=E_onehot, y=None).mask(node_mask), node_mask, max_num_nodes


def cal_GRNBOOST2(data_mRNA, data_gene_id=None):
    if data_gene_id is None:
        data_gene_id = [f"G{i}" for i in range(0, data_mRNA.shape[0])]
    from arboreto.algo import grnboost2
    mean = np.mean(data_mRNA)
    std = np.std(data_mRNA)
    data_mRNA = (data_mRNA - mean) / std
    dat1 = pd.DataFrame(np.transpose(data_mRNA), columns=data_gene_id)
    # with suppress_output():
    network = grnboost2(expression_data=dat1, gene_names=data_gene_id)
    adj_matrix = pd.DataFrame(np.zeros((len(data_gene_id), len(data_gene_id))),
                              index=data_gene_id, columns=data_gene_id)
    # 3-col to adj
    for _, row in network.iterrows():
        source = row['TF']
        target = row['target']
        weight = row['importance']
        adj_matrix.loc[source, target] = weight
    adj_matrix = adj_matrix.to_numpy()
    return adj_matrix


class PlaceHolder:
    def __init__(self, X, E, y):
        self.X = X
        self.E = E
        self.y = y

    def type_as(self, x: torch.Tensor):
        """ Changes the device and dtype of X, E, y. """
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        if self.y is not None:
            self.y = self.y.type_as(x)
        return self

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1

        if collapse:
            self.E = torch.argmax(self.E, dim=-1)  # 沿着这个维度找到具有最大值的索引
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1 #这些边被视为无效或不存在。
        else:
            self.X = self.X * x_mask             # 保留有效节点
            self.E = self.E * e_mask1 * e_mask2  # 保留有效边
           # assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))  # 保证对称性
        return self

