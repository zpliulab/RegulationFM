import torch
import torch.nn as nn
import torch.nn.functional as F
from ..noise_predefined import PredefinedNoiseScheduleDiscrete



class GraphAttention_Encode(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden1_dim,
                 hidden2_dim,
                 hidden3_dim,
                 num_head1,
                 num_head2,
                 alpha,
                 reduction):
        super(GraphAttention_Encode, self).__init__()
        self.num_head1 = num_head1
        self.num_head2 = num_head2
        self.alpha = alpha
        self.type = type
        self.reduction = reduction

        if self.reduction == 'mean':
            self.hidden1_dim = hidden1_dim
            self.hidden2_dim = hidden2_dim
        elif self.reduction == 'concate':
            self.hidden1_dim = num_head1*hidden1_dim
            self.hidden2_dim = num_head2*hidden2_dim

        # 创建多头Attention层
        self.ConvLayer1 = [AttentionLayer(input_dim, hidden1_dim, alpha) for _ in range(num_head1)]  # _（占位符），表示忽略
        for i, attention in enumerate(self.ConvLayer1):
            self.add_module('ConvLayer1_AttentionHead{}'.format(i), attention)

        # 创建多头Attention层
        self.ConvLayer2 = [AttentionLayer(self.hidden1_dim, hidden2_dim, alpha) for _ in range(num_head2)]
        for i, attention in enumerate(self.ConvLayer2):
            self.add_module('ConvLayer2_AttentionHead{}'.format(i),attention)

        self.tf_linear1 = nn.Linear(hidden2_dim, hidden3_dim)
        self.tf_linear2 = nn.Linear(hidden3_dim, input_dim)
        self.time_linear = nn.Linear(1, 1)
        self.Linear_cat = nn.Linear(input_dim*2, input_dim)
        self.reset_parameters()

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(timesteps=1000,
                                                              device='cpu',
                                                              noise='cos')
    def reset_parameters(self):
        for attention in self.ConvLayer1:
            attention.reset_parameters()

        for attention in self.ConvLayer2:
            attention.reset_parameters()

    def encode(self, x, adj):
        if self.reduction =='concate':
            x = torch.cat([att(x, adj) for att in self.ConvLayer1], dim=-1)
            x = F.elu(x)
        elif self.reduction =='mean':
            x = torch.mean(torch.stack([att(x, adj) for att in self.ConvLayer1]), dim=0)
            x = F.elu(x)
        else:
            raise TypeError
        out = torch.mean(torch.stack([att(x, adj) for att in self.ConvLayer2]), dim=0)
        return out

    def forward(self, x, adj, t=None):
        embed = self.encode(x,adj)
        tf_embed = self.tf_linear1(embed)
        tf_embed = F.gelu(tf_embed)
        tf_embed = F.dropout(tf_embed, p=0.1)
        tf_embed = self.tf_linear2(tf_embed)
        tf_embed = F.gelu(tf_embed)
        if t is not None:
            t_para = self.time_linear(t)
            t_para = torch.sigmoid(t_para)
            t_para = t_para.unsqueeze(-1)
            tf_embed = (1 - t_para) * tf_embed + t_para * x
        out_embed = self.Linear_cat(torch.cat([x, tf_embed], dim=-1))  # 对基因特征进行处理
        return out_embed

class AttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, alpha=0.2,bias=True):
        super(AttentionLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha

        self.weight = nn.Parameter(torch.FloatTensor(self.input_dim, self.output_dim))
        self.weight_interact = nn.Parameter(torch.FloatTensor(self.input_dim,self.output_dim))
        self.a = nn.Parameter(torch.zeros(size=(2*self.output_dim,1)))     # nn.Parameter表示该张量是一个可学习的模型参数

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(self.output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.weight_interact.data, gain=1.414)
        if self.bias is not None:
            self.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def _prepare_attentional_mechanism_input(self, x):

        Wh1 = torch.matmul(x, self.a[:self.output_dim, :])   # (0 ：output-1)行 * X
        Wh2 = torch.matmul(x, self.a[self.output_dim:, :])   # (output ：end)行 * X
        Wh2 = Wh2.permute(0, 2, 1)
        # e = F.leaky_relu(Wh1 + Wh2,negative_slope=self.alpha)
        e = F.gelu(Wh1 + Wh2)
        return e

    def forward(self, x, adj):
        h = torch.matmul(x, self.weight)
        e = self._prepare_attentional_mechanism_input(h)
        adj = adj.squeeze(-1)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj.to_dense() > 0, e, zero_vec)    # to_dense  = 稀疏2密集
        attention = F.softmax(attention, dim=1)

        attention = F.dropout(attention, p=0.1, training=self.training)
        output_data = torch.matmul(attention, h)
        # output_data = F.leaky_relu(output_data, negative_slope=self.alpha)
        output_data = F.gelu(output_data)
        output_data = F.normalize(output_data, p=2, dim=1)

        if self.bias is not None:
            output_data = output_data + self.bias

        return output_data


if __name__ == '__main__':
    def generate_synthetic_data(num_nodes, input_dim):
        # 随机生成节点特征
        x = torch.rand(num_nodes, input_dim)
        # 生成简单的邻接矩阵，这里我们生成一个简单的链式结构
        edge_index = torch.tensor([[i, i + 1] for i in range(num_nodes - 1)], dtype=torch.long).t().contiguous()
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
        adj[edge_index[0], edge_index[1]] = 1
        adj[edge_index[1], edge_index[0]] = 1  # 使其无向
        return x, adj


    # 假设参数
    input_dim = 30
    hidden1_dim = 128
    hidden2_dim = 64
    hidden3_dim = 64
    num_head1 = 3
    num_head2 = 3
    alpha = 0.2
    device = 'cpu'
    reduction = 'concate'

    # 创建模型实例
    model = GraphAttention_Encode(input_dim=30, hidden1_dim=128, hidden2_dim=64, hidden3_dim=64,
                                  num_head1=3, num_head2=3, alpha=0.2, reduction='mean')

    # 生成数据
    num_nodes = 10
    x, adj = generate_synthetic_data(num_nodes, input_dim)

    # 测试模型
    embed = model(x, adj)
    print("Embedding output shape:", embed.shape)
