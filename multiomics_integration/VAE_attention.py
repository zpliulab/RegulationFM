import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossAttention(nn.Module):
    def __init__(self, latent_dim):
        super(CrossAttention, self).__init__()
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)
        self.layer_norm = nn.LayerNorm(latent_dim)  # 添加 LayerNorm

    def forward(self, z_q, z_k, z_v):
        Q = self.query(z_q)
        K = self.key(z_k)
        V = self.value(z_v)

        attention_weights = F.softmax(
            torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(K.size(-1), dtype=torch.float32)), dim=-1)
        z_fused = torch.matmul(attention_weights, V)
        z_fused = self.layer_norm(z_fused)
        return z_fused


# 层级式CrossAttention模块
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, latent_dim, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        assert latent_dim % num_heads == 0, "latent_dim must be divisible by num_heads"
        self.head_dim = latent_dim // num_heads

        # 多头的线性变换
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)

        # 头的输出拼接后的线性变换
        self.fc_out = nn.Linear(latent_dim, latent_dim)

    def forward(self, z_q, z_k, z_v):
        batch_size = z_q.shape[0]

        # 将query, key, value进行多头拆分
        Q = self.query(z_q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(z_k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(z_v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attention_weights = F.softmax(
            torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim), dim=-1
        )

        # 加权求和得到注意力输出
        z_fused = torch.matmul(attention_weights, V)

        # 将多头输出拼接
        z_fused = z_fused.transpose(1, 2).contiguous().view(batch_size, -1, self.latent_dim)

        # 通过线性层得到最终输出
        out = self.fc_out(z_fused)
        return out, attention_weights

class AttentionFusion(torch.nn.Module):
    def __init__(self, z_dim, latent_dim):
        super(AttentionFusion, self).__init__()
        # self.attention = torch.nn.MultiheadAttention(embed_dim=z_dim, num_heads=1)
        self.joint_linear = nn.Sequential(
            nn.Linear(z_dim, latent_dim*2),
            nn.LeakyReLU(),
            nn.Linear(latent_dim*2, latent_dim)  # 输出均值和对数方差
        )

        transformer_layer = nn.TransformerEncoderLayer(d_model=z_dim, nhead=4)  # 4头注意力
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=2)  # 6层Transformer Encoder

    def forward(self, primary_z, aux_z_list):
        z_fused = torch.cat([primary_z] + aux_z_list, dim=-1)
        joint_z = self.transformer_encoder(z_fused)
        # 注意力机制可以根据输入对不同模态进行加权融合
        # joint_z, _ = self.attention(z_fused, z_fused, z_fused)
        joint_z = self.joint_linear(joint_z)
        return joint_z


# 多模态VAE模型
class MultimodalVAE(nn.Module):
    def __init__(self, input_dims, hidden_dim, latent_dim):
        super(MultimodalVAE, self).__init__()
        self.num_modalities = len(input_dims)

        # 编码器
        self.encoders = nn.ModuleList([
            self._build_encoder(input_dim, hidden_dim, latent_dim)
            for input_dim in input_dims
        ])

        # Cross-Attention模块
        # self.cross_attentions = nn.ModuleList([
        #     CrossAttention(latent_dim) for _ in range(1, self.num_modalities)
        # ])
        self.cross_attentions_mu = nn.ModuleList([
            MultiHeadCrossAttention(latent_dim, num_heads=4) for _ in range(1, self.num_modalities)
        ])
        self.cross_attentions_log = nn.ModuleList([
            MultiHeadCrossAttention(latent_dim, num_heads=4) for _ in range(1, self.num_modalities)
        ])
        self.transformer_fusion_mu = AttentionFusion(z_dim=latent_dim*len(input_dims), latent_dim=latent_dim)
        self.transformer_fusion_log = AttentionFusion(z_dim=latent_dim*len(input_dims), latent_dim=latent_dim)

        # 解码器
        # self.decoders = nn.ModuleList([
        #     self._build_decoder(latent_dim, hidden_dim, input_dim)
        #     for input_dim in input_dims
        # ])
        self.decoders_joint = nn.ModuleList([
            self._build_decoder(latent_dim, hidden_dim, input_dim)
            for input_dim in input_dims
        ])

    def _build_encoder(self, input_dim, hidden_dim, latent_dim):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.3),  # 添加 Dropout
            nn.Linear(hidden_dim, latent_dim * 2)  # 输出均值和对数方差
        )

    def _build_decoder(self, latent_dim, hidden_dim, output_dim):
        return nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.3),  # 添加 Dropout
            nn.Linear(hidden_dim, output_dim),
            nn.LeakyReLU()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


    def forward(self, x_list):
        primary_mu, primary_logvar = self._get_mu_logvar(self.encoders[0], x_list[0])
        # primary_z = self.reparameterize(primary_mu, primary_logvar)

        aux_log_list = []
        aux_mu_list = []
        aux_z_list = []
        for i in range(1, self.num_modalities):
            aux_mu, aux_logvar = self._get_mu_logvar(self.encoders[i], x_list[i])
            aux_log_attended, attention_weights = self.cross_attentions_log[i - 1](primary_logvar, aux_logvar, aux_logvar)
            aux_log_list.append(aux_log_attended)

            aux_mu_attended, attention_weights2 = self.cross_attentions_mu[i - 1](primary_mu, aux_mu, aux_mu)
            aux_mu_list.append(aux_mu_attended)

            aux_z_attended = self.reparameterize(aux_mu_attended, aux_log_attended)
            aux_z_list.append(aux_z_attended)

        # 拼接所有的潜在表示
        # z_list = [primary_z] + aux_z_list

        joint_mu = self.transformer_fusion_mu(primary_mu, aux_mu_list)
        joint_log = self.transformer_fusion_log(primary_mu, aux_log_list)

        joint_primary_z = self.reparameterize(joint_mu, joint_log)

        # recon_list = [decoder(z) for decoder, z in zip(self.decoders, z_list)]

        recon_joint_z_list = [decoder(joint_primary_z) for decoder in self.decoders_joint]

        return joint_primary_z, recon_joint_z_list, primary_mu, primary_logvar, attention_weights
    def _get_mu_logvar(self, encoder, x):
        h = encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        return mu, logvar


    def compute_kernel(self,x, y):
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        x = x.unsqueeze(1)  # (x_size, 1, dim)
        y = y.unsqueeze(0)  # (1, y_size, dim)
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)
        kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(dim)
        return torch.exp(-kernel_input)  # (x_size, y_size)

    def compute_mmd(self,x, y):
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
        return mmd

    # def compute_mmd(self, x, y):
    #     # 生成标准正态分布的样本
    #     # 使用高斯核函数计算 MMD
    #     sigma = 1.0  # RBF 核的参数
    #
    #     def rbf_kernel(x, y, sigma):
    #         # x 的 shape 为 [batch_size, latent_dim]
    #         # y 的 shape 为 [batch_size, latent_dim]
    #         diff = x.unsqueeze(1) - y.unsqueeze(0)
    #         dist_sq = torch.sum(diff ** 2, dim=-1)
    #         return torch.exp(-dist_sq / (2 * sigma ** 2))
    #
    #     mmd_x_x = torch.mean(rbf_kernel(x, x, sigma))
    #     mmd_y_y = torch.mean(rbf_kernel(y, y, sigma))
    #     mmd_x_y = torch.mean(rbf_kernel(x, y, sigma))
    #
    #     mmd_loss = mmd_x_x + mmd_y_y - 2 * mmd_x_y
    #     return mmd_loss

    def loss_function(self, recon_joint_list, x_list, mu_list, logvar_list, joint_z, beta=1.0):

        batch_size = x_list[0].size(0)  # 获取批次大小
        dim = mu_list[0].size(-1)  # 获取潜在空间的维度
        recon_loss_jonit = sum(F.mse_loss(recon, x, reduction='mean')
                         for recon, x in zip(recon_joint_list, x_list))

        # kl_loss = sum(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / (batch_size * dim)
        #               for mu, logvar in zip(mu_list, logvar_list))
        joint_z = joint_z.squeeze()
        true_samples = torch.randn(joint_z.shape[0], joint_z.shape[1]).to(joint_z.device)
        kl_loss = torch.sum(self.compute_mmd(true_samples, joint_z))
        return recon_loss_jonit + beta * kl_loss

    def get_specific_and_joint_representations(self, x_list):
        x_list = [tensor.unsqueeze(0) for tensor in x_list]
        joint_z, recon_list, _, _ = self.forward(x_list)
        joint_z = joint_z.squeeze(0)
        recon_list = [tensor.squeeze(0) for tensor in recon_list]
        z_fused = joint_z.cpu().detach().numpy()
        recon_list_np = [data.cpu().detach().numpy() for data in recon_list]
        # 联合表示
        return recon_list_np, z_fused  # 返回特异性表示和联合表示

