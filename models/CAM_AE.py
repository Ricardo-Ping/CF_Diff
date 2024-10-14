import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CAM_AE(nn.Module):
    """
    CAM-AE: 该神经网络架构用于在扩散模型的逆向过程中学习数据分布。
    一跳邻居（直接邻居）信息将会被集成。
    """

    def __init__(self, d_model, num_heads, num_layers, in_dims, emb_size, time_type="cat", norm=False, dropout=0.5):
        super(CAM_AE, self).__init__()
        self.in_dims = in_dims
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm
        self.num_layers = num_layers

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        # 输入层和输出层
        self.in_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip([d_model, d_model], [d_model, d_model])])
        self.out_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip([d_model, d_model], [d_model, d_model])])
        # 多层前馈层
        self.forward_layers = nn.ModuleList([nn.Linear(d_model, d_model) for i in range(num_layers)])
        # 内部维度定义，用于编码用户-物品交互
        self.dim_inters = 650

        # 一跳邻居嵌入和解码层
        self.first_hop_embedding = nn.Linear(1, d_model)  # Expend dimension
        self.first_hop_decoding = nn.Linear(d_model, 1)

        # 二跳邻居嵌入层
        self.second_hop_embedding = nn.Linear(1, d_model)  # Expend dimension

        # 输出层，结合时间嵌入和内部特征
        self.final_out = nn.Linear(self.dim_inters + emb_size, self.dim_inters)

        # Dropout层，用于防止过拟合
        self.drop = nn.Dropout(dropout)
        self.drop1 = nn.Dropout(0.8)  # 第一层 Dropout
        self.drop2 = nn.Dropout(dropout)  # 第二层 Dropout

        # 自编码器的编码器和解码器
        self.encoder = nn.Linear(self.in_dims, self.dim_inters)  # 编码用户-物品交互数据
        self.decoder = nn.Linear(self.dim_inters + emb_size, self.in_dims)  # 解码层，将数据映射回输入维度
        self.encoder2 = nn.Linear(self.in_dims, self.dim_inters)

        # 注意力层
        self.self_attentions = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=0.5, batch_first=True)
            for i in range(num_layers)
        ])

        self.time_emb_dim = emb_size
        self.d_model = d_model

        # LayerNorm层，用于标准化输入和输出
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, x_sec_hop, timesteps):
        """
        前向传播过程
        :param x: 输入的一跳邻居数据
        :param x_sec_hop: 输入的二跳邻居数据
        :param timesteps: 时间步，用于生成时间嵌入
        """

        # Step 1: 编码一跳和二跳邻居信息
        x = self.encoder(x)  # 对一跳邻居数据进行编码
        h_sec_hop = self.encoder(x_sec_hop)  # 对二跳邻居数据进行编码

        # Step 2: 生成时间步嵌入
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)

        # Step 3: 如果设置了归一化，则对输入进行归一化
        if self.norm:
            x = F.normalize(x)

        # Step 4: 添加 Dropout 进行正则化
        x = self.drop(x)

        # Step 5: 将一跳邻居信息和时间嵌入拼接
        h = torch.cat([x, emb], dim=-1)
        h = h.unsqueeze(-1)  # 增加一维度
        h = self.first_hop_embedding(h)  # 对一跳信息进行扩展维度

        # Step 6: 处理二跳邻居信息
        h_sec_hop = torch.cat([h_sec_hop, emb], dim=-1)
        h_sec_hop = h_sec_hop.unsqueeze(-1)  # 增加一维度
        h_sec_hop = self.second_hop_embedding(h_sec_hop)  # 对二跳信息进行扩展维度

        # Step 7: 多层自注意力机制
        for i in range(self.num_layers):
            attention_layer = self.self_attentions[i]  # 选择当前层的多头自注意力层
            attention, attn_output_weights = attention_layer(h_sec_hop, h, h)  # 计算注意力

            # 添加注意力结果并进行残差连接
            attention = self.drop1(attention)
            h = h + attention
            # h = self.norm1(h)

            # 第二层 Dropout
            h = self.drop2(h)

            # 前馈层
            forward_pass = self.forward_layers[i]
            h = forward_pass(h)

            # 使用 tanh 激活函数
            if i != self.num_layers - 1:
                h = torch.tanh(h)

        # Step 8: 解码一跳信息
        h = self.first_hop_decoding(h)
        h = torch.squeeze(h)  # 去掉多余的维度
        h = torch.tanh(h)
        h = self.decoder(h)  # 解码成输出

        return h


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
