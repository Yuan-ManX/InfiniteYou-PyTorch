import math
import torch
import torch.nn as nn


def FeedForward(dim, mult=4):
    """
    前馈神经网络层，用于Transformer中的FFN部分。

    参数:
        dim (int): 输入和输出的维度。
        mult (int, optional): 中间层的维度倍数，默认为4。

    返回:
        nn.Sequential: 包含前馈神经网络层序列的模块。
    """
    # 计算中间层的维度
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim), # 对输入进行层归一化
        nn.Linear(dim, inner_dim, bias=False), # 线性变换，维度从dim变为inner_dim，不使用偏置
        nn.GELU(), # GELU激活函数
        nn.Linear(inner_dim, dim, bias=False), # 线性变换，维度从inner_dim变回dim，不使用偏置
    )


def reshape_tensor(x, heads):
    """
    重塑输入张量以适应多头注意力机制。

    参数:
        x (torch.Tensor): 输入张量，形状为 (batch_size, 序列长度, 宽度)。
        heads (int): 多头注意力的头数。

    返回:
        torch.Tensor: 重塑后的张量，形状为 (batch_size * heads, 序列长度, dim_per_head)。
    """
    # 分离批次大小、序列长度和宽度
    bs, length, width = x.shape
    # 将张量重塑为 (batch_size, 序列长度, 头数, 每头的维度)
    x = x.view(bs, length, heads, -1)
    # 转置维度顺序为 (batch_size, 头数, 序列长度, 每头的维度)
    x = x.transpose(1, 2)
    # 展平批次和头数维度，得到 (batch_size * 头数, 序列长度, 每头的维度)
    x = x.reshape(bs, heads, length, -1)
    return x


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        """
        Perceiver注意力机制的实现。

        参数:
            dim (int): 输入和输出的维度。
            dim_head (int, optional): 每个注意力头的维度，默认为64。
            heads (int, optional): 注意力头的数量，默认为8。
        """
        super().__init__()
        # 缩放因子，用于缩放注意力得分
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        # 计算多头注意力的总内部维度
        inner_dim = dim_head * heads

        # 对输入进行层归一化
        self.norm1 = nn.LayerNorm(dim)
        # 对潜在变量进行层归一化
        self.norm2 = nn.LayerNorm(dim)

        # 线性变换，将输入映射到查询向量
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        # 线性变换，将输入映射到键和值向量
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        # 线性变换，将多头注意力输出映射回原始维度
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        前向传播过程。

        参数:
            x (torch.Tensor): 输入特征张量，形状为 (batch_size, n1, D)。
            latents (torch.Tensor): 潜在特征张量，形状为 (batch_size, n2, D)。

        返回:
            torch.Tensor: 注意力机制的输出，形状为 (batch_size, n2, D)。
        """
        # 对输入进行层归一化
        x = self.norm1(x)
        # 对潜在变量进行层归一化
        latents = self.norm2(latents)
        
        # 获取潜在变量的批次大小和序列长度
        b, l, _ = latents.shape

        # 计算查询向量
        q = self.to_q(latents)
        # 将输入和潜在变量在最后一个维度上拼接，用于计算键和值
        kv_input = torch.cat((x, latents), dim=-2)
        # 将拼接后的张量拆分为键和值
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        
        # 重塑查询张量以适应多头注意力
        q = reshape_tensor(q, self.heads)
        # 重塑键张量以适应多头注意力
        k = reshape_tensor(k, self.heads)
        # 重塑值张量以适应多头注意力
        v = reshape_tensor(v, self.heads)

        # 计算注意力权重
        # 计算缩放因子
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        # 计算注意力得分
        weight = (q * scale) @ (k * scale).transpose(-2, -1) 
        # 对注意力得分进行softmax归一化
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        # 计算多头注意力的输出
        out = weight @ v
        
        # 重塑输出张量形状
        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        # 通过线性变换映射回原始维度
        return self.to_out(out)


class Resampler(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
    ):
        """
        重采样器模块，用于对输入特征进行重采样。

        参数:
            dim (int, optional): 模型的维度，默认为1024。
            depth (int, optional): 层的深度，即重复的注意力层和前馈层的数量，默认为8。
            dim_head (int, optional): 每个注意力头的维度，默认为64。
            heads (int, optional): 注意力头的数量，默认为16。
            num_queries (int, optional): 查询的数量，默认为8。
            embedding_dim (int, optional): 输入嵌入的维度，默认为768。
            output_dim (int, optional): 输出维度，默认为1024。
            ff_mult (int, optional): 前馈层中中间层的维度倍数，默认为4。
        """
        super().__init__()
        
        # 初始化潜在变量张量，形状为 (1, num_queries, dim)，并除以sqrt(dim)进行初始化
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)
        
        # 线性变换，将输入嵌入映射到模型维度
        self.proj_in = nn.Linear(embedding_dim, dim)

        # 线性变换，将模型维度映射到输出维度
        self.proj_out = nn.Linear(dim, output_dim)
        # 对输出进行层归一化
        self.norm_out = nn.LayerNorm(output_dim)
        
        # 初始化层的列表
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            # 为每一层添加注意力机制和前馈层
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x):
        """
        前向传播过程。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, sequence_length, embedding_dim)。

        返回:
            torch.Tensor: 重采样后的输出张量，形状为 (batch_size, num_queries, output_dim)。
        """
        # 将潜在变量张量重复batch_size次以匹配批次大小
        latents = self.latents.repeat(x.size(0), 1, 1)

        # 将输入张量投影到模型维度
        x = self.proj_in(x)
        
        for attn, ff in self.layers:
            # 应用注意力机制并与原latents相加
            latents = attn(x, latents) + latents
            # 应用前馈层并与原latents相加
            latents = ff(latents) + latents
        
        # 将latents投影到输出维度
        latents = self.proj_out(latents)

        # 对输出进行层归一化
        return self.norm_out(latents)
