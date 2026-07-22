"""
seg_mask_head.py - 分割掩码头模块

本模块实现了一个基于Transformer的分割掩码头（SegMaskHead），用于根据查询嵌入
和编码器记忆生成分割掩码。该模块是分割任务（如全景分割）的核心组件之一，
负责将检测头输出的查询嵌入转换为像素级的分割掩码预测。

模块的核心架构：
1. Mlp: 多层感知机（MLP），用于特征变换。
2. SelfAttention: 自注意力模块，对查询自身进行注意力计算。
3. Attention: 交叉注意力模块，在查询和编码器记忆之间进行注意力计算，
   同时输出注意力掩码（attention mask）。
4. AttentionTail: 轻量级的注意力尾部模块，用于生成额外的掩码预测，
   使掩码解码器能多一层深度。
5. Block: Transformer解码器块，包含自注意力（可选）、交叉注意力和MLP。
6. drop_path / DropPath: 随机深度（Stochastic Depth）正则化，用于训练时
   随机丢弃网络层，增强泛化能力。
7. SegMaskHead: 分割掩码头主类，组合多个Block和一个AttentionTail，
   通过迭代精炼查询嵌入并生成逐层的注意力掩码。

整体流程：
- 输入：编码器记忆（memory）、位置编码（pos_memory）、查询嵌入（query_embed）、
  查询位置编码（pos_query）、掩码（mask_memory）。
- 处理：通过多个Block迭代精炼查询嵌入，每个Block输出一个注意力掩码。
- 最后通过AttentionTail生成最终的注意力掩码，用于后续的掩码预测。
"""

import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from functools import partial
from mmdet.models.utils.builder import TRANSFORMER
import math
from mmcv.runner import force_fp32

# 全局计数器（未在当前代码中使用，可能用于调试或统计）
count = 0


class Mlp(nn.Module):
    """
    多层感知机（MLP）模块。

    实现一个简单的两层全连接网络，结构为：
    fc1 -> GELU激活 -> Dropout -> fc2 -> Dropout

    用于在Transformer块中进行特征变换，通常是自注意力和交叉注意力之后
    的逐位置前馈网络（Position-wise Feed-Forward Network）。

    Args:
        in_features (int): 输入特征的维度。
        hidden_features (int, optional): 隐藏层维度，默认为 in_features。
        out_features (int, optional): 输出特征维度，默认为 in_features。
        act_layer (nn.Module): 激活函数类，默认 nn.GELU。
        drop (float): Dropout概率，默认 0.0。
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        self.fp16_enabled = False
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # 第一层全连接：in_features -> hidden_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        # 激活函数
        self.act = act_layer()
        # 第二层全连接：hidden_features -> out_features
        self.fc2 = nn.Linear(hidden_features, out_features)
        # Dropout层
        self.drop = nn.Dropout(drop)

    @force_fp32(apply_to=('x', ))
    def forward(self, x):
        """
        MLP前向传播。

        Args:
            x (Tensor): 输入特征，形状为 (..., in_features)。

        Returns:
            Tensor: 输出特征，形状为 (..., out_features)。
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SelfAttention(nn.Module):
    """
    自注意力（Self-Attention）模块。

    实现标准的缩放点积自注意力机制，公式为：
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

    其中 Q、K、V 都来自同一个输入 x（通过 qkv 线性层投影）。

    该模块用于查询嵌入自身的注意力计算，帮助查询之间交换信息。

    Args:
        cfg: 配置对象（未在当前代码中直接使用，保留用于扩展）。
        dim (int): 输入特征维度。
        num_heads (int): 注意力头数，默认 2。
        qkv_bias (bool): QKV投影是否使用偏置，默认 False。
        qk_scale (float, optional): QK缩放因子，默认为 head_dim^{-0.5}。
        attn_drop (float): 注意力权重的Dropout概率，默认 0.0。
        proj_drop (float): 输出投影的Dropout概率，默认 0.0。
    """

    def __init__(self,
                 cfg,
                 dim,
                 num_heads=2,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.fp16_enabled = False
        # 缩放因子：如果未指定，默认为 1/sqrt(head_dim)
        self.scale = qk_scale or head_dim**-0.5

        # 合并的QKV投影：从 dim 维线性投影到 3*dim 维
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        # 输出投影：将多头注意力结果映射回 dim 维
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    @force_fp32(apply_to=('x', ))
    def forward(self, x):
        """
        自注意力前向传播。

        处理流程：
        1. 通过 qkv 层生成 Q、K、V。
        2. 重塑为多头格式：(B, num_heads, N, head_dim)。
        3. 计算缩放点积注意力：softmax(Q * K^T / sqrt(d)) * V。
        4. 重塑回原始形状并通过输出投影。

        Args:
            x (Tensor): 输入特征，形状为 (B, N, C)，
                其中 B 为批次大小，N 为序列长度，C 为特征维度。

        Returns:
            Tensor: 自注意力输出，形状为 (B, N, C)。
        """
        B, N, C = x.shape

        # 生成 QKV：reshape 为 (B, N, 3, num_heads, head_dim)
        # 然后 permute 为 (3, B, num_heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1,
                                                               4).contiguous()
        # 分离 Q, K, V
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        # 计算注意力分数：Q @ K^T，然后缩放
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # softmax归一化
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # 加权求和：attn @ V，然后 reshape 回 (B, N, C)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # 输出投影和Dropout
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Attention(nn.Module):
    """
    交叉注意力（Cross-Attention）模块。

    与自注意力不同，该模块的查询（Q）、键（K）、值（V）来自不同的输入：
    - Q 来自查询嵌入（query_embed）
    - K 和 V 来自编码器记忆（memory）

    除此之外，该模块还额外输出一个注意力掩码（attention mask），
    通过对注意力权重进行额外的线性变换得到。这个掩码可以用于后续的
    分割掩码预测。

    处理流程：
    1. 分别对 Q、K、V 进行线性投影。
    2. 计算交叉注意力分数。
    3. 对注意力分数进行额外的线性变换（linear_l1 + linear），
       生成注意力掩码。这是一个学习到的从注意力权重到掩码的映射。
    4. 对注意力分数进行softmax归一化，然后加权求和V。
    5. 输出投影。

    Args:
        cfg: 配置对象。
        dim (int): 输入特征维度。
        num_heads (int): 注意力头数，默认 2。
        qkv_bias (bool): QKV投影是否使用偏置，默认 False。
        qk_scale (float, optional): QK缩放因子。
        attn_drop (float): 注意力权重的Dropout概率，默认 0.0。
        proj_drop (float): 输出投影的Dropout概率，默认 0.0。
    """

    def __init__(self,
                 cfg,
                 dim,
                 num_heads=2,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.fp16_enabled = False
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # 分别对 Q、K、V 进行线性投影（而非合并的qkv投影）
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 注意力掩码生成网络：
        # 第一层：对每个注意力头进行变换（num_heads -> num_heads）
        self.linear_l1 = nn.Sequential(
            nn.Linear(self.num_heads, self.num_heads),
            nn.ReLU(),
        )
        # 第二层：将多头注意力合并为单个掩码值（num_heads -> 1）
        self.linear = nn.Sequential(
            nn.Linear(self.num_heads, 1),
            nn.ReLU(),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        """
        重置参数：对维度大于1的参数使用 Xavier 均匀初始化。
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @force_fp32(apply_to=('query', 'key', 'value'))
    def forward(self, query, key, value, key_padding_mask, hw_lvl):
        """
        交叉注意力前向传播。

        处理流程：
        1. 对 Q、K、V 分别进行线性投影，重塑为多头格式。
        2. 计算注意力分数：Q @ K^T / sqrt(d_k)。
        3. 对注意力分数进行额外的线性变换（permute + linear_l1 + linear），
           生成注意力掩码（mask）。这个掩码会随查询嵌入一起返回。
        4. 对注意力分数进行softmax归一化，然后加权求和V。
        5. 输出投影。

        Args:
            query (Tensor): 查询嵌入，形状为 (B, N, C)，
                其中 N 为查询数量，C 为特征维度。
            key (Tensor): 键（编码器记忆），形状为 (B, L, C)，
                其中 L 为编码器输出的序列长度。
            value (Tensor): 值（编码器记忆），形状为 (B, L, C)。
            key_padding_mask (Tensor): 键的padding掩码，形状为 (B, L)。
                True 表示需要忽略的位置。
            hw_lvl: 高度和宽度的层级信息（未在当前代码中直接使用，
                保留用于扩展）。

        Returns:
            tuple:
                - x (Tensor): 交叉注意力输出，形状为 (B, N, C)。
                - mask (Tensor): 注意力掩码，形状为 (B, N, L)。
                  通过对注意力分数进行线性变换得到，可用于后续的掩码预测。
        """
        B, N, C = query.shape
        _, L, _ = key.shape

        # 对Q进行投影：reshape为 (B, num_heads, N, head_dim)
        q = self.q(query).reshape(B, N,
                                  self.num_heads, C // self.num_heads).permute(
                                      0, 2, 1,
                                      3).contiguous()
        # 对K进行投影：reshape为 (B, num_heads, L, head_dim)
        k = self.k(key).reshape(B, L,
                                self.num_heads, C // self.num_heads).permute(
                                    0, 2, 1,
                                    3).contiguous()
        # 对V进行投影：reshape为 (B, num_heads, L, head_dim)
        v = self.v(value).reshape(B, L,
                                  self.num_heads, C // self.num_heads).permute(
                                      0, 2, 1,
                                      3).contiguous()

        # 计算注意力分数：(B, num_heads, N, L)
        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale

        # 将注意力分数 permute 为 (B, N, L, num_heads)，
        # 以便在最后一个维度（num_heads）上应用线性变换
        attn = attn.permute(0, 2, 3, 1)

        # 生成注意力掩码：通过两层线性变换将多头注意力映射为单通道掩码
        new_feats = self.linear_l1(attn)  # (B, N, L, num_heads)
        mask = self.linear(new_feats)     # (B, N, L, 1)，然后squeeze

        # 将注意力分数 permute 回 (B, num_heads, N, L) 以进行softmax和加权求和
        attn = attn.permute(0, 3, 1, 2)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 加权求和：attn @ V，然后 reshape 回 (B, N, C)
        x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)
        # 输出投影和Dropout
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, mask


class AttentionTail(nn.Module):
    """
    轻量级注意力尾部模块。

    该模块是 Attention 的简化版本，只计算注意力掩码（mask），
    不计算加权求和后的输出。它用于在掩码解码器的最后一层生成额外的
    注意力掩码预测，使掩码解码器能多一层深度，但计算开销更小。

    与 Attention 的主要区别：
    - 没有 V 的投影和加权求和步骤。
    - 没有输出投影（proj）。
    - 只返回注意力掩码（mask），不返回特征。

    Args:
        cfg: 配置对象。
        dim (int): 输入特征维度。
        num_heads (int): 注意力头数，默认 2。
        qkv_bias (bool): QKV投影是否使用偏置，默认 False。
        qk_scale (float, optional): QK缩放因子。
        attn_drop (float): 注意力权重的Dropout概率，默认 0.0。
        proj_drop (float): 输出投影的Dropout概率（此处未使用），默认 0.0。
    """

    def __init__(self,
                 cfg,
                 dim,
                 num_heads=2,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.fp16_enabled = False
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # 只对 Q 和 K 进行投影（不需要 V）
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)

        # 注意力掩码生成网络（与 Attention 中的结构相同）
        self.linear_l1 = nn.Sequential(
            nn.Linear(self.num_heads, self.num_heads),
            nn.ReLU(),
        )

        self.linear = nn.Sequential(
            nn.Linear(self.num_heads, 1),
            nn.ReLU(),
        )
        self._reset_parameters()

    def _reset_parameters(self):
        """
        重置参数：对维度大于1的参数使用 Xavier 均匀初始化。
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @force_fp32(apply_to=('query', 'key'))
    def forward(self, query, key, key_padding_mask, hw_lvl=None):
        """
        注意力尾部前向传播。

        只计算注意力掩码，不计算特征输出。

        Args:
            query (Tensor): 查询嵌入，形状为 (B, N, C)。
            key (Tensor): 键（编码器记忆），形状为 (B, L, C)。
            key_padding_mask (Tensor): 键的padding掩码，形状为 (B, L)。
            hw_lvl: 高度和宽度的层级信息（未直接使用）。

        Returns:
            Tensor: 注意力掩码，形状为 (B, N, L)。
        """
        B, N, C = query.shape
        _, L, _ = key.shape

        # 对Q进行投影：reshape为 (B, num_heads, N, head_dim)
        q = self.q(query).reshape(B, N,
                                  self.num_heads, C // self.num_heads).permute(
                                      0, 2, 1,
                                      3).contiguous()
        # 对K进行投影：reshape为 (B, num_heads, L, head_dim)
        k = self.k(key).reshape(B, L,
                                self.num_heads, C // self.num_heads).permute(
                                    0, 2, 1,
                                    3).contiguous()
        # 计算注意力分数：(B, num_heads, N, L)
        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale

        # 将注意力分数 permute 为 (B, N, L, num_heads)
        attn = attn.permute(0, 2, 3, 1)

        # 生成注意力掩码
        new_feats = self.linear_l1(attn)
        mask = self.linear(new_feats)

        return mask


class Block(nn.Module):
    """
    Transformer解码器块。

    该模块实现了一个标准的Transformer解码器层，包含以下组件：
    1. 可选的自注意力（SelfAttention）：对查询嵌入进行自注意力计算。
    2. 交叉注意力（Attention）：在查询嵌入和编码器记忆之间进行注意力计算。
    3. MLP（多层感知机）：对交叉注意力输出进行非线性变换。

    每个组件都包含残差连接（Residual Connection）和层归一化（LayerNorm），
    结构如下：

    self_attn (可选):
        query = query + DropPath(SelfAttention(query))
        query = LayerNorm(query)

    cross_attn:
        x, mask = Attention(query, key, value)
        query = query + DropPath(x)
        query = LayerNorm(query)

    mlp:
        query = query + DropPath(MLP(query))
        query = LayerNorm(query)

    Args:
        cfg: 配置对象。
        dim (int): 特征维度。
        num_heads (int): 注意力头数。
        mlp_ratio (float): MLP隐藏层维度相对于输入维度的比例，默认 4.0。
        qkv_bias (bool): QKV投影是否使用偏置，默认 False。
        qk_scale (float, optional): QK缩放因子。
        drop (float): MLP中的Dropout概率，默认 0.0。
        attn_drop (float): 注意力中的Dropout概率，默认 0.0。
        drop_path (float): 随机深度（DropPath）的概率，默认 0.0。
        act_layer (nn.Module): 激活函数类，默认 nn.GELU。
        norm_layer (nn.Module): 归一化层类，默认 nn.LayerNorm。
        self_attn (bool): 是否使用自注意力，默认 False。
    """

    def __init__(self,
                 cfg,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 self_attn=False):
        super().__init__()
        self.fp16_enabled = False
        # 交叉注意力之后的第一个层归一化
        self.head_norm1 = norm_layer(dim)
        self.self_attn = self_attn

        # 交叉注意力模块
        self.attn = Attention(cfg,
                              dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop=attn_drop,
                              proj_drop=drop)

        # 随机深度（DropPath）：以一定概率丢弃整个残差分支
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        # MLP之后的第二个层归一化
        self.head_norm2 = norm_layer(dim)

        # MLP：隐藏层维度 = dim * mlp_ratio
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

        # 如果启用自注意力
        if self.self_attn:
            self.self_attention = SelfAttention(cfg,
                                                dim,
                                                num_heads=num_heads,
                                                qkv_bias=qkv_bias,
                                                qk_scale=qk_scale,
                                                attn_drop=attn_drop,
                                                proj_drop=drop)
            # 自注意力之后的第三个层归一化
            self.norm3 = norm_layer(dim)

    @force_fp32(apply_to=('query', 'key', 'value'))
    def forward(self, query, key, value, key_padding_mask=None, hw_lvl=None):
        """
        Transformer解码器块的前向传播。

        处理流程：
        1. 可选的自注意力：query = self_attention(query) + query
        2. 交叉注意力：计算 query 和 key/value 之间的注意力，
           输出精炼后的 query 和注意力掩码 mask。
        3. MLP：对精炼后的 query 进行非线性变换。

        Args:
            query (Tensor): 查询嵌入，形状为 (B, N, C)。
            key (Tensor): 键（编码器记忆），形状为 (B, L, C)。
            value (Tensor): 值（编码器记忆），形状为 (B, L, C)。
            key_padding_mask (Tensor, optional): 键的padding掩码，
                形状为 (B, L)。默认 None。
            hw_lvl: 高度和宽度的层级信息，传入 Attention 子模块。

        Returns:
            tuple:
                - query (Tensor): 精炼后的查询嵌入，形状为 (B, N, C)。
                - mask (Tensor): 交叉注意力产生的注意力掩码，
                  形状为 (B, N, L)。
        """
        # 可选：自注意力
        if self.self_attn:
            query = query + self.drop_path(self.self_attention(query))
            query = self.norm3(query)

        # 交叉注意力：query与key/value交互
        x, mask = self.attn(query, key, value, key_padding_mask, hw_lvl=hw_lvl)
        # 残差连接 + DropPath + LayerNorm
        query = query + self.drop_path(x)
        query = self.head_norm1(query)

        # MLP：非线性变换
        query = query + self.drop_path(self.mlp(query))
        query = self.head_norm2(query)

        return query, mask


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    随机深度（Stochastic Depth / DropPath）函数。

    在训练时，以概率 drop_prob 随机丢弃整个残差分支（将其输出设为零），
    这是一种正则化技术，可以增强模型的泛化能力，并允许训练更深的网络。

    原理：
    - 在训练时，每个样本有 drop_prob 的概率被丢弃（输出为零）。
    - 存活下来的样本的输出会被放大（除以 keep_prob），
      以保持期望值不变。
    - 在推理时，不进行丢弃。

    参考：
    - Deep Networks with Stochastic Depth (https://arxiv.org/abs/1603.09382)
    - 该实现与 EfficientNet 等网络中的 DropConnect 实现相同，
      但原名称有误导性，因此改名为 'drop path'。

    Args:
        x (Tensor): 输入张量。
        drop_prob (float): 丢弃概率，范围 [0, 1]。默认 0.0。
        training (bool): 是否处于训练模式。默认 False。

    Returns:
        Tensor: 应用随机深度后的输出张量。
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # 生成与输入张量具有相同批次维度的随机掩码
    shape = (x.shape[0], ) + (1, ) * (
        x.ndim - 1)  # 支持不同维度的张量，不仅仅是2D ConvNets
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # 二值化：大于1的为1，小于1的为0
    # 存活样本的输出除以 keep_prob，以保持期望值
    output = x.div(keep_prob) * random_tensor
    return output


def _get_clones(module, N):
    """
    克隆一个模块 N 次，返回 ModuleList。

    Args:
        module (nn.Module): 要克隆的模块。
        N (int): 克隆数量。

    Returns:
        nn.ModuleList: 包含 N 个深拷贝模块的列表。
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DropPath(nn.Module):
    """
    随机深度（DropPath）模块。

    将 drop_path 函数封装为 nn.Module，方便嵌入到模型中。
    在训练时，以概率 drop_prob 随机丢弃整个残差分支。

    Args:
        drop_prob (float, optional): 丢弃概率，默认 None（即不丢弃）。
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    @force_fp32(apply_to=('x', ))
    def forward(self, x):
        """
        DropPath 前向传播。

        Args:
            x (Tensor): 输入张量。

        Returns:
            Tensor: 应用随机深度后的输出张量。
        """
        return drop_path(x, self.drop_prob, self.training)


@TRANSFORMER.register_module()
class SegMaskHead(nn.Module):
    """
    分割掩码头（SegMaskHead）。

    该模块是分割任务中用于生成分割掩码的核心组件。它接收编码器记忆
    和查询嵌入，通过多个Transformer解码器块迭代精炼查询嵌入，
    并从交叉注意力权重中生成注意力掩码，这些掩码可用于后续的
    分割掩码预测。

    架构组成：
    1. 多个 Block（Transformer解码器块）：每个块包含可选的
       自注意力、交叉注意力和MLP，输出精炼后的查询嵌入和注意力掩码。
    2. 一个 AttentionTail：在最后一层生成额外的注意力掩码，
       使掩码解码器能多一层深度。

    每一层都会输出一个注意力掩码（masks列表），最后一层通过
    AttentionTail 输出一个额外的掩码。这些掩码可以被后续的
    掩码上采样和预测模块使用。

    Args:
        cfg: 配置对象。
        d_model (int): 模型特征维度，默认 16。
        nhead (int): 注意力头数，默认 2。
        num_encoder_layers (int): 编码器层数（未在当前实现中使用），默认 6。
        num_decoder_layers (int): 解码器层数（即 Block 的数量），默认 1。
        dim_feedforward (int): 前馈网络维度（未在当前实现中使用），默认 64。
        dropout (float): Dropout概率（未在当前实现中使用），默认 0.1。
        activation (str): 激活函数类型（未在当前实现中使用），默认 "relu"。
        normalize_before (bool): 是否在注意力和FFN之前进行归一化
            （未在当前实现中使用），默认 False。
        return_intermediate_dec (bool): 是否返回中间解码器层输出
            （未在当前实现中使用），默认 False。
        self_attn (bool): Block中是否启用自注意力，默认 False。
    """

    def __init__(self,
                 cfg=None,
                 d_model=16,
                 nhead=2,
                 num_encoder_layers=6,
                 num_decoder_layers=1,
                 dim_feedforward=64,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False,
                 return_intermediate_dec=False,
                 self_attn=False):
        super().__init__()

        self.fp16_enabled = False
        # 设置固定参数
        mlp_ratio = 4        # MLP隐藏层维度比例
        qkv_bias = True      # QKV投影使用偏置
        qk_scale = None      # 使用默认缩放因子
        drop_rate = 0        # Dropout概率
        attn_drop_rate = 0   # 注意力Dropout概率

        # 归一化层：使用 LayerNorm，eps=1e-6
        norm_layer = None
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        # 激活函数：使用 GELU
        act_layer = None
        act_layer = act_layer or nn.GELU

        # 创建 Block 模板，并克隆 num_decoder_layers 次
        block = Block(cfg,
                      dim=d_model,
                      num_heads=nhead,
                      mlp_ratio=mlp_ratio,
                      qkv_bias=qkv_bias,
                      qk_scale=qk_scale,
                      drop=drop_rate,
                      attn_drop=attn_drop_rate,
                      drop_path=0,
                      norm_layer=norm_layer,
                      act_layer=act_layer,
                      self_attn=self_attn)
        self.blocks = _get_clones(block, num_decoder_layers)

        # 注意力尾部：用于生成额外的注意力掩码
        self.attnen = AttentionTail(cfg,
                                    d_model,
                                    num_heads=nhead,
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    attn_drop=attn_drop_rate,
                                    proj_drop=0)

        self._reset_parameters()

    def _reset_parameters(self):
        """
        重置参数：对维度大于1的参数使用 Xavier 均匀初始化。
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        """
        将位置编码添加到张量上。

        如果位置编码为 None，则直接返回原张量。

        Args:
            tensor (Tensor): 输入张量。
            pos (Tensor, optional): 位置编码张量。

        Returns:
            Tensor: 添加位置编码后的张量。
        """
        if pos is None:
            return tensor
        else:
            return tensor + pos

    @force_fp32(apply_to=('memory', 'mask_memory', 'pos_memory', 'query_embed',
                          'mask_query', 'pos_query'))
    def forward(self, memory, mask_memory, pos_memory, query_embed, mask_query,
                pos_query, hw_lvl):
        """
        分割掩码头的前向传播。

        处理流程：
        1. 如果 mask_memory 是张量，将其转为布尔类型。
        2. 遍历每个 Block（解码器块）：
           - 将位置编码添加到查询嵌入和编码器记忆上。
           - 通过 Block 进行交叉注意力计算，精炼查询嵌入。
           - 收集每个 Block 输出的注意力掩码和中间查询嵌入。
        3. 通过 AttentionTail 生成最终的注意力掩码。

        Args:
            memory (Tensor): 编码器记忆（编码器输出），形状为 (B, L, C)，
                其中 L 为编码器输出的序列长度，C 为特征维度。
            mask_memory (Tensor): 编码器记忆的padding掩码，形状为 (B, L)。
                True 表示需要忽略的位置。
            pos_memory (Tensor): 编码器记忆的位置编码，形状为 (B, L, C)。
            query_embed (Tensor): 查询嵌入，形状为 (B, N, C)，
                其中 N 为查询数量。
            mask_query (Tensor): 查询的padding掩码（未在当前代码中直接使用，
                保留用于扩展）。
            pos_query (Tensor): 查询的位置编码，形状为 (B, N, C)。
            hw_lvl: 高度和宽度的层级信息，传入 Block 子模块。

        Returns:
            tuple:
                - attn (Tensor): AttentionTail 生成的最终注意力掩码，
                  形状为 (B, N, L)。
                - masks (list[Tensor]): 每个 Block 生成的注意力掩码列表，
                  每个元素的形状为 (B, N, L)。
                - inter_query (list[Tensor]): 每个 Block 精炼后的查询嵌入列表，
                  每个元素的形状为 (B, N, C)。
        """
        # 将掩码转为布尔类型（如果还不是）
        if mask_memory is not None and isinstance(mask_memory, torch.Tensor):
            mask_memory = mask_memory.to(torch.bool)

        masks = []       # 存储每个Block生成的注意力掩码
        inter_query = [] # 存储每个Block精炼后的查询嵌入

        # 遍历每个解码器块
        for i, block in enumerate(self.blocks):
            # 为查询和编码器记忆添加位置编码，然后进行交叉注意力计算
            query_embed, mask = block(self.with_pos_embed(
                query_embed, pos_query),
                                      self.with_pos_embed(memory, pos_memory),
                                      memory,
                                      key_padding_mask=mask_memory,
                                      hw_lvl=hw_lvl)
            # 收集当前Block的注意力掩码和精炼后的查询嵌入
            masks.append(mask)
            inter_query.append(query_embed)

        # 通过 AttentionTail 生成最终的注意力掩码
        attn = self.attnen(self.with_pos_embed(query_embed, pos_query),
                           self.with_pos_embed(memory, pos_memory),
                           key_padding_mask=mask_memory,
                           hw_lvl=hw_lvl)

        return attn, masks, inter_query