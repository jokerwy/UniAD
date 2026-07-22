#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

"""
运动预测可变形注意力模块 (Motion Deformable Attention)

本模块实现了 UniAD 运动预测任务中使用的可变形注意力机制，包含以下组件：

1. MotionTransformerAttentionLayer: 运动预测 Transformer 的基础注意力层。
   基于 mmcv 的 BaseTransformerLayer，支持灵活的操作顺序配置
   （如 self_attn -> norm -> cross_attn -> norm -> ffn -> norm）。
   支持 pre-norm 和 post-norm 两种归一化方式。

2. MotionDeformableAttention: 运动预测的可变形注意力模块。
   这是 UniAD 运动预测中智能体与 BEV 特征交互的核心组件。
   它根据参考轨迹（reference_trajs）在 BEV 特征图上进行可变形采样，
   让智能体能够感知其未来轨迹路径上的环境特征。
   关键特性：
   - 支持多步采样（multi-step sampling）：沿轨迹的多个时间步进行采样
   - 支持多层级采样（multi-level sampling）：在多个 BEV 特征层级上进行采样
   - 支持多智能体坐标转换：将 agent 坐标系的轨迹转换到 ego（自车）坐标系
   - 自定义权重初始化：采样偏移量使用均匀分布的圆形初始化

3. CustomModeMultiheadAttention: 自定义的多模态多头注意力模块。
   封装了 PyTorch 的 MultiheadAttention，支持多模态（multi-mode）维度的处理。
   在运动预测中，每个智能体有多个运动模态，该模块能够正确处理这种多模态的查询结构。
"""

import copy
import warnings
import torch
import math
import torch.nn as nn

from einops import rearrange, repeat
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import ATTENTION, TRANSFORMER_LAYER
from mmcv.cnn.bricks.transformer import build_attention, build_feedforward_network, build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmcv.utils import ConfigDict, deprecated_api_warning
from projects.mmdet3d_plugin.uniad.modules.multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32


@TRANSFORMER_LAYER.register_module()
class MotionTransformerAttentionLayer(BaseModule):
    """
    运动预测 Transformer 的基础注意力层。

    基于 mmcv 的 BaseTransformerLayer 实现，支持灵活的 operation_order 配置。
    可以包含 self_attn、cross_attn、ffn 和 norm 等操作，按指定顺序执行。

    支持 pre-norm 和 post-norm 两种架构：
    - pre-norm: 操作顺序以 'norm' 开头，在 attention/ffn 之前先做归一化
    - post-norm: 操作顺序不以 'norm' 开头，在 attention/ffn 之后做归一化

    支持多个 attention 和 ffn 模块，可以灵活配置注意力机制和前馈网络的组合。

    Args:
        attn_cfgs (list[ConfigDict] | ConfigDict | None): 注意力模块的配置列表。
            列表中每个元素对应 operation_order 中的一个 attention 操作。
            如果是单个 dict，则所有 attention 操作共享同一配置。
        ffn_cfgs (list[ConfigDict] | ConfigDict | None): 前馈网络模块的配置列表。
            列表中每个元素对应 operation_order 中的一个 ffn 操作。
            如果是单个 dict，则所有 ffn 操作共享同一配置。
            默认配置: embed_dims=256, feedforward_channels=1024, num_fcs=2, ffn_drop=0.
        operation_order (tuple[str]): 操作执行顺序，如 ('self_attn', 'norm', 'ffn', 'norm')。
            支持 'self_attn', 'cross_attn', 'ffn', 'norm'。
            如果第一个元素是 'norm'，则启用 pre-norm 模式。
        norm_cfg (dict): 归一化层配置。默认: dict(type='LN')
        init_cfg (ConfigDict): 初始化配置。
        batch_first (bool): 是否为 batch_first 格式。Key/Query/Value 的形状：
            - True: (batch, n, embed_dim)
            - False: (n, batch, embed_dim)
            默认 False
    """

    def __init__(self,
                 attn_cfgs=None,
                 ffn_cfgs=dict(
                     type='FFN',
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ),
                 operation_order=None,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 batch_first=False,
                 **kwargs):

        # 处理已弃用的参数名称，向后兼容旧版本配置
        deprecated_args = dict(
            feedforward_channels='feedforward_channels',
            ffn_dropout='ffn_drop',
            ffn_num_fcs='num_fcs')
        for ori_name, new_name in deprecated_args.items():
            if ori_name in kwargs:
                warnings.warn(
                    f'The arguments `{ori_name}` in BaseTransformerLayer '
                    f'has been deprecated, now you should set `{new_name}` '
                    f'and other FFN related arguments '
                    f'to a dict named `ffn_cfgs`. ', DeprecationWarning)
                ffn_cfgs[new_name] = kwargs[ori_name]

        super().__init__(init_cfg)

        self.batch_first = batch_first

        # 统计 operation_order 中 attention 操作的数量
        num_attn = operation_order.count('self_attn') + operation_order.count(
            'cross_attn')
        if isinstance(attn_cfgs, dict):
            # 如果 attn_cfgs 是单个 dict，则复制为每个 attention 操作创建副本
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        else:
            assert num_attn == len(attn_cfgs), f'The length ' \
                f'of attn_cfg {num_attn} is ' \
                f'not consistent with the number of attention' \
                f'in operation_order {operation_order}.'

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.pre_norm = operation_order[0] == 'norm'  # 判断是否 pre-norm 模式
        self.attentions = ModuleList()

        # 按 operation_order 顺序构建各个操作模块
        index = 0
        for operation_name in operation_order:
            if operation_name in ['self_attn', 'cross_attn']:
                if 'batch_first' in attn_cfgs[index]:
                    assert self.batch_first == attn_cfgs[index]['batch_first']
                else:
                    attn_cfgs[index]['batch_first'] = self.batch_first
                attention = build_attention(attn_cfgs[index])
                # 记录操作名称，某些自定义注意力模块需要知道自己是 self_attn 还是 cross_attn
                attention.operation_name = operation_name
                self.attentions.append(attention)
                index += 1

        self.embed_dims = self.attentions[0].embed_dims

        # 构建前馈网络模块
        self.ffns = ModuleList()
        num_ffns = operation_order.count('ffn')
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = ConfigDict(ffn_cfgs)
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
        assert len(ffn_cfgs) == num_ffns
        for ffn_index in range(num_ffns):
            if 'embed_dims' not in ffn_cfgs[ffn_index]:
                ffn_cfgs[ffn_index]['embed_dims'] = self.embed_dims
            else:
                assert ffn_cfgs[ffn_index]['embed_dims'] == self.embed_dims
            self.ffns.append(
                build_feedforward_network(ffn_cfgs[ffn_index],
                                          dict(type='FFN')))

        # 构建归一化层
        self.norms = ModuleList()
        num_norms = operation_order.count('norm')
        for _ in range(num_norms):
            self.norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])

    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        """
        MotionTransformerAttentionLayer 的前向传播。

        按照 operation_order 定义的顺序依次执行各个操作。
        支持 pre-norm 和 post-norm 两种模式。

        Args:
            query (Tensor): 查询张量，形状取决于 batch_first:
                - batch_first=False: (num_queries, bs, embed_dims)
                - batch_first=True: (bs, num_queries, embed_dims)
            key (Tensor): 键张量，形状同 query
            value (Tensor): 值张量，形状同 key
            query_pos (Tensor): 查询的位置编码，默认 None
            key_pos (Tensor): 键的位置编码，默认 None
            attn_masks (List[Tensor] | None): 注意力掩码列表，长度应与 attention 数量一致
            query_key_padding_mask (Tensor): 查询的 padding 掩码，形状 (bs, num_queries)。
                仅用于 self_attn 层
            key_padding_mask (Tensor): 键的 padding 掩码，形状 (bs, num_keys)

        Returns:
            Tensor: 前向传播结果，形状与 query 相同
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query  # 保存残差连接的原始输入

        # 处理注意力掩码
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        f'to the number of attention in ' \
                        f'operation_order {self.num_attn}'

        # 按 operation_order 顺序执行操作
        for layer in self.operation_order:
            if layer == 'self_attn':
                # Self-Attention: query, key, value 都来自自身
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,  # pre-norm: 传入 identity 用于残差
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query  # 更新 identity 为当前输出

            elif layer == 'norm':
                # 归一化层
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                # Cross-Attention: query 来自自身，key/value 来自外部
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                # 前馈网络
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query


@ATTENTION.register_module()
class MotionDeformableAttention(BaseModule):
    """
    运动预测的可变形注意力模块。

    这是 UniAD 运动预测中智能体与 BEV 特征交互的核心组件。它基于参考轨迹
    （reference_trajs）在 BEV 特征图上进行可变形采样，使智能体能够感知其未来
    轨迹路径上的环境特征。

    核心特性：
    1. 多步采样 (multi-step sampling): 沿轨迹的多个时间步进行采样，
       采样偏移量由 query 通过线性层预测，维度为 num_steps。
    2. 多层级采样 (multi-level sampling): 在多个 BEV 特征层级上采样，
       每个层级对应不同的空间分辨率。
    3. 坐标转换: 将 agent 坐标系的轨迹转换到 ego（自车）坐标系，
       因为 BEV 特征图是在 ego 坐标系下定义的。
    4. 注意力权重: 每个采样点有一个注意力权重，由 query 通过线性层预测，
       用于加权聚合采样点特征。
    5. 圆形初始化: 采样偏移量使用均匀分布的圆形模式初始化，
       使初始采样点围绕参考点均匀分布。

    参考论文: Deformable DETR: Deformable Transformers for End-to-End Object Detection
    (https://arxiv.org/pdf/2010.04159.pdf)

    Args:
        embed_dims (int): 嵌入维度，默认 256。必须能被 num_heads 整除。
        num_heads (int): 并行注意力头数，默认 8。
        num_levels (int): 多尺度特征层级数，默认 4。
        num_points (int): 每个查询在每个注意力头中的采样点数，默认 4。
        num_steps (int): 沿轨迹的采样步数，默认 1。
        sample_index (int): 从参考轨迹中采样的时间步索引，默认 -1（最后一步）。
        im2col_step (int): image_to_column 操作的步长，默认 64。
        dropout (float): Dropout 比率，默认 0.1。
        bev_range (list): BEV 范围，格式 [x_min, y_min, z_min, x_max, y_max, z_max]。
            默认 [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        voxel_size (list): 体素大小，格式 [x_size, y_size, z_size]。
            默认 [0.2, 0.2, 8]
        batch_first (bool): 是否 batch 维度在前，默认 True。
        norm_cfg (dict): 归一化层配置。
        init_cfg (dict): 初始化配置。
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 num_steps=1,
                 sample_index=-1,
                 im2col_step=64,
                 dropout=0.1,
                 bev_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 voxel_size=[0.2, 0.2, 8],
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads  # 每个注意力头的维度
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.fp16_enabled = False
        self.bev_range = bev_range

        # 检查每个注意力头的维度是否是 2 的幂次方
        # CUDA 实现中，2 的幂次方更高效
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_steps = num_steps
        self.sample_index = sample_index

        # 采样偏移量预测: 对每个查询，预测每个注意力头、每个时间步、每个层级、每个采样点的 (x, y) 偏移
        # 输出维度: num_heads * num_steps * num_levels * num_points * 2
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_steps * num_levels * num_points * 2)

        # 注意力权重预测: 对每个查询，预测每个注意力头、每个时间步、每个层级、每个采样点的权重
        # 输出维度: num_heads * num_steps * num_levels * num_points
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_steps * num_levels * num_points)

        # 值投影: 将 BEV 特征投影到嵌入空间
        self.value_proj = nn.Linear(embed_dims, embed_dims)

        # 输出投影: 将多步采样的特征拼接后投影回嵌入空间
        # 输入维度: num_steps * embed_dims（多步特征拼接），输出: embed_dims
        self.output_proj = Sequential(
            nn.Linear(num_steps*embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True)
        )

        self.init_weights()

    def init_weights(self):
        """
        初始化模块参数。

        初始化策略：
        - sampling_offsets: 偏置使用圆形模式初始化，使初始采样点围绕参考点均匀分布。
          每个注意力头有不同的初始角度，每个采样点有不同的半径距离。
        - attention_weights: 初始化为 0，使初始注意力权重均匀。
        - value_proj 和 output_proj: 使用 Xavier 均匀初始化。
        """
        # 采样偏移量的权重初始化为 0
        constant_init(self.sampling_offsets, 0.)

        # 采样偏移量的偏置使用圆形模式初始化
        # 每个注意力头分配一个均匀分布的角度
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        # 生成单位圆上的点 (cos(theta), sin(theta))
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        # 归一化到 [-1, 1] 范围
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1, 1,
            2).repeat(1, self.num_steps, self.num_levels, self.num_points, 1)
        # 不同的采样点有不同的半径距离（i+1），实现多尺度采样
        for i in range(self.num_points):
            grid_init[:, :, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)

        # 注意力权重初始化为 0
        constant_init(self.attention_weights, val=0., bias=0.)

        # 值投影和输出投影使用 Xavier 均匀初始化
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiScaleDeformableAttention')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                spatial_shapes=None,
                level_start_index=None,
                bbox_results=None,
                reference_trajs=None,
                flag='decoder',
                **kwargs):
        """
        运动可变形注意力的前向传播。

        处理流程：
        1. 添加位置编码到查询
        2. 展平多模态查询维度
        3. 对 BEV 值特征进行投影
        4. 预测采样偏移量和注意力权重
        5. 将参考轨迹从 agent 坐标系转换到 ego 坐标系，并归一化
        6. 计算采样位置 = 参考点 + 采样偏移量 / 归一化因子
        7. 执行多尺度可变形注意力操作
        8. 合并多步采样结果并投影
        9. 恢复多模态查询形状，加入残差和 dropout

        Args:
            query (Tensor): 查询张量，形状 (B, A, P, D)
                B=batch_size, A=agent数量, P=模态数, D=embed_dims
            key (Tensor): 键张量（未使用）
            value (Tensor): 值张量，即 BEV 特征，形状 (B, H*W, D)
            identity (Tensor): 残差连接的输入，形状同 query
            query_pos (Tensor): 查询的位置编码，形状 (B, A, P, D)
            key_padding_mask (Tensor): BEV 特征的 padding 掩码
            spatial_shapes (Tensor): 每个特征层级的空间形状，形状 (num_levels, 2)
            level_start_index (Tensor): 每个层级的起始索引，形状 (num_levels,)
            bbox_results: 检测框结果列表，用于坐标转换
            reference_trajs (Tensor): 参考轨迹，形状 (B, A, P, S, num_levels, 2)
                S=预测步数, num_levels=层级数（通常为1）
            flag (str): 标记，'decoder' 表示用于解码器

        Returns:
            Tensor: 可变形注意力输出，形状 (B, A, P, D)
        """
        bs, num_agent, num_mode, _ = query.shape
        num_query = num_agent * num_mode  # 展平后的查询数量

        # 如果 value 未提供，使用 query 作为 value
        if value is None:
            value = query
        # 如果 identity 未提供，使用 query 作为 identity
        if identity is None:
            identity = query

        # 添加位置编码
        if query_pos is not None:
            query = query + query_pos

        # 展平 agent 和 mode 维度: (B, A, P, D) -> (B, A*P, D)
        query = torch.flatten(query, start_dim=1, end_dim=2)

        # 调整 value 维度: (B, H*W, D) -> (H*W, B, D) 适配多尺度可变形注意力
        value = value.permute(1, 0, 2)
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        # 对 BEV 值特征进行线性投影
        value = self.value_proj(value)
        # 应用 padding 掩码
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        # 重塑为多头格式: (H*W, B, D) -> (B, H*W, num_heads, D//num_heads)
        value = value.view(bs, num_value, self.num_heads, -1)

        # 预测采样偏移量: (B, A*P, D) -> (B, A*P, num_heads, num_steps, num_levels, num_points, 2)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_steps, self.num_levels, self.num_points, 2)

        # 预测注意力权重: (B, A*P, D) -> (B, A*P, num_heads, num_steps, num_levels*num_points)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_steps, self.num_levels * self.num_points)
        # 对注意力权重做 softmax 归一化
        attention_weights = attention_weights.softmax(-1)

        # 恢复形状: (B, A*P, num_heads, num_steps, num_levels, num_points)
        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_steps,
                                                   self.num_levels,
                                                   self.num_points)

        # 处理参考轨迹以计算采样位置
        if reference_trajs.shape[-1] == 2:
            # 从参考轨迹中选取指定时间步的采样点
            # reference_trajs: (B, A, P, S, num_levels, 2) -> (B, A, P, num_levels, 2)
            reference_trajs = reference_trajs[:, :, :, [self.sample_index], :, :]

            # 将参考轨迹从 agent 坐标系转换到 ego 坐标系
            # agent 坐标系的轨迹是相对于 agent 当前位置的偏移量
            # ego 坐标系是全局坐标系，BEV 特征图在 ego 坐标系下定义
            reference_trajs_ego = self.agent_coords_to_ego_coords(
                copy.deepcopy(reference_trajs), bbox_results).detach()

            # 展平 agent 和 mode 维度: (B, A, P, num_levels, 2) -> (B, A*P, num_levels, 2)
            reference_trajs_ego = torch.flatten(reference_trajs_ego, start_dim=1, end_dim=2)

            # 扩展维度以匹配采样偏移量的形状
            # (B, A*P, num_levels, 2) -> (B, A*P, 1, num_levels, 1, 2)
            reference_trajs_ego = reference_trajs_ego[:, :, None, :, :, None, :]

            # 将参考点归一化到 [0, 1] 范围（相对于 BEV 范围）
            reference_trajs_ego[..., 0] -= self.bev_range[0]  # 减去 x_min
            reference_trajs_ego[..., 1] -= self.bev_range[1]  # 减去 y_min
            reference_trajs_ego[..., 0] /= (self.bev_range[3] - self.bev_range[0])  # 除以 x 范围
            reference_trajs_ego[..., 1] /= (self.bev_range[4] - self.bev_range[1])  # 除以 y 范围

            # 计算偏移量归一化因子: 每个层级的 (width, height)
            # 采样偏移量除以归一化因子，确保偏移量在不同分辨率的特征层级上尺度一致
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)

            # 计算最终采样位置 = 归一化参考点 + 归一化偏移量
            sampling_locations = reference_trajs_ego \
                + sampling_offsets \
                / offset_normalizer[None, None, None, None, :, None, :]

            # 调整维度顺序: (B, A*P, nh, ns, nl, np, 2) -> (B, A*P, ns, nh, nl, np, 2)
            # 将 num_steps 维度移到前面，便于后续 reshape
            sampling_locations = rearrange(
                sampling_locations, 'bs nq nh ns nl np c -> bs nq ns nh nl np c')
            attention_weights = rearrange(
                attention_weights, 'bs nq nh ns nl np -> bs nq ns nh nl np')

            # 合并 num_query 和 num_steps 维度: (B, A*P*ns, nh, nl, np, 2)
            sampling_locations = sampling_locations.reshape(
                bs, num_query*self.num_steps, self.num_heads, self.num_levels, self.num_points, 2)
            attention_weights = attention_weights.reshape(
                bs, num_query*self.num_steps, self.num_heads, self.num_levels, self.num_points)

        else:
            raise ValueError(
                f'Last dim of reference_trajs must be'
                f' 2 or 4, but get {reference_trajs.shape[-1]} instead.')

        # 执行多尺度可变形注意力操作
        if torch.cuda.is_available() and value.is_cuda:
            # 使用 CUDA 实现的可变形注意力（fp32 版本，因为 fp16 下不稳定）
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            # 使用 PyTorch 纯 Python 实现的可变形注意力
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        # 恢复形状并合并多步特征
        # output: (B, A*P*ns, D//num_heads * num_heads) -> (B, A*P, ns, D)
        output = output.view(bs, num_query, self.num_steps, -1)
        # 合并 num_steps 和 embed_dims: (B, A*P, ns*D)
        output = torch.flatten(output, start_dim=2, end_dim=3)
        # 通过输出投影层: (B, A*P, ns*D) -> (B, A*P, D)
        output = self.output_proj(output)

        # 恢复多模态形状: (B, A*P, D) -> (B, A, P, D)
        output = output.view(bs, num_agent, num_mode, -1)

        # 残差连接 + dropout
        return self.dropout(output) + identity

    def agent_coords_to_ego_coords(self, reference_trajs, bbox_results):
        """
        将参考轨迹从 agent 坐标系转换到 ego（自车）坐标系。

        agent 坐标系的轨迹是相对于每个 agent 当前位置的偏移量，
        而 ego 坐标系是全局坐标系（以自车为原点）。
        BEV 特征图是在 ego 坐标系下定义的，因此需要将轨迹转换到 ego 坐标系。

        转换方式：将 agent 坐标系下的轨迹偏移量加上 agent 在 ego 坐标系下的位置。

        Args:
            reference_trajs (Tensor): agent 坐标系下的参考轨迹，
                形状 (B, A, P, num_levels, 2)
            bbox_results: 检测框结果列表，包含每个 agent 在 ego 坐标系下的位置

        Returns:
            Tensor: ego 坐标系下的参考轨迹，形状 (B, A, P, num_levels, 2)
        """
        batch_size = len(bbox_results)
        reference_trajs_ego = []
        for i in range(batch_size):
            # 从检测结果中提取 agent 的 3D 边界框
            boxes_3d, scores, labels, bbox_index, mask = bbox_results[i]
            # 获取 agent 在 ego 坐标系下的重力中心点坐标
            det_centers = boxes_3d.gravity_center.to(reference_trajs.device)
            # 提取当前 batch 样本的参考轨迹
            batch_reference_trajs = reference_trajs[i]
            # 将 agent 坐标系下的偏移量加上 agent 在 ego 坐标系下的位置
            # det_centers 形状: (A, 2) 或 (A, 3)，只取前 2 维 (x, y)
            batch_reference_trajs += det_centers[:, None, None, None, :2]
            reference_trajs_ego.append(batch_reference_trajs)
        return torch.stack(reference_trajs_ego)

    def rot_2d(self, yaw):
        """
        生成 2D 旋转矩阵。

        用于将坐标绕原点旋转指定的偏航角 (yaw)。

        Args:
            yaw (Tensor): 偏航角（弧度），形状可以是任意维度

        Returns:
            Tensor: 2D 旋转矩阵，形状 (..., 2, 2)
                旋转矩阵格式: [[cos(yaw), -sin(yaw)], [sin(yaw), cos(yaw)]]
        """
        sy, cy = torch.sin(yaw), torch.cos(yaw)
        out = torch.stack([torch.stack([cy, -sy]), torch.stack([sy, cy])]).permute([2, 0, 1])
        return out


@ATTENTION.register_module()
class CustomModeMultiheadAttention(BaseModule):
    """
    自定义的多模态多头注意力模块。

    封装了 PyTorch 的 nn.MultiheadAttention，支持多模态（multi-mode）维度的处理。
    在 UniAD 的运动预测中，每个智能体有多个运动模态（P 个），查询的形状为 (B, A, P, D)。
    该模块正确处理这种结构，将 B 和 A 维度合并后输入标准的 MultiheadAttention，
    然后在输出时恢复原始形状。

    主要特点：
    - 支持残差连接（identity）
    - 支持查询和键的位置编码
    - 支持 dropout（注意力 dropout 和投影 dropout）
    - 正确处理 batch_first 格式

    Args:
        embed_dims (int): 嵌入维度
        num_heads (int): 并行注意力头数
        attn_drop (float): 注意力权重的 dropout 比率，默认 0.0
        proj_drop (float): 输出投影后的 dropout 比率，默认 0.0
        dropout_layer (dict): 残差连接后的 dropout 配置，默认 dict(type='Dropout', drop_prob=0.)
        init_cfg (dict): 初始化配置
        **kwargs: 传递给 nn.MultiheadAttention 的其他参数
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg)

        # 处理已弃用的 dropout 参数，向后兼容
        if 'dropout' in kwargs:
            warnings.warn(
                'The arguments `dropout` in MultiheadAttention '
                'has been deprecated, now you can separately '
                'set `attn_drop`(float), proj_drop(float), '
                'and `dropout_layer`(dict) ', DeprecationWarning)
            attn_drop = kwargs['dropout']
            dropout_layer['drop_prob'] = kwargs.pop('dropout')

        self.embed_dims = embed_dims
        self.num_heads = num_heads

        # 构建 PyTorch 标准的 MultiheadAttention
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop, **kwargs)

        # 输出投影后的 dropout
        self.proj_drop = nn.Dropout(proj_drop)

        # 残差连接后的 dropout（如果未配置则使用恒等映射）
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()

    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiheadAttention')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):
        """
        自定义多模态多头注意力的前向传播。

        处理流程：
        1. 添加位置编码
        2. 展平 B 和 A 维度，适配 PyTorch MultiheadAttention 的输入格式
        3. 转置为 (num_query, batch, embed_dims) 格式
        4. 执行多头注意力
        5. 转置回 (batch, num_query, embed_dims) 格式
        6. 恢复 (B, A, P, D) 形状
        7. 添加残差连接和 dropout

        Args:
            query (Tensor): 查询张量，形状 (B, A, P, D)
            key (Tensor): 键张量，形状 (B, A, P, D)。如果为 None，使用 query
            value (Tensor): 值张量，形状同 key。如果为 None，使用 key
            identity (Tensor): 残差连接的输入，形状同 query。如果为 None，使用 query
            query_pos (Tensor): 查询位置编码，形状 (B, A, P, D)
            key_pos (Tensor): 键位置编码，形状 (B, A, P, D)
            attn_mask (Tensor): 注意力掩码，形状 (num_queries, num_keys)
            key_padding_mask (Tensor): 键的 padding 掩码，形状 (bs, num_keys)

        Returns:
            Tensor: 注意力输出，形状 (B, A, P, D)
        """
        # 为位置编码扩展维度: (B, A, D) -> (B, A, 1, D)
        # 因为 query 有模态维度 P，需要将位置编码广播到 P 维度
        query_pos = query_pos.unsqueeze(1)
        key_pos = key_pos.unsqueeze(1)

        bs, n_agent, n_query, D = query.shape

        # 设置默认值
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query

        # 如果 key_pos 未提供，尝试使用 query_pos
        if key_pos is None:
            if query_pos is not None:
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f'position encoding of key is'
                                  f'missing in {self.__class__.__name__}.')

        # 添加位置编码
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # 展平 B 和 A 维度: (B, A, P, D) -> (B*A, P, D)
        # 因为 PyTorch 的 MultiheadAttention 不支持 (B, A, P, D) 这种格式
        query = torch.flatten(query, start_dim=0, end_dim=1)
        key = torch.flatten(key, start_dim=0, end_dim=1)
        value = torch.flatten(value, start_dim=0, end_dim=1)
        identity = torch.flatten(identity, start_dim=0, end_dim=1)

        # 转置为 (num_query, batch, embed_dims) 格式
        # PyTorch MultiheadAttention 默认 batch_first=False
        query = query.transpose(0, 1)  # (P, B*A, D)
        key = key.transpose(0, 1)      # (P, B*A, D)
        value = value.transpose(0, 1)  # (P, B*A, D)

        # 执行多头注意力
        out = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask)[0]

        # 转置回 (batch, num_query, embed_dims) 格式
        out = out.transpose(0, 1)  # (B*A, P, D)

        # 残差连接: identity + dropout(proj_drop(out))
        out = identity + self.dropout_layer(self.proj_drop(out))

        # 恢复形状: (B*A, P, D) -> (B, A, P, D)
        return out.view(bs, n_agent, n_query, D)