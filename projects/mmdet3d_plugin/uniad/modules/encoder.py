
# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

"""
BEVFormer 编码器 (Encoder)
==========================
包含两个核心类:

1. BEVFormerEncoder: 编码器序列，由多层 BEVFormerLayer 堆叠而成
   负责将多视角图像特征通过时空注意力转换为 BEV 鸟瞰图特征

2. BEVFormerLayer: 编码器的单层实现
   每层依次执行: 时序自注意力 (TSA) → 归一化 → 空间交叉注意力 (SCA) → 归一化 → FFN

核心流程 (Encoder 单层):
    BEV Query (当前帧 BEV 查询)
        │
        ├── 时序自注意力 (TSA): 与上一帧 BEV 特征交互
        │   └── 将当前帧 BEV 查询与旋转对齐后的历史 BEV 做可变形注意力
        │
        ├── 空间交叉注意力 (SCA): 与多视角图像特征交互
        │   └── 将 BEV 3D 参考点投影到 6 个相机图像上采样特征
        │
        └── FFN: 前馈网络进行特征变换
"""

from .custom_base_transformer_layer import MyCustomBaseTransformerLayer
import copy
import warnings
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.runner import force_fp32, auto_fp16
import numpy as np
import torch
import cv2 as cv
import mmcv
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.utils import ext_loader
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class BEVFormerEncoder(TransformerLayerSequence):
    """BEVFormer 编码器

    由多层 BEVFormerLayer 堆叠而成，每层都包含时序自注意力和空间交叉注意力。
    负责将多视角图像特征转换为 BEV 鸟瞰图特征。

    编码器每层的计算:
        1. 时序自注意力 (TSA): BEV queries 与上一帧 BEV 特征交互
        2. 空间交叉注意力 (SCA): BEV queries 在图像特征上采样
        3. 前馈网络 (FFN): 特征变换

    Args:
        pc_range: 点云范围 [x_min, y_min, z_min, x_max, y_max, z_max]
        num_points_in_pillar: 每个 pillar 中的采样点数 (用于 3D 参考点)
        return_intermediate: 是否返回中间层输出
    """

    def __init__(self, *args, pc_range=None, num_points_in_pillar=4, return_intermediate=False, dataset_type='nuscenes',
                 **kwargs):

        super(BEVFormerEncoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range
        self.fp16_enabled = False

    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """生成 BEV 网格的参考点

        参考点用于引导注意力机制在何处采样特征:
        - 3D 参考点: 用于空间交叉注意力 (SCA)，将 BEV 网格点投影到图像上
        - 2D 参考点: 用于时序自注意力 (TSA)，在 BEV 平面上定位

        Args:
            H, W: BEV 网格的空间尺寸 (如 200×200)
            Z: BEV pillar 的高度 (通常为 8，表示在高度方向上采样 8 个点)
            num_points_in_pillar: 每个 pillar 中采样点数 (默认 4)
            dim: 参考点维度
                '3d' → 3D 参考点 (x, y, z)，用于 SCA
                '2d' → 2D 参考点 (x, y)，用于 TSA
            bs: batch size
            device: 计算设备
            dtype: 数据类型

        Returns:
            3D 参考点: (bs, num_points_in_pillar, bev_h*bev_w, 3)
                归一化到 [0, 1] 范围，表示在 BEV 空间中的相对位置
            2D 参考点: (bs, bev_h*bev_w, 1, 2)
                归一化到 [0, 1] 范围，表示在 BEV 平面上的相对位置
        """

        # ---- 3D 参考点: 用于空间交叉注意力 (SCA) ----
        # 在 BEV 网格的每个位置生成 num_points_in_pillar 个不同高度的采样点
        if dim == '3d':
            # z 坐标: 在 [0.5, Z-0.5] 范围内均匀采样 num_points_in_pillar 个点
            # 归一化到 [0, 1]，形状 (num_points_in_pillar, H, W)
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z

            # x 坐标: 在 [0.5, W-0.5] 范围内均匀采样，归一化到 [0, 1]
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W

            # y 坐标: 在 [0.5, H-0.5] 范围内均匀采样，归一化到 [0, 1]
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H

            # 堆叠 (x, y, z) → (num_points_in_pillar, H, W, 3)
            ref_3d = torch.stack((xs, ys, zs), -1)

            # 展平空间维度: (num_points_in_pillar, H, W, 3) → (num_points_in_pillar, H*W, 3)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)

            # 扩展 batch 维度: (1, num_points_in_pillar, H*W, 3)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # ---- 2D 参考点: 用于时序自注意力 (TSA) ----
        elif dim == '2d':
            # 使用 meshgrid 生成 BEV 平面上的网格坐标
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
            )

            # 展平并归一化到 [0, 1]
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W

            # 堆叠: (bs, H*W, 1, 2)
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    # This function must use fp32!!!
    @force_fp32(apply_to=('reference_points', 'img_metas'))
    def point_sampling(self, reference_points, pc_range, img_metas):
        """将 3D BEV 参考点投影到多视角图像上

        核心步骤:
        1. 将归一化的 BEV 参考点反归一化到实际 3D 坐标
        2. 通过 lidar2img 变换矩阵将 3D 点投影到每个相机图像平面
        3. 生成 bev_mask 标记哪些参考点落在图像范围内

        Args:
            reference_points: 3D 归一化参考点 (bs, num_points, H*W, 3)
            pc_range: 点云范围 [x_min, y_min, z_min, x_max, y_max, z_max]
            img_metas: 图像元信息 (包含 lidar2img 变换矩阵)

        Returns:
            reference_points_cam: 投影到图像平面的参考点坐标
                形状 (num_cam, bs, H*W, num_points, 2)
                值在 [0, 1] 范围内，表示图像上的归一化像素坐标
            bev_mask: 每个参考点在各相机图像上的可见性掩码
                形状 (num_cam, bs, H*W, num_points)
                True 表示该参考点投影后在该相机图像内
        """

        # 提取 lidar→image 变换矩阵: (B, N, 4, 4)
        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)
        lidar2img = reference_points.new_tensor(lidar2img)
        reference_points = reference_points.clone()

        # Step 1: 反归一化 - 将归一化坐标恢复到实际 3D 坐标
        # [0,1] → [pc_range[0], pc_range[3]] 等
        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (pc_range[5] - pc_range[2]) + pc_range[2]

        # Step 2: 齐次坐标变换 - 添加齐次维度 (x, y, z, 1)
        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1)

        # 调整维度: (D, B, num_query, 4) 其中 D=num_points_in_pillar
        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)

        # 扩展维度以匹配每个相机: (D, B, 1, num_query, 4) → (D, B, num_cam, num_query, 4)
        reference_points = reference_points.view(
            D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)

        lidar2img = lidar2img.view(
            1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

        # Step 3: 投影 - 3D lidar 坐标 → 2D 图像像素坐标
        # (D, B, num_cam, num_query, 4) × (D, B, num_cam, num_query, 4, 4)
        # → (D, B, num_cam, num_query, 4) 即 (x, y, z, 1)
        reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                            reference_points.to(torch.float32)).squeeze(-1)
        eps = 1e-5

        # Step 4: 透视除法 - (x, y, z) → (x/z, y/z)
        # 深度 z > eps 的点才是有效的 (在相机前方)
        bev_mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)

        # Step 5: 归一化到图像尺寸范围 [0, 1]
        reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]  # 除以图像宽度
        reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]  # 除以图像高度

        # Step 6: 生成可见性掩码 - 投影点必须在图像边界内
        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))

        # 处理 NaN 值
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            bev_mask = torch.nan_to_num(bev_mask)
        else:
            bev_mask = bev_mask.new_tensor(
                np.nan_to_num(bev_mask.cpu().numpy()))

        # 调整维度顺序: (num_cam, bs, H*W, num_points, 2)
        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        return reference_points_cam, bev_mask

    @auto_fp16()
    def forward(self,
                bev_query,
                key,
                value,
                *args,
                bev_h=None,
                bev_w=None,
                bev_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                valid_ratios=None,
                prev_bev=None,
                shift=0.,
                img_metas=None,
                **kwargs):
        """编码器前向传播

        将 BEV queries 通过多层 BEVFormerLayer，每层依次执行:
        TSA (时序自注意力) → 归一化 → SCA (空间交叉注意力) → 归一化 → FFN

        时序 BEV 融合机制:
        1. 将上一帧 BEV (prev_bev) 和当前帧 BEV query 拼接在一起
        2. 对上一帧 BEV 的 2D 参考点施加 shift 偏移 (自车运动补偿)
        3. 通过 TSA，当前帧 BEV query 可以关注上一帧 BEV 中相关的空间位置

        Args:
            bev_query: BEV 查询向量 (num_query, bs, embed_dims)
            key/value: 多视角图像特征 (num_cam, ΣH*W, bs, embed_dims)
            bev_h, bev_w: BEV 网格尺寸
            bev_pos: BEV 位置编码
            spatial_shapes: 多尺度特征空间形状
            level_start_index: 多尺度特征起始索引
            prev_bev: 上一帧 BEV 特征
            shift: 自车位移偏移 (bs, 2)
            img_metas: 图像元信息

        Returns:
            output: BEV 特征 (num_query, bs, embed_dims)
        """

        output = bev_query
        intermediate = []

        # 生成 3D 参考点 (用于 SCA 投影到图像)
        ref_3d = self.get_reference_points(
            bev_h, bev_w, self.pc_range[5]-self.pc_range[2], self.num_points_in_pillar, dim='3d', bs=bev_query.size(1),  device=bev_query.device, dtype=bev_query.dtype)

        # 生成 2D 参考点 (用于 TSA 在 BEV 平面上定位)
        ref_2d = self.get_reference_points(
            bev_h, bev_w, dim='2d', bs=bev_query.size(1), device=bev_query.device, dtype=bev_query.dtype)

        # 将 3D 参考点投影到 6 个相机图像上
        reference_points_cam, bev_mask = self.point_sampling(
            ref_3d, self.pc_range, img_metas)

        # 对上一帧 BEV 的 2D 参考点施加自车位移偏移
        # 这样 TSA 在上一帧 BEV 上采样时，能够补偿自车运动带来的位置变化
        shift_ref_2d = ref_2d.clone()
        shift_ref_2d += shift[:, None, None, :]

        # 调整 BEV query 和位置编码的维度
        bev_query = bev_query.permute(1, 0, 2)  # (num_query, bs, C) → (bs, num_query, C)
        bev_pos = bev_pos.permute(1, 0, 2)
        bs, len_bev, num_bev_level, _ = ref_2d.shape

        # 构建混合参考点: 将上一帧 BEV 和当前帧 BEV 的参考点拼接
        if prev_bev is not None:
            prev_bev = prev_bev.permute(1, 0, 2)

            # 将 prev_bev 和当前 bev_query 拼接: (bs, 2, len_bev, C) → (bs*2, len_bev, C)
            prev_bev = torch.stack(
                [prev_bev, bev_query], 1).reshape(bs*2, len_bev, -1)

            # 混合参考点: 上一帧用 shift_ref_2d (带位移补偿), 当前帧用 ref_2d
            hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(
                bs*2, len_bev, num_bev_level, 2)
        else:
            # 没有上一帧 BEV → 当前帧参考点复用两次 (保持维度一致)
            hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(
                bs*2, len_bev, num_bev_level, 2)

        # 逐层前向传播
        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                key,
                value,
                *args,
                bev_pos=bev_pos,
                ref_2d=hybird_ref_2d,          # 混合 2D 参考点 (TSA 使用)
                ref_3d=ref_3d,                  # 3D 参考点 (SCA 使用)
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,  # 投影到图像上的参考点
                bev_mask=bev_mask,              # 可见性掩码
                prev_bev=prev_bev,              # 上一帧 BEV (TSA 使用)
                **kwargs)

            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


@TRANSFORMER_LAYER.register_module()
class BEVFormerLayer(MyCustomBaseTransformerLayer):
    """BEVFormer 编码器的单层实现

    继承自 MyCustomBaseTransformerLayer，实现了 BEVFormer 特有的注意力机制。

    每层的操作顺序 (operation_order):
        1. self_attn (时序自注意力 TSA): 当前 BEV 查询关注上一帧 BEV 特征
        2. norm: LayerNorm 归一化
        3. cross_attn (空间交叉注意力 SCA): BEV 查询在图像特征上采样
        4. norm: LayerNorm 归一化
        5. ffn: 前馈网络

    Args:
        attn_cfgs: 注意力模块配置列表，包含 2 个注意力配置:
            [0] 时序自注意力 (TemporalSelfAttention) 配置
            [1] 空间交叉注意力 (SpatialCrossAttention) 配置
        feedforward_channels: FFN 隐藏层维度
        ffn_dropout: FFN dropout 概率
        operation_order: 操作执行顺序
            BEVFormer 使用 ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(BEVFormerLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False
        # BEVFormer 层必须有 6 个操作: self_attn, norm, cross_attn, norm, ffn, norm
        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])

    def forward(self,
                query,
                key=None,
                value=None,
                bev_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                reference_points_cam=None,
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
                prev_bev=None,
                **kwargs):
        """BEVFormer 层的前向传播

        按照 operation_order 依次执行:
        1. 时序自注意力 (TSA): query 关注 prev_bev 中相关空间位置
        2. 空间交叉注意力 (SCA): query 在图像特征上采样
        3. FFN: 特征变换

        Args:
            query: BEV 查询 (bs, num_query, C)
            key/value: 图像特征 (num_cam, ΣH*W, bs, C)
            bev_pos: BEV 位置编码
            ref_2d: 2D 参考点 (用于 TSA)
            ref_3d: 3D 参考点 (用于 SCA)
            reference_points_cam: 投影到图像平面的参考点
            mask: 可见性掩码 (bev_mask)
            prev_bev: 上一帧 BEV 特征 (TSA 的 key/value)
            spatial_shapes: 多尺度特征空间形状
            level_start_index: 多尺度特征起始索引

        Returns:
            query: 更新后的 BEV 特征 (bs, num_query, C)
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
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

        for layer in self.operation_order:
            # ---- 时序自注意力 (TSA) ----
            # 当前帧 BEV query 关注上一帧 BEV 特征 (prev_bev)
            # 通过 2D 参考点，在 BEV 平面上的对应位置采样历史特征
            if layer == 'self_attn':

                query = self.attentions[attn_index](
                    query,
                    prev_bev,          # key: 上一帧 BEV
                    prev_bev,          # value: 上一帧 BEV
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=ref_2d,           # 2D 参考点
                    spatial_shapes=torch.tensor(
                        [[bev_h, bev_w]], device=query.device),
                    level_start_index=torch.tensor([0], device=query.device),
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            # ---- 空间交叉注意力 (SCA) ----
            # BEV query 关注多视角图像特征
            # 通过 3D 参考点投影到每个相机，在图像特征上采样
            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,               # 图像特征
                    value,             # 图像特征
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,                  # 3D 参考点
                    reference_points_cam=reference_points_cam, # 投影后的图像坐标
                    mask=mask,                                # 可见性掩码
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query
