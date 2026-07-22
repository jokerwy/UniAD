"""
seg_deformable_transformer.py - 分割任务中的可变形Transformer模块

本模块实现了用于分割任务的可变形DETR Transformer（SegDeformableTransformer），
继承自 mmdet 的 Transformer 基类。

可变形DETR（Deformable DETR）是一种改进的目标检测Transformer架构，其核心思想是：
- 使用可变形注意力机制（Deformable Attention），在特征图上只关注稀疏的采样点，
  而非全局注意力，从而大幅降低计算复杂度。
- 支持多尺度特征图输入，通过 level_embeds 区分不同层级。
- 支持两阶段（as_two_stage）模式：先从编码器输出生成初始候选框，再送入解码器精炼。

本模块的核心组件：
1. SegDeformableTransformer: 主Transformer类，包含编码器和解码器。
2. 参考点生成：为可变形注意力提供每个查询位置的参考点。
3. 位置编码生成：为候选框生成正弦位置编码。
4. 两阶段候选生成：从编码器输出提取初始候选框（top-k选择）。
"""

from mmcv.runner.fp16_utils import force_fp32
from mmdet.models.utils.builder import TRANSFORMER
from mmdet.models.utils import Transformer
import warnings
import math
import copy
import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_norm_layer, xavier_init
from mmcv.cnn.bricks.registry import (TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         MultiScaleDeformableAttention,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)
from mmcv.runner.base_module import BaseModule
from torch.nn.init import normal_

from mmdet.models.utils.builder import TRANSFORMER
from mmcv.cnn.bricks.registry import ATTENTION
from torch import einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


@TRANSFORMER.register_module()
class SegDeformableTransformer(Transformer):
    """
    分割任务专用的可变形DETR Transformer。

    该类实现了可变形DETR（Deformable DETR）的Transformer架构，用于分割任务中的
    特征编码和解码。与标准DETR Transformer相比，主要特点包括：

    1. 支持多尺度特征层级（FPN输出），通过 level_embeds 为每层特征添加可学习的位置编码。
    2. 使用可变形注意力机制，在特征图上稀疏采样，降低计算量。
    3. 支持两阶段模式（as_two_stage）：先从编码器输出生成初始候选框，
       再送入解码器进行精炼。
    4. 支持参考点（reference points）生成，为可变形注意力提供空间锚点。

    Args:
        as_two_stage (bool): 是否启用两阶段模式。若为True，从编码器特征生成初始查询。
            默认: False。
        num_feature_levels (int): FPN输出的特征图数量。默认: 4。
        two_stage_num_proposals (int): 两阶段模式下生成的候选框数量。默认: 300。
        **kwargs: 传递给父类 Transformer 的其他参数。
    """

    def __init__(self,
                 as_two_stage=False,
                 num_feature_levels=4,
                 two_stage_num_proposals=300,
                 **kwargs):
        super(SegDeformableTransformer, self).__init__(**kwargs)
        # 禁用FP16自动转换（使用 force_fp32 装饰器手动控制精度）
        self.fp16_enabled = False
        self.as_two_stage = as_two_stage
        self.num_feature_levels = num_feature_levels
        self.two_stage_num_proposals = two_stage_num_proposals
        # 从编码器配置中获取嵌入维度
        self.embed_dims = self.encoder.embed_dims
        # 初始化各层
        self.init_layers()

    def init_layers(self):
        """
        初始化可变形DETR Transformer的额外层。

        包括：
        - level_embeds: 为每个特征层级学习的嵌入向量，形状 (num_feature_levels, embed_dims)。
          用于区分不同尺度的特征图。
        - 两阶段模式下：编码器输出投影层（enc_output）、位置变换层（pos_trans）及其归一化层。
        - 非两阶段模式下：参考点预测层（reference_points），从查询嵌入预测参考点坐标。
        """
        # 可学习的层级嵌入：为每个特征层级添加独特的标识，帮助模型区分不同尺度
        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))

        if self.as_two_stage:
            # 两阶段模式：需要将编码器输出映射到解码器输入空间
            # enc_output: 对编码器输出进行线性投影
            self.enc_output = nn.Linear(self.embed_dims, self.embed_dims)
            self.enc_output_norm = nn.LayerNorm(self.embed_dims)
            # pos_trans: 将位置编码从 2*embed_dims 映射到 2*embed_dims（包含查询和位置两部分）
            self.pos_trans = nn.Linear(self.embed_dims * 2,
                                       self.embed_dims * 2)
            self.pos_trans_norm = nn.LayerNorm(self.embed_dims * 2)
        else:
            # 非两阶段模式：直接从查询嵌入预测2D参考点坐标
            self.reference_points = nn.Linear(self.embed_dims, 2)

    def init_weights(self):
        """
        初始化Transformer的权重。

        初始化策略：
        - 所有维度大于1的参数使用 Xavier 均匀初始化。
        - 多尺度可变形注意力模块使用其自身的初始化方法。
        - 非两阶段模式下，参考点预测层使用均匀分布的 Xavier 初始化，偏置为0。
        - 层级嵌入使用正态分布初始化。
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                try:
                    m.init_weight()
                except:
                    m.init_weights()
        if not self.as_two_stage:
            # 参考点预测层使用均匀分布初始化，偏置为0
            xavier_init(self.reference_points, distribution='uniform', bias=0.)
        # 层级嵌入使用正态分布初始化
        normal_(self.level_embeds)

    def gen_encoder_output_proposals(self, memory, memory_padding_mask,
                                     spatial_shapes):
        """
        从编码器输出生成候选框。

        在两阶段模式下，利用编码器输出的特征图生成初始候选框（proposals），
        这些候选框将作为解码器的初始查询。候选框的生成逻辑：

        1. 对于每个特征层级，在特征图上生成均匀网格点作为初始位置。
        2. 每个位置的候选框大小与该层级的分辨率相关（高层级特征对应更大的框）。
        3. 对候选框进行逆sigmoid变换，确保其在有效范围内。
        4. 过滤掉无效位置（padding区域和边界位置）的候选框。
        5. 通过 enc_output 和 enc_output_norm 对编码器输出进行投影和归一化。

        Args:
            memory (Tensor): 编码器的输出特征，形状为 (bs, num_key, embed_dim)。
                其中 num_key 是所有层级特征图上的点的总数。
            memory_padding_mask (Tensor): 编码器输出的padding掩码，
                形状为 (bs, num_key)。True 表示需要忽略的位置。
            spatial_shapes (Tensor): 所有特征图的形状，形状为 (num_level, 2)，
                每行表示 (H, W)。

        Returns:
            tuple:
                - output_memory (Tensor): 处理后的编码器输出，用于解码器输入。
                  形状为 (bs, num_key, embed_dim)。
                - output_proposals (Tensor): 逆sigmoid变换后的归一化候选框。
                  形状为 (bs, num_keys, 4)，格式为 (cx, cy, w, h)。
        """

        N, S, C = memory.shape
        proposals = []
        _cur = 0  # 当前层级在展平特征中的起始索引
        for lvl, (H, W) in enumerate(spatial_shapes):
            # 获取当前层级的padding掩码，重塑为 (N, H, W, 1)
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H * W)].view(
                N, H, W, 1)
            # 计算每个样本在当前层级上的有效高度和宽度
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            # 生成特征图上的网格坐标（均匀分布）
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0,
                               H - 1,
                               H,
                               dtype=torch.float32,
                               device=memory.device),
                torch.linspace(0,
                               W - 1,
                               W,
                               dtype=torch.float32,
                               device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            # 将网格坐标归一化到 [0, 1]：加上0.5偏移后除以有效尺寸
            scale = torch.cat([valid_W.unsqueeze(-1),
                               valid_H.unsqueeze(-1)], 1).view(N, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N, -1, -1, -1) + 0.5) / scale
            # 候选框的宽高：与层级相关，每层放大2倍，基础大小为0.05
            wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)
            proposal = torch.cat((grid, wh), -1).view(N, -1, 4)
            proposals.append(proposal)
            _cur += (H * W)

        # 拼接所有层级的候选框
        output_proposals = torch.cat(proposals, 1)
        # 检查候选框是否在有效范围内（0.01 < value < 0.99）
        output_proposals_valid = ((output_proposals > 0.01) &
                                  (output_proposals < 0.99)).all(-1,
                                                                 keepdim=True)
        # 逆sigmoid变换：将 [0,1] 范围的坐标映射到实数空间，
        # 这样解码器可以更容易地回归精炼值
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        # 将padding区域和无效区域的候选框设为无穷大（表示无效）
        output_proposals = output_proposals.masked_fill(
            memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(
            ~output_proposals_valid, float('inf'))

        # 处理编码器输出：将padding和无效位置的特征清零，然后投影+归一化
        output_memory = memory
        output_memory = output_memory.masked_fill(
            memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid,
                                                  float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """
        获取解码器中使用的参考点。

        参考点是可变形注意力机制中的关键概念：解码器的每个查询位置需要一组
        参考点来确定在编码器特征图上何处进行采样。该函数为每个特征层级上
        的每个空间位置生成归一化的参考点坐标。

        生成逻辑：
        1. 对于每个特征层级，在特征图上生成均匀网格点（从0.5到H-0.5）。
        2. 将网格坐标归一化到 [0, 1]。
        3. 再乘以 valid_ratios，确保无效区域（padding）的参考点也被调整。

        Args:
            spatial_shapes (Tensor): 所有特征图的形状，形状为 (num_level, 2)。
            valid_ratios (Tensor): 有效区域的比例，形状为 (bs, num_levels, 2)，
                表示每个特征层级上沿高度和宽度方向的有效比例。
            device (torch.device): 目标设备。

        Returns:
            Tensor: 解码器使用的参考点，形状为 (bs, num_keys, num_levels, 2)。
        """
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            # 生成特征图上的网格坐标，从0.5开始（网格中心）
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5,
                               H - 0.5,
                               H,
                               dtype=torch.float32,
                               device=device),
                torch.linspace(0.5,
                               W - 0.5,
                               W,
                               dtype=torch.float32,
                               device=device))
            # 归一化 y 坐标
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] *
                                               H)
            # 归一化 x 坐标
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] *
                                               W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        # 拼接所有层级的参考点，形状: (bs, total_num_keys, num_levels, 2)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def get_valid_ratio(self, mask):
        """
        计算特征图的有效区域比例。

        该函数用于处理padding问题：在批处理中，不同图像可能被填充到相同尺寸，
        有效区域比例可以帮助模型知道哪些位置是真实的图像内容，哪些是padding。

        Args:
            mask (Tensor): 特征图的padding掩码，形状为 (bs, H, W)。
                True 表示padding位置，False 表示有效位置。

        Returns:
            Tensor: 有效区域比例，形状为 (bs, 2)，
                第一列是宽度方向的有效比例 (valid_ratio_w)，
                第二列是高度方向的有效比例 (valid_ratio_h)。
        """
        _, H, W = mask.shape
        # 统计高度方向上的有效像素数
        valid_H = torch.sum(~mask[:, :, 0], 1)
        # 统计宽度方向上的有效像素数
        valid_W = torch.sum(~mask[:, 0, :], 1)
        # 计算有效比例
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def get_proposal_pos_embed(self,
                               proposals,
                               num_pos_feats=128,
                               temperature=10000):
        """
        获取候选框的位置编码（正弦位置编码）。

        该函数为候选框的4个坐标值（cx, cy, w, h）生成正弦位置编码，
        使用类似Transformer中标准位置编码的正弦/余弦函数。

        处理流程：
        1. 对候选框坐标应用sigmoid，将其映射到 [0, 1]，然后乘以 2*pi 缩放到 [0, 2*pi]。
        2. 对每个坐标值，使用不同频率的正弦和余弦函数进行编码。
        3. 将4个坐标的编码拼接在一起，得到最终的候选框位置编码。

        Args:
            proposals (Tensor): 候选框坐标，形状为 (N, L, 4)，
                格式为 (cx, cy, w, h)，经过逆sigmoid变换。
            num_pos_feats (int): 每个坐标值的位置编码维度，默认128。
            temperature (int): 温度参数，控制频率范围，默认10000。

        Returns:
            Tensor: 候选框的位置编码，形状为 (N, L, 4*num_pos_feats) = (N, L, 512)。
        """
        scale = 2 * math.pi
        # 生成频率维度上的缩放因子，模拟不同频率的正弦波
        dim_t = torch.arange(num_pos_feats,
                             dtype=torch.float32,
                             device=proposals.device)
        dim_t = temperature**(2 * (dim_t // 2) / num_pos_feats)
        # 对候选框坐标应用sigmoid并缩放到 [0, 2*pi]
        proposals = proposals.sigmoid() * scale
        # 将每个坐标值除以不同频率的缩放因子，形状: (N, L, 4, 128)
        pos = proposals[:, :, :, None] / dim_t
        # 对偶数索引用sin，奇数索引用cos，重塑为 (N, L, 4*128) = (N, L, 512)
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()),
                          dim=4).flatten(2)
        return pos

    @force_fp32(apply_to=('mlvl_feats', 'query_embed', 'mlvl_pos_embeds'))
    def forward(self,
                mlvl_feats,
                mlvl_masks,
                query_embed,
                mlvl_pos_embeds,
                reg_branches=None,
                cls_branches=None,
                **kwargs):
        """
        Transformer的前向传播函数。

        完整的处理流程：
        1. 多尺度特征预处理：展平多层特征图，添加层级嵌入，拼接所有层级的特征。
        2. 计算参考点：为可变形注意力生成参考点。
        3. 编码器：对多尺度特征进行编码，提取全局上下文信息。
        4. 两阶段处理（可选）：
           - 从编码器输出生成初始候选框。
           - 使用top-k选择质量最高的候选框。
           - 生成候选框的位置编码作为解码器查询。
        5. 非两阶段处理：使用可学习的查询嵌入作为解码器输入。
        6. 解码器：基于编码器输出和查询嵌入进行解码，输出精炼后的特征。

        Args:
            mlvl_feats (list[Tensor]): 不同层级的输入特征图。
                每个元素的形状为 [bs, embed_dims, h, w]。
            mlvl_masks (list[Tensor]): 不同层级的padding掩码，用于编码器和解码器。
                每个元素的形状为 [bs, h, w]。True 表示padding位置。
            query_embed (Tensor): 解码器的查询嵌入，形状为 [num_query, c]。
            mlvl_pos_embeds (list[Tensor]): 不同层级的特征位置编码。
                每个元素的形状为 [bs, embed_dims, h, w]。
            reg_branches (nn.ModuleList, optional): 各解码器层的回归头。
                仅在 `with_box_refine` 为 True 时传入。默认 None。
            cls_branches (nn.ModuleList, optional): 各解码器层的分类头。
                仅在 `as_two_stage` 为 True 时传入。默认 None。

        Returns:
            tuple:
                - (memory, lvl_pos_embed_flatten, mask_flatten, query_pos):
                  编码器输出和相关信息，用于后续处理（如掩码头）。
                - inter_states: 解码器的中间输出。
                  如果 return_intermediate_dec 为 True，形状为
                  (num_dec_layers, bs, num_query, embed_dims)，否则为
                  (1, bs, num_query, embed_dims)。
                - init_reference_out: 参考点的初始值，形状为 (bs, num_queries, 4)。
                - inter_references_out: 解码器中参考点的中间值，
                  形状为 (num_dec_layers, bs, num_query, embed_dims)。
                - enc_outputs_class: 编码器生成的候选框的分类分数。
                  仅在 as_two_stage 为 True 时返回，否则为 None。
                - enc_outputs_coord_unact: 编码器生成的候选框的回归结果。
                  仅在 as_two_stage 为 True 时返回，否则为 None。
        """

        assert self.as_two_stage or query_embed is not None
        # 初始化存储列表
        feat_flatten = []          # 展平后的特征
        mask_flatten = []          # 展平后的掩码
        lvl_pos_embed_flatten = [] # 展平后的位置编码（含层级嵌入）
        spatial_shapes = []        # 各层级的空间形状

        # 遍历每个特征层级，进行展平和拼接
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            # 将特征从 (bs, c, h, w) 展平为 (bs, h*w, c)
            feat = feat.flatten(2).transpose(1, 2)
            # 将掩码从 (bs, h, w) 展平为 (bs, h*w)
            mask = mask.flatten(1)
            # 将位置编码从 (bs, c, h, w) 展平为 (bs, h*w, c)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            # 添加层级嵌入，使模型能区分不同尺度的特征
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)

        # 拼接所有层级的特征、掩码和位置编码
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        # 将空间形状列表转为张量
        spatial_shapes = torch.as_tensor(spatial_shapes,
                                         dtype=torch.long,
                                         device=feat_flatten.device)
        # 计算每个层级在展平特征中的起始索引
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        # 计算每个特征层级的有效比例
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)

        # 为编码器生成参考点
        reference_points = \
            self.get_reference_points(spatial_shapes,
                                      valid_ratios,
                                      device=feat.device)

        # 将特征和位置编码转为 (seq_len, bs, embed_dims) 格式，
        # 这是Transformer编码器期望的输入格式
        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(
            1, 0, 2)  # (H*W, bs, embed_dims)

        # 编码器：对多尺度特征进行编码，提取全局上下文
        memory = self.encoder(query=feat_flatten,
                              key=None,
                              value=None,
                              query_pos=lvl_pos_embed_flatten,
                              query_key_padding_mask=mask_flatten,
                              spatial_shapes=spatial_shapes,
                              reference_points=reference_points,
                              level_start_index=level_start_index,
                              valid_ratios=valid_ratios,
                              **kwargs)

        # 将编码器输出恢复为 (bs, seq_len, embed_dims) 格式
        memory = memory.permute(1, 0, 2)
        bs, _, c = memory.shape

        if self.as_two_stage:
            # ===== 两阶段模式 =====
            # 阶段1: 从编码器输出生成初始候选框
            output_memory, output_proposals = \
                self.gen_encoder_output_proposals(
                    memory, mask_flatten, spatial_shapes)
            # 对候选框进行分类和回归
            enc_outputs_class = cls_branches[self.decoder.num_layers](
                output_memory)
            enc_outputs_coord_unact = \
                reg_branches[
                    self.decoder.num_layers](output_memory) + output_proposals

            # Top-k 选择：按分类分数选取最优的候选框
            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk,
                                        dim=1)[1]
            topk_coords_unact = torch.gather(
                enc_outputs_coord_unact, 1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points

            # 生成候选框的位置编码，并拆分为查询位置编码和查询内容
            pos_trans_out = self.pos_trans_norm(
                self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            query_pos, query = torch.split(pos_trans_out, c, dim=2)
        else:
            # ===== 非两阶段模式 =====
            # 将查询嵌入拆分为位置部分和内容部分
            # query_embed 的形状为 (num_query, 2*C)，前C维是位置编码，后C维是查询内容
            query_pos, query = torch.split(query_embed, c, dim=1)
            # 扩展到批次维度
            query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
            query = query.unsqueeze(0).expand(bs, -1, -1)
            # 从查询位置编码预测参考点，并使用sigmoid归一化到 [0, 1]
            reference_points = self.reference_points(query_pos).sigmoid()
            init_reference_out = reference_points

        # ===== 解码器 =====
        # 将查询和编码器输出转为 (seq_len, bs, embed_dims) 格式
        query = query.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        # 解码器：基于编码器输出对查询进行精炼
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=memory,
            query_pos=query_pos,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=reg_branches,
            **kwargs)
        inter_references_out = inter_references

        if self.as_two_stage:
            return (memory,lvl_pos_embed_flatten,mask_flatten,query_pos), inter_states, init_reference_out,\
                inter_references_out, enc_outputs_class,\
                enc_outputs_coord_unact
        return (memory,lvl_pos_embed_flatten,mask_flatten,query_pos), inter_states, init_reference_out, \
            inter_references_out, None, None