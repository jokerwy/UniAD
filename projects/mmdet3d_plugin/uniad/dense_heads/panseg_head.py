#----------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)   #
# Source code: https://github.com/OpenDriveLab/UniAD                               #
# Copyright (c) OpenDriveLab. All rights reserved.                                 #
# Modified from panoptic_segformer (https://github.com/zhiqi-li/Panoptic-SegFormer)#
#--------------------------------------------------------------------------------- #

"""
全景分割头 (PansegformerHead)
=============================
基于 Deformable DETR 架构的全景分割头，对 BEV 特征进行全景分割。
将场景中的元素分为"物体"(things, 可数目标，如车道线、车辆)和"背景"(stuff, 不可数区域，如可行驶区域)。

核心流程:
    1. Location Decoder (位置解码器): 对 BEV 特征进行物体检测，产出 class/bbox/query
    2. Query Filter (查询过滤): 使用 Hungarian 匹配，过滤低质量查询
    3. Mask Decoder for Things (物体掩码解码器): 对过滤后的物体查询生成掩码
    4. Mask Decoder for Stuff (背景掩码解码器): 对 stuff 查询生成掩码
    5. 合并: 将 things 和 stuff 掩码合并为全景分割结果

与原始 DETR 的区别:
    - 输入是 BEV 特征而非图像特征
    - 同时处理 things 和 stuff 两种查询
    - 使用 query filter 机制减少计算量
    - 支持车道线、可行驶区域等多种分割任务
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob, constant_init
from mmcv.runner import force_fp32, auto_fp16
from mmdet.core import multi_apply
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models.builder import HEADS, build_loss
from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.models.utils import build_transformer
from .seg_head_plugin import SegDETRHead, IOU

@HEADS.register_module()
class PansegformerHead(SegDETRHead):
    """
    全景分割头 (Panoptic SegFormer Head)

    继承自 SegDETRHead，在 DETR 检测头的基础上增加了全景分割能力。
    将分割任务分为 things (物体/可数目标) 和 stuff (背景/不可数区域) 两部分处理。

    Args:
        bev_h, bev_w: BEV 特征图的高度和宽度
        canvas_size: 画布尺寸 (用于分割结果输出)
        pc_range: 点云范围 [x_min, y_min, z_min, x_max, y_max, z_max]
        with_box_refine: 是否在 decoder 中逐层精修参考点
        as_two_stage: 是否使用两阶段 (encoder 特征生成 proposal)
        transformer: Encoder 和 Decoder 的配置
        quality_threshold_things: things 类别的质量阈值 (score 过滤)
        quality_threshold_stuff: stuff 类别的质量阈值
        overlap_threshold_things: things 类别的重叠阈值 (NMS)
        overlap_threshold_stuff: stuff 类别的重叠阈值
        thing_transformer_head: 物体掩码解码器的 Transformer 配置
        stuff_transformer_head: 背景掩码解码器的 Transformer 配置
        loss_mask: 掩码损失配置 (DiceLoss)
        train_cfg: 训练配置 (assigner, sampler)
    """

    def __init__(
            self,
            *args,
            bev_h,                          # BEV 特征图高度
            bev_w,                          # BEV 特征图宽度
            canvas_size,                    # 输出画布尺寸
            pc_range,                       # 点云范围 [x_min, y_min, z_min, x_max, y_max, z_max]
            with_box_refine=False,          # 是否在 decoder 中逐层精修参考点
            as_two_stage=False,             # 是否使用两阶段
            transformer=None,               # Transformer 配置
            quality_threshold_things=0.25,  # things 类别质量阈值
            quality_threshold_stuff=0.25,   # stuff 类别质量阈值
            overlap_threshold_things=0.4,   # things 类别重叠阈值
            overlap_threshold_stuff=0.2,    # stuff 类别重叠阈值
            thing_transformer_head=dict(
                type='TransformerHead',  # mask decoder for things - 物体掩码解码器
                d_model=256,
                nhead=8,
                num_decoder_layers=6),
            stuff_transformer_head=dict(
                type='TransformerHead',  # mask decoder for stuff - 背景掩码解码器
                d_model=256,
                nhead=8,
                num_decoder_layers=6),
            loss_mask=dict(type='DiceLoss', weight=2.0),  # 掩码损失: Dice Loss
            train_cfg=dict(
                assigner=dict(type='HungarianAssigner',       # 匈牙利匹配器
                              cls_cost=dict(type='ClassificationCost',  # 分类代价
                                            weight=1.),
                              reg_cost=dict(type='BBoxL1Cost', weight=5.0),  # 回归 L1 代价
                              iou_cost=dict(type='IoUCost',            # IoU 代价
                                            iou_mode='giou',
                                            weight=2.0)),
                sampler=dict(type='PseudoSampler'),  # 伪采样器 (不做实际采样)
            ),
            **kwargs):
        self.bev_h = bev_h                              # BEV 高度
        self.bev_w = bev_w                              # BEV 宽度
        self.canvas_size = canvas_size                  # 输出画布尺寸
        self.pc_range = pc_range                        # 点云范围
        self.real_w = self.pc_range[3] - self.pc_range[0]  # 实际宽度 (x 方向)
        self.real_h = self.pc_range[4] - self.pc_range[1]  # 实际高度 (y 方向)

        self.with_box_refine = with_box_refine          # 是否逐层精修框
        self.as_two_stage = as_two_stage                # 是否两阶段
        self.quality_threshold_things = 0.1             # things 质量阈值 (注释显示为 0.1)
        self.quality_threshold_stuff = quality_threshold_stuff    # stuff 质量阈值
        self.overlap_threshold_things = overlap_threshold_things  # things 重叠阈值
        self.overlap_threshold_stuff = overlap_threshold_stuff    # stuff 重叠阈值
        self.fp16_enabled = False                       # 禁用 fp16 (数值稳定性)

        # 两阶段模式下，encoder 输出作为 proposal
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        self.num_dec_things = thing_transformer_head['num_decoder_layers']  # things 解码器层数
        self.num_dec_stuff = stuff_transformer_head['num_decoder_layers']   # stuff 解码器层数
        super(PansegformerHead, self).__init__(*args,
                                            transformer=transformer,
                                            train_cfg=train_cfg,
                                            **kwargs)
        # 训练配置: 构建带掩码匹配的采样器和分配器
        if train_cfg:
            sampler_cfg = train_cfg['sampler_with_mask']
            self.sampler_with_mask = build_sampler(sampler_cfg, context=self)  # 带掩码的采样器
            assigner_cfg = train_cfg['assigner_with_mask']
            self.assigner_with_mask = build_assigner(assigner_cfg)  # 带掩码的分配器
            # 查询过滤器: 使用 Hungarian 匹配过滤低质量查询
            self.assigner_filter = build_assigner(
                dict(
                    type='HungarianAssigner_filter',     # 自定义的匈牙利过滤分配器
                    cls_cost=dict(type='FocalLossCost', weight=2.0),  # 使用 Focal Loss 作为分类代价
                    reg_cost=dict(type='BBoxL1Cost',
                                  weight=5.0,
                                  box_format='xywh'),
                    iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
                    max_pos=
                    3  # 最大正样本数: 取决于 GPU 内存，设为 1 可以在 1080Ti 上训练
                ), )

        self.loss_mask = build_loss(loss_mask)              # 构建掩码损失 (DiceLoss)
        self.things_mask_head = build_transformer(thing_transformer_head)  # 物体掩码解码器
        self.stuff_mask_head = build_transformer(stuff_transformer_head)    # 背景掩码解码器
        self.count = 0                                      # 计数 (用于调试)

    def _init_layers(self):
        """初始化分类分支和回归分支

        构建三层网络结构:
        1. Location Decoder 分支: cls_branches (分类) + reg_branches (回归)
        2. Mask Decoder for Things: cls_thing_branches + reg_branches2 (掩码解码器中的分类和回归)
        3. Mask Decoder for Stuff: cls_stuff_branches (stuff 分类)
        以及 query_embedding (物体查询) 和 stuff_query (背景查询)
        """
        if not self.as_two_stage:
            # BEV 位置嵌入: 为 BEV 的每个格子提供可学习的位置编码
            self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)

        fc_cls = Linear(self.embed_dims, self.cls_out_channels)  # 物体分类头 (things classes)
        fc_cls_stuff = Linear(self.embed_dims, 1)                # 背景分类头 (stuff, 每个类别独立做二分类)
        # 回归分支: 多个 FC + ReLU → 输出 4 维 (cx, cy, w, h)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, 4))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            """深拷贝模块 N 次"""
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # 预测层数: 两阶段模式下多一层 (encoder 也输出预测)
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            # 每层使用独立的分类/回归分支 (支持逐层精修)
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            # 所有层共享分类/回归分支
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])
        if not self.as_two_stage:
            # 物体查询嵌入: 每个查询有 2*embed_dims 维 (query + query_pos)
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)
        # 背景查询嵌入: 每个 stuff 类别一个查询
        self.stuff_query = nn.Embedding(self.num_stuff_classes,
                                        self.embed_dims * 2)
        self.reg_branches2 = _get_clones(reg_branch, self.num_dec_things)   # 掩码解码器中的回归分支
        self.cls_thing_branches = _get_clones(fc_cls, self.num_dec_things)  # 掩码解码器中的 things 分类分支
        self.cls_stuff_branches = _get_clones(fc_cls_stuff, self.num_dec_stuff)  # 掩码解码器中的 stuff 分类分支

    def init_weights(self):
        """初始化 DeformDETR head 的权重

        分类分支: 使用 bias_init_with_prob 初始化偏置，使初始输出概率接近 0.01
        回归分支: 最后一层初始化为 0
        特殊处理: 第一个回归分支的 (w, h) 偏置初始化为 -2.0 (使初始框较小)
        两阶段模式下: 回归偏置初始化为 0
        """
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)  # 初始概率 0.01
            for m in self.cls_branches:
                nn.init.constant_(m.bias, bias_init)
            for m in self.cls_thing_branches:
                nn.init.constant_(m.bias, bias_init)
            for m in self.cls_stuff_branches:
                nn.init.constant_(m.bias, bias_init)
        # 回归分支最后一层权重和偏置初始化为 0
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        for m in self.reg_branches2:
            constant_init(m[-1], 0, bias=0)
        # 第一个回归分支的 (w, h) 偏置初始化为 -2.0 (使初始预测框较小)
        nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)

        if self.as_two_stage:
            for m in self.reg_branches:
                nn.init.constant_(m[-1].bias.data[2:], 0.0)

    @force_fp32(apply_to=('bev_embed', ))
    def forward(self, bev_embed):
        """前向传播: 位置解码器 (Location Decoder)

        对 BEV 特征进行物体检测，输出分类分数、边界框和编解码器结果。

        流程:
        1. 将 BEV 特征 reshape 为多尺度特征列表
        2. 构建位置编码和掩码
        3. 经过 Deformable Transformer (encoder + decoder)
        4. 逐层预测分类和回归
        5. 返回包含 args_tuple 的输出字典 (供后续 mask decoder 使用)

        Args:
            bev_embed (Tensor): BEV 特征 (N, C, H*W)，来自上游网络

        Returns:
            dict: 包含以下键
                - bev_embed: 原始 BEV 特征
                - outputs_classes: 分类分数 [nb_dec, bs, num_query, cls_out_channels]
                - outputs_coords: 边界框预测 [nb_dec, bs, num_query, 4] (归一化坐标 cx,cy,w,h)
                - enc_outputs_class: encoder 输出的分类分数 (仅两阶段模式)
                - enc_outputs_coord: encoder 输出的坐标 (仅两阶段模式)
                - args_tuple: (memory, memory_mask, memory_pos, query, None, query_pos, hw_lvl) 供 mask decoder 使用
                - reference: 参考点
        """
        _, bs, _ = bev_embed.shape

        # 将 BEV 特征 reshape 为 (B, C, H, W) 格式
        mlvl_feats = [torch.reshape(bev_embed, (bs, self.bev_h, self.bev_w ,-1)).permute(0, 3, 1, 2)]
        img_masks = mlvl_feats[0].new_zeros((bs, self.bev_h, self.bev_w))  # 全零掩码 (BEV 无 padding)

        hw_lvl = [feat_lvl.shape[-2:] for feat_lvl in mlvl_feats]  # 每层特征图的空间尺寸
        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            # 构建掩码 (全 False，表示全部有效)
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            # 正弦位置编码
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))

        query_embeds = None
        if not self.as_two_stage:
            # 使用可学习的 query embedding
            query_embeds = self.query_embedding.weight
        # Deformable Transformer 前向传播
        (memory, memory_pos, memory_mask, query_pos), hs, init_reference, inter_references, \
        enc_outputs_class, enc_outputs_coord = self.transformer(
            mlvl_feats,
            mlvl_masks,
            query_embeds,
            mlvl_positional_encodings,
            reg_branches=self.reg_branches if self.with_box_refine else None,  # 逐层精修时需要回归分支
            cls_branches=self.cls_branches if self.as_two_stage else None       # 两阶段时需要分类分支
        )

        # 调整维度顺序，为 mask decoder 准备
        memory = memory.permute(1, 0, 2)
        query = hs[-1].permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        memory_pos = memory_pos.permute(1, 0, 2)

        # 传递给 mask decoder 的参数元组
        args_tuple = [memory, memory_mask, memory_pos, query, None, query_pos, hw_lvl]

        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        # 逐层预测分类和回归
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference              # 初始参考点
            else:
                reference = inter_references[lvl - 1]   # 中间层精修的参考点
            reference = inverse_sigmoid(reference)      # 反 sigmoid 变换
            outputs_class = self.cls_branches[lvl](hs[lvl])  # 分类预测
            tmp = self.reg_branches[lvl](hs[lvl])            # 回归预测 (偏移量)

            # 将偏移量加到参考点上
            if reference.shape[-1] == 4:
                tmp += reference  # 4 维参考点: cx, cy, w, h
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference  # 2 维参考点: 仅中心点
            outputs_coord = tmp.sigmoid()  # sigmoid 归一化到 [0, 1]
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        outs = {
                'bev_embed': None if self.as_two_stage else bev_embed,
                'outputs_classes': outputs_classes,             # 分类分数
                'outputs_coords': outputs_coords,               # 边界框预测
                'enc_outputs_class': enc_outputs_class if self.as_two_stage else None,
                'enc_outputs_coord': enc_outputs_coord.sigmoid() if self.as_two_stage else None,
                'args_tuple': args_tuple,                       # 传递给 mask decoder 的参数
                'reference': reference,
            }

        return outs

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list',
                          'args_tuple', 'reference'))
    def loss(
        self,
        all_cls_scores,
        all_bbox_preds,
        enc_cls_scores,
        enc_bbox_preds,
        args_tuple,
        reference,
        gt_labels_list,
        gt_bboxes_list,
        gt_masks_list,
        img_metas=None,
        gt_bboxes_ignore=None,
    ):
        """"全景分割损失函数

        计算 Location Decoder 和 Mask Decoder 的联合损失。

        流程:
        1. 分离 things 和 stuff 的 GT 标签
        2. 对前 L-1 层 decoder 计算检测损失 (cls, bbox, iou)
        3. 对最后一层 decoder 计算全景分割损失 (things mask + stuff mask + cls + bbox + iou)
        4. 如果 encoder 有输出，也计算 encoder 的损失
        5. 使用 things_ratio 和 stuff_ratio 动态平衡两类损失

        Args:
            all_cls_scores: 所有 decoder 层的分类分数 [nb_dec, bs, num_query, cls_out_channels]
            all_bbox_preds: 所有 decoder 层的回归预测 [nb_dec, bs, num_query, 4] (归一化 cx,cy,w,h)
            enc_cls_scores: encoder 输出的分类分数 (仅两阶段模式)
            enc_bbox_preds: encoder 输出的回归预测 (仅两阶段模式)
            args_tuple: (memory, memory_mask, memory_pos, query, None, query_pos, hw_lvl)
            reference: location decoder 的参考点
            gt_bboxes_list: GT 边界框 [num_gts, 4] (x1,y1,x2,y2 格式)
            gt_labels_list: GT 类别标签 [num_gts]
            gt_masks_list: GT 掩码 [num_gts, H, W]
            img_metas: 图像元信息
            gt_bboxes_ignore: 忽略的边界框

        Returns:
            dict[str, Tensor]: 损失字典
        """
        img_metas[0]['img_shape'] = (self.canvas_size[0], self.canvas_size[1], 3)

        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        ### 分离 things 和 stuff 的 GT 标签
        # things: id < num_things_classes 的类别 (可数目标)
        # stuff: 其他类别 (不可数区域)
        gt_things_lables_list = []
        gt_things_bboxes_list = []
        gt_things_masks_list = []
        gt_stuff_labels_list = []
        gt_stuff_masks_list = []
        for i, each in enumerate(gt_labels_list):
            # MDS: 对 COCO 数据集，id < 80 (连续 id) 的是 things，其他数据集可能不同
            things_selected = each < self.num_things_classes

            stuff_selected = things_selected == False

            gt_things_lables_list.append(gt_labels_list[i][things_selected])
            gt_things_bboxes_list.append(gt_bboxes_list[i][things_selected])
            gt_things_masks_list.append(gt_masks_list[i][things_selected])

            gt_stuff_labels_list.append(gt_labels_list[i][stuff_selected])
            gt_stuff_masks_list.append(gt_masks_list[i][stuff_selected])

        num_dec_layers = len(all_cls_scores)
        # 前 L-1 层 decoder 的 GT 列表
        all_gt_bboxes_list = [
            gt_things_bboxes_list for _ in range(num_dec_layers - 1)
        ]
        all_gt_labels_list = [
            gt_things_lables_list for _ in range(num_dec_layers - 1)
        ]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers - 1)
        ]
        img_metas_list = [img_metas for _ in range(num_dec_layers - 1)]

        # 对前 L-1 层 decoder 计算检测损失
        losses_cls, losses_bbox, losses_iou = multi_apply(
            self.loss_single, all_cls_scores[:-1], all_bbox_preds[:-1],
            all_gt_bboxes_list, all_gt_labels_list, img_metas_list,
            all_gt_bboxes_ignore_list)

        # 对最后一层 decoder 计算全景分割损失 (包含 things + stuff 掩码)
        losses_cls_f, losses_bbox_f, losses_iou_f, losses_masks_things_f, losses_masks_stuff_f, loss_mask_things_list_f, loss_mask_stuff_list_f, loss_iou_list_f, loss_bbox_list_f, loss_cls_list_f, loss_cls_stuff_list_f, things_ratio, stuff_ratio = self.loss_single_panoptic(
            all_cls_scores[-1], all_bbox_preds[-1], args_tuple, reference,
            gt_things_bboxes_list, gt_things_lables_list, gt_things_masks_list,
            (gt_stuff_labels_list, gt_stuff_masks_list), img_metas,
            gt_bboxes_ignore)

        loss_dict = dict()
        # encoder 输出的 proposal 损失
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_things_lables_list[i])
                for i in range(len(img_metas))
            ]
            enc_loss_cls, enc_losses_bbox, enc_losses_iou = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_things_bboxes_list, binary_labels_list,
                                 img_metas, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls * things_ratio
            loss_dict['enc_loss_bbox'] = enc_losses_bbox * things_ratio
            loss_dict['enc_loss_iou'] = enc_losses_iou * things_ratio
        # 最后一层 decoder 的损失
        loss_dict['loss_cls'] = losses_cls_f * things_ratio
        loss_dict['loss_bbox'] = losses_bbox_f * things_ratio
        loss_dict['loss_iou'] = losses_iou_f * things_ratio
        loss_dict['loss_mask_things'] = losses_masks_things_f * things_ratio
        loss_dict['loss_mask_stuff'] = losses_masks_stuff_f * stuff_ratio
        # mask decoder 中间层损失
        num_dec_layer = 0
        for i in range(len(loss_mask_things_list_f)):
            loss_dict[f'd{i}.loss_mask_things_f'] = loss_mask_things_list_f[
                i] * things_ratio
            loss_dict[f'd{i}.loss_iou_f'] = loss_iou_list_f[i] * things_ratio
            loss_dict[f'd{i}.loss_bbox_f'] = loss_bbox_list_f[i] * things_ratio
            loss_dict[f'd{i}.loss_cls_f'] = loss_cls_list_f[i] * things_ratio
        for i in range(len(loss_mask_stuff_list_f)):
            loss_dict[f'd{i}.loss_mask_stuff_f'] = loss_mask_stuff_list_f[
                i] * stuff_ratio
            loss_dict[f'd{i}.loss_cls_stuff_f'] = loss_cls_stuff_list_f[
                i] * stuff_ratio
        # 前 L-1 层 decoder 的检测损失
        for loss_cls_i, loss_bbox_i, loss_iou_i in zip(
                losses_cls,
                losses_bbox,
                losses_iou,
        ):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i * things_ratio
            loss_dict[
                f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i * things_ratio
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i * things_ratio

            num_dec_layer += 1
        return loss_dict

    def filter_query(self,
                     cls_scores_list,
                     bbox_preds_list,
                     gt_bboxes_list,
                     gt_labels_list,
                     img_metas,
                     gt_bboxes_ignore_list=None):
        '''
        查询过滤函数 (Query Filter)

        使用 Location Decoder 的匹配代价来过滤低质量的物体查询。
        只有匹配代价较低的查询才会被送入 Mask Decoder，从而减少计算量。
        使用 HungarianAssigner_filter 进行多轮匹配，最多匹配 max_pos 个正样本。

        流程:
        1. 对每张图像调用 _filter_query_single
        2. 汇总正负样本索引和标签

        Returns:
            pos_inds_mask_list: 正样本索引掩码列表
            neg_inds_mask_list: 负样本索引掩码列表
            labels_list: 标签列表
            label_weights_list: 标签权重列表
            bbox_targets_list: 边界框目标列表
            bbox_weights_list: 边界框权重列表
            num_total_pos: 总正样本数
            num_total_neg: 总负样本数
            pos_inds_list: 正样本索引
            neg_inds_list: 负样本索引
        '''
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (pos_inds_mask_list, neg_inds_mask_list, labels_list,
         label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
             self._filter_query_single, cls_scores_list, bbox_preds_list,
             gt_bboxes_list, gt_labels_list, img_metas, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))

        return pos_inds_mask_list, neg_inds_mask_list, labels_list, label_weights_list, bbox_targets_list, \
               bbox_weights_list, num_total_pos, num_total_neg, pos_inds_list, neg_inds_list

    def _filter_query_single(self,
                             cls_score,
                             bbox_pred,
                             gt_bboxes,
                             gt_labels,
                             img_meta,
                             gt_bboxes_ignore=None):
        """
        单张图像的查询过滤

        使用 HungarianAssigner_filter 进行匹配，返回正负样本的索引和标签。
        分配器支持多轮匈牙利匹配 (最多 max_pos 轮)，每轮匹配到的查询被标记为正样本。

        Args:
            cls_score: 分类分数 [num_query, cls_out_channels]
            bbox_pred: 边界框预测 [num_query, 4] (归一化 cx,cy,w,h)
            gt_bboxes: GT 边界框 [num_gts, 4] (x1,y1,x2,y2)
            gt_labels: GT 标签 [num_gts]
            img_meta: 图像元信息
            gt_bboxes_ignore: 忽略的边界框

        Returns:
            pos_ind_mask: 正样本索引掩码
            neg_ind_mask: 负样本索引掩码
            labels: 标签 (背景类为 num_things_classes)
            label_weights: 标签权重
            bbox_targets: 边界框目标 (归一化 cx,cy,w,h)
            bbox_weights: 边界框权重
            pos_inds: 正样本索引
            neg_inds: 负样本索引
        """
        num_bboxes = bbox_pred.size(0)
        # 使用 filter 分配器进行匹配
        pos_ind_mask, neg_ind_mask, assign_result = self.assigner_filter.assign(
            bbox_pred, cls_score, gt_bboxes, gt_labels, img_meta,
            gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        # 标签目标: 默认背景类
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_things_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # 边界框目标: 归一化为 cxcywh 格式
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        img_h, img_w, _ = img_meta['img_shape']

        # DETR 回归的是相对位置 (cxcywh)，需要归一化到 [0, 1]
        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        pos_gt_bboxes_normalized = sampling_result.pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets

        return (pos_ind_mask, neg_ind_mask, labels, label_weights,
                bbox_targets, bbox_weights, pos_inds, neg_inds)

    def get_targets_with_mask(self,
                              cls_scores_list,
                              bbox_preds_list,
                              masks_preds_list_thing,
                              gt_bboxes_list,
                              gt_labels_list,
                              gt_masks_list,
                              img_metas,
                              gt_bboxes_ignore_list=None):
        """"为全景分割计算回归和分类目标

        使用带掩码的分配器 (assigner_with_mask) 和采样器 (sampler_with_mask)，
        同时考虑分类、回归和掩码三个维度的代价进行匹配。

        Args:
            cls_scores_list: 分类分数 [num_query, cls_out_channels]
            bbox_preds_list: 边界框预测 [num_query, 4] (归一化 cx,cy,w,h)
            masks_preds_list_thing: 物体掩码预测
            gt_bboxes_list: GT 边界框 [num_gts, 4] (x1,y1,x2,y2)
            gt_labels_list: GT 标签 [num_gts]
            gt_masks_list: GT 掩码 [num_gts, H, W]
            img_metas: 图像元信息
            gt_bboxes_ignore_list: 忽略的边界框

        Returns:
            labels_list, label_weights_list: 标签和权重
            bbox_targets_list, bbox_weights_list: 边界框目标和权重
            mask_targets_list, mask_weights_list: 掩码目标和权重
            num_total_pos_thing, num_total_neg_thing: 正负样本总数
            pos_inds_list: 正样本索引
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         mask_targets_list, mask_weights_list, pos_inds_list,
         neg_inds_list) = multi_apply(self._get_target_single_with_mask,
                                      cls_scores_list, bbox_preds_list,
                                      masks_preds_list_thing, gt_bboxes_list,
                                      gt_labels_list, gt_masks_list, img_metas,
                                      gt_bboxes_ignore_list)
        num_total_pos_thing = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg_thing = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, mask_targets_list, mask_weights_list,
                num_total_pos_thing, num_total_neg_thing, pos_inds_list)

    def _get_target_single_with_mask(self,
                                     cls_score,
                                     bbox_pred,
                                     masks_preds_things,
                                     gt_bboxes,
                                     gt_labels,
                                     gt_masks,
                                     img_meta,
                                     gt_bboxes_ignore=None):
        """
        单张图像的带掩码目标计算

        使用 assigner_with_mask 进行匹配，同时考虑 classification + regression + mask 代价。
        然后构建标签、边界框和掩码的目标。

        Args:
            cls_score: 分类分数 [num_query, cls_out_channels]
            bbox_pred: 边界框预测 [num_query, 4] (归一化 cx,cy,w,h)
            masks_preds_things: 物体掩码预测 [num_query, H, W]
            gt_bboxes: GT 边界框 [num_gts, 4] (x1,y1,x2,y2)
            gt_labels: GT 标签 [num_gts]
            gt_masks: GT 掩码 [num_gts, H, W]
            img_meta: 图像元信息
            gt_bboxes_ignore: 忽略的边界框

        Returns:
            labels: 标签
            label_weights: 标签权重
            bbox_targets: 边界框目标 (归一化 cxcywh)
            bbox_weights: 边界框权重
            mask_target: 掩码目标
            mask_weights: 掩码权重
            pos_inds: 正样本索引
            neg_inds: 负样本索引
        """

        num_bboxes = bbox_pred.size(0)
        gt_masks = gt_masks.float()

        # 使用带掩码的分配器: 同时考虑 cls + reg + mask 代价
        assign_result = self.assigner_with_mask.assign(bbox_pred, cls_score,
                                                       masks_preds_things,
                                                       gt_bboxes, gt_labels,
                                                       gt_masks, img_meta,
                                                       gt_bboxes_ignore)
        sampling_result = self.sampler_with_mask.sample(
            assign_result, bbox_pred, gt_bboxes, gt_masks)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # 标签目标: 默认背景类
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_things_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # 边界框目标: 归一化为 cxcywh
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        img_h, img_w, _ = img_meta['img_shape']

        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        pos_gt_bboxes_normalized = sampling_result.pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets

        # 掩码目标: 正样本位置填充 GT 掩码
        mask_weights = masks_preds_things.new_zeros(num_bboxes)
        mask_weights[pos_inds] = 1.0
        pos_gt_masks = sampling_result.pos_gt_masks
        _, w, h = pos_gt_masks.shape
        mask_target = masks_preds_things.new_zeros([num_bboxes, w, h])
        mask_target[pos_inds] = pos_gt_masks

        return (labels, label_weights, bbox_targets, bbox_weights, mask_target,
                mask_weights, pos_inds, neg_inds)

    def get_filter_results_and_loss(self, cls_scores, bbox_preds,
                                    cls_scores_list, bbox_preds_list,
                                    gt_bboxes_list, gt_labels_list, img_metas,
                                    gt_bboxes_ignore_list):
        """
        获取过滤结果并计算检测损失

        先通过 filter_query 过滤低质量查询，然后计算过滤后的:
        - 分类损失 (FocalLoss)
        - IoU 损失 (GIoU)
        - L1 回归损失

        Returns:
            loss_cls, loss_iou, loss_bbox: 三种损失
            pos_inds_mask_list: 正样本索引掩码
            num_total_pos_thing: 正样本总数
        """

        pos_inds_mask_list, neg_inds_mask_list, labels_list, label_weights_list, bbox_targets_list, \
        bbox_weights_list, num_total_pos_thing, num_total_neg_thing, pos_inds_list, neg_inds_list = self.filter_query(
            cls_scores_list, bbox_preds_list,
            gt_bboxes_list, gt_labels_list,
            img_metas, gt_bboxes_ignore_list)
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # 分类损失: 使用加权平均因子 (与官方 DETR 一致)
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        cls_avg_factor = num_total_pos_thing * 1.0 + \
                         num_total_neg_thing * self.bg_cls_weight  # 背景类权重
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(cls_scores,
                                 labels,
                                 label_weights,
                                 avg_factor=cls_avg_factor)

        # 跨 GPU 归一化: 计算平均正样本数
        num_total_pos_thing = loss_cls.new_tensor([num_total_pos_thing])
        num_total_pos_thing = torch.clamp(reduce_mean(num_total_pos_thing),
                                          min=1).item()

        # 构建缩放因子: 用于将归一化坐标转换为实际坐标
        factors = []
        for img_meta, bbox_pred in zip(img_metas, bbox_preds):
            img_h, img_w, _ = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR 回归相对位置 (cxcywh)，需要反归一化来计算 IoU 损失
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # IoU 损失 (默认 GIoU)
        loss_iou = self.loss_iou(bboxes,
                                 bboxes_gt,
                                 bbox_weights,
                                 avg_factor=num_total_pos_thing)

        # L1 回归损失
        loss_bbox = self.loss_bbox(bbox_preds,
                                   bbox_targets,
                                   bbox_weights,
                                   avg_factor=num_total_pos_thing)
        return loss_cls, loss_iou, loss_bbox,\
            pos_inds_mask_list, num_total_pos_thing

    def loss_single_panoptic(self,
                             cls_scores,
                             bbox_preds,
                             args_tuple,
                             reference,
                             gt_bboxes_list,
                             gt_labels_list,
                             gt_masks_list,
                             gt_panoptic_list,
                             img_metas,
                             gt_bboxes_ignore_list=None):
        """"单层 decoder 的全景分割损失函数

        这是全景分割的核心损失函数，包含三个子任务:
        1. Location Decoder 检测损失 (cls + bbox + iou) — 通过 Query Filter
        2. Mask Decoder for Things 损失 (mask + cls + bbox + iou 中间层)
        3. Mask Decoder for Stuff 损失 (mask + cls 中间层)

        流程:
        1. 通过 Query Filter 过滤低质量查询，计算检测损失
        2. 构建 thing_query (过滤后的查询) 和 stuff_query (可学习查询)
        3. Things Mask Head 和 Stuff Mask Head 分别生成掩码
        4. 逐层计算中间层损失
        5. 使用 things_ratio / stuff_ratio 动态平衡两类损失

        Args:
            cls_scores: 分类分数 [bs, num_query, cls_out_channels]
            bbox_preds: 边界框预测 [bs, num_query, 4] (归一化 cx,cy,w,h)
            args_tuple: (memory, memory_mask, memory_pos, query, None, query_pos, hw_lvl)
            reference: Location Decoder 的参考点
            gt_bboxes_list: GT 边界框 [num_gts, 4] (x1,y1,x2,y2)
            gt_labels_list: GT 标签 [num_gts]
            gt_masks_list: GT 物体掩码 [num_gts, H, W]
            gt_panoptic_list: (gt_stuff_labels_list, gt_stuff_masks_list)
            img_metas: 图像元信息
            gt_bboxes_ignore_list: 忽略的边界框

        Returns:
            loss_cls, loss_bbox, loss_iou: 检测损失
            loss_mask_things: things 掩码损失
            loss_mask_stuff: stuff 掩码损失
            loss_mask_things_list: things 掩码中间层损失列表
            loss_mask_stuff_list: stuff 掩码中间层损失列表
            loss_iou_list, loss_bbox_list: 中间层 IoU 和 bbox 损失列表
            loss_cls_thing_list, loss_cls_stuff_list: 中间层分类损失列表
            things_ratio, stuff_ratio: 动态权重比例
        """
        num_imgs = cls_scores.size(0)
        gt_stuff_labels_list, gt_stuff_masks_list = gt_panoptic_list
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        loss_cls, loss_iou, loss_bbox, pos_inds_mask_list, num_total_pos_thing = self.get_filter_results_and_loss(
            cls_scores, bbox_preds, cls_scores_list, bbox_preds_list, gt_bboxes_list, gt_labels_list, img_metas, gt_bboxes_ignore_list)

        memory, memory_mask, memory_pos, query, _, query_pos, hw_lvl = args_tuple

        BS, _, dim_query = query.shape[0], query.shape[1], query.shape[-1]

        len_query = max([len(pos_ind) for pos_ind in pos_inds_mask_list])
        thing_query = torch.zeros([BS, len_query, dim_query],
                                  device=query.device)

        stuff_query, stuff_query_pos = torch.split(self.stuff_query.weight,
                                                   self.embed_dims,
                                                   dim=1)
        stuff_query_pos = stuff_query_pos.unsqueeze(0).expand(BS, -1, -1)
        stuff_query = stuff_query.unsqueeze(0).expand(BS, -1, -1)

        for i in range(BS):
            thing_query[i, :len(pos_inds_mask_list[i])] = query[
                i, pos_inds_mask_list[i]]

        mask_preds_things = []
        mask_preds_stuff = []
        # mask_preds_inter = [[],[],[]]
        mask_preds_inter_things = [[] for _ in range(self.num_dec_things)]
        mask_preds_inter_stuff = [[] for _ in range(self.num_dec_stuff)]
        cls_thing_preds = [[] for _ in range(self.num_dec_things)]
        cls_stuff_preds = [[] for _ in range(self.num_dec_stuff)]
        BS, NQ, L = bbox_preds.shape
        new_bbox_preds = [
            torch.zeros([BS, len_query, L]).to(bbox_preds.device)
            for _ in range(self.num_dec_things)
        ]

        mask_things, mask_inter_things, query_inter_things = self.things_mask_head(
            memory, memory_mask, None, thing_query, None, None, hw_lvl=hw_lvl)

        mask_stuff, mask_inter_stuff, query_inter_stuff = self.stuff_mask_head(
            memory,
            memory_mask,
            None,
            stuff_query,
            None,
            stuff_query_pos,
            hw_lvl=hw_lvl)

        mask_things = mask_things.squeeze(-1)
        mask_inter_things = torch.stack(mask_inter_things, 0).squeeze(-1)

        mask_stuff = mask_stuff.squeeze(-1)
        mask_inter_stuff = torch.stack(mask_inter_stuff, 0).squeeze(-1)

        for i in range(BS):
            tmp_i = mask_things[i][:len(pos_inds_mask_list[i])].reshape(
                -1, *hw_lvl[0])
            mask_preds_things.append(tmp_i)
            pos_ind = pos_inds_mask_list[i]
            reference_i = reference[i:i + 1, pos_ind, :]

            for j in range(self.num_dec_things):
                tmp_i_j = mask_inter_things[j][i][:len(pos_inds_mask_list[i]
                                                       )].reshape(
                                                           -1, *hw_lvl[0])
                mask_preds_inter_things[j].append(tmp_i_j)

                # mask_preds_inter_things[j].append(mask_inter_things[j].reshape(-1, *hw_lvl[0]))
                query_things = query_inter_things[j]
                t1, t2, t3 = query_things.shape
                tmp = self.reg_branches2[j](query_things.reshape(t1 * t2, t3)).reshape(t1, t2, 4)
                if len(pos_ind) == 0:
                    tmp = tmp.sum(
                    ) + reference_i  # for reply bug of pytorch broadcast
                elif reference_i.shape[-1] == 4:
                    tmp += reference_i
                else:
                    assert reference_i.shape[-1] == 2
                    tmp[..., :2] += reference_i

                outputs_coord = tmp.sigmoid()

                new_bbox_preds[j][i][:len(pos_inds_mask_list[i])] = outputs_coord
                cls_thing_preds[j].append(self.cls_thing_branches[j](
                    query_things.reshape(t1 * t2, t3)))

            # stuff
            tmp_i = mask_stuff[i].reshape(-1, *hw_lvl[0])
            mask_preds_stuff.append(tmp_i)
            for j in range(self.num_dec_stuff):
                tmp_i_j = mask_inter_stuff[j][i].reshape(-1, *hw_lvl[0])
                mask_preds_inter_stuff[j].append(tmp_i_j)

                query_stuff = query_inter_stuff[j]
                s1, s2, s3 = query_stuff.shape
                cls_stuff_preds[j].append(self.cls_stuff_branches[j](
                    query_stuff.reshape(s1 * s2, s3)))

        masks_preds_list_thing = [
            mask_preds_things[i] for i in range(num_imgs)
        ]
        mask_preds_things = torch.cat(mask_preds_things, 0)
        mask_preds_inter_things = [
            torch.cat(each, 0) for each in mask_preds_inter_things
        ]
        cls_thing_preds = [torch.cat(each, 0) for each in cls_thing_preds]
        cls_stuff_preds = [torch.cat(each, 0) for each in cls_stuff_preds]
        mask_preds_stuff = torch.cat(mask_preds_stuff, 0)
        mask_preds_inter_stuff = [
            torch.cat(each, 0) for each in mask_preds_inter_stuff
        ]
        cls_scores_list = [
            cls_scores_list[i][pos_inds_mask_list[i]] for i in range(num_imgs)
        ]

        bbox_preds_list = [
            bbox_preds_list[i][pos_inds_mask_list[i]] for i in range(num_imgs)
        ]

        gt_targets = self.get_targets_with_mask(cls_scores_list,
                                                bbox_preds_list,
                                                masks_preds_list_thing,
                                                gt_bboxes_list, gt_labels_list,
                                                gt_masks_list, img_metas,
                                                gt_bboxes_ignore_list)

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         mask_targets_list, mask_weights_list, _, _,
         pos_inds_list) = gt_targets

        thing_labels = torch.cat(labels_list, 0)
        things_weights = torch.cat(label_weights_list, 0)

        bboxes_taget = torch.cat(bbox_targets_list)
        bboxes_weights = torch.cat(bbox_weights_list)

        factors = []
        for img_meta, bbox_pred in zip(img_metas, bbox_preds_list):
            img_h, img_w, _ = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        bboxes_gt = bbox_cxcywh_to_xyxy(bboxes_taget) * factors

        mask_things_gt = torch.cat(mask_targets_list, 0).to(torch.float)

        mask_weight_things = torch.cat(mask_weights_list,
                                       0).to(thing_labels.device)

        mask_stuff_gt = []
        mask_weight_stuff = []
        stuff_labels = []
        num_total_pos_stuff = 0
        for i in range(BS):
            num_total_pos_stuff += len(gt_stuff_labels_list[i])  ## all stuff

            select_stuff_index = gt_stuff_labels_list[
                i] - self.num_things_classes
            mask_weight_i_stuff = torch.zeros([self.num_stuff_classes])
            mask_weight_i_stuff[select_stuff_index] = 1
            stuff_masks = torch.zeros(
                (self.num_stuff_classes, *mask_targets_list[i].shape[-2:]),
                device=mask_targets_list[i].device).to(torch.bool)
            stuff_masks[select_stuff_index] = gt_stuff_masks_list[i].to(
                torch.bool)
            mask_stuff_gt.append(stuff_masks)
            select_stuff_index = torch.cat([
                select_stuff_index,
                torch.tensor([self.num_stuff_classes],
                             device=select_stuff_index.device)
            ])

            stuff_labels.append(1 - mask_weight_i_stuff)
            mask_weight_stuff.append(mask_weight_i_stuff)

        mask_weight_stuff = torch.cat(mask_weight_stuff,
                                      0).to(thing_labels.device)
        stuff_labels = torch.cat(stuff_labels, 0).to(thing_labels.device)
        mask_stuff_gt = torch.cat(mask_stuff_gt, 0).to(torch.float)

        num_total_pos_stuff = loss_cls.new_tensor([num_total_pos_stuff])
        num_total_pos_stuff = torch.clamp(reduce_mean(num_total_pos_stuff),
                                          min=1).item()
        if mask_preds_things.shape[0] == 0:
            loss_mask_things = (0 * mask_preds_things).sum()
        else:
            mask_preds = F.interpolate(mask_preds_things.unsqueeze(0),
                                       scale_factor=2.0,
                                       mode='bilinear').squeeze(0)
            mask_targets_things = F.interpolate(mask_things_gt.unsqueeze(0),
                                                size=mask_preds.shape[-2:],
                                                mode='bilinear').squeeze(0)
            loss_mask_things = self.loss_mask(mask_preds,
                                              mask_targets_things,
                                              mask_weight_things,
                                              avg_factor=num_total_pos_thing)
        if mask_preds_stuff.shape[0] == 0:
            loss_mask_stuff = (0 * mask_preds_stuff).sum()
        else:
            mask_preds = F.interpolate(mask_preds_stuff.unsqueeze(0),
                                       scale_factor=2.0,
                                       mode='bilinear').squeeze(0)
            mask_targets_stuff = F.interpolate(mask_stuff_gt.unsqueeze(0),
                                               size=mask_preds.shape[-2:],
                                               mode='bilinear').squeeze(0)

            loss_mask_stuff = self.loss_mask(mask_preds,
                                             mask_targets_stuff,
                                             mask_weight_stuff,
                                             avg_factor=num_total_pos_stuff)

        loss_mask_things_list = []
        loss_mask_stuff_list = []
        loss_iou_list = []
        loss_bbox_list = []
        for j in range(len(mask_preds_inter_things)):
            mask_preds_this_level = mask_preds_inter_things[j]
            if mask_preds_this_level.shape[0] == 0:
                loss_mask_j = (0 * mask_preds_this_level).sum()
            else:
                mask_preds_this_level = F.interpolate(
                    mask_preds_this_level.unsqueeze(0),
                    scale_factor=2.0,
                    mode='bilinear').squeeze(0)
                loss_mask_j = self.loss_mask(mask_preds_this_level,
                                             mask_targets_things,
                                             mask_weight_things,
                                             avg_factor=num_total_pos_thing)
            loss_mask_things_list.append(loss_mask_j)
            bbox_preds_this_level = new_bbox_preds[j].reshape(-1, 4)
            bboxes_this_level = bbox_cxcywh_to_xyxy(
                bbox_preds_this_level) * factors
            # We let this loss be 0. We didn't predict bbox in our mask decoder. Predicting bbox in the mask decoder is basically useless
            loss_iou_j = self.loss_iou(bboxes_this_level,
                                       bboxes_gt,
                                       bboxes_weights,
                                       avg_factor=num_total_pos_thing) * 0
            if bboxes_taget.shape[0] != 0:
                loss_bbox_j = self.loss_bbox(
                    bbox_preds_this_level,
                    bboxes_taget,
                    bboxes_weights,
                    avg_factor=num_total_pos_thing) * 0
            else:
                loss_bbox_j = bbox_preds_this_level.sum() * 0
            loss_iou_list.append(loss_iou_j)
            loss_bbox_list.append(loss_bbox_j)
        for j in range(len(mask_preds_inter_stuff)):
            mask_preds_this_level = mask_preds_inter_stuff[j]
            if mask_preds_this_level.shape[0] == 0:
                loss_mask_j = (0 * mask_preds_this_level).sum()
            else:
                mask_preds_this_level = F.interpolate(
                    mask_preds_this_level.unsqueeze(0),
                    scale_factor=2.0,
                    mode='bilinear').squeeze(0)
                loss_mask_j = self.loss_mask(mask_preds_this_level,
                                             mask_targets_stuff,
                                             mask_weight_stuff,
                                             avg_factor=num_total_pos_stuff)
            loss_mask_stuff_list.append(loss_mask_j)

        loss_cls_thing_list = []
        loss_cls_stuff_list = []
        thing_labels = thing_labels.reshape(-1)
        for j in range(len(mask_preds_inter_things)):
            # We let this loss be 0. When using "query-filter", only partial thing queries are feed to the mask decoder. This will cause imbalance when supervising these queries.
            cls_scores = cls_thing_preds[j]

            if cls_scores.shape[0] == 0:
                loss_cls_thing_j = cls_scores.sum() * 0
            else:
                loss_cls_thing_j = self.loss_cls(
                    cls_scores,
                    thing_labels,
                    things_weights,
                    avg_factor=num_total_pos_thing) * 2 * 0
            loss_cls_thing_list.append(loss_cls_thing_j)

        for j in range(len(mask_preds_inter_stuff)):
            cls_scores = cls_stuff_preds[j]
            if cls_scores.shape[0] == 0:
                loss_cls_stuff_j = cls_stuff_preds[j].sum() * 0
            else:
                loss_cls_stuff_j = self.loss_cls(
                    cls_stuff_preds[j],
                    stuff_labels.to(torch.long),
                    avg_factor=num_total_pos_stuff) * 2
            loss_cls_stuff_list.append(loss_cls_stuff_j)

        ## dynamic adjusting the weights
        things_ratio, stuff_ratio = num_total_pos_thing / (
            num_total_pos_stuff + num_total_pos_thing), num_total_pos_stuff / (
                num_total_pos_stuff + num_total_pos_thing)

        return loss_cls, loss_bbox, loss_iou, loss_mask_things, loss_mask_stuff, loss_mask_things_list, loss_mask_stuff_list, loss_iou_list, loss_bbox_list, loss_cls_thing_list, loss_cls_stuff_list, things_ratio, stuff_ratio
    
    def forward_test(self,
                    pts_feats=None,
                    gt_lane_labels=None,
                    gt_lane_masks=None,
                    img_metas=None,
                    rescale=False):
        """测试前向传播

        对 BEV 特征进行全景分割，并计算与 GT 的 IoU 指标。
        包括可行驶区域 (drivable) 和车道线 (lane) 两大类，其中车道线分为:
        - divider (分隔线)
        - crossing (斑马线)
        - contour (轮廓线)

        Returns:
            bbox_list: 包含分割结果和 IoU 指标的列表
        """
        bbox_list = [dict() for i in range(len(img_metas))]

        pred_seg_dict = self(pts_feats)
        results = self.get_bboxes(pred_seg_dict['outputs_classes'],
                                           pred_seg_dict['outputs_coords'],
                                           pred_seg_dict['enc_outputs_class'],
                                           pred_seg_dict['enc_outputs_coord'],
                                           pred_seg_dict['args_tuple'],
                                           pred_seg_dict['reference'],
                                           img_metas,
                                           rescale=rescale)

        with torch.no_grad():
            # 可行驶区域 IoU 计算
            drivable_pred = results[0]['drivable']
            drivable_gt = gt_lane_masks[0][0, -1]  # 最后一帧是可行驶区域
            drivable_iou, drivable_intersection, drivable_union = IOU(drivable_pred.view(1, -1), drivable_gt.view(1, -1))

            # 车道线 IoU 计算 (合并所有车道线类别)
            lane_pred = results[0]['lane']
            lanes_pred = (results[0]['lane'].sum(0) > 0).int()  # 合并所有车道线
            lanes_gt = (gt_lane_masks[0][0][:-1].sum(0) > 0).int()  # 合并所有车道线 GT
            lanes_iou, lanes_intersection, lanes_union = IOU(lanes_pred.view(1, -1), lanes_gt.view(1, -1))

            # 细分车道线类别: divider, crossing, contour
            divider_gt = (gt_lane_masks[0][0][gt_lane_labels[0][0] == 0].sum(0) > 0).int()
            crossing_gt = (gt_lane_masks[0][0][gt_lane_labels[0][0] == 1].sum(0) > 0).int()
            contour_gt = (gt_lane_masks[0][0][gt_lane_labels[0][0] == 2].sum(0) > 0).int()
            divider_iou, divider_intersection, divider_union = IOU(lane_pred[0].view(1, -1), divider_gt.view(1, -1))
            crossing_iou, crossing_intersection, crossing_union = IOU(lane_pred[1].view(1, -1), crossing_gt.view(1, -1))
            contour_iou, contour_intersection, contour_union = IOU(lane_pred[2].view(1, -1), contour_gt.view(1, -1))


            ret_iou = {'drivable_intersection': drivable_intersection,
                       'drivable_union': drivable_union,
                       'lanes_intersection': lanes_intersection,
                       'lanes_union': lanes_union,
                       'divider_intersection': divider_intersection,
                       'divider_union': divider_union,
                       'crossing_intersection': crossing_intersection,
                       'crossing_union': crossing_union,
                       'contour_intersection': contour_intersection,
                       'contour_union': contour_union,
                       'drivable_iou': drivable_iou,
                       'lanes_iou': lanes_iou,
                       'divider_iou': divider_iou,
                       'crossing_iou': crossing_iou,
                       'contour_iou': contour_iou}
        for result_dict, pts_bbox in zip(bbox_list, results):
            result_dict['pts_bbox'] = pts_bbox
            result_dict['ret_iou'] = ret_iou
            result_dict['args_tuple'] = pred_seg_dict['args_tuple']
        return bbox_list


    @auto_fp16(apply_to=("bev_feat", "prev_bev"))
    def forward_train(self,
                          bev_feat=None,
                          img_metas=None,
                          gt_lane_labels=None,
                          gt_lane_bboxes=None,
                          gt_lane_masks=None,
                         ):
        """
        Forward pass of the segmentation model during training.

        Args:
            bev_feat (torch.Tensor): Bird's eye view feature maps. Shape [batch_size, channels, height, width].
            img_metas (list[dict]): List of image meta information dictionaries.
            gt_lane_labels (list[torch.Tensor]): Ground-truth lane class labels. Shape [batch_size, num_lanes, max_lanes].
            gt_lane_bboxes (list[torch.Tensor]): Ground-truth lane bounding boxes. Shape [batch_size, num_lanes, 4].
            gt_lane_masks (list[torch.Tensor]): Ground-truth lane masks. Shape [batch_size, num_lanes, height, width].
            prev_bev (torch.Tensor): Previous bird's eye view feature map. Shape [batch_size, channels, height, width].

        Returns:
            tuple:
                - losses_seg (torch.Tensor): Total segmentation loss.
                - pred_seg_dict (dict): Dictionary of predicted segmentation outputs.
        """
        pred_seg_dict = self(bev_feat)
        loss_inputs = [
            pred_seg_dict['outputs_classes'],
            pred_seg_dict['outputs_coords'],
            pred_seg_dict['enc_outputs_class'],
            pred_seg_dict['enc_outputs_coord'],
            pred_seg_dict['args_tuple'],
            pred_seg_dict['reference'],
            gt_lane_labels,
            gt_lane_bboxes,
            gt_lane_masks
        ]
        losses_seg = self.loss(*loss_inputs, img_metas=img_metas)
        return losses_seg, pred_seg_dict

    def _get_bboxes_single(self,
                           cls_score,
                           bbox_pred,
                           img_shape,
                           scale_factor,
                           rescale=False):
        """
        单张图像的边界框检测

        从分类分数和边界框预测中提取 top-k 检测结果。

        流程:
        1. 如果使用 sigmoid 分类: 取 top-k 高分预测
        2. 如果使用 softmax 分类: 排除背景类后取 top-k
        3. 将归一化坐标 (cxcywh) 转换为绝对坐标 (x1y1x2y2)
        4. 裁剪到图像范围内

        Args:
            cls_score: 分类分数 [num_query, cls_out_channels]
            bbox_pred: 边界框预测 [num_query, 4] (归一化 cx,cy,w,h)
            img_shape: 图像尺寸 (H, W, 3)
            scale_factor: 缩放因子
            rescale: 是否反归一化到原图尺寸

        Returns:
            bbox_index: 选中的边界框索引
            det_bboxes: 检测结果 [num_query, 5] (x1,y1,x2,y2,score)
            det_labels: 检测标签 [num_query]
        """
        assert len(cls_score) == len(bbox_pred)
        max_per_img = self.test_cfg.get('max_per_img', self.num_query)

        # 排除背景类
        if self.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            det_labels = indexes % self.num_things_classes
            bbox_index = indexes // self.num_things_classes
            bbox_pred = bbox_pred[bbox_index]
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            bbox_pred = bbox_pred[bbox_index]
            det_labels = det_labels[bbox_index]

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]  # 恢复 x 坐标
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]  # 恢复 y 坐标
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])  # 裁剪 x
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])  # 裁剪 y
        if rescale:
            det_bboxes /= det_bboxes.new_tensor(scale_factor)
        det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(1)), -1)  # 拼接分数

        return bbox_index, det_bboxes, det_labels

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list',
                          'args_tuple'))
    def get_bboxes(
        self,
        all_cls_scores,
        all_bbox_preds,
        enc_cls_scores,
        enc_bbox_preds,
        args_tuple,
        reference,
        img_metas,
        rescale=False,
    ):
        """
        全景分割推理: 获取边界框和分割掩码

        这是测试时的核心函数，从 Location Decoder 和 Mask Decoder 的输出
        生成最终的全景分割结果。

        流程:
        1. 从 Location Decoder 获取检测结果 (bbox, label, score)
        2. 构建 joint_query (thing_query + stuff_query)
        3. Things Mask Head 和 Stuff Mask Head 生成掩码
        4. mask-wise 合并: 使用掩码分数重新加权检测分数
        5. 按分数排序，进行重叠过滤 (类似 NMS)
        6. 为每个目标分配唯一的实例 ID

        输出:
        - bbox: 物体边界框
        - segm: 物体分割掩码
        - labels: 物体标签
        - panoptic: 全景分割结果
        - drivable: 可行驶区域
        - lane: 车道线
        - lane_score: 车道线分数
        """
        cls_scores = all_cls_scores[-1]  # 取最后一层 decoder 输出
        bbox_preds = all_bbox_preds[-1]
        memory, memory_mask, memory_pos, query, _, query_pos, hw_lvl = args_tuple

        seg_list = []
        stuff_score_list = []
        panoptic_list = []
        bbox_list = []
        labels_list = []
        drivable_list = []
        lane_list = []
        lane_score_list = []
        score_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_shape = (self.canvas_size[0], self.canvas_size[1], 3)
            ori_shape = (self.canvas_size[0], self.canvas_size[1], 3)
            scale_factor = 1

            # 第一步: 获取检测结果 (bbox, label, score)
            index, bbox, labels = self._get_bboxes_single(
                cls_score, bbox_pred, img_shape, scale_factor, rescale)

            i = img_id
            # 构建 joint_query: thing_query + stuff_query
            thing_query = query[i:i + 1, index, :]  # 选中的物体查询
            thing_query_pos = query_pos[i:i + 1, index, :]
            joint_query = torch.cat([
                thing_query, self.stuff_query.weight[None, :, :self.embed_dims]
            ], 1)

            stuff_query_pos = self.stuff_query.weight[None, :,
                                                      self.embed_dims:]

            # Things Mask Head: 生成物体掩码
            mask_things, mask_inter_things, query_inter_things = self.things_mask_head(
                memory[i:i + 1],
                memory_mask[i:i + 1],
                None,
                joint_query[:, :-self.num_stuff_classes],
                None,
                None,
                hw_lvl=hw_lvl)
            # Stuff Mask Head: 生成背景掩码
            mask_stuff, mask_inter_stuff, query_inter_stuff = self.stuff_mask_head(
                memory[i:i + 1],
                memory_mask[i:i + 1],
                None,
                joint_query[:, -self.num_stuff_classes:],
                None,
                stuff_query_pos,
                hw_lvl=hw_lvl)

            attn_map = torch.cat([mask_things, mask_stuff], 1)
            attn_map = attn_map.squeeze(-1)  # BS, NQ, N_head, LEN

            # stuff 分类分数
            stuff_query = query_inter_stuff[-1]
            scores_stuff = self.cls_stuff_branches[-1](
                stuff_query).sigmoid().reshape(-1)

            # 掩码预测: reshape 并上采样到原始尺寸
            mask_pred = attn_map.reshape(-1, *hw_lvl[0])
            mask_pred = F.interpolate(mask_pred.unsqueeze(0),
                                      size=ori_shape[:2],
                                      mode='bilinear').squeeze(0)

            # 可行驶区域: 最后一个 stuff 类别
            masks_all = mask_pred
            score_list.append(masks_all)
            drivable_list.append(masks_all[-1] > 0.5)
            masks_all = masks_all[:-self.num_stuff_classes]  # 去掉 stuff 类别
            seg_all = masks_all > 0.5
            sum_seg_all = seg_all.sum((1, 2)).float() + 1  # 避免除零

            scores_all = bbox[:, -1]  # 检测分数
            bboxes_all = bbox
            labels_all = labels

            # mask-wise 合并: 掩码内平均分数加权检测分数
            seg_scores = (masks_all * seg_all.float()).sum(
                (1, 2)) / sum_seg_all
            scores_all *= (seg_scores**2)  # 平方加权

            scores_all, index = torch.sort(scores_all, descending=True)  # 按分数降序排列

            masks_all = masks_all[index]
            labels_all = labels_all[index]
            bboxes_all = bboxes_all[index]
            seg_all = seg_all[index]

            bboxes_all[:, -1] = scores_all

            # 分离 things 和 stuff
            things_selected = labels_all < self.num_things_classes
            stuff_selected = labels_all >= self.num_things_classes
            bbox_th = bboxes_all[things_selected][:100]  # 最多 100 个物体
            labels_th = labels_all[things_selected][:100]
            seg_th = seg_all[things_selected][:100]
            labels_st = labels_all[stuff_selected]
            scores_st = scores_all[stuff_selected]
            masks_st = masks_all[stuff_selected]

            stuff_score_list.append(scores_st)

            # 全景分割合并: 按分数排序，逐个处理
            results = torch.zeros((2, *mask_pred.shape[-2:]),
                                  device=mask_pred.device).to(torch.long)
            id_unique = 1
            lane = torch.zeros((self.num_things_classes, *mask_pred.shape[-2:]), device=mask_pred.device).to(torch.long)
            lane_score = torch.zeros((self.num_things_classes, *mask_pred.shape[-2:]), device=mask_pred.device).to(mask_pred.dtype)
            for i, scores in enumerate(scores_all):
                # things 和 stuff 使用不同的阈值
                if labels_all[i] < self.num_things_classes and scores < self.quality_threshold_things:
                    continue
                elif labels_all[i] >= self.num_things_classes and scores < self.quality_threshold_stuff:
                    continue
                _mask = masks_all[i] > 0.5
                mask_area = _mask.sum().item()
                intersect = _mask & (results[0] > 0)
                intersect_area = intersect.sum().item()
                # 重叠过滤: 如果重叠比例过高，跳过
                if labels_all[i] < self.num_things_classes:
                    if mask_area == 0 or (intersect_area * 1.0 / mask_area
                                          ) > self.overlap_threshold_things:
                        continue
                else:
                    if mask_area == 0 or (intersect_area * 1.0 / mask_area
                                          ) > self.overlap_threshold_stuff:
                        continue
                if intersect_area > 0:
                    _mask = _mask & (results[0] == 0)  # 只保留非重叠区域
                results[0, _mask] = labels_all[i]       # 语义标签
                if labels_all[i] < self.num_things_classes:
                    lane[labels_all[i], _mask] = 1                     # 车道线类别
                    lane_score[labels_all[i], _mask] = masks_all[i][_mask]  # 车道线分数
                    results[1, _mask] = id_unique  # 实例 ID
                    id_unique += 1

            file_name = img_metas[img_id]['pts_filename'].split('/')[-1].split('.')[0]
            panoptic_list.append(
                (results.permute(1, 2, 0).cpu().numpy(), file_name, ori_shape))

            bbox_list.append(bbox_th)
            labels_list.append(labels_th)
            seg_list.append(seg_th)
            lane_list.append(lane)
            lane_score_list.append(lane_score)
        results = []
        for i in range(len(img_metas)):
            results.append({
                'bbox': bbox_list[i],
                'segm': seg_list[i],
                'labels': labels_list[i],
                'panoptic': panoptic_list[i],
                'drivable': drivable_list[i],
                'score_list': score_list[i],
                'lane': lane_list[i],
                'lane_score': lane_score_list[i],
                'stuff_score_list': stuff_score_list[i],
            })
        return results
