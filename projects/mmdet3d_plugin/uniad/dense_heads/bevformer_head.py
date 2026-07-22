"""
BEVFormer 检测头 (BEVFormerHead)
================================
BEVFormer 的基础检测头，负责将图像特征通过 Transformer 转换为 BEV 特征，
并在 BEV 特征上进行 3D 目标检测。

核心流程:
    1. Encoder: 图像特征 → BEV 特征 (BEVFormerEncoder)
    2. Decoder: BEV 特征 + object queries → 检测结果 (DetectionTransformerDecoder)
    3. 分类/回归分支: 每层 decoder 输出 → 分类分数 + 3D 框

检测框编码格式 (code_size=10):
    0: cx    中心 x 坐标 (归一化后反归一化到 pc_range)
    1: cy    中心 y 坐标
    2: w     宽度
    3: l     长度
    4: cz    中心 z 坐标
    5: h     高度
    6: sinθ  朝向角 sin
    7: cosθ  朝向角 cos
    8: vx    x 方向速度
    9: vy    y 方向速度

BEVFormerHead_GroupDETR:
    继承自 BEVFormerHead，使用 Group DETR 策略。
    训练时使用 group_detr 组 query，推理时只使用一组，增加训练效率。
"""

import copy
import torch
import torch.nn as nn

from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.utils import TORCH_VERSION, digit_version
from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet.models.dense_heads import DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from mmcv.runner import force_fp32, auto_fp16


@HEADS.register_module()
class BEVFormerHead(DETRHead):
    """BEVFormer 基础检测头

    继承自 DETRHead，实现了 BEV 特征生成和 3D 目标检测功能。

    Args:
        with_box_refine: 是否在 Decoder 中逐层 refine 框位置
            True  → 每层 Decoder 都有独立的回归分支，逐层更新参考点
            False → 所有 Decoder 层共享回归分支
        as_two_stage: 是否使用两阶段检测 (Encoder 生成初始 proposal)
        transformer: PerceptionTransformer 配置
        bbox_coder: 检测框编码/解码器配置
        num_cls_fcs: 分类分支的全连接层数
        code_weights: 框回归各维度的损失权重
        bev_h, bev_w: BEV 特征图的空间尺寸
    """

    def __init__(self,
                 *args,
                 with_box_refine=False,       # 是否逐层 refine 框
                 as_two_stage=False,           # 是否两阶段检测
                 transformer=None,             # Transformer 配置
                 bbox_coder=None,              # 框编码器配置
                 num_cls_fcs=2,               # 分类分支 FC 层数
                 code_weights=None,            # 回归损失权重
                 bev_h=30,                     # BEV 高度
                 bev_w=30,                     # BEV 宽度
                 **kwargs):

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage

        # code_size: 框编码的维度 (10: cx,cy,w,l,cz,h,sinθ,cosθ,vx,vy)
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10

        # code_weights: 框回归各维度的损失权重
        # 前 8 维权重为 1.0，速度维度 (vx,vy) 权重为 0.2
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        # 构建框编码器
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]  # 真实世界宽度
        self.real_h = self.pc_range[4] - self.pc_range[1]  # 真实世界高度
        self.num_cls_fcs = num_cls_fcs - 1
        super(BEVFormerHead, self).__init__(
            *args, transformer=transformer, **kwargs)
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)

    def _init_layers(self):
        """初始化分类分支和回归分支

        分类分支结构:
            Linear(256,256) → LayerNorm → ReLU → ... → Linear(256, num_classes)
            使用 LayerNorm 稳定训练

        回归分支结构:
            Linear(256,256) → ReLU → ... → Linear(256, 10)
            不使用 LayerNorm (回归对归一化更敏感)

        with_box_refine=True 时:
            每层 decoder 都有独立的分支 (num_pred 个副本)
        with_box_refine=False 时:
            所有层共享分支 (但每层仍有独立副本)
        """
        # 分类分支: 多层 FC + LayerNorm + ReLU
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        # 回归分支: 多层 FC + ReLU (无 LayerNorm)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # 需要预测的层数:
        # as_two_stage=True  → decoder_layers + 1 (多一层 encoder 输出)
        # as_two_stage=False → decoder_layers
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])

        # 如果不是两阶段模式，需要创建可学习的 BEV 和 object queries
        if not self.as_two_stage:
            # bev_embedding: BEV 网格查询 (bev_h*bev_w, C)
            self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)
            # query_embedding: 目标查询 (num_query, 2*C)
            # 前半部分为位置编码，后半部分为内容编码
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)

    def init_weights(self):
        """初始化权重

        - Transformer: 调用自身的 init_weights
        - 分类分支: 最后一层 bias 初始化为 -log((1-0.01)/0.01)
           这对应初始预测概率约为 0.01，避免训练初期分类过于自信
        """
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, img_metas, prev_bev=None, only_bev=False):
        """检测头前向传播

        两种模式:
        1. only_bev=True:  只使用 Encoder 生成 BEV 特征 (用于获取历史 BEV)
        2. only_bev=False: Encoder + Decoder → 完整检测结果

        Args:
            mlvl_feats: 多尺度图像特征，每个形状 (B, N, C, H, W)
            img_metas: 图像元信息
            prev_bev: 上一帧 BEV 特征 (时序融合)
            only_bev: 是否只生成 BEV 特征

        Returns:
            outs: 字典，包含:
                - bev_embed: BEV 特征
                - all_cls_scores: 各层分类分数 (nb_dec, bs, num_query, num_cls)
                - all_bbox_preds: 各层框预测 (nb_dec, bs, num_query, 10)
        """
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype

        # 获取可学习的 query embeddings
        object_query_embeds = self.query_embedding.weight.to(dtype)  # (num_query, 2*C)
        bev_queries = self.bev_embedding.weight.to(dtype)            # (bev_h*bev_w, C)

        # 生成 BEV 位置编码
        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        if only_bev:
            # 模式 1: 只生成 BEV 特征 (跳过 Decoder)
            return self.transformer.get_bev_features(
                mlvl_feats, bev_queries,
                self.bev_h, self.bev_w, self.real_h, self.real_w,
                grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
                bev_pos=bev_pos, img_metas=img_metas, prev_bev=prev_bev)
        else:
            # 模式 2: 完整检测流程 (Encoder + Decoder)
            outputs = self.transformer(
                mlvl_feats, bev_queries, object_query_embeds,
                self.bev_h, self.bev_w, self.real_h, self.real_w,
                grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=self.reg_branches if self.with_box_refine else None,
                cls_branches=self.cls_branches if self.as_two_stage else None,
                img_metas=img_metas, prev_bev=prev_bev)

        # 处理 Decoder 输出
        bev_embed, hs, init_reference, inter_references = outputs
        hs = hs.permute(0, 2, 1, 3)  # (num_layers, num_query, bs, C) → (num_layers, bs, num_query, C)
        outputs_classes = []
        outputs_coords = []

        for lvl in range(hs.shape[0]):
            # 获取当前层的参考点
            if lvl == 0:
                reference = init_reference          # 第一层用初始参考点
            else:
                reference = inter_references[lvl - 1]  # 后续层用上一层的参考点

            # 将参考点从 [0,1] 转回 inverse sigmoid 空间
            reference = inverse_sigmoid(reference)

            # 分类和回归预测
            outputs_class = self.cls_branches[lvl](hs[lvl])  # (bs, num_query, num_cls)
            tmp = self.reg_branches[lvl](hs[lvl])            # (bs, num_query, 10)

            assert reference.shape[-1] == 3

            # 解码框坐标: 预测值 → 实际坐标
            # cx, cy = sigmoid(Δcx + ref_cx) → 归一化坐标
            tmp[..., 0:2] += reference[..., 0:2]   # 加上参考点
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()  # sigmoid 到 [0,1]
            # cz = sigmoid(Δcz + ref_cz) → 归一化高度
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()

            # 反归一化到实际坐标范围
            tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])

            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        outs = {
            'bev_embed': bev_embed,
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
        }

        return outs

    def _get_target_single(self, cls_score, bbox_pred, gt_labels, gt_bboxes, gt_bboxes_ignore=None):
        """为单张图像计算分类和回归目标

        通过匈牙利匹配将预测框与 GT 框关联，生成训练目标。

        Args:
            cls_score: 分类分数 (num_query, num_cls)
            bbox_pred: 预测框 (num_query, code_size)
            gt_labels: GT 标签 (num_gts,)
            gt_bboxes: GT 框 (num_gts, code_size)

        Returns:
            labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds
        """
        num_bboxes = bbox_pred.size(0)
        gt_c = gt_bboxes.shape[-1]

        # 匈牙利匹配: 将预测框与 GT 框进行最优匹配
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)
        # 采样: 选取正负样本
        sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)
        pos_inds = sampling_result.pos_inds  # 正样本索引
        neg_inds = sampling_result.neg_inds  # 负样本索引

        # 标签: 正样本为 GT 类别，负样本为背景 (num_classes)
        labels = gt_bboxes.new_full((num_bboxes,), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # 框目标: 正样本为 GT 框，负样本为 0
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds)

    def get_targets(self, cls_scores_list, bbox_preds_list, gt_bboxes_list, gt_labels_list, gt_bboxes_ignore_list=None):
        """为 batch 中的所有图像计算目标"""
        assert gt_bboxes_ignore_list is None, 'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def loss_single(self, cls_scores, bbox_preds, gt_bboxes_list, gt_labels_list, gt_bboxes_ignore_list=None):
        """单层 Decoder 的损失计算

        计算分类损失 (Focal Loss) 和框回归损失 (L1 Loss)。

        Args:
            cls_scores: 分类分数 (bs, num_query, num_cls)
            bbox_preds: 预测框 (bs, num_query, 10)
            gt_bboxes_list: GT 框列表
            gt_labels_list: GT 标签列表

        Returns:
            loss_cls: 分类损失
            loss_bbox: 框回归损失
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list, gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # 分类损失 (Focal Loss)
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # 框回归损失 (L1 Loss)
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights  # 应用维度权重

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan, :10],
            bbox_weights[isnotnan, :10], avg_factor=num_total_pos)

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, gt_bboxes_list, gt_labels_list, preds_dicts, gt_bboxes_ignore=None, img_metas=None):
        """检测损失计算

        对每层 Decoder 的输出分别计算损失，实现逐层监督。
        - 最后一层: key 为 'loss_cls', 'loss_bbox'
        - 中间层: key 为 'd0.loss_cls', 'd0.loss_bbox', ...

        Args:
            gt_bboxes_list: GT 3D 框列表
            gt_labels_list: GT 标签列表
            preds_dicts: 预测结果字典 (all_cls_scores, all_bbox_preds)

        Returns:
            loss_dict: 损失字典
        """
        assert gt_bboxes_ignore is None, f'{self.__class__.__name__} only supports for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        # 将 GT 框转换为 (cx, cy, w, l, cz, h, sinθ, cosθ, vx, vy) 格式
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1).to(device)
            for gt_bboxes in gt_bboxes_list]

        # 为每层 Decoder 复制 GT (逐层监督)
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers)]

        # 计算各层损失
        losses_cls, losses_bbox = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list, all_gt_bboxes_ignore_list)

        loss_dict = dict()

        # Encoder 输出的损失 (两阶段模式)
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i]) for i in range(len(all_gt_labels_list))]
            enc_loss_cls, enc_losses_bbox = self.loss_single(
                enc_cls_scores, enc_bbox_preds, gt_bboxes_list, binary_labels_list, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        # 最后一层 Decoder 的损失
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        # 中间层 Decoder 的损失 (辅助损失)
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1], losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """从预测结果生成最终检测框

        通过 bbox_coder 解码预测结果，生成 3D 检测框。

        Args:
            preds_dicts: 预测结果字典
            img_metas: 图像元信息

        Returns:
            ret_list: 每个样本的 [bboxes, scores, labels]
        """
        preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']

            # 将框中心从底部调整到几何中心 (cz - h/2)
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5

            code_size = bboxes.shape[-1]
            bboxes = img_metas[i]['box_type_3d'](bboxes, code_size)
            scores = preds['scores']
            labels = preds['labels']

            ret_list.append([bboxes, scores, labels])

        return ret_list


@HEADS.register_module()
class BEVFormerHead_GroupDETR(BEVFormerHead):
    """Group DETR 版本的 BEVFormerHead

    使用 Group DETR 策略: 训练时使用 group_detr 组独立的 query，
    每组 query 通过匈牙利匹配与 GT 关联，相当于增加了正样本数量。
    推理时只使用一组 query。

    优势:
    - 训练时更多正样本 → 更快收敛
    - 推理时计算量不变 (只使用一组 query)

    Args:
        group_detr: query 组数，默认 1
    """
    def __init__(self, *args, group_detr=1, **kwargs):
        self.group_detr = group_detr
        assert 'num_query' in kwargs
        kwargs['num_query'] = group_detr * kwargs['num_query']  # 总 query 数 = 组数 × 每组 query 数
        super().__init__(*args, **kwargs)

    def forward(self, mlvl_feats, img_metas, prev_bev=None, only_bev=False):
        """Group DETR 前向传播

        与 BEVFormerHead 的区别:
        推理时只取前 num_query/group_detr 个 query，减少计算量。
        """
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        object_query_embeds = self.query_embedding.weight.to(dtype)
        if not self.training:
            # 推理时只使用一组 query
            object_query_embeds = object_query_embeds[:self.num_query // self.group_detr]
        bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        if only_bev:
            return self.transformer.get_bev_features(
                mlvl_feats, bev_queries, self.bev_h, self.bev_w,
                grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
                bev_pos=bev_pos, img_metas=img_metas, prev_bev=prev_bev)
        else:
            outputs = self.transformer(
                mlvl_feats, bev_queries, object_query_embeds,
                self.bev_h, self.bev_w,
                grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=self.reg_branches if self.with_box_refine else None,
                cls_branches=self.cls_branches if self.as_two_stage else None,
                img_metas=img_metas, prev_bev=prev_bev)

        # 解码输出 (与 BEVFormerHead 相同)
        bev_embed, hs, init_reference, inter_references = outputs
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])
            assert reference.shape[-1] == 3
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        outs = {
            'bev_embed': bev_embed,
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
        }

        return outs

    def loss(self, gt_bboxes_list, gt_labels_list, preds_dicts, gt_bboxes_ignore=None, img_metas=None):
        """Group DETR 损失计算

        将 query 按 group_detr 分组，每组独立计算损失，最后取平均。
        这样每组 query 都通过匈牙利匹配与所有 GT 关联，相当于增加了正样本数量。
        """
        assert gt_bboxes_ignore is None

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']
        assert enc_cls_scores is None and enc_bbox_preds is None

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1).to(device)
            for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers)]

        loss_dict = dict()
        loss_dict['loss_cls'] = 0
        loss_dict['loss_bbox'] = 0
        for num_dec_layer in range(all_cls_scores.shape[0] - 1):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = 0
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = 0

        # 按组计算损失并取平均
        num_query_per_group = self.num_query // self.group_detr
        for group_index in range(self.group_detr):
            group_query_start = group_index * num_query_per_group
            group_query_end = (group_index+1) * num_query_per_group
            group_cls_scores = all_cls_scores[:, :, group_query_start:group_query_end, :]
            group_bbox_preds = all_bbox_preds[:, :, group_query_start:group_query_end, :]
            losses_cls, losses_bbox = multi_apply(
                self.loss_single, group_cls_scores, group_bbox_preds,
                all_gt_bboxes_list, all_gt_labels_list, all_gt_bboxes_ignore_list)
            loss_dict['loss_cls'] += losses_cls[-1] / self.group_detr
            loss_dict['loss_bbox'] += losses_bbox[-1] / self.group_detr
            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1], losses_bbox[:-1]):
                loss_dict[f'd{num_dec_layer}.loss_cls'] += loss_cls_i / self.group_detr
                loss_dict[f'd{num_dec_layer}.loss_bbox'] += loss_bbox_i / self.group_detr
                num_dec_layer += 1
        return loss_dict