"""
seg_detr_head.py - 分割DETR检测头模块

本模块实现了用于分割任务的DETR（DEtection TRansformer）检测头（SegDETRHead），
继承自 mmdet 的 AnchorFreeHead。

DETR是一种基于Transformer的端到端目标检测框架，其核心思想是将目标检测
建模为集合预测问题，通过Transformer编码器-解码器结构直接输出预测结果，
无需锚框（anchor）和非极大值抑制（NMS）等后处理步骤。

本模块的核心功能：
1. 构建Transformer编解码器，对多尺度特征进行编码和解码。
2. 通过可学习的查询嵌入（query embedding）与编码器输出进行交互，
   直接预测目标的类别和边界框。
3. 支持 things 类别和 stuff 类别的预测（常用于全景分割任务）。
4. 使用匈牙利算法进行预测与真值的一对一匹配。
5. 计算分类损失、回归L1损失和GIoU损失。

参考论文：End-to-End Object Detection with Transformers
<https://arxiv.org/pdf/2005.12872>
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, Linear, build_activation_layer
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding
from mmcv.runner import force_fp32

from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.models.utils import build_transformer

from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet.models.builder import HEADS, build_loss


@HEADS.register_module()
class SegDETRHead(
        AnchorFreeHead
):
    """
    分割DETR检测头 —— 为全景分割任务修改的DETR检测头。

    该类在标准DETRHead的基础上进行了扩展，支持同时预测 things 类别（可数物体，
    如人、车等）和 stuff 类别（不可数区域，如天空、道路等），常用于全景分割任务。

    与标准DETRHead的主要区别：
    - 区分 things_classes 和 stuff_classes 两种类别。
    - 支持掩码（mask）预测（通过与其他模块配合）。
    - 分类权重中单独处理背景类（background class）的权重。

    Args:
        num_classes (int): 总类别数（things + stuff），不包括背景。
        num_things_classes (int): things 类别数（可数物体类别）。
        num_stuff_classes (int): stuff 类别数（不可数区域类别）。
        in_channels (int): 输入特征图的通道数。
        num_query (int): Transformer中查询（query）的数量，默认100。
        num_reg_fcs (int, optional): 回归FFN中全连接层数，默认2。
        transformer (dict): Transformer的配置字典。
        sync_cls_avg_factor (bool): 是否在多卡间同步分类损失的归一化因子。
            默认 False。
        positional_encoding (dict): 位置编码的配置字典。
        loss_cls (dict): 分类损失的配置字典。
        loss_bbox (dict): 回归L1损失的配置字典。
        loss_iou (dict): GIoU损失的配置字典。
        train_cfg (dict): 训练配置，包含分配器（assigner）配置。
        test_cfg (dict): 测试配置，包含 max_per_img 等参数。
        init_cfg (dict or list[dict], optional): 初始化配置字典。
        **kwargs: 其他传递给父类的参数。
    """

    _version = 2

    def __init__(
            self,
            num_classes,
            num_things_classes,
            num_stuff_classes,
            in_channels,
            num_query=100,
            num_reg_fcs=2,
            transformer=None,
            sync_cls_avg_factor=False,
            positional_encoding=dict(type='SinePositionalEncoding',
                                     num_feats=128,
                                     normalize=True),
            loss_cls=dict(type='CrossEntropyLoss',
                          bg_cls_weight=0.1,
                          use_sigmoid=False,
                          loss_weight=1.0,
                          class_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=5.0),
            loss_iou=dict(type='GIoULoss', loss_weight=2.0),
            train_cfg=dict(assigner=dict(
                type='HungarianAssigner',
                cls_cost=dict(type='ClassificationCost', weight=1.),
                reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
            test_cfg=dict(max_per_img=100),
            init_cfg=None,
            **kwargs):
        # 注意：这里使用 AnchorFreeHead 而不是 TransformerHead 作为父类，
        # 因为 TransformerHead 的初始化会带来不便
        super(AnchorFreeHead, self).__init__(init_cfg)
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor

        # 处理分类损失中的类别权重
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__ is SegDETRHead):
            assert isinstance(class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(class_weight)}.'
            # 注意：遵循官方DETR的实现，bg_cls_weight 表示无物体类别
            #（背景类）的相对分类权重
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(bg_cls_weight)}.'
            # 构建类别权重张量：things 类别使用统一的 class_weight，
            # 背景类使用 bg_cls_weight
            class_weight = torch.ones(num_things_classes + 1) * class_weight
            # 将背景类设置为最后一个类别
            class_weight[num_things_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight

        # 构建分配器和采样器
        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided '\
                'when train_cfg is set.'
            assigner = train_cfg['assigner']
            self.assigner = build_assigner(assigner)
            # DETR 中 sampling=False，因此使用伪采样器（PseudoSampler）
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        # 存储基本配置参数
        self.num_query = num_query                  # 查询数量
        self.num_classes = num_classes              # 总类别数（things + stuff）
        self.num_things_classes = num_things_classes  # things 类别数
        self.num_stuff_classes = num_stuff_classes    # stuff 类别数
        self.in_channels = in_channels              # 输入通道数
        self.num_reg_fcs = num_reg_fcs              # 回归FFN中的全连接层数
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False                   # 禁用FP16自动转换

        # 构建损失函数
        self.loss_cls = build_loss(loss_cls)    # 分类损失
        self.loss_bbox = build_loss(loss_bbox)  # 回归L1损失
        self.loss_iou = build_loss(loss_iou)    # GIoU损失

        # 确定分类输出通道数
        if self.loss_cls.use_sigmoid:
            # 使用sigmoid时，每个类别独立预测，输出通道数 = things类别数
            self.cls_out_channels = num_things_classes
        else:
            # 使用softmax时，需要额外的背景类，输出通道数 = things类别数 + 1
            self.cls_out_channels = num_things_classes + 1

        # 构建激活函数（默认ReLU）
        self.act_cfg = transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=True))
        self.activate = build_activation_layer(self.act_cfg)

        # 构建位置编码
        self.positional_encoding = build_positional_encoding(
            positional_encoding)

        # 构建Transformer（SegDeformableTransformer）
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims

        # 验证位置编码维度与嵌入维度的一致性
        assert 'num_feats' in positional_encoding
        num_feats = positional_encoding['num_feats']
        assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
            f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
            f' and {num_feats}.'

        # 初始化各层
        self._init_layers()

    def _init_layers(self):
        """
        初始化检测头的各层。

        包括：
        - input_proj: 输入投影层，将backbone特征通道数映射到Transformer的嵌入维度。
        - fc_cls: 分类全连接层，从嵌入维度预测类别logits。
        - reg_ffn: 回归前馈网络（FFN），对嵌入特征进行非线性变换。
        - fc_reg: 回归全连接层，预测边界框坐标（cx, cy, w, h）。
        - query_embedding: 可学习的查询嵌入，作为解码器的初始查询。
        """
        # 输入投影：1x1卷积将输入通道数映射到嵌入维度
        self.input_proj = Conv2d(self.in_channels,
                                 self.embed_dims,
                                 kernel_size=1)
        # 分类头：线性层从嵌入维度预测类别logits
        self.fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        # 回归FFN：多层全连接网络对特征进行非线性变换
        self.reg_ffn = FFN(self.embed_dims,
                           self.embed_dims,
                           self.num_reg_fcs,
                           self.act_cfg,
                           dropout=0.0,
                           add_residual=False)
        # 回归头：线性层预测边界框坐标 (cx, cy, w, h)
        self.fc_reg = Linear(self.embed_dims, 4)
        # 可学习的查询嵌入：每个查询对应一个嵌入向量
        self.query_embedding = nn.Embedding(self.num_query, self.embed_dims)

    def init_weights(self):
        """
        初始化检测头的权重。

        Transformer的初始化非常重要，这里调用Transformer自身的初始化方法。
        """
        self.transformer.init_weights()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """
        从状态字典加载权重（兼容旧版本checkpoint）。

        由于版本升级，某些参数的名称发生了变化，该函数负责名称映射：
        - .self_attn. -> .attentions.0.
        - .ffn. -> .ffns.0.
        - .multihead_attn. -> .attentions.1.
        - .decoder.norm. -> .decoder.post_norm.

        Args:
            state_dict (dict): 模型状态字典。
            prefix (str): 参数名前缀。
            local_metadata (dict): 本地元数据，包含版本信息。
            strict (bool): 是否严格匹配。
            missing_keys (list): 缺失的键列表。
            unexpected_keys (list): 多余的键列表。
            error_msgs (list): 错误信息列表。
        """
        # 注意：这里使用 AnchorFreeHead 而不是 TransformerHead，
        # 因为 AnchorFreeHead._load_from_state_dict 不应被调用。
        # 调用默认的 Module._load_from_state_dict 就足够了。

        version = local_metadata.get('version', None)
        if (version is None or version < 2) and self.__class__ is SegDETRHead:
            # 旧版本参数名到新版本参数名的映射表
            convert_dict = {
                '.self_attn.': '.attentions.0.',
                '.ffn.': '.ffns.0.',
                '.multihead_attn.': '.attentions.1.',
                '.decoder.norm.': '.decoder.post_norm.'
            }
            for k in state_dict.keys():
                for ori_key, convert_key in convert_dict.items():
                    if ori_key in k:
                        convert_key = k.replace(ori_key, convert_key)
                        state_dict[convert_key] = state_dict[k]
                        del state_dict[k]

        super(AnchorFreeHead,
              self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)

    def forward(self, feats, img_metas):
        """
        前向传播函数。

        对每个特征层级调用 forward_single 进行单层级的处理。

        Args:
            feats (tuple[Tensor]): 上游网络输出的多尺度特征图，
                每个元素的形状为 [bs, c, h, w]。
            img_metas (list[dict]): 图像的元信息列表。

        Returns:
            tuple[list[Tensor], list[Tensor]]: 所有尺度层级的输出。
                - all_cls_scores_list (list[Tensor]): 各尺度层级的分类分数。
                  每个元素的形状为 [nb_dec, bs, num_query, cls_out_channels]。
                - all_bbox_preds_list (list[Tensor]): 各尺度层级的回归输出。
                  每个元素的形状为 [nb_dec, bs, num_query, 4]，
                  格式为归一化的 (cx, cy, w, h)。
        """
        num_levels = len(feats)
        img_metas_list = [img_metas for _ in range(num_levels)]
        return multi_apply(self.forward_single, feats, img_metas_list)

    def forward_single(self, x, img_metas):
        """
        单个特征层级的前向传播。

        处理流程：
        1. 构建二值掩码（binary mask），标记padding位置。
        2. 通过 input_proj 将输入特征映射到嵌入维度。
        3. 对掩码进行插值，与特征图空间尺寸对齐。
        4. 计算位置编码。
        5. 通过Transformer进行编解码。
        6. 对解码器输出进行分类和回归预测。

        Args:
            x (Tensor): backbone单层级的输入特征，形状为 [bs, c, h, w]。
            img_metas (list[dict]): 图像的元信息列表。

        Returns:
            all_cls_scores (Tensor): 分类头的输出，
                形状为 [nb_dec, bs, num_query, cls_out_channels]。
            all_bbox_preds (Tensor): 回归头的sigmoid输出，
                形状为 [nb_dec, bs, num_query, 4]，
                格式为归一化的 (cx, cy, w, h)。
        """
        # 构建二值掩码：用于Transformer中标记padding/忽略位置
        # 注意：遵循官方DETR的实现，非零值表示忽略位置，零值表示有效位置
        batch_size = x.size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        masks = x.new_ones((batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            # 将有效图像区域设为0（有效），padding区域保持为1（忽略）
            masks[img_id, :img_h, :img_w] = 0

        # 输入投影：将通道数从 in_channels 映射到 embed_dims
        x = self.input_proj(x)
        # 将掩码插值到与特征图相同的空间尺寸，并转为布尔类型
        masks = F.interpolate(masks.unsqueeze(1),
                              size=x.shape[-2:]).to(torch.bool).squeeze(1)
        # 位置编码：基于掩码生成正弦位置编码，形状为 [bs, embed_dim, h, w]
        pos_embed = self.positional_encoding(masks)

        # Transformer 编解码
        # outs_dec: 解码器输出，形状为 [nb_dec, bs, num_query, embed_dim]
        outs_dec, _ = self.transformer(x, masks, self.query_embedding.weight,
                                       pos_embed)

        # 分类预测：线性层将嵌入维度映射到类别logits
        all_cls_scores = self.fc_cls(outs_dec)
        # 回归预测：先通过 FFN 和激活函数，再通过线性层，最后 sigmoid 归一化到 [0, 1]
        all_bbox_preds = self.fc_reg(self.activate(
            self.reg_ffn(outs_dec))).sigmoid()
        return all_cls_scores, all_bbox_preds

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def loss(self,
             all_cls_scores_list,
             all_bbox_preds_list,
             gt_bboxes_list,
             gt_labels_list,
             img_metas,
             gt_bboxes_ignore=None):
        """
        损失函数。

        默认仅使用最后一个特征层级的输出来计算损失。对每个解码器层分别计算损失，
        包括最后一层（主损失）和中间层（辅助损失）。

        计算流程：
        1. 取最后一个特征层级的分类和回归输出。
        2. 为每个解码器层复制真实标签。
        3. 通过 multi_apply 对每个解码器层调用 loss_single。
        4. 收集并组织所有损失。

        Args:
            all_cls_scores_list (list[Tensor]): 各特征层级的分类输出。
                每个元素的形状为 [nb_dec, bs, num_query, cls_out_channels]。
            all_bbox_preds_list (list[Tensor]): 各特征层级的回归输出。
                每个元素的形状为 [nb_dec, bs, num_query, 4]。
            gt_bboxes_list (list[Tensor]): 每张图像的真实边界框，
                形状为 (num_gts, 4)，格式为 [tl_x, tl_y, br_x, br_y]。
            gt_labels_list (list[Tensor]): 每张图像的真实类别索引，
                形状为 (num_gts,)。
            img_metas (list[dict]): 图像的元信息列表。
            gt_bboxes_ignore (list[Tensor], optional): 需要忽略的真实框。
                默认 None。

        Returns:
            dict[str, Tensor]: 损失组件字典，包括：
                - loss_cls: 最后一层解码器的分类损失
                - loss_bbox: 最后一层解码器的回归L1损失
                - loss_iou: 最后一层解码器的GIoU损失
                - d{num}.loss_cls: 第num层中间解码器的分类损失
                - d{num}.loss_bbox: 第num层中间解码器的回归L1损失
                - d{num}.loss_iou: 第num层中间解码器的GIoU损失
        """
        # 默认仅使用最后一个特征层级的输出
        all_cls_scores = all_cls_scores_list[-1]
        all_bbox_preds = all_bbox_preds_list[-1]
        assert gt_bboxes_ignore is None, \
            'Only supports for gt_bboxes_ignore setting to None.'

        num_dec_layers = len(all_cls_scores)
        # 为每个解码器层复制真实标签
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]

        # 对每个解码器层计算损失
        losses_cls, losses_bbox, losses_iou = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list, img_metas_list,
            all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # 最后一层解码器的损失（主损失）
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        # 中间解码器层的损失（辅助损失），用于深监督（deep supervision）
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i in zip(losses_cls[:-1],
                                                       losses_bbox[:-1],
                                                       losses_iou[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            num_dec_layer += 1
        return loss_dict

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):
        """
        单个解码器层、单个特征层级的损失函数。

        计算流程：
        1. 获取分配目标：通过匈牙利匹配将预测分配给真实框。
        2. 计算分类损失：使用交叉熵损失，加权平均因子结合正负样本数。
        3. 计算GIoU损失：将预测框坐标还原到图像尺度后计算。
        4. 计算回归L1损失：直接对归一化坐标计算L1距离。

        Args:
            cls_scores (Tensor): 单个解码器层的分类logits，
                形状为 [bs, num_query, cls_out_channels]。
            bbox_preds (Tensor): 单个解码器层的sigmoid回归输出，
                形状为 [bs, num_query, 4]，格式为归一化的 (cx, cy, w, h)。
            gt_bboxes_list (list[Tensor]): 每张图像的真实边界框，
                形状为 (num_gts, 4)，格式为 [tl_x, tl_y, br_x, br_y]。
            gt_labels_list (list[Tensor]): 每张图像的真实类别索引，
                形状为 (num_gts,)。
            img_metas (list[dict]): 图像的元信息列表。
            gt_bboxes_ignore_list (list[Tensor], optional): 需要忽略的真实框。
                默认 None。

        Returns:
            tuple:
                - loss_cls (Tensor): 分类损失标量
                - loss_bbox (Tensor): 回归L1损失标量
                - loss_iou (Tensor): GIoU损失标量
        """
        num_imgs = cls_scores.size(0)
        # 按图像拆分预测结果
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]

        # 获取分类和回归的目标值
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           img_metas, gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        # 拼接所有图像的目标
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # ===== 分类损失 =====
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # 构建加权平均因子：与官方DETR实现保持一致
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            # 多卡训练时同步归一化因子
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(cls_scores,
                                 labels,
                                 label_weights,
                                 avg_factor=cls_avg_factor)

        # 计算所有GPU上的平均真实框数量，用于归一化
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # ===== 构建用于还原边界框的缩放因子 =====
        factors = []
        for img_meta, bbox_pred in zip(img_metas, bbox_preds):
            img_h, img_w, _ = img_meta['img_shape']
            # 缩放因子: [img_w, img_h, img_w, img_h]
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR回归的是图像中的相对位置（cxcywh），目标值被归一化到[0,1]。
        # 因此计算IoU损失时需要将其还原到图像尺度
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # ===== GIoU损失 =====
        loss_iou = self.loss_iou(bboxes,
                                 bboxes_gt,
                                 bbox_weights,
                                 avg_factor=num_total_pos)

        # ===== 回归L1损失 =====
        loss_bbox = self.loss_bbox(bbox_preds,
                                   bbox_targets,
                                   bbox_weights,
                                   avg_factor=num_total_pos)
        return loss_cls, loss_bbox, loss_iou

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):
        """
        计算一批图像的回归和分类目标。

        对每张图像调用 _get_target_single，汇总所有图像的目标。

        Args:
            cls_scores_list (list[Tensor]): 每张图像单个解码器层的分类logits，
                每个元素的形状为 [num_query, cls_out_channels]。
            bbox_preds_list (list[Tensor]): 每张图像单个解码器层的回归输出，
                每个元素的形状为 [num_query, 4]。
            gt_bboxes_list (list[Tensor]): 每张图像的真实边界框，
                形状为 (num_gts, 4)，格式为 [tl_x, tl_y, br_x, br_y]。
            gt_labels_list (list[Tensor]): 每张图像的真实类别索引，
                形状为 (num_gts,)。
            img_metas (list[dict]): 图像的元信息列表。
            gt_bboxes_ignore_list (list[Tensor], optional): 需要忽略的真实框。
                默认 None。

        Returns:
            tuple:
                - labels_list (list[Tensor]): 每张图像的标签。
                - label_weights_list (list[Tensor]): 每张图像的标签权重。
                - bbox_targets_list (list[Tensor]): 每张图像的边界框目标。
                - bbox_weights_list (list[Tensor]): 每张图像的边界框权重。
                - num_total_pos (int): 所有图像中正样本的总数。
                - num_total_neg (int): 所有图像中负样本的总数。
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single, cls_scores_list, bbox_preds_list,
             gt_bboxes_list, gt_labels_list, img_metas, gt_bboxes_ignore_list)
        # 统计正样本和负样本总数
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_bboxes,
                           gt_labels,
                           img_meta,
                           gt_bboxes_ignore=None):
        """
        计算单张图像的回归和分类目标。

        处理流程：
        1. 使用匈牙利算法（HungarianAssigner）将预测框分配给真实框。
        2. 使用伪采样器（PseudoSampler）获取正负样本。
        3. 构建标签目标：正样本分配对应的真实类别，负样本分配背景类。
        4. 构建边界框目标：将真实框坐标归一化并转为 (cx, cy, w, h) 格式。

        Args:
            cls_score (Tensor): 单张图像单个解码器层的分类logits，
                形状为 [num_query, cls_out_channels]。
            bbox_pred (Tensor): 单张图像单个解码器层的回归输出，
                形状为 [num_query, 4]，格式为归一化的 (cx, cy, w, h)。
            gt_bboxes (Tensor): 单张图像的真实边界框，
                形状为 (num_gts, 4)，格式为 [tl_x, tl_y, br_x, br_y]。
            gt_labels (Tensor): 单张图像的真实类别索引，
                形状为 (num_gts,)。
            img_meta (dict): 单张图像的元信息。
            gt_bboxes_ignore (Tensor, optional): 需要忽略的真实框。
                默认 None。

        Returns:
            tuple[Tensor]:
                - labels (Tensor): 每个查询的标签，形状为 (num_query,)。
                - label_weights (Tensor): 每个查询的标签权重，形状为 (num_query,)。
                - bbox_targets (Tensor): 每个查询的边界框目标，
                  形状为 (num_query, 4)，格式为归一化的 (cx, cy, w, h)。
                - bbox_weights (Tensor): 每个查询的边界框权重，
                  形状为 (num_query, 4)。
                - pos_inds (Tensor): 正样本的索引。
                - neg_inds (Tensor): 负样本的索引。
        """
        num_bboxes = bbox_pred.size(0)

        # 步骤1: 使用匈牙利匹配进行分配
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                            gt_labels, img_meta,
                                            gt_bboxes_ignore)
        # 步骤2: 通过伪采样器获取正负样本
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # 步骤3: 构建标签目标
        # 默认所有查询分配为背景类（num_things_classes 是背景类的索引）
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_things_classes,
                                    dtype=torch.long)
        # 正样本分配对应的真实类别
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        # 标签权重全为1
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # 步骤4: 构建边界框目标
        bbox_targets = torch.zeros_like(bbox_pred)
        bbox_weights = torch.zeros_like(bbox_pred)
        # 正样本的边界框权重为1
        bbox_weights[pos_inds] = 1.0
        img_h, img_w, _ = img_meta['img_shape']

        # DETR回归的是图像中的相对位置（cxcywh），需要将目标归一化到[0,1]
        # 同时将格式从 (x1,y1,x2,y2) 转换为 (cx,cy,w,h)
        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        pos_gt_bboxes_normalized = sampling_result.pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        训练模式的前向传播函数。

        重写父类方法的原因是 img_metas 需要作为检测头的输入。

        Args:
            x (list[Tensor]): backbone输出的特征图列表。
            img_metas (list[dict]): 每张图像的元信息，包括图像尺寸、缩放因子等。
            gt_bboxes (Tensor): 图像的真实边界框，形状为 (num_gts, 4)。
            gt_labels (Tensor): 每个真实框的类别标签，形状为 (num_gts,)。
            gt_bboxes_ignore (Tensor): 需要忽略的真实框，形状为 (num_ignored_gts, 4)。
            proposal_cfg (mmcv.Config): 测试/后处理配置，如果为None则使用test_cfg。
            **kwargs: 其他参数。

        Returns:
            dict[str, Tensor]: 损失组件字典。
        """
        assert proposal_cfg is None, '"proposal_cfg" must be None'
        outs = self(x, img_metas)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def get_bboxes(self,
                   all_cls_scores_list,
                   all_bbox_preds_list,
                   img_metas,
                   rescale=False):
        """
        将网络输出转换为一组边界框预测。

        使用最后一个特征层级、最后一个解码器层的输出。

        Args:
            all_cls_scores_list (list[Tensor]): 各特征层级的分类输出。
                每个元素的形状为 [nb_dec, bs, num_query, cls_out_channels]。
            all_bbox_preds_list (list[Tensor]): 各特征层级的回归输出。
                每个元素的形状为 [nb_dec, bs, num_query, 4]。
            img_metas (list[dict]): 每张图像的元信息。
            rescale (bool, optional): 如果为True，将边界框还原到原始图像空间。
                默认 False。

        Returns:
            list[list[Tensor, Tensor]]: 每张图像的检测结果。
                每个元素是包含两个Tensor的列表：
                - 第一个Tensor形状为 (n, 5)：前4列是边界框坐标
                  (tl_x, tl_y, br_x, br_y)，第5列是置信度分数。
                - 第二个Tensor形状为 (n,)：每个框对应的预测类别标签。
        """
        # 默认仅使用最后一个特征层级、最后一个解码器层的输出
        cls_scores = all_cls_scores_list[-1][-1]
        bbox_preds = all_bbox_preds_list[-1][-1]

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score, bbox_pred,
                                                img_shape, scale_factor,
                                                rescale)
            result_list.append(proposals)

        return result_list

    def _get_bboxes_single(self,
                           cls_score,
                           bbox_pred,
                           img_shape,
                           scale_factor,
                           rescale=False):
        """
        将单张图像最后一个解码器层的输出转换为边界框预测。

        处理流程：
        1. 根据测试配置确定最大检测数量（max_per_img）。
        2. 排除背景类：
           - 如果使用sigmoid：对所有类别分数展平后取top-k。
           - 如果使用softmax：排除背景类后取每个位置的最大分数。
        3. 将边界框格式从 (cx, cy, w, h) 转换为 (x1, y1, x2, y2)。
        4. 将归一化坐标还原到图像尺度。
        5. 裁剪边界框到图像范围内。
        6. 如果 rescale=True，将边界框还原到原始图像尺度。

        Args:
            cls_score (Tensor): 单张图像最后一个解码器层的分类logits，
                形状为 [num_query, cls_out_channels]。
            bbox_pred (Tensor): 单张图像最后一个解码器层的回归输出，
                形状为 [num_query, 4]，格式为 (cx, cy, w, h)。
            img_shape (tuple[int]): 输入图像的形状 (height, width, 3)。
            scale_factor (ndarray): 图像的缩放因子 (w_scale, h_scale, w_scale, h_scale)。
            rescale (bool, optional): 如果为True，返回原始图像空间的边界框。
                默认 False。

        Returns:
            tuple[Tensor]:
                - det_bboxes: 检测到的边界框，形状为 [num_query, 5]，
                  前4列是边界框位置 (tl_x, tl_y, br_x, br_y)，
                  第5列是置信度分数。
                - det_labels: 检测到的类别标签，形状为 [num_query]。
        """
        assert len(cls_score) == len(bbox_pred)
        max_per_img = self.test_cfg.get('max_per_img', self.num_query)

        # 排除背景类，选择最优预测
        if self.loss_cls.use_sigmoid:
            # 使用sigmoid模式：每个类别独立预测，展平所有分数后取top-k
            cls_score = cls_score.sigmoid()
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            det_labels = indexes % self.num_things_classes
            bbox_index = indexes // self.num_things_classes
            bbox_pred = bbox_pred[bbox_index]
        else:
            # 使用softmax模式：排除背景类（最后一类），取每个位置的最大分数
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            bbox_pred = bbox_pred[bbox_index]
            det_labels = det_labels[bbox_index]

        # 将边界框格式从 (cx, cy, w, h) 转换为 (x1, y1, x2, y2)
        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        # 将归一化坐标还原到图像尺度
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]  # x坐标
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]  # y坐标
        # 裁剪到图像范围内
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        # 如果需要，缩放到原始图像空间
        if rescale:
            det_bboxes /= det_bboxes.new_tensor(scale_factor)
        # 拼接置信度分数
        det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(1)), -1)

        return det_bboxes, det_labels