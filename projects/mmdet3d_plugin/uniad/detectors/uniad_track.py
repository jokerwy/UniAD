#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

"""
UniAD 跟踪模块 (Tracking Module)
========================================================================
本模块实现了 UniAD 中的多目标跟踪功能。核心流程：

1. 图像特征提取 → 多视角图像经过 backbone + neck 提取特征
2. BEV 特征生成 → 利用 BEVFormer 编码器将多视角特征转为鸟瞰图特征
3. 目标检测 → Transformer Decoder 在 BEV 特征上进行 3D 目标检测
4. 时序关联 → 利用 Memory Bank 和 Query Interaction 实现跨帧目标关联
5. 轨迹预测 → 预测每个目标的过去轨迹，用于后续运动预测

训练时：逐帧处理视频片段，通过匈牙利匹配关联 GT 与预测，计算跟踪损失
推理时：逐帧在线处理，维护 track instances 实现持续跟踪
"""

import torch
import torch.nn as nn
from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
import copy
import math
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from mmdet.models import build_loss
from einops import rearrange
from mmdet.models.utils.transformer import inverse_sigmoid
from ..dense_heads.track_head_plugin import MemoryBank, QueryInteractionModule, Instances, RuntimeTrackerBase

@DETECTORS.register_module()
class UniADTrack(MVXTwoStageDetector):
    """UniAD 跟踪检测器

    继承自 MVXTwoStageDetector，负责多目标跟踪的完整流程。
    核心组件包括：
    - BEVFormer Head: 生成 BEV 特征并进行目标检测
    - Memory Bank: 存储历史目标的特征，用于时序关联
    - Query Interaction Module: 融合历史 query 与当前 query
    - RuntimeTrackerBase: 管理跟踪 ID 的分配与消亡
    - Track Criterion: 计算跟踪相关的损失函数
    """
    def __init__(
        self,
        use_grid_mask=False,          # 是否对图像使用 grid mask 数据增强
        img_backbone=None,            # 图像 backbone 网络（如 ResNet-101）
        img_neck=None,                # 图像 neck 网络（如 FPN）
        pts_bbox_head=None,           # 点云/BEV 检测头（BEVFormerTrackHead）
        train_cfg=None,               # 训练配置
        test_cfg=None,                # 测试配置
        pretrained=None,              # 预训练权重路径
        video_test_mode=False,        # 是否启用视频时序测试模式
        loss_cfg=None,                # 损失函数配置（TrackLoss）
        qim_args=dict(                # Query Interaction Module 参数
            qim_type="QIMBase",       # QIM 类型
            merger_dropout=0,         # dropout 概率
            update_query_pos=False,   # 是否更新 query 位置编码
            fp_ratio=0.3,             # 假阳性比例
            random_drop=0.1,          # 随机丢弃比例
        ),
        mem_args=dict(                # Memory Bank 参数
            memory_bank_type="MemoryBank",  # Memory Bank 类型
            memory_bank_score_thresh=0.0,   # 存入 memory 的分数阈值
            memory_bank_len=4,              # memory bank 存储的历史帧数
        ),
        bbox_coder=dict(              # 检测框编码解码器参数
            type="DETRTrack3DCoder",  # 编码器类型
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],  # 后处理有效范围
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],             # 点云范围
            max_num=300,              # 最大检测数量
            num_classes=10,           # 类别数
            score_threshold=0.0,      # 分数阈值
            with_nms=False,           # 是否使用 NMS
            iou_thres=0.3,            # IoU 阈值
        ),
        pc_range=None,                # 点云范围 [x_min, y_min, z_min, x_max, y_max, z_max]
        embed_dims=256,               # 特征嵌入维度
        num_query=900,                # object query 数量（检测目标的上限）
        num_classes=10,               # 目标类别数
        vehicle_id_list=None,         # 车辆类别 ID 列表
        score_thresh=0.2,             # 跟踪分数阈值（高于此值才视为有效跟踪）
        filter_score_thresh=0.1,      # 过滤分数阈值（低于此值的 query 进入休眠）
        miss_tolerance=5,             # 目标丢失容忍帧数（超过此帧数未匹配则删除）
        gt_iou_threshold=0.0,         # GT 匹配的 IoU 阈值
        freeze_img_backbone=False,    # 是否冻结图像 backbone
        freeze_img_neck=False,        # 是否冻结图像 neck
        freeze_bn=False,              # 是否冻结 BatchNorm
        freeze_bev_encoder=False,     # 是否冻结 BEV 编码器
        queue_length=3,               # 训练时使用的视频片段帧数
    ):
        # 初始化父类 MVXTwoStageDetector
        super(UniADTrack, self).__init__(
            img_backbone=img_backbone,
            img_neck=img_neck,
            pts_bbox_head=pts_bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
        )

        # GridMask 数据增强：随机遮挡图像部分区域，防止过拟合
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
        )
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        self.embed_dims = embed_dims
        self.num_query = num_query
        self.num_classes = num_classes
        self.vehicle_id_list = vehicle_id_list
        self.pc_range = pc_range
        self.queue_length = queue_length

        # 冻结图像 backbone：冻结后 backbone 参数不参与梯度更新
        if freeze_img_backbone:
            if freeze_bn:
                self.img_backbone.eval()
            for param in self.img_backbone.parameters():
                param.requires_grad = False

        # 冻结图像 neck
        if freeze_img_neck:
            if freeze_bn:
                self.img_neck.eval()
            for param in self.img_neck.parameters():
                param.requires_grad = False

        # 时序推理模式：必须开启，用于维护帧间 BEV 特征和跟踪状态
        self.video_test_mode = video_test_mode
        assert self.video_test_mode

        # 前一帧信息缓存：用于推理时的时序连续性
        self.prev_frame_info = {
            "prev_bev": None,          # 前一帧的 BEV 特征
            "scene_token": None,       # 当前场景 token
            "prev_pos": 0,             # 前一帧自车位置
            "prev_angle": 0,           # 前一帧自车角度
        }

        # query_embedding: 可学习的检测查询向量
        # 前 num_query 个用于普通目标检测，最后一个 (索引 900) 用于自车 (SDC) 查询
        # 每个 query 维度为 embed_dims*2，前半部分为位置编码，后半部分为内容编码
        self.query_embedding = nn.Embedding(self.num_query+1, self.embed_dims * 2)

        # reference_points: 根据 query 内容预测 3D 参考点 (cx, cy, cz)
        self.reference_points = nn.Linear(self.embed_dims, 3)

        # Memory Bank 长度：存储历史帧数，用于时序特征融合
        self.mem_bank_len = mem_args["memory_bank_len"]

        # RuntimeTrackerBase: 运行时跟踪管理器
        # 负责分配和回收跟踪 ID，管理目标的生命周期（出现→跟踪→丢失→删除）
        self.track_base = RuntimeTrackerBase(
            score_thresh=score_thresh,              # 目标初始化分数阈值
            filter_score_thresh=filter_score_thresh, # 目标休眠分数阈值
            miss_tolerance=miss_tolerance,           # 丢失容忍帧数
        )

        # QueryInteractionModule: Query 交互模块
        # 将历史帧的 query（来自 Memory Bank）与当前帧的新 query 进行融合
        self.query_interact = QueryInteractionModule(
            qim_args,
            dim_in=embed_dims,
            hidden_dim=embed_dims,
            dim_out=embed_dims,
        )

        # bbox_coder: 检测框编码解码器
        # 负责将归一化的预测框解码为实际 3D 坐标，以及后处理过滤
        self.bbox_coder = build_bbox_coder(bbox_coder)

        # MemoryBank: 记忆库
        # 存储每个跟踪目标的历史特征，用于时序一致性建模
        self.memory_bank = MemoryBank(
            mem_args,
            dim_in=embed_dims,
            hidden_dim=embed_dims,
            dim_out=embed_dims,
        )

        self.mem_bank_len = (
            0 if self.memory_bank is None else self.memory_bank.max_his_length
        )

        # criterion: 跟踪损失函数
        # 包括匈牙利匹配、分类损失、框回归损失、轨迹损失等
        self.criterion = build_loss(loss_cfg)

        self.test_track_instances = None   # 推理时维护的跟踪实例
        self.l2g_r_mat = None              # 上一帧 lidar→global 旋转矩阵
        self.l2g_t = None                  # 上一帧 lidar→global 平移向量
        self.gt_iou_threshold = gt_iou_threshold

        # BEV 特征图的空间尺寸（来自 pts_bbox_head 的配置）
        self.bev_h, self.bev_w = self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w
        self.freeze_bev_encoder = freeze_bev_encoder

    def extract_img_feat(self, img, len_queue=None):
        """提取多视角图像特征

        将 6 视角图像通过 backbone + neck 提取多尺度特征。

        Args:
            img: 输入图像，形状为 (B, N, C, H, W)
                 B=batch_size, N=相机数量(6), C=通道数(3), H=高, W=宽
            len_queue: 时序队列长度（训练时表示视频片段的帧数）

        Returns:
            img_feats_reshaped: 多尺度图像特征列表
                每个尺度形状为 (B/len_queue, len_queue, N, C, H, W)
        """
        if img is None:
            return None
        assert img.dim() == 5
        B, N, C, H, W = img.size()

        # 将 batch 和相机维度合并，统一送入 backbone
        # (B, N, C, H, W) → (B*N, C, H, W)
        img = img.reshape(B * N, C, H, W)

        # GridMask 数据增强：随机遮挡图像的网格区域
        if self.use_grid_mask:
            img = self.grid_mask(img)

        # 经过图像 backbone（如 ResNet-101）提取特征
        img_feats = self.img_backbone(img)
        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())

        # 经过图像 neck（如 FPN）进行多尺度特征融合
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        # 将特征 reshape 回多帧、多相机的结构
        # (B*N, C, H, W) → (B/len_queue, len_queue, N, C, H, W)
        img_feats_reshaped = []
        for img_feat in img_feats:
            _, c, h, w = img_feat.size()
            if len_queue is not None:
                img_feat_reshaped = img_feat.view(B//len_queue, len_queue, N, c, h, w)
            else:
                img_feat_reshaped = img_feat.view(B, N, c, h, w)
            img_feats_reshaped.append(img_feat_reshaped)
        return img_feats_reshaped

    def _generate_empty_tracks(self):
        """生成空的跟踪实例

        初始化一个全新的 track_instances 对象，所有字段置为默认值。
        该函数在以下场景被调用：
        - 训练开始时初始化跟踪状态
        - 推理时新场景开始时初始化跟踪状态
        - QueryInteractionModule 中作为新 query 的来源

        Returns:
            track_instances: Instances 对象，包含以下关键字段：
                - query: 可学习的检测查询向量 (num_query+1, 2*embed_dims)
                - ref_pts: 3D 参考点 (num_query+1, 3)
                - pred_boxes: 预测的 3D 框 (num_query+1, 10)
                - pred_logits: 预测的分类分数 (num_query+1, num_classes)
                - obj_idxes: 跟踪 ID，-1 表示未分配 (num_query+1,)
                - score: 检测分数 (num_query+1,)
                - mem_bank: 记忆库 (num_query+1, mem_bank_len, embed_dims)
                - mem_padding_mask: 记忆库填充掩码 (num_query+1, mem_bank_len)
                - disappear_time: 目标消失计时器 (num_query+1,)
        """
        track_instances = Instances((1, 1))
        num_queries, dim = self.query_embedding.weight.shape  # (901, 512)
        device = self.query_embedding.weight.device
        query = self.query_embedding.weight

        # 根据 query 的前半部分（位置编码）预测初始 3D 参考点
        track_instances.ref_pts = self.reference_points(query[..., : dim // 2])

        # 初始化预测框为全零，10 维分别表示:
        # cx, cy, w, l, cz, h, sin(θ), cos(θ), vx, vy
        pred_boxes_init = torch.zeros(
            (len(track_instances), 10), dtype=torch.float, device=device
        )
        track_instances.query = query

        # output_embedding: 用于 Memory Bank 存储的特征
        track_instances.output_embedding = torch.zeros(
            (num_queries, dim >> 1), device=device
        )

        # obj_idxes: 跟踪对象 ID，-1 表示未激活/未匹配到目标
        track_instances.obj_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long, device=device
        )

        # matched_gt_idxes: 匹配到的 GT 索引，-1 表示未匹配
        track_instances.matched_gt_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long, device=device
        )

        # disappear_time: 目标连续丢失帧数计数器
        track_instances.disappear_time = torch.zeros(
            (len(track_instances),), dtype=torch.long, device=device
        )

        # iou: 与 GT 匹配的 IoU 值
        track_instances.iou = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )

        # scores: 检测置信度分数
        track_instances.scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )

        # track_scores: 跟踪分数
        track_instances.track_scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )

        track_instances.pred_boxes = pred_boxes_init

        track_instances.pred_logits = torch.zeros(
            (len(track_instances), self.num_classes), dtype=torch.float, device=device
        )

        # Memory Bank: 初始化为全零
        mem_bank_len = self.mem_bank_len
        track_instances.mem_bank = torch.zeros(
            (len(track_instances), mem_bank_len, dim // 2),
            dtype=torch.float32,
            device=device,
        )

        # mem_padding_mask: True 表示该位置为空（待填充），False 表示有有效数据
        track_instances.mem_padding_mask = torch.ones(
            (len(track_instances), mem_bank_len), dtype=torch.bool, device=device
        )

        # save_period: 记录每个目标在 memory bank 中已存储的帧数
        track_instances.save_period = torch.zeros(
            (len(track_instances),), dtype=torch.float32, device=device
        )

        return track_instances.to(self.query_embedding.weight.device)

    def velo_update(
        self, ref_pts, velocity, l2g_r1, l2g_t1, l2g_r2, l2g_t2, time_delta
    ):
        """速度更新：根据目标速度和时间差更新参考点位置

        将参考点从当前帧的 lidar 坐标系变换到下一帧的 lidar 坐标系。
        变换流程：
        1. 当前 lidar 坐标系 → 世界坐标系 (通过 l2g_r1, l2g_t1)
        2. 加上速度引起的位置偏移
        3. 世界坐标系 → 下一帧 lidar 坐标系 (通过 l2g_r2, l2g_t2)

        Args:
            ref_pts: 参考点 (num_query, 3)，在 inverse sigmoid 空间中
            velocity: 目标速度 (num_query, 2)，vx, vy，单位 m/s
            l2g_r1: 当前帧 lidar→global 旋转矩阵
            l2g_t1: 当前帧 lidar→global 平移向量
            l2g_r2: 下一帧 lidar→global 旋转矩阵
            l2g_t2: 下一帧 lidar→global 平移向量
            time_delta: 两帧之间的时间差 (秒)

        Returns:
            ref_pts: 更新后的参考点 (num_query, 3)，在 inverse sigmoid 空间中
        """
        time_delta = time_delta.type(torch.float)
        num_query = ref_pts.size(0)

        # 速度补零到 3D (vx, vy, 0)
        velo_pad_ = velocity.new_zeros((num_query, 1))
        velo_pad = torch.cat((velocity, velo_pad_), dim=-1)

        # Step 1: 将参考点从 inverse sigmoid 空间转换到实际 3D 坐标
        reference_points = ref_pts.sigmoid().clone()
        pc_range = self.pc_range
        reference_points[..., 0:1] = (
            reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        )
        reference_points[..., 1:2] = (
            reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        )
        reference_points[..., 2:3] = (
            reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        )

        # Step 2: 加上速度引起的位置偏移 (位置 += 速度 × 时间)
        reference_points = reference_points + velo_pad * time_delta

        # Step 3: 当前 lidar 坐标系 → 世界坐标系
        # 然后减去 l2g_t2，准备转换到下一帧的 lidar 坐标系
        ref_pts = reference_points @ l2g_r1 + l2g_t1 - l2g_t2

        # Step 4: 世界坐标系 → 下一帧 lidar 坐标系
        g2l_r = torch.linalg.inv(l2g_r2).type(torch.float)
        ref_pts = ref_pts @ g2l_r

        # Step 5: 重新归一化到 [0, 1] 范围，然后转回 inverse sigmoid 空间
        ref_pts[..., 0:1] = (ref_pts[..., 0:1] - pc_range[0]) / (
            pc_range[3] - pc_range[0]
        )
        ref_pts[..., 1:2] = (ref_pts[..., 1:2] - pc_range[1]) / (
            pc_range[4] - pc_range[1]
        )
        ref_pts[..., 2:3] = (ref_pts[..., 2:3] - pc_range[2]) / (
            pc_range[5] - pc_range[2]
        )

        ref_pts = inverse_sigmoid(ref_pts)

        return ref_pts

    def _copy_tracks_for_loss(self, tgt_instances):
        """为损失计算复制跟踪实例

        在 decoder 的每一层都需要独立的 track_instances 来计算损失。
        此函数复制关键字段（跟踪 ID、匹配信息等），但重置预测相关的字段。

        Args:
            tgt_instances: 源跟踪实例

        Returns:
            track_instances: 复制后的跟踪实例，预测字段被清零
        """
        device = self.query_embedding.weight.device
        track_instances = Instances((1, 1))

        # 深拷贝跟踪状态信息（ID、匹配索引、消失时间）
        track_instances.obj_idxes = copy.deepcopy(tgt_instances.obj_idxes)
        track_instances.matched_gt_idxes = copy.deepcopy(tgt_instances.matched_gt_idxes)
        track_instances.disappear_time = copy.deepcopy(tgt_instances.disappear_time)

        # 预测相关字段初始化为零，等待各层 decoder 填充
        track_instances.scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.track_scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.pred_boxes = torch.zeros(
            (len(track_instances), 10), dtype=torch.float, device=device
        )
        track_instances.iou = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.pred_logits = torch.zeros(
            (len(track_instances), self.num_classes), dtype=torch.float, device=device
        )

        track_instances.save_period = copy.deepcopy(tgt_instances.save_period)
        return track_instances.to(device)

    def get_history_bev(self, imgs_queue, img_metas_list):
        """获取历史 BEV 特征

        迭代处理历史帧图像，生成最新的 BEV 特征。
        为了节省 GPU 显存，此过程不计算梯度。

        Args:
            imgs_queue: 历史帧图像队列，形状 (bs, len_queue, num_cams, C, H, W)
            img_metas_list: 每帧的图像元信息列表

        Returns:
            prev_bev: 最后一帧的 BEV 特征，形状 (bev_h*bev_w, bs, C)
        """
        self.eval()
        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape

            # 将所有帧合并在一起提取特征，提高效率
            imgs_queue = imgs_queue.reshape(bs * len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_img_feat(img=imgs_queue, len_queue=len_queue)

            # 逐帧迭代生成 BEV 特征
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]

                # 调用 BEVFormer Head 的 encoder 生成 BEV 特征
                # 传入 prev_bev 实现时序融合
                prev_bev, _ = self.pts_bbox_head.get_bev_features(
                    mlvl_feats=img_feats,
                    img_metas=img_metas,
                    prev_bev=prev_bev)

        self.train()
        return prev_bev

    def get_bevs(self, imgs, img_metas, prev_img=None, prev_img_metas=None, prev_bev=None):
        """生成 BEV 特征

        这是 BEV 特征生成的统一入口。支持两种模式：
        1. 提供历史帧图像 → 自动计算历史 BEV
        2. 直接提供 prev_bev → 跳过历史 BEV 计算

        Args:
            imgs: 当前帧图像，形状 (B, N, C, H, W)
            img_metas: 当前帧图像元信息
            prev_img: 历史帧图像（可选）
            prev_img_metas: 历史帧图像元信息（可选）
            prev_bev: 上一帧的 BEV 特征（可选，优先级高于 prev_img）

        Returns:
            bev_embed: BEV 特征，形状 (bev_h*bev_w, B, C)
            bev_pos: BEV 位置编码，形状 (B, C, bev_h, bev_w)
        """
        # 如果提供了历史帧图像，先计算历史 BEV
        if prev_img is not None and prev_img_metas is not None:
            assert prev_bev is None
            prev_bev = self.get_history_bev(prev_img, prev_img_metas)

        # 提取当前帧图像特征
        img_feats = self.extract_img_feat(img=imgs)

        # 如果冻结 BEV 编码器，不计算梯度
        if self.freeze_bev_encoder:
            with torch.no_grad():
                bev_embed, bev_pos = self.pts_bbox_head.get_bev_features(
                    mlvl_feats=img_feats, img_metas=img_metas, prev_bev=prev_bev)
        else:
            bev_embed, bev_pos = self.pts_bbox_head.get_bev_features(
                    mlvl_feats=img_feats, img_metas=img_metas, prev_bev=prev_bev)

        # 统一 BEV 特征的形状为 (bev_h*bev_w, B, C)
        if bev_embed.shape[1] == self.bev_h * self.bev_w:
            bev_embed = bev_embed.permute(1, 0, 2)

        assert bev_embed.shape[0] == self.bev_h * self.bev_w
        return bev_embed, bev_pos

    @auto_fp16(apply_to=("img", "prev_bev"))
    def _forward_single_frame_train(
        self,
        img,
        img_metas,
        track_instances,
        prev_img,
        prev_img_metas,
        l2g_r1=None,
        l2g_t1=None,
        l2g_r2=None,
        l2g_t2=None,
        time_delta=None,
        all_query_embeddings=None,
        all_matched_indices=None,
        all_instances_pred_logits=None,
        all_instances_pred_boxes=None,
    ):
        """训练时的单帧前向传播

        处理一帧图像的完整训练流程：
        1. 生成 BEV 特征
        2. 在 BEV 上进行目标检测
        3. 匈牙利匹配 → 关联预测与 GT
        4. 速度更新参考点（为下一帧做准备）
        5. Memory Bank 更新
        6. Query Interaction 融合历史与当前 query

        Args:
            img: 当前帧图像，形状 [B, num_cam, 3, H, W]
            img_metas: 图像元信息
            track_instances: 当前跟踪实例（包含历史 query 和状态）
            prev_img: 历史帧图像
            prev_img_metas: 历史帧图像元信息
            l2g_r1: 当前帧 lidar→global 旋转矩阵
            l2g_t1: 当前帧 lidar→global 平移向量
            l2g_r2: 下一帧 lidar→global 旋转矩阵（最后一帧为 None）
            l2g_t2: 下一帧 lidar→global 平移向量（最后一帧为 None）
            time_delta: 两帧间时间差（最后一帧为 None）
            all_query_embeddings: 收集各层 decoder 的 query embedding
            all_matched_indices: 收集各层 decoder 的匹配索引
            all_instances_pred_logits: 收集各层 decoder 的分类预测
            all_instances_pred_boxes: 收集各层 decoder 的框预测

        Returns:
            out: 字典，包含检测结果、跟踪状态、BEV 特征等
        """
        # ---- Step 1: 生成 BEV 特征 ----
        # 利用 BEVFormer Encoder 将多视角图像转为鸟瞰图特征
        bev_embed, bev_pos = self.get_bevs(
            img, img_metas,
            prev_img=prev_img, prev_img_metas=prev_img_metas,
        )

        # ---- Step 2: 目标检测 ----
        # 在 BEV 特征上执行 Decoder，检测目标并输出分类、回归、轨迹预测
        det_output = self.pts_bbox_head.get_detections(
            bev_embed,
            object_query_embeds=track_instances.query,
            ref_points=track_instances.ref_pts,
            img_metas=img_metas,
        )

        # 提取检测结果
        output_classes = det_output["all_cls_scores"]       # (nb_dec, bs, num_query, num_cls)
        output_coords = det_output["all_bbox_preds"]        # (nb_dec, bs, num_query, 10)
        output_past_trajs = det_output["all_past_traj_preds"]  # (nb_dec, bs, num_query, past_steps, 2)
        last_ref_pts = det_output["last_ref_points"]        # (bs, num_query, 3)
        query_feats = det_output["query_feats"]             # (nb_dec, bs, num_query, C)

        # 构建输出字典
        out = {
            "pred_logits": output_classes[-1],
            "pred_boxes": output_coords[-1],
            "pred_past_trajs": output_past_trajs[-1],
            "ref_pts": last_ref_pts,
            "bev_embed": bev_embed,
            "bev_pos": bev_pos
        }

        # 计算跟踪分数：取最后一层 decoder 分类分数的 sigmoid 最大值
        with torch.no_grad():
            track_scores = output_classes[-1, 0, :].sigmoid().max(dim=-1).values

        # ---- Step 3: 更新跟踪实例 ----
        nb_dec = output_classes.size(0)  # decoder 层数

        # 为每一层 decoder 创建独立的 track_instances 副本用于损失计算
        track_instances_list = [
            self._copy_tracks_for_loss(track_instances) for i in range(nb_dec - 1)
        ]

        # 保存最后一层 decoder 的 query embedding，后续存入 Memory Bank
        track_instances.output_embedding = query_feats[-1][0]  # [300, feat_dim]

        # ---- Step 4: 速度更新参考点 ----
        # 利用目标速度和时间差，将参考点从当前帧变换到下一帧的坐标系
        velo = output_coords[-1, 0, :, -2:]  # [num_query, 2] 提取 vx, vy
        if l2g_r2 is not None:
            ref_pts = self.velo_update(
                last_ref_pts[0],
                velo,
                l2g_r1,
                l2g_t1,
                l2g_r2,
                l2g_t2,
                time_delta=time_delta,
            )
        else:
            ref_pts = last_ref_pts[0]

        # 更新 reference points（保留原始 query 预测的 z 分量）
        dim = track_instances.query.shape[-1]
        track_instances.ref_pts = self.reference_points(track_instances.query[..., :dim//2])
        track_instances.ref_pts[...,:2] = ref_pts[...,:2]

        track_instances_list.append(track_instances)

        # ---- Step 5: 匈牙利匹配 ----
        # 对每一层 decoder 的输出，通过匈牙利算法匹配预测与 GT
        for i in range(nb_dec):
            track_instances = track_instances_list[i]

            track_instances.scores = track_scores
            track_instances.pred_logits = output_classes[i, 0]       # [300, num_cls]
            track_instances.pred_boxes = output_coords[i, 0]         # [300, box_dim]
            track_instances.pred_past_trajs = output_past_trajs[i, 0]  # [300, past_steps, 2]

            out["track_instances"] = track_instances

            # 匈牙利匹配：将预测框与 GT 框进行最优匹配
            track_instances, matched_indices = self.criterion.match_for_single_frame(
                out, i, if_step=(i == (nb_dec - 1))
            )

            # 收集各层 decoder 的结果
            all_query_embeddings.append(query_feats[i][0])
            all_matched_indices.append(matched_indices)
            all_instances_pred_logits.append(output_classes[i, 0])
            all_instances_pred_boxes.append(output_coords[i, 0])

        # ---- Step 6: 筛选活跃的跟踪目标 ----
        # 活跃条件：obj_idxes >= 0（已分配 ID）且 IoU >= 阈值且匹配到了 GT
        active_index = (track_instances.obj_idxes>=0) & (track_instances.iou >= self.gt_iou_threshold) & (track_instances.matched_gt_idxes >=0)
        out.update(self.select_active_track_query(track_instances, active_index, img_metas))

        # 提取自车 (SDC) 查询结果（索引 900 为自车专用 query）
        out.update(self.select_sdc_track_query(track_instances[900], img_metas))

        # ---- Step 7: Memory Bank 更新 ----
        # 将当前帧的目标特征存入 Memory Bank，用于后续帧的时序关联
        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)

        # ---- Step 8: Query Interaction ----
        # 将 Memory Bank 中的历史 query 与空 query 融合，生成下一帧的初始 query
        tmp = {}
        tmp["init_track_instances"] = self._generate_empty_tracks()
        tmp["track_instances"] = track_instances
        out_track_instances = self.query_interact(tmp)
        out["track_instances"] = out_track_instances
        return out

    def select_active_track_query(self, track_instances, active_index, img_metas, with_mask=True):
        """筛选活跃的跟踪查询

        从所有 track instances 中提取活跃的（有 ID 且匹配到 GT 的）目标，
        将其解码为 3D 检测框格式，用于后续的运动预测和规划模块。

        Args:
            track_instances: 所有跟踪实例
            active_index: 布尔索引，标记哪些实例是活跃的
            img_metas: 图像元信息
            with_mask: 是否返回有效性掩码

        Returns:
            result_dict: 包含活跃目标的 3D 框、分数、标签、embedding 等
        """
        result_dict = self._track_instances2results(track_instances[active_index], img_metas, with_mask=with_mask)
        result_dict["track_query_embeddings"] = track_instances.output_embedding[active_index][result_dict['bbox_index']][result_dict['mask']]
        result_dict["track_query_matched_idxes"] = track_instances.matched_gt_idxes[active_index][result_dict['bbox_index']][result_dict['mask']]
        return result_dict

    def select_sdc_track_query(self, sdc_instance, img_metas):
        """提取自车 (SDC) 跟踪查询结果

        自车 query（索引 900）专门用于建模自车状态，输出自车的 3D 框和 embedding。

        Args:
            sdc_instance: 自车跟踪实例
            img_metas: 图像元信息

        Returns:
            out: 包含自车 3D 框、分数、embedding 的字典
        """
        out = dict()
        result_dict = self._track_instances2results(sdc_instance, img_metas, with_mask=False)
        out["sdc_boxes_3d"] = result_dict['boxes_3d']
        out["sdc_scores_3d"] = result_dict['scores_3d']
        out["sdc_track_scores"] = result_dict['track_scores']
        out["sdc_track_bbox_results"] = result_dict['track_bbox_results']
        out["sdc_embedding"] = sdc_instance.output_embedding[0]
        return out

    @auto_fp16(apply_to=("img", "points"))
    def forward_track_train(self,
                            img,
                            gt_bboxes_3d,
                            gt_labels_3d,
                            gt_past_traj,
                            gt_past_traj_mask,
                            gt_inds,
                            gt_sdc_bbox,
                            gt_sdc_label,
                            l2g_t,
                            l2g_r_mat,
                            img_metas,
                            timestamp):
        """训练时的多帧跟踪前向传播

        处理一个完整视频片段（num_frame 帧）的跟踪训练。
        逐帧调用 _forward_single_frame_train，维护跨帧的 track_instances 状态。

        Args:
            img: 视频片段图像，形状 (B, num_frame, N_cam, C, H, W)
            gt_bboxes_3d: 每帧的 GT 3D 框 (B, num_frame, ...)
            gt_labels_3d: 每帧的 GT 标签 (B, num_frame, ...)
            gt_past_traj: 每帧目标的过去轨迹 (B, num_frame, num_objs, past_steps, 2)
            gt_past_traj_mask: 过去轨迹的有效性掩码 (B, num_frame, num_objs, past_steps, 2)
            gt_inds: 每帧目标的全局跟踪 ID (B, num_frame, num_objs)
            gt_sdc_bbox: 每帧自车的 3D 框 (B, num_frame, ...)
            gt_sdc_label: 每帧自车的标签 (B, num_frame, ...)
            l2g_t: lidar→global 平移向量 (B, num_frame, 3)
            l2g_r_mat: lidar→global 旋转矩阵 (B, num_frame, 3, 3)
            img_metas: 每帧的图像元信息
            timestamp: 每帧的时间戳 (B, num_frame)

        Returns:
            losses: 跟踪损失字典（分类损失、框回归损失、轨迹损失、跟踪损失）
            out: 跟踪输出字典（BEV 特征、活跃目标的检测结果等）
        """
        # 初始化空的跟踪实例
        track_instances = self._generate_empty_tracks()
        num_frame = img.size(1)

        # ---- 构建 GT 实例列表 ----
        # 将每帧的 GT 数据组织成 Instances 对象，供后续匈牙利匹配使用
        gt_instances_list = []

        for i in range(num_frame):
            gt_instances = Instances((1, 1))
            boxes = gt_bboxes_3d[0][i].tensor.to(img.device)

            # 归一化 GT 框坐标到 [0, 1] 范围
            boxes = normalize_bbox(boxes, self.pc_range)

            sd_boxes = gt_sdc_bbox[0][i].tensor.to(img.device)
            sd_boxes = normalize_bbox(sd_boxes, self.pc_range)

            gt_instances.boxes = boxes
            gt_instances.labels = gt_labels_3d[0][i]
            gt_instances.obj_ids = gt_inds[0][i]                   # 全局跟踪 ID
            gt_instances.past_traj = gt_past_traj[0][i].float()    # 过去轨迹
            gt_instances.past_traj_mask = gt_past_traj_mask[0][i].float()  # 轨迹掩码

            # 自车框和标签：复制 N 份以匹配当前帧的目标数量
            # N = boxes.shape[0]，当帧内无目标时为 0
            gt_instances.sdc_boxes = torch.cat([sd_boxes for _ in range(boxes.shape[0])], dim=0)
            gt_instances.sdc_labels = torch.cat([gt_sdc_label[0][i] for _ in range(gt_labels_3d[0][i].shape[0])], dim=0)
            gt_instances_list.append(gt_instances)

        # 初始化跟踪损失模块，传入 GT 实例列表
        self.criterion.initialize_for_single_clip(gt_instances_list)

        out = dict()

        # ---- 逐帧处理 ----
        for i in range(num_frame):
            # 历史帧图像：当前帧之前的所有帧
            prev_img = img[:, :i, ...] if i != 0 else img[:, :1, ...]
            prev_img_metas = copy.deepcopy(img_metas)

            # 当前帧图像
            img_single = torch.stack([img_[i] for img_ in img], dim=0)
            img_metas_single = [copy.deepcopy(img_metas[0][i])]

            # 获取 lidar→global 变换矩阵
            # 最后一帧没有下一帧，所以 l2g_r2/l2g_t2 为 None
            if i == num_frame - 1:
                l2g_r2 = None
                l2g_t2 = None
                time_delta = None
            else:
                l2g_r2 = l2g_r_mat[0][i + 1]
                l2g_t2 = l2g_t[0][i + 1]
                time_delta = timestamp[0][i + 1] - timestamp[0][i]

            # 用于收集各层 decoder 的结果
            all_query_embeddings = []
            all_matched_idxes = []
            all_instances_pred_logits = []
            all_instances_pred_boxes = []

            # 单帧训练前向传播
            frame_res = self._forward_single_frame_train(
                img_single,
                img_metas_single,
                track_instances,
                prev_img,
                prev_img_metas,
                l2g_r_mat[0][i],
                l2g_t[0][i],
                l2g_r2,
                l2g_t2,
                time_delta,
                all_query_embeddings,
                all_matched_idxes,
                all_instances_pred_logits,
                all_instances_pred_boxes,
            )

            # 更新跟踪实例，传入下一帧使用
            # 这样实现了跨帧的时序信息传递
            track_instances = frame_res["track_instances"]

        # 收集输出：BEV 特征、跟踪结果、自车信息等
        get_keys = ["bev_embed", "bev_pos",
                    "track_query_embeddings", "track_query_matched_idxes", "track_bbox_results",
                    "sdc_boxes_3d", "sdc_scores_3d", "sdc_track_scores", "sdc_track_bbox_results", "sdc_embedding"]
        out.update({k: frame_res[k] for k in get_keys})

        # 获取累积的损失
        losses = self.criterion.losses_dict
        return losses, out

    def upsample_bev_if_tiny(self, outs_track):
        """对小型模型的 BEV 特征进行上采样

        当 BEV 分辨率为 100×100 时，上采样到 200×200，
        以匹配下游模块（运动预测、规划）期望的 BEV 尺寸。

        Args:
            outs_track: 跟踪输出字典

        Returns:
            outs_track: 上采样后的输出字典
        """
        if outs_track["bev_embed"].size(0) == 100 * 100:
            # 上采样 BEV 特征: (10000, 1, 256) → (40000, 1, 256)
            bev_embed = outs_track["bev_embed"]
            dim, _, _ = bev_embed.size()
            w = h = int(math.sqrt(dim))
            assert h == w == 100

            bev_embed = rearrange(bev_embed, '(h w) b c -> b c h w', h=h, w=w)  # [1, 256, 100, 100]
            bev_embed = nn.Upsample(scale_factor=2)(bev_embed)                   # [1, 256, 200, 200]
            bev_embed = rearrange(bev_embed, 'b c h w -> (h w) b c')             # [40000, 1, 256]
            outs_track["bev_embed"] = bev_embed

            # 上采样历史 BEV 特征
            prev_bev = outs_track.get("prev_bev", None)
            if prev_bev is not None:
                if self.training:
                    # 训练时形状为 [1, 10000, 256]
                    prev_bev = rearrange(prev_bev, 'b (h w) c -> b c h w', h=h, w=w)
                    prev_bev = nn.Upsample(scale_factor=2)(prev_bev)
                    prev_bev = rearrange(prev_bev, 'b c h w -> b (h w) c')
                    outs_track["prev_bev"] = prev_bev
                else:
                    # 推理时形状为 [10000, 1, 256]
                    prev_bev = rearrange(prev_bev, '(h w) b c -> b c h w', h=h, w=w)
                    prev_bev = nn.Upsample(scale_factor=2)(prev_bev)
                    prev_bev = rearrange(prev_bev, 'b c h w -> (h w) b c')
                    outs_track["prev_bev"] = prev_bev

            # 上采样 BEV 位置编码
            bev_pos  = outs_track["bev_pos"]  # [1, 256, 100, 100]
            bev_pos = nn.Upsample(scale_factor=2)(bev_pos)  # [1, 256, 200, 200]
            outs_track["bev_pos"] = bev_pos
        return outs_track


    def _forward_single_frame_inference(
        self,
        img,
        img_metas,
        track_instances,
        prev_bev=None,
        l2g_r1=None,
        l2g_t1=None,
        l2g_r2=None,
        l2g_t2=None,
        time_delta=None,
    ):
        """推理时的单帧跟踪前向传播

        与训练版本不同，推理时：
        1. 不需要匈牙利匹配
        2. 使用 RuntimeTrackerBase 管理跟踪 ID
        3. 利用 Memory Bank 和 Query Interaction 实现时序关联

        Args:
            img: 当前帧图像，形状 (B, num_cam, C, H, W)
            img_metas: 图像元信息
            track_instances: 上一帧的跟踪实例
            prev_bev: 上一帧的 BEV 特征
            l2g_r1/l2g_t1: 上一帧 lidar→global 变换
            l2g_r2/l2g_t2: 当前帧 lidar→global 变换
            time_delta: 两帧间时间差

        Returns:
            out: 包含检测结果、跟踪状态、BEV 特征的字典
        """
        # ---- Step 1: 速度更新 ----
        # 将活跃目标的参考点从上一帧坐标系变换到当前帧坐标系
        active_inst = track_instances[track_instances.obj_idxes >= 0]
        other_inst = track_instances[track_instances.obj_idxes < 0]

        if l2g_r2 is not None and len(active_inst) > 0 and l2g_r1 is not None:
            ref_pts = active_inst.ref_pts
            velo = active_inst.pred_boxes[:, -2:]
            ref_pts = self.velo_update(
                ref_pts, velo, l2g_r1, l2g_t1, l2g_r2, l2g_t2, time_delta=time_delta
            )
            ref_pts = ref_pts.squeeze(0)
            dim = active_inst.query.shape[-1]
            active_inst.ref_pts = self.reference_points(active_inst.query[..., :dim//2])
            active_inst.ref_pts[...,:2] = ref_pts[...,:2]

        # 合并活跃和非活跃实例
        track_instances = Instances.cat([other_inst, active_inst])

        # ---- Step 2: 生成 BEV 特征并进行目标检测 ----
        bev_embed, bev_pos = self.get_bevs(img, img_metas, prev_bev=prev_bev)
        det_output = self.pts_bbox_head.get_detections(
            bev_embed,
            object_query_embeds=track_instances.query,
            ref_points=track_instances.ref_pts,
            img_metas=img_metas,
        )
        output_classes = det_output["all_cls_scores"]
        output_coords = det_output["all_bbox_preds"]
        last_ref_pts = det_output["last_ref_points"]
        query_feats = det_output["query_feats"]

        out = {
            "pred_logits": output_classes,
            "pred_boxes": output_coords,
            "ref_pts": last_ref_pts,
            "bev_embed": bev_embed,
            "query_embeddings": query_feats,
            "all_past_traj_preds": det_output["all_past_traj_preds"],
            "bev_pos": bev_pos,
        }

        # ---- Step 3: 更新跟踪实例 ----
        # 用当前帧的检测结果更新跟踪实例的状态
        track_scores = output_classes[-1, 0, :].sigmoid().max(dim=-1).values

        track_instances.scores = track_scores
        track_instances.pred_logits = output_classes[-1, 0]       # [300, num_cls]
        track_instances.pred_boxes = output_coords[-1, 0]         # [300, box_dim]
        track_instances.output_embedding = query_feats[-1][0]     # [300, feat_dim]
        track_instances.ref_pts = last_ref_pts[0]

        # 自车 query（索引 900）标记为特殊对象
        track_instances.obj_idxes[900] = -2

        # ---- Step 4: 更新跟踪管理器 ----
        # RuntimeTrackerBase 负责分配/回收跟踪 ID，管理目标生命周期
        self.track_base.update(track_instances, None)

        # ---- Step 5: 筛选活跃目标 ----
        # 条件：obj_idxes >= 0（已分配 ID）且分数 >= 过滤阈值
        active_index = (track_instances.obj_idxes>=0) & (track_instances.scores >= self.track_base.filter_score_thresh)
        out.update(self.select_active_track_query(track_instances, active_index, img_metas))

        # 提取自车查询结果
        out.update(self.select_sdc_track_query(track_instances[track_instances.obj_idxes==-2], img_metas))

        # ---- Step 6: Memory Bank 更新 ----
        # 将当前帧的目标特征存入 Memory Bank
        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)

        # ---- Step 7: Query Interaction ----
        # 融合历史 query 与空 query，生成下一帧的初始 query
        tmp = {}
        tmp["init_track_instances"] = self._generate_empty_tracks()
        tmp["track_instances"] = track_instances
        out_track_instances = self.query_interact(tmp)
        out["track_instances_fordet"] = track_instances    # 用于检测结果输出
        out["track_instances"] = out_track_instances       # 传入下一帧使用
        out["track_obj_idxes"] = track_instances.obj_idxes
        return out

    def simple_test_track(
        self,
        img=None,
        l2g_t=None,
        l2g_r_mat=None,
        img_metas=None,
        timestamp=None,
    ):
        """推理时的跟踪入口

        仅支持 batch_size=1 的时序推理。维护场景级别的跟踪状态，
        当场景切换时自动重置跟踪实例。

        Args:
            img: 当前帧图像
            l2g_t: lidar→global 平移向量
            l2g_r_mat: lidar→global 旋转矩阵
            img_metas: 图像元信息
            timestamp: 当前帧时间戳

        Returns:
            results: 包含检测和跟踪结果的列表
        """
        bs = img.size(0)

        # ---- 场景切换检测 ----
        # 以下情况需要初始化/重置跟踪实例：
        # 1. 首次推理 (test_track_instances is None)
        # 2. 场景切换 (scene_token 改变)
        if (
            self.test_track_instances is None
            or img_metas[0]["scene_token"] != self.scene_token
        ):
            self.timestamp = timestamp
            self.scene_token = img_metas[0]["scene_token"]
            self.prev_bev = None
            track_instances = self._generate_empty_tracks()
            # 新场景开始，没有上一帧的变换信息
            time_delta, l2g_r1, l2g_t1, l2g_r2, l2g_t2 = None, None, None, None, None

        else:
            # 续接上一帧的跟踪状态
            track_instances = self.test_track_instances
            time_delta = timestamp - self.timestamp
            l2g_r1 = self.l2g_r_mat
            l2g_t1 = self.l2g_t
            l2g_r2 = l2g_r_mat
            l2g_t2 = l2g_t

        # ---- 保存当前帧信息，供下一帧使用 ----
        self.timestamp = timestamp
        self.l2g_t = l2g_t
        self.l2g_r_mat = l2g_r_mat

        # ---- 单帧推理 ----
        prev_bev = self.prev_bev
        frame_res = self._forward_single_frame_inference(
            img,
            img_metas,
            track_instances,
            prev_bev,
            l2g_r1,
            l2g_t1,
            l2g_r2,
            l2g_t2,
            time_delta,
        )

        # ---- 更新全局状态 ----
        self.prev_bev = frame_res["bev_embed"]           # 保存当前帧 BEV 供下一帧使用
        track_instances = frame_res["track_instances"]    # 下一帧的跟踪实例
        track_instances_fordet = frame_res["track_instances_fordet"]  # 本帧的检测结果

        self.test_track_instances = track_instances

        # ---- 构建输出 ----
        results = [dict()]
        get_keys = ["bev_embed", "bev_pos",
                    "track_query_embeddings", "track_bbox_results",
                    "boxes_3d", "scores_3d", "labels_3d", "track_scores", "track_ids"]
        if self.with_motion_head:
            get_keys += ["sdc_boxes_3d", "sdc_scores_3d", "sdc_track_scores", "sdc_track_bbox_results", "sdc_embedding"]
        results[0].update({k: frame_res[k] for k in get_keys})

        # 将检测实例转换为最终输出格式（3D 框、分数、标签、跟踪 ID）
        results = self._det_instances2results(track_instances_fordet, results, img_metas)
        return results

    def _track_instances2results(self, track_instances, img_metas, with_mask=True):
        """将跟踪实例转换为检测结果

        通过 bbox_coder 将归一化的预测框解码为实际 3D 坐标，
        并提取分数、标签、跟踪 ID 等信息。

        Args:
            track_instances: 跟踪实例
            img_metas: 图像元信息
            with_mask: 是否返回有效性掩码

        Returns:
            result_dict: 包含以下字段的字典：
                - boxes_3d: 3D 检测框 (LiDARInstance3DBoxes)
                - scores_3d: 检测分数
                - labels_3d: 类别标签
                - track_scores: 跟踪分数
                - track_ids: 跟踪 ID
                - bbox_index: 框索引
                - mask: 有效性掩码
                - track_bbox_results: 跟踪检测结果列表
        """
        bbox_dict = dict(
            cls_scores=track_instances.pred_logits,    # 分类分数
            bbox_preds=track_instances.pred_boxes,     # 归一化预测框
            track_scores=track_instances.scores,        # 跟踪分数
            obj_idxes=track_instances.obj_idxes,        # 跟踪 ID
        )

        # bbox_coder.decode: 将归一化预测框解码为实际 3D 坐标
        bboxes_dict = self.bbox_coder.decode(bbox_dict, with_mask=with_mask, img_metas=img_metas)[0]

        bboxes = bboxes_dict["bboxes"]
        bboxes = img_metas[0]["box_type_3d"](bboxes, 9)  # 构建 LiDARInstance3DBoxes
        labels = bboxes_dict["labels"]
        scores = bboxes_dict["scores"]
        bbox_index = bboxes_dict["bbox_index"]

        track_scores = bboxes_dict["track_scores"]
        obj_idxes = bboxes_dict["obj_idxes"]

        result_dict = dict(
            boxes_3d=bboxes.to("cpu"),
            scores_3d=scores.cpu(),
            labels_3d=labels.cpu(),
            track_scores=track_scores.cpu(),
            bbox_index=bbox_index.cpu(),
            track_ids=obj_idxes.cpu(),
            mask=bboxes_dict["mask"].cpu(),
            track_bbox_results=[[bboxes.to("cpu"), scores.cpu(), labels.cpu(), bbox_index.cpu(), bboxes_dict["mask"].cpu()]]
        )
        return result_dict

    def _det_instances2results(self, instances, results, img_metas):
        """将检测实例转换为最终输出格式

        用于推理时，将跟踪实例的解码结果输出为评估所需的格式。

        Args:
            instances: 检测实例，包含以下关键字段：
                - pred_logits: 分类分数
                - pred_boxes: 归一化预测框
                - scores: 检测分数
                - obj_idxes: 跟踪 ID
            results: 已有的结果列表
            img_metas: 图像元信息

        Returns:
            results: 更新后的结果列表，包含：
                - boxes_3d_det: 3D 检测框
                - scores_3d_det: 检测分数
                - labels_3d_det: 类别标签
                - track_ids: 跟踪 ID
                - tracking_score: 跟踪分数
        """
        # 如果没有检测到任何目标，返回 None
        if instances.pred_logits.numel() == 0:
            return [None]

        bbox_dict = dict(
            cls_scores=instances.pred_logits,
            bbox_preds=instances.pred_boxes,
            track_scores=instances.scores,
            obj_idxes=instances.obj_idxes,
        )

        bboxes_dict = self.bbox_coder.decode(bbox_dict, img_metas=img_metas)[0]
        bboxes = bboxes_dict["bboxes"]
        bboxes = img_metas[0]["box_type_3d"](bboxes, 9)
        labels = bboxes_dict["labels"]
        scores = bboxes_dict["scores"]

        track_scores = bboxes_dict["track_scores"]
        obj_idxes = bboxes_dict["obj_idxes"]

        result_dict = results[0]
        result_dict_det = dict(
            boxes_3d_det=bboxes.to("cpu"),
            scores_3d_det=scores.cpu(),
            labels_3d_det=labels.cpu(),
        )
        if result_dict is not None:
            result_dict.update(result_dict_det)
        else:
            result_dict = None

        return [result_dict]

