# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

"""
BEVFormer 检测器 (Bird's Eye View Transformer)
==============================================
BEVFormer 是一个基于 Transformer 的鸟瞰图 (BEV) 3D 目标检测器。

核心思想:
    将多视角 2D 图像特征通过时空注意力机制转换为统一的 BEV 特征，
    然后在 BEV 空间中进行 3D 目标检测。

工作流程:
    1. 图像特征提取: Backbone + Neck 提取多尺度图像特征
    2. 时序 BEV 生成: 获取历史帧的 BEV 特征（用于时序融合）
    3. BEV 特征编码: Encoder 通过空间交叉注意力和时序自注意力生成 BEV 特征
    4. 目标检测: Decoder 在 BEV 特征上检测 3D 目标
    5. 损失计算: 匈牙利匹配 + 分类损失 + 回归损失

网络架构:
    多视角图像 (N_cam=6, 3, H, W)
        │
        ▼
    [Image Backbone]  ─── ResNet-101 + DCN
        │
        ▼
    [Image Neck]      ─── FPN 多尺度融合
        │
        ▼
    [BEVFormerHead]   ─── Encoder + Decoder
        │
        ├── Encoder:  图像特征 → BEV 特征 (空间交叉注意力 + 时序自注意力)
        └── Decoder:  BEV 特征 → 3D 检测框 (可变形注意力 + 逐层 refine)
        │
        ▼
    3D 检测结果: (cx, cy, w, l, cz, h, θ, vx, vy)

参考论文:
    BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images
    via Spatiotemporal Transformers (ECCV 2022)
"""

import torch
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
import time
import copy
import numpy as np
import mmdet3d
from projects.mmdet3d_plugin.models.utils.bricks import run_time

@DETECTORS.register_module()
class BEVFormer(MVXTwoStageDetector):
    """BEVFormer: 基于时空 Transformer 的 BEV 3D 目标检测器

    继承自 MVXTwoStageDetector，负责:
    - 图像特征提取 (Backbone + Neck)
    - 时序 BEV 特征管理
    - 训练/推理流程编排
    - 将核心计算委托给 BEVFormerHead

    Args:
        video_test_mode: 是否在推理时使用时序信息。
            True  → 启用时序，利用上一帧 BEV 特征
            False → 不使用时序，每帧独立推理
    """

    def __init__(self,
                 use_grid_mask=False,          # 是否使用 GridMask 数据增强
                 pts_voxel_layer=None,         # 点云体素化层（BEVFormer 不使用点云）
                 pts_voxel_encoder=None,       # 点云体素编码器（不使用）
                 pts_middle_encoder=None,       # 点云中间编码器（不使用）
                 pts_fusion_layer=None,        # 点云融合层（不使用）
                 img_backbone=None,            # 图像 backbone (如 ResNet-101)
                 pts_backbone=None,            # 点云 backbone（不使用）
                 img_neck=None,                # 图像 neck (如 FPN)
                 pts_neck=None,                # 点云 neck（不使用）
                 pts_bbox_head=None,           # BEV 检测头 (BEVFormerHead)
                 img_roi_head=None,            # 图像 ROI head（不使用）
                 img_rpn_head=None,            # 图像 RPN head（不使用）
                 train_cfg=None,               # 训练配置
                 test_cfg=None,                # 测试配置
                 pretrained=None,              # 预训练权重路径
                 video_test_mode=False         # 是否启用视频时序推理模式
                 ):

        # 初始化父类 MVXTwoStageDetector
        # 注意: pts_voxel_layer, pts_backbone, pts_neck 等点云相关参数在 BEVFormer 中不使用
        # 但为了兼容 mmdet3d 框架接口，仍需保留这些参数
        super(BEVFormer,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)

        # GridMask: 在图像上随机遮挡网格区域，增强模型鲁棒性
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False

        # ---- 时序推理相关 ----
        # video_test_mode: 是否在推理时使用时序信息
        self.video_test_mode = video_test_mode

        # prev_frame_info: 缓存前一帧的 BEV 特征和自车位姿
        # 用于推理时的时序一致性
        self.prev_frame_info = {
            'prev_bev': None,          # 前一帧的 BEV 特征 (bev_h*bev_w, bs, C)
            'scene_token': None,       # 当前场景的 token，用于检测场景切换
            'prev_pos': 0,             # 前一帧自车位置 (x, y, z)
            'prev_angle': 0,           # 前一帧自车朝向角 (yaw)
        }


    def extract_img_feat(self, img, img_metas, len_queue=None):
        """提取多视角图像特征

        将多视角图像通过 Backbone + Neck 提取多尺度特征。

        Args:
            img: 输入图像
                - 单帧: (B, N, C, H, W)  B=batch, N=相机数(6)
                - 多帧: (B*len_queue, N, C, H, W)  多帧合并后的 batch
            img_metas: 图像元信息
            len_queue: 时序队列长度（多帧合并时使用）

        Returns:
            img_feats_reshaped: 多尺度图像特征列表
                每个尺度形状为 (B/len_queue, len_queue, N, C, H, W)
                或 (B, N, C, H, W)
        """
        B = img.size(0)
        if img is not None:

            # 处理 shape 为 5D 的情况
            # B=1 时 squeeze 掉 batch 维度（backbone 内部会处理）
            # B>1 时 reshape 为 (B*N, C, H, W)
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)

            # GridMask 数据增强: 随机遮挡网格区域
            if self.use_grid_mask:
                img = self.grid_mask(img)

            # 图像 Backbone: 提取多尺度特征
            # 输入: (B*N, 3, H, W)
            # 输出: 多尺度特征图列表
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None

        # 图像 Neck (FPN): 多尺度特征融合
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        # 将特征 reshape 回多帧结构
        # (B*N, C, H, W) → (B/len_queue, len_queue, N, C, H, W)
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """提取图像特征（统一入口）

        封装 extract_img_feat，供外部调用。
        """
        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        return img_feats


    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          prev_bev=None):
        """训练时的 BEV 检测前向传播

        将图像特征送入 BEVFormerHead，生成检测结果并计算损失。

        Args:
            pts_feats: 图像特征列表（经过 Backbone + Neck）
            gt_bboxes_3d: GT 3D 框
            gt_labels_3d: GT 类别标签
            img_metas: 图像元信息
            gt_bboxes_ignore: 忽略的 GT 框
            prev_bev: 上一帧的 BEV 特征（用于时序融合）

        Returns:
            losses: 检测损失字典
        """
        # Step 1: Head 前向传播
        # 输入: 图像特征 + 上一帧 BEV
        # 输出: BEV 特征 + 分类分数 + 回归预测
        outs = self.pts_bbox_head(
            pts_feats, img_metas, prev_bev)

        # Step 2: 计算损失
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses

    def forward_dummy(self, img):
        """占位前向传播，用于统计模型参数和 FLOPs"""
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """统一前向入口

        Args:
            return_loss: True=训练模式，False=推理模式
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """获取历史 BEV 特征

        迭代处理历史帧图像，通过 BEVFormerHead 的 Encoder 逐帧生成 BEV 特征。
        为了节省 GPU 显存，此过程不计算梯度。

        处理流程:
        Frame 1 → prev_bev=None → Encoder → BEV_1
        Frame 2 → prev_bev=BEV_1 → Encoder → BEV_2
        ...
        Frame N → prev_bev=BEV_{N-1} → Encoder → BEV_N

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

            # 将所有帧的图像合并，批量提取特征（提高效率）
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)

            # 逐帧迭代生成 BEV
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]

                # 如果当前帧是场景的第一帧，不存在上一帧 BEV
                if not img_metas[0]['prev_bev_exists']:
                    prev_bev = None

                # 提取第 i 帧的图像特征
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]

                # 调用 Head 的 Encoder 部分（only_bev=True 表示只生成 BEV，不执行 Decoder）
                # 输入: 图像特征 + 上一帧 BEV
                # 输出: 当前帧 BEV 特征
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev, only_bev=True)

            self.train()
            return prev_bev

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      ):
        """训练时的前向传播

        处理流程:
        1. 将多帧图像拆分为历史帧和当前帧
        2. 从历史帧获取时序 BEV 特征
        3. 在当前帧上执行检测
        4. 计算检测损失

        Args:
            img: 多帧图像，形状 (B, len_queue, N, C, H, W)
                 其中 len_queue 是时序窗口长度
            gt_bboxes_3d: GT 3D 框
            gt_labels_3d: GT 标签
            img_metas: 图像元信息
            prev_bev: 上一帧 BEV 特征

        Returns:
            losses: 检测损失字典
        """
        # ---- 拆分历史帧和当前帧 ----
        len_queue = img.size(1)                     # 时序窗口长度
        prev_img = img[:, :-1, ...]                # 历史帧: 前 N-1 帧
        img = img[:, -1, ...]                       # 当前帧: 最后一帧

        # ---- 获取历史 BEV 特征 ----
        # 从历史帧图像中逐帧迭代生成 BEV 特征
        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)

        # ---- 提取当前帧图像特征 ----
        img_metas = [each[len_queue-1] for each in img_metas]

        # 如果当前帧是场景的第一帧，prev_bev 置为 None
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None

        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        # ---- BEV 检测 ----
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore, prev_bev)

        losses.update(losses_pts)
        return losses

    def forward_test(self, img_metas, img=None, **kwargs):
        """推理时的前向传播

        时序推理流程:
        1. 检测场景切换 → 重置 BEV 缓存
        2. 计算自车运动增量 (相对上一帧的位移和旋转)
        3. 利用上一帧 BEV 特征进行时序推理
        4. 保存当前帧 BEV 特征，供下一帧使用

        Args:
            img_metas: 图像元信息（双层列表，外层为 test-time augmentation）
            img: 当前帧图像

        Returns:
            bbox_results: 检测结果列表
        """
        # 校验输入类型
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        # ---- 场景切换检测 ----
        # 如果 scene_token 改变，说明进入了新场景，重置上一帧 BEV
        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            self.prev_frame_info['prev_bev'] = None
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # 如果不启用时序模式，不使用上一帧 BEV
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # ---- 计算自车运动增量 ----
        # can_bus: 自车状态信息 [x, y, z, ..., yaw]
        # 保存当前帧的绝对位姿
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])

        if self.prev_frame_info['prev_bev'] is not None:
            # 有上一帧 BEV → 计算相对位姿差
            # can_bus 变为增量值，供 Encoder 中的时序对齐使用
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            # 无上一帧 BEV → 增量设为 0
            img_metas[0][0]['can_bus'][-1] = 0
            img_metas[0][0]['can_bus'][:3] = 0

        # ---- 执行推理 ----
        new_prev_bev, bbox_results = self.simple_test(
            img_metas[0], img[0], prev_bev=self.prev_frame_info['prev_bev'], **kwargs)

        # ---- 保存当前帧状态，供下一帧使用 ----
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev

        return bbox_results

    def simple_test_pts(self, x, img_metas, prev_bev=None, rescale=False):
        """推理时的 BEV 检测（点云/图像分支）

        调用 Head 进行前向传播，然后解码检测结果。

        Args:
            x: 图像特征列表
            img_metas: 图像元信息
            prev_bev: 上一帧 BEV 特征
            rescale: 是否将结果缩放回原始坐标系

        Returns:
            bev_embed: 当前帧 BEV 特征（用于下一帧的时序融合）
            bbox_results: 检测结果列表
        """
        # Head 前向传播
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev)

        # 解码检测结果: 归一化预测 → 实际 3D 坐标
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)

        # 格式化为最终输出: 每个检测 → (bboxes, scores, labels)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return outs['bev_embed'], bbox_results

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False):
        """推理时的简化测试（无 test-time augmentation）

        流程:
        1. 提取图像特征
        2. Head 前向传播 + 解码
        3. 返回 BEV 特征（供下一帧使用）和检测结果

        Args:
            img_metas: 图像元信息
            img: 当前帧图像
            prev_bev: 上一帧 BEV 特征
            rescale: 是否缩放回原始坐标系

        Returns:
            new_prev_bev: 当前帧 BEV 特征
            bbox_list: 检测结果列表
        """
        # 提取图像特征
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]

        # BEV 检测
        new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats, img_metas, prev_bev, rescale=rescale)

        # 合并结果
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox

        return new_prev_bev, bbox_list
