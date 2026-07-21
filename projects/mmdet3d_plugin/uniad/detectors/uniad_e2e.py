#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

"""
UniAD 端到端自动驾驶模型 (End-to-End Autonomous Driving)
========================================================
本模块实现了 UniAD 的完整端到端流程，继承自 UniADTrack，统一了以下六大任务：

1. 检测与跟踪 (Track)  → 继承自 UniADTrack，检测 3D 目标并维护跟踪 ID
2. 在线建图 (Map)       → SegHead: 预测车道线、人行横道等地图元素
3. 运动预测 (Motion)    → MotionHead: 预测每个目标的未来轨迹
4. 占据预测 (Occ)       → OccHead: 预测 3D 占据栅格，推断遮挡区域
5. 规划 (Planning)      → PlanningHead: 基于感知结果生成自车行驶轨迹
6. 所有任务共享 BEV 特征 → 由 Track 模块生成，各任务 Head 复用

数据流:
    多视角图像 → BEV 特征 → 各任务 Head → 各任务输出
                         → Track (检测+跟踪)
                         → Map (建图)
                         → Motion (运动预测)
                         → Occ (占据预测)
                         → Planning (规划)
"""

import torch
from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS
import copy
import os
from ..dense_heads.seg_head_plugin import IOU
from .uniad_track import UniADTrack
from mmdet.models.builder import build_head
from ..dense_heads.seg_head_plugin.seg_detr_head import SegDETRHead

@DETECTORS.register_module()
class UniAD(UniADTrack):
    """UniAD 端到端自动驾驶模型

    继承自 UniADTrack，在跟踪模块的基础上增加了：
    - SegHead:     在线建图（车道线、人行横道等）
    - MotionHead:  运动预测（目标未来轨迹预测）
    - OccHead:     占据预测（3D 占据栅格预测）
    - PlanningHead: 轨迹规划（自车行驶轨迹生成）

    所有任务共享同一个 BEV 特征图，实现"感知-预测-规划"的统一。
    """
    def __init__(
        self,
        seg_head=None,              # 建图头 (SegHead) 配置
        motion_head=None,           # 运动预测头 (MotionHead) 配置
        occ_head=None,              # 占据预测头 (OccHead) 配置
        planning_head=None,         # 规划头 (PlanningHead) 配置
        task_loss_weight=dict(      # 各任务损失权重
            track=1.0,              # 跟踪损失权重
            map=1.0,                # 建图损失权重
            motion=1.0,             # 运动预测损失权重
            occ=1.0,                # 占据预测损失权重
            planning=1.0            # 规划损失权重
        ),
        **kwargs,
    ):
        # 初始化父类 UniADTrack（包含 BEV 特征生成、检测、跟踪）
        super(UniAD, self).__init__(**kwargs)

        # 按需构建各任务 Head（通过配置文件控制是否启用）
        if seg_head:
            self.seg_head = build_head(seg_head)
        if occ_head:
            self.occ_head = build_head(occ_head)
        if motion_head:
            self.motion_head = build_head(motion_head)
        if planning_head:
            self.planning_head = build_head(planning_head)

        self.task_loss_weight = task_loss_weight
        assert set(task_loss_weight.keys()) == \
               {'track', 'occ', 'motion', 'map', 'planning'}

        # 类型注解：声明各 Head 类型为 SegDETRHead
        self.seg_head: SegDETRHead = build_head(seg_head)
        self.occ_head: SegDETRHead = build_head(occ_head)
        self.motion_head: SegDETRHead = build_head(motion_head)
        self.planning_head: SegDETRHead = build_head(planning_head)

    @property
    def with_planning_head(self):
        """是否启用规划头"""
        return hasattr(self, 'planning_head') and self.planning_head is not None

    @property
    def with_occ_head(self):
        """是否启用占据预测头"""
        return hasattr(self, 'occ_head') and self.occ_head is not None

    @property
    def with_motion_head(self):
        """是否启用运动预测头"""
        return hasattr(self, 'motion_head') and self.motion_head is not None

    @property
    def with_seg_head(self):
        """是否启用建图头"""
        return hasattr(self, 'seg_head') and self.seg_head is not None

    def forward_dummy(self, img):
        """占位前向传播，用于模型参数统计"""
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """统一前向入口

        根据 return_loss 标志分发到训练或推理流程。
        - return_loss=True  → forward_train（训练模式）
        - return_loss=False → forward_test（推理模式）

        Args:
            return_loss: 是否返回损失（True=训练，False=推理）
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      img=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_inds=None,
                      l2g_t=None,
                      l2g_r_mat=None,
                      timestamp=None,
                      # ---- 建图 (Map) 真值 ----
                      gt_lane_labels=None,       # 车道线标签
                      gt_lane_bboxes=None,       # 车道线框
                      gt_lane_masks=None,        # 车道线掩码
                      # ---- 运动预测 (Motion) 真值 ----
                      gt_fut_traj=None,          # 目标未来轨迹 (num_objs, fut_steps, 2)
                      gt_fut_traj_mask=None,     # 未来轨迹有效性掩码
                      gt_past_traj=None,         # 目标过去轨迹 (num_objs, past_steps+2, 2)
                      gt_past_traj_mask=None,    # 过去轨迹有效性掩码
                      gt_sdc_bbox=None,          # 自车 3D 框
                      gt_sdc_label=None,         # 自车标签
                      gt_sdc_fut_traj=None,      # 自车未来轨迹
                      gt_sdc_fut_traj_mask=None, # 自车未来轨迹掩码
                      # ---- 占据预测 (Occ) 真值 ----
                      gt_segmentation=None,      # 分割真值
                      gt_instance=None,          # 实例分割真值
                      gt_occ_img_is_valid=None,  # 图像是否有效（用于占据预测）
                      # ---- 规划 (Planning) 真值 ----
                      sdc_planning=None,         # 自车规划轨迹真值
                      sdc_planning_mask=None,    # 自车规划轨迹掩码
                      command=None,              # 高层指令 (左转/右转/直行)
                      # ---- 规划用的未来真值 ----
                      gt_future_boxes=None,      # 未来帧的 GT 框
                      **kwargs,
                      ):
        """训练时的多任务联合前向传播

        处理流程（按顺序）:
        1. Track:  检测 + 跟踪 → 输出 BEV 特征和跟踪结果
        2. Map:    在 BEV 特征上进行在线建图
        3. Motion: 基于跟踪结果和建图结果预测目标未来轨迹
        4. Occ:    基于运动预测结果预测 3D 占据栅格
        5. Planning: 综合所有感知结果生成自车规划轨迹

        各任务损失通过 task_loss_weight 加权后汇总。

        Returns:
            losses: 所有任务的加权损失字典
        """
        losses = dict()
        len_queue = img.size(1)  # 训练视频片段的帧数

        # ============================================================
        # Step 1: 检测与跟踪 (Track)
        # ============================================================
        # 调用父类 UniADTrack 的 forward_track_train，处理多帧跟踪训练
        # 输入: 多帧图像 + GT 3D 框 + GT 轨迹 + GT 跟踪 ID
        # 输出: BEV 特征 (bev_embed) + 跟踪结果 (track_query_embeddings 等)
        losses_track, outs_track = self.forward_track_train(
            img, gt_bboxes_3d, gt_labels_3d,
            gt_past_traj, gt_past_traj_mask, gt_inds,
            gt_sdc_bbox, gt_sdc_label,
            l2g_t, l2g_r_mat, img_metas, timestamp)

        # 给跟踪损失加上 'track.' 前缀并加权
        losses_track = self.loss_weighted_and_prefixed(losses_track, prefix='track')
        losses.update(losses_track)

        # 如果使用小型模型 (BEV 100×100)，上采样到 200×200
        # 确保 BEV 尺寸与下游任务一致
        outs_track = self.upsample_bev_if_tiny(outs_track)

        # 提取 BEV 特征：所有下游任务共享这个特征
        bev_embed = outs_track["bev_embed"]  # (bev_h*bev_w, B, C)
        bev_pos  = outs_track["bev_pos"]     # (B, C, bev_h, bev_w)

        # 取最后一帧的 img_metas（下游任务只需要最后一帧的信息）
        img_metas = [each[len_queue-1] for each in img_metas]

        # ============================================================
        # Step 2: 在线建图 (Map)
        # ============================================================
        outs_seg = dict()
        if self.with_seg_head:
            # SegHead 在 BEV 特征上预测车道线、人行横道等地图元素
            losses_seg, outs_seg = self.seg_head.forward_train(
                bev_embed, img_metas,
                gt_lane_labels, gt_lane_bboxes, gt_lane_masks)

            losses_seg = self.loss_weighted_and_prefixed(losses_seg, prefix='map')
            losses.update(losses_seg)

        # ============================================================
        # Step 3: 运动预测 (Motion)
        # ============================================================
        outs_motion = dict()
        if self.with_motion_head:
            # MotionHead 输入:
            #   - BEV 特征
            #   - 跟踪结果 (outs_track): 目标的 3D 框、embedding 等
            #   - 建图结果 (outs_seg): 车道线等地图信息
            # 输出: 每个目标的未来轨迹预测
            ret_dict_motion = self.motion_head.forward_train(
                bev_embed,
                gt_bboxes_3d, gt_labels_3d,
                gt_fut_traj, gt_fut_traj_mask,
                gt_sdc_fut_traj, gt_sdc_fut_traj_mask,
                outs_track=outs_track, outs_seg=outs_seg)

            losses_motion = ret_dict_motion["losses"]
            outs_motion = ret_dict_motion["outs_motion"]
            outs_motion['bev_pos'] = bev_pos
            losses_motion = self.loss_weighted_and_prefixed(losses_motion, prefix='motion')
            losses.update(losses_motion)

        # ============================================================
        # Step 4: 占据预测 (Occ)
        # ============================================================
        if self.with_occ_head:
            # 如果运动预测没有输出任何目标 query（场景中无目标），创建占位张量
            # 防止 OccHead 因空输入而报错
            if outs_motion['track_query'].shape[1] == 0:
                outs_motion['track_query'] = torch.zeros((1, 1, 256)).to(bev_embed)
                outs_motion['track_query_pos'] = torch.zeros((1,1, 256)).to(bev_embed)
                outs_motion['traj_query'] = torch.zeros((3, 1, 1, 6, 256)).to(bev_embed)
                outs_motion['all_matched_idxes'] = [[-1]]

            # OccHead 输入:
            #   - BEV 特征
            #   - 运动预测结果 (outs_motion): 目标轨迹和 query
            # 输出: 3D 占据栅格 + 实例分割
            losses_occ = self.occ_head.forward_train(
                bev_embed,
                outs_motion,
                gt_inds_list=gt_inds,
                gt_segmentation=gt_segmentation,
                gt_instance=gt_instance,
                gt_img_is_valid=gt_occ_img_is_valid,
            )
            losses_occ = self.loss_weighted_and_prefixed(losses_occ, prefix='occ')
            losses.update(losses_occ)

        # ============================================================
        # Step 5: 规划 (Planning)
        # ============================================================
        if self.with_planning_head:
            # PlanningHead 输入:
            #   - BEV 特征
            #   - 运动预测结果 (outs_motion)
            #   - 高层指令 (command): 左转/右转/直行
            # 输出: 自车未来行驶轨迹
            outs_planning = self.planning_head.forward_train(
                bev_embed, outs_motion,
                sdc_planning, sdc_planning_mask, command, gt_future_boxes)

            losses_planning = outs_planning['losses']
            losses_planning = self.loss_weighted_and_prefixed(losses_planning, prefix='planning')
            losses.update(losses_planning)

        # 将损失中的 NaN 替换为 0，防止反向传播出错
        for k,v in losses.items():
            losses[k] = torch.nan_to_num(v)
        return losses

    def loss_weighted_and_prefixed(self, loss_dict, prefix=''):
        """为损失字典添加任务前缀并加权

        处理流程:
        1. 给每个损失项加上任务前缀 (如 'track.loss_cls')
        2. 乘以对应任务的权重 (如 task_loss_weight['track'])

        Args:
            loss_dict: 原始损失字典，key 为损失名称，value 为损失值
            prefix: 任务前缀，如 'track', 'map', 'motion', 'occ', 'planning'

        Returns:
            加权并加前缀后的损失字典

        Example:
            >>> loss_weighted_and_prefixed({'loss_cls': 0.5}, prefix='track')
            {'track.loss_cls': 0.5 * 1.0}
        """
        loss_factor = self.task_loss_weight[prefix]
        loss_dict = {f"{prefix}.{k}" : v*loss_factor for k, v in loss_dict.items()}
        return loss_dict

    def forward_test(self,
                     img=None,
                     img_metas=None,
                     l2g_t=None,
                     l2g_r_mat=None,
                     timestamp=None,
                     gt_lane_labels=None,
                     gt_lane_masks=None,
                     rescale=False,
                     # ---- 规划真值（仅用于评估） ----
                     sdc_planning=None,
                     sdc_planning_mask=None,
                     command=None,
                     # ---- 占据真值（仅用于评估） ----
                     gt_segmentation=None,
                     gt_instance=None,
                     gt_occ_img_is_valid=None,
                     **kwargs
                    ):
        """推理时的多任务联合前向传播

        与训练流程的差异:
        1. 不需要计算损失
        2. 各任务按顺序执行，下游任务依赖上游任务输出
        3. 需要处理场景切换（重置时序状态）
        4. 需要计算自车运动的相对位姿差

        时序推理流程:
        第 1 帧: 初始化跟踪状态 + BEV 特征
        第 N 帧: 利用上一帧的 BEV 特征和跟踪状态，进行时序推理

        Returns:
            result: 包含所有任务输出的列表
        """
        # 校验输入类型
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        # ---- 场景切换检测 ----
        # 如果 scene_token 改变，说明进入了新场景，需要重置时序状态
        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            self.prev_frame_info['prev_bev'] = None
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # 如果不启用时序模式，不使用上一帧 BEV
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # ---- 计算自车运动增量 ----
        # can_bus 包含自车位置 (x,y,z) 和朝向角 (yaw)
        # 保存当前帧的绝对位姿
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])

        # 第一帧：位姿增量设为 0
        if self.prev_frame_info['scene_token'] is None:
            img_metas[0][0]['can_bus'][:3] = 0
            img_metas[0][0]['can_bus'][-1] = 0
        # 后续帧：计算相对于上一帧的位姿差
        else:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']

        # 保存当前帧位姿，供下一帧计算增量
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle

        img = img[0]
        img_metas = img_metas[0]
        timestamp = timestamp[0] if timestamp is not None else None

        result = [dict() for i in range(len(img_metas))]

        # ============================================================
        # Step 1: 检测与跟踪 (Track) - 推理模式
        # ============================================================
        # 调用父类的 simple_test_track，进行时序跟踪推理
        # 输出: BEV 特征 + 跟踪结果 (3D 框、分数、跟踪 ID)
        result_track = self.simple_test_track(img, l2g_t, l2g_r_mat, img_metas, timestamp)

        # 上采样 BEV（小型模型需要）
        result_track[0] = self.upsample_bev_if_tiny(result_track[0])

        bev_embed = result_track[0]["bev_embed"]

        # ============================================================
        # Step 2: 在线建图 (Map) - 推理模式
        # ============================================================
        if self.with_seg_head:
            result_seg = self.seg_head.forward_test(
                bev_embed, gt_lane_labels, gt_lane_masks, img_metas, rescale)

        # ============================================================
        # Step 3: 运动预测 (Motion) - 推理模式
        # ============================================================
        if self.with_motion_head:
            result_motion, outs_motion = self.motion_head.forward_test(
                bev_embed, outs_track=result_track[0], outs_seg=result_seg[0])
            outs_motion['bev_pos'] = result_track[0]['bev_pos']

        # ============================================================
        # Step 4: 占据预测 (Occ) - 推理模式
        # ============================================================
        outs_occ = dict()
        if self.with_occ_head:
            occ_no_query = outs_motion['track_query'].shape[1] == 0
            outs_occ = self.occ_head.forward_test(
                bev_embed,
                outs_motion,
                no_query=occ_no_query,
                gt_segmentation=gt_segmentation,
                gt_instance=gt_instance,
                gt_img_is_valid=gt_occ_img_is_valid,
            )
            result[0]['occ'] = outs_occ

        # ============================================================
        # Step 5: 规划 (Planning) - 推理模式
        # ============================================================
        if self.with_planning_head:
            planning_gt = dict(
                segmentation=gt_segmentation,
                sdc_planning=sdc_planning,
                sdc_planning_mask=sdc_planning_mask,
                command=command)
            result_planning = self.planning_head.forward_test(
                bev_embed, outs_motion, outs_occ, command)
            result[0]['planning'] = dict(
                planning_gt=planning_gt,
                result_planning=result_planning,
            )

        # ---- 清理输出：移除不需要传递给下游的大张量 ----
        # 这些内容在推理时不需要保存在最终结果中，移除以节省内存
        pop_track_list = ['prev_bev', 'bev_pos', 'bev_embed',
                          'track_query_embeddings', 'sdc_embedding']
        result_track[0] = pop_elem_in_result(result_track[0], pop_track_list)

        if self.with_seg_head:
            result_seg[0] = pop_elem_in_result(result_seg[0], pop_list=['pts_bbox', 'args_tuple'])
        if self.with_motion_head:
            result_motion[0] = pop_elem_in_result(result_motion[0])
        if self.with_occ_head:
            result[0]['occ'] = pop_elem_in_result(
                result[0]['occ'],
                pop_list=['seg_out_mask', 'flow_out', 'future_states_occ',
                          'pred_ins_masks', 'pred_raw_occ', 'pred_ins_logits',
                          'pred_ins_sigmoid'])

        # ---- 合并所有任务的结果 ----
        for i, res in enumerate(result):
            res['token'] = img_metas[i]['sample_idx']
            res.update(result_track[i])
            if self.with_motion_head:
                res.update(result_motion[i])
            if self.with_seg_head:
                res.update(result_seg[i])

        return result


def pop_elem_in_result(task_result: dict, pop_list: list = None):
    """清理推理结果中的冗余字段

    移除以下类型的字段以节省内存和传输带宽:
    1. 以 'query', 'query_pos', 'embedding' 结尾的字段（中间特征）
    2. pop_list 中指定的字段

    Args:
        task_result: 任务输出字典
        pop_list: 需要额外移除的字段列表

    Returns:
        清理后的字典
    """
    all_keys = list(task_result.keys())
    for k in all_keys:
        if k.endswith('query') or k.endswith('query_pos') or k.endswith('embedding'):
            task_result.pop(k)

    if pop_list is not None:
        for pop_k in pop_list:
            task_result.pop(pop_k, None)
    return task_result
