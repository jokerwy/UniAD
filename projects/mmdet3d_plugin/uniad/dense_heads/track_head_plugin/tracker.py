"""
运行时跟踪管理器 (RuntimeTrackerBase)
=====================================
负责跟踪 ID 的生命周期管理: 分配新 ID、监控目标状态、回收丢失目标的 ID。

跟踪 ID 状态管理:
    1. 新目标出现: scores >= score_thresh 且 obj_idxes == -1
       → 分配新的 obj_id (从 max_obj_id 递增)

    2. 目标活跃: scores >= score_thresh 且 obj_idxes >= 0
       → 重置 disappear_time = 0

    3. 目标暂时丢失: scores < filter_score_thresh 且 obj_idxes >= 0
       → disappear_time += 1 (容忍帧数计数器)

    4. 目标永久丢失: disappear_time >= miss_tolerance
       → 回收 obj_id (obj_idxes = -1)

Args:
    score_thresh: 新目标初始化分数阈值 (高于此值才分配 ID)
    filter_score_thresh: 目标休眠分数阈值 (低于此值开始计数消失)
    miss_tolerance: 目标丢失容忍帧数 (超过此帧数未匹配则删除)
"""

from .track_instance import Instances
from mmdet3d.core.bbox.iou_calculators.iou3d_calculator import (
    bbox_overlaps_nearest_3d as iou_3d, )
from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox

class RuntimeTrackerBase(object):
    def __init__(self, score_thresh=0.5, filter_score_thresh=0.4, miss_tolerance=5):
        self.score_thresh = score_thresh           # 新目标初始化分数阈值
        self.filter_score_thresh = filter_score_thresh  # 目标休眠阈值
        self.miss_tolerance = miss_tolerance         # 丢失容忍帧数
        self.max_obj_id = 0                          # 当前最大 ID

    def clear(self):
        """重置 ID 计数器"""
        self.max_obj_id = 0

    def update(self, track_instances: Instances, iou_thre=None):
        """更新跟踪实例的 ID 状态

        对每个检测到的目标:
        1. 如果分数 >= score_thresh 且未分配 ID → 分配新 ID
        2. 如果分数 >= score_thresh 且已有 ID → 重置消失计时器
        3. 如果分数 < filter_score_thresh → 增加消失计时器
        4. 如果消失计时器 >= miss_tolerance → 回收 ID

        Args:
            track_instances: 跟踪实例
            iou_thre: IoU 阈值 (可选，用于避免重复检测)
        """
        # 分数足够高的目标，重置消失计时器
        track_instances.disappear_time[track_instances.scores >= self.score_thresh] = 0

        for i in range(len(track_instances)):
            # 情况 1: 新目标 → 分配 ID
            if (
                track_instances.obj_idxes[i] == -1
                and track_instances.scores[i] >= self.score_thresh
            ):
                # 如果设置了 IoU 阈值，检查与已有目标的重叠
                if iou_thre is not None and track_instances.pred_boxes[track_instances.obj_idxes>=0].shape[0]!=0:
                    iou3ds = iou_3d(
                        denormalize_bbox(track_instances.pred_boxes[i].unsqueeze(0), None)[...,:7],
                        denormalize_bbox(track_instances.pred_boxes[track_instances.obj_idxes>=0], None)[...,:7])
                    if iou3ds.max() > iou_thre:
                        continue  # 与已有目标重叠过大，跳过
                # 分配新 ID
                track_instances.obj_idxes[i] = self.max_obj_id
                self.max_obj_id += 1

            # 情况 3: 目标丢失 → 增加消失计时器
            elif (
                track_instances.obj_idxes[i] >= 0
                and track_instances.scores[i] < self.filter_score_thresh
            ):
                track_instances.disappear_time[i] += 1

                # 情况 4: 消失太久 → 回收 ID
                if track_instances.disappear_time[i] >= self.miss_tolerance:
                    track_instances.obj_idxes[i] = -1