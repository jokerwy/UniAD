"""
检测头包 (Dense Heads)
======================
包含 UniAD 所有任务的检测头模块:

    - BEVFormerHead:          BEVFormer 基础检测头 (Encoder + Decoder)
    - BEVFormerTrackHead:     跟踪检测头 (增加轨迹预测分支)
    - MotionHead:             运动预测头 (预测目标未来轨迹)
    - OccHead:                占据预测头 (3D 占据栅格预测)
    - PlanningHeadSingleMode: 规划头 (自车轨迹规划)
    - PansegformerHead:       全景分割头 (实例分割)
"""

from .track_head import BEVFormerTrackHead
from .panseg_head import PansegformerHead
from .motion_head import MotionHead
from .occ_head import OccHead
from .planning_head import PlanningHeadSingleMode
from .bevformer_head import BEVFormerHead