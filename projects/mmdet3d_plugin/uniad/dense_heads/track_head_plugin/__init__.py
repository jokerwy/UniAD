"""
跟踪头插件 (Track Head Plugin)
==============================
包含跟踪任务的核心组件:

    - Instances:               跟踪实例容器，存储目标的属性字段
    - RuntimeTrackerBase:      运行时跟踪管理器，负责 ID 分配与回收
    - MemoryBank:              记忆库，存储历史目标特征用于时序关联
    - QueryInteractionModule:  Query 交互模块，融合历史和当前 query
"""

from .modules import MemoryBank, QueryInteractionModule
from .track_instance import Instances
from .tracker import RuntimeTrackerBase