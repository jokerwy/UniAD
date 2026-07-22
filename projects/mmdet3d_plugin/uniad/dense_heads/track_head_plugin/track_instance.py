"""
跟踪实例 (Instances)
====================
一个灵活的目标实例容器，用于存储一帧图像中所有目标的属性信息。

核心设计:
    - 使用动态字段 (fields) 存储任意属性
    - 所有字段具有相同的长度 (目标数量)
    - 支持索引、切片、拼接等操作
    - 支持 .to(device) 和 .numpy() 转换

在 UniAD 跟踪中的使用:
    track_instances 是一个 Instances 对象，包含以下关键字段:
    - query:              可学习的检测查询向量 (num_query, 2*C)
    - ref_pts:            3D 参考点 (num_query, 3)
    - pred_boxes:         预测的 3D 框 (num_query, 10)
    - pred_logits:        预测的分类分数 (num_query, num_classes)
    - scores:             检测置信度 (num_query,)
    - obj_idxes:          跟踪 ID，-1=未分配，-2=自车 (num_query,)
    - output_embedding:   输出特征嵌入 (num_query, C)
    - mem_bank:           记忆库 (num_query, mem_len, C)
    - mem_padding_mask:   记忆库填充掩码 (num_query, mem_len)
    - disappear_time:     目标消失计时器 (num_query,)
    - matched_gt_idxes:   匹配到的 GT 索引 (num_query,)
    - iou:                与 GT 的 IoU (num_query,)
"""

import itertools
from typing import Any, Dict, List, Tuple, Union
import torch


class Instances:
    """目标实例容器

    存储一帧图像中所有目标的属性信息，每个属性是一个字段 (field)。
    支持索引操作，可以像列表一样对目标进行筛选。

    使用示例:
        # 创建实例
        instances = Instances((1, 1))
        instances.boxes = boxes_tensor     # 设置字段
        instances.scores = scores_tensor

        # 索引筛选
        valid = instances[instances.scores > 0.5]  # 筛选高分目标
        first_10 = instances[:10]                   # 取前 10 个

        # 拼接
        merged = Instances.cat([instances_a, instances_b])
    """

    def __init__(self, image_size: Tuple[int, int], **kwargs: Any):
        """创建实例容器

        Args:
            image_size: (height, width) 图像尺寸 (用于记录，实际不限制)
            kwargs: 初始字段，如 boxes=..., scores=...
        """
        self._image_size = image_size
        self._fields: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.set(k, v)

    @property
    def image_size(self) -> Tuple[int, int]:
        """返回图像尺寸 (height, width)"""
        return self._image_size

    def __setattr__(self, name: str, val: Any) -> None:
        """属性设置: 私有属性 (以_开头) 直接设置，其他视为字段"""
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name: str) -> Any:
        """属性访问: 从字段中查找"""
        if name == "_fields" or name not in self._fields:
            raise AttributeError("Cannot find field '{}' in the given Instances!".format(name))
        return self._fields[name]

    def set(self, name: str, value: Any) -> None:
        """设置字段值

        Args:
            name: 字段名
            value: 字段值，其长度必须与现有字段一致
        """
        data_len = len(value)
        if len(self._fields):
            assert len(self) == data_len, \
                "Adding a field of length {} to a Instances of length {}".format(data_len, len(self))
        self._fields[name] = value

    def has(self, name: str) -> bool:
        """检查字段是否存在"""
        return name in self._fields

    def remove(self, name: str) -> None:
        """删除字段"""
        del self._fields[name]

    def get(self, name: str) -> Any:
        """获取字段值"""
        return self._fields[name]

    def get_fields(self) -> Dict[str, Any]:
        """获取所有字段的字典"""
        return self._fields

    def to(self, *args: Any, **kwargs: Any) -> "Instances":
        """将所有支持 .to() 的字段转移到指定设备"""
        ret = Instances(self._image_size)
        for k, v in self._fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            ret.set(k, v)
        return ret

    def numpy(self):
        """将所有支持 .numpy() 的字段转换为 numpy"""
        ret = Instances(self._image_size)
        for k, v in self._fields.items():
            if hasattr(v, "numpy"):
                v = v.numpy()
            ret.set(k, v)
        return ret

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Instances":
        """索引操作: 对所有字段应用相同的索引

        支持:
        - 整数索引: instances[0] → 第一个目标
        - 切片: instances[:5] → 前 5 个目标
        - 布尔掩码: instances[scores > 0.5] → 筛选高分目标
        """
        if type(item) == int:
            if item >= len(self) or item < -len(self):
                raise IndexError("Instances index out of range!")
            else:
                item = slice(item, None, len(self))

        ret = Instances(self._image_size)
        for k, v in self._fields.items():
            # 特殊处理 kalman_models (列表类型)
            if k == 'kalman_models' and isinstance(item, torch.Tensor):
                ret_list = []
                for i, if_true in enumerate(item):
                    if if_true:
                        ret_list.append(self.kalman_models[i])
                ret.set(k, ret_list)
            else:
                ret.set(k, v[item])
        return ret

    def __len__(self) -> int:
        """返回目标数量 (所有字段应具有相同长度)"""
        for v in self._fields.values():
            return v.__len__()
        raise NotImplementedError("Empty Instances does not support __len__!")

    def __iter__(self):
        raise NotImplementedError("`Instances` object is not iterable!")

    @staticmethod
    def cat(instance_lists: List["Instances"]) -> "Instances":
        """拼接多个 Instances 对象

        将所有字段沿第一维拼接，支持:
        - torch.Tensor: torch.cat
        - list: itertools.chain
        - 自定义类型: type.cat()

        Args:
            instance_lists: Instances 列表

        Returns:
            拼接后的 Instances
        """
        assert all(isinstance(i, Instances) for i in instance_lists)
        assert len(instance_lists) > 0
        if len(instance_lists) == 1:
            return instance_lists[0]

        image_size = instance_lists[0].image_size
        for i in instance_lists[1:]:
            assert i.image_size == image_size
        ret = Instances(image_size)
        for k in instance_lists[0]._fields.keys():
            values = [i.get(k) for i in instance_lists]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                values = torch.cat(values, dim=0)
            elif isinstance(v0, list):
                values = list(itertools.chain(*values))
            elif hasattr(type(v0), "cat"):
                values = type(v0).cat(values)
            else:
                raise ValueError("Unsupported type {} for concatenation".format(type(v0)))
            ret.set(k, values)
        return ret

    def __str__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self))
        s += "image_height={}, ".format(self._image_size[0])
        s += "image_width={}, ".format(self._image_size[1])
        s += "fields=[{}])".format(", ".join((f"{k}: {v}" for k, v in self._fields.items())))
        return s

    __repr__ = __str__