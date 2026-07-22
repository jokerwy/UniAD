"""
seg_utils.py - 分割工具函数模块

本模块提供分割任务中使用的辅助工具函数，目前主要包含IoU（交并比）计算函数。
IoU用于评估预测分割掩码与真实掩码之间的重叠程度，是分割任务中常用的评估指标。
"""


def IOU(intputs, targets):
    """
    计算输入张量与目标张量之间的IoU（Intersection over Union，交并比）。

    IoU定义为：交集面积 / 并集面积
    即：(inputs * targets).sum() / (inputs.sum() + targets.sum() - (inputs * targets).sum())

    该函数在CPU上计算，返回IoU值、分子（交集）和分母（并集），
    可用于后续的损失计算或评估指标统计。

    Args:
        intputs (Tensor): 预测的分割掩码或概率图，形状为 (N, ...)，每个元素表示预测值。
        targets (Tensor): 真实的分割掩码，形状与inputs相同，每个元素表示真实标签（通常为0或1）。

    Returns:
        tuple:
            - loss (Tensor, CPU): IoU值，形状为 (N,)，范围在[0, 1]之间。
            - numerator (Tensor, CPU): 交集部分，即分子：(inputs * targets).sum(dim=1)
            - denominator (Tensor, CPU): 并集部分，即分母（添加了极小值eps防止除零错误）
    """
    # 计算交集：预测值与真实值逐元素相乘后沿第1维求和
    numerator = (intputs * targets).sum(dim=1)
    # 计算并集：预测值之和 + 真实值之和 - 交集（避免重复计算）
    # 添加极小值 1e-13 防止分母为零导致除零错误
    denominator = intputs.sum(dim=1) + targets.sum(dim=1) - numerator
    loss = numerator / (denominator + 0.0000000000001)
    # 将结果移动到CPU上，便于后续处理（如numpy转换、日志记录等）
    return loss.cpu(), numerator.cpu(), denominator.cpu()