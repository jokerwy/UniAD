# UniAD 代码库阅读学习指南

> **文档版本**: v1.0  
> **适用版本**: UniAD 2.0 (mmdet3d 1.x + torch 2.x)  
> **创建日期**: 2026-07-16

---

## 📚 阅读指南说明

本文档提供**系统性的 UniAD 代码库阅读顺序**，按照"**从整体到局部、从数据流到底层实现**"的原则编排。建议按照推荐的阶段顺序阅读，每个阶段标注了：

- ⭐ **重要程度**: 必读/重要/参考
- 🎯 **阅读重点**: 关键代码段和核心概念
- ⚠️ **注意事项**: 容易混淆或出错的地方
- 🔗 **相关文件**: 关联的上下游模块

---

## 🗺️ 整体架构速览

```
UniAD 端到端架构（6大任务）

输入: 6相机 × 3帧图像
  │
  ├── 特征提取 ─────────┐
  │  ├─ img_backbone (ResNet101+DCN)
  │  └─ img_neck (FPN)
  │
  ├── BEV编码 ──────────┤ Perception
  │  └─ PerceptionTransformer      │ 感知
  │     ├─ BEVFormerEncoder ───────┤
  │     │   ├─ TemporalSelfAttention (时序融合)
  │     │   └─ SpatialCrossAttention (图像→BEV投影)
  │     └─ DetectionTransformerDecoder
  │
  ├── 跟踪 ─────────────┤ TrackFormer
  │  └─ BEVFormerTrackHead + QIM + MemoryBank
  │
  ├── 建图 ─────────────┤ MapFormer
  │  └─ PansegformerHead (车道线+可行驶区域)
  │
  ├── 运动预测 ─────────┤ MotionFormer
  │  └─ MotionHead (6模态 × 12步轨迹)
  │
  ├── 占用预测 ─────────┤ OccFormer
  │  └─ OccHead (未来4帧占据栅格)
  │
  └── 轨迹规划 ─────────┤ Planner
     └─ PlanningHeadSingleMode + 碰撞优化
```

---

## 第一阶段：项目概览与环境准备 ⭐必读

### 1.1 项目文档阅读

| 文件 | 重要程度 | 预计时间 | 核心内容 |
|------|---------|---------|---------|
| [README.md](README.md) | ⭐⭐⭐ | 15min | 项目介绍、训练流程、模型结构 |
| [docs/INSTALL.md](docs/INSTALL.md) | ⭐⭐⭐ | 20min | 环境安装、依赖版本、常见问题 |
| [docs/DATA_PREP.md](docs/DATA_PREP.md) | ⭐⭐⭐ | 15min | 数据集准备、预处理脚本 |
| [docs/TRAIN_EVAL.md](docs/TRAIN_EVAL.md) | ⭐⭐⭐ | 20min | 训练/评估/可视化完整流程 |
| UniAD_Paper_Key_Info.md | ⭐⭐⭐ | 30min | 论文核心概念和算法细节 |
| CODEBASE_ANALYSIS.md | ⭐⭐⭐ | 30min | 代码架构深度分析 |

**🎯 阅读重点**:
- 理解两阶段训练策略（stage1: 感知 / stage2: 端到端）
- 掌握 query-based 设计的核心理念
- 熟悉各任务的评价指标和数据格式

**⚠️ 注意事项**:
- UniAD v2.0 使用 mmdet3d 1.x 和 torch 2.x，与 v1.0 不兼容
- 阶段1需要约 50GB GPU 显存，阶段2约 17GB

---

### 1.2 目录结构概览

```
UniAD/
├── docs/                    # 文档（必读）
├── ckpts/                   # 预训练权重目录
├── projects/
│   ├── configs/             # 配置文件
│   │   ├── stage1_track_map/   # 阶段1配置
│   │   └── stage2_e2e/         # 阶段2配置
│   └── mmdet3d_plugin/      # ⭐ 核心代码
│       ├── datasets/         # 数据集实现
│       ├── losses/           # 损失函数
│       ├── models/           # 模型组件
│       └── uniad/            # ⭐ UniAD主实现
│           ├── apis/         # 训练/测试API
│           ├── detectors/    # ⭐ 检测器入口
│           ├── dense_heads/  # ⭐ 任务头
│           ├── modules/      # ⭐ BEV Transformer
│           └── hooks/        # 训练钩子
├── tools/                    # 训练/评估/可视化脚本
└── data/                     # 数据集目录（需自行准备）
```

---

## 第二阶段：入口与核心模型 ⭐⭐⭐ 必读

### 2.1 训练/测试入口

#### 文件: `tools/train.py`

| 属性 | 内容 |
|------|------|
| 重要程度 | ⭐⭐⭐ |
| 阅读目标 | 理解整体训练流程 |
| 预计时间 | 15min |

**🎯 阅读重点**:
```python
# 1. 配置文件解析
args = parse_args()
cfg = Config.fromfile(args.config)

# 2. 模型构建
model = build_model(cfg.model)

# 3. 数据集构建
datasets = [build_dataset(cfg.data.train)]

# 4. 训练启动
train_model(model, datasets, cfg, ...)
```

**⚠️ 注意事项**:
- 使用 `mmcv` 的配置系统，支持继承和动态修改
- 分布式训练需要特殊处理：`init_dist()`

---

### 2.2 核心模型：UniAD

#### 文件: `projects/mmdet3d_plugin/uniad/detectors/uniad_e2e.py` ⭐⭐⭐ 最重要

| 属性 | 内容 |
|------|------|
| 重要程度 | ⭐⭐⭐⭐⭐ |
| 阅读目标 | 理解端到端模型整体结构和数据流 |
| 预计时间 | 1-2小时 |
| 前置知识 | 了解 mmdet3d 的检测器基类 |

**🎯 核心代码段阅读顺序**:

**1. 类定义与初始化 (Line 16-65)**
```python
@DETECTORS.register_module()
class UniAD(UniADTrack):
    """
    UniAD: Unifying Detection, Tracking, Segmentation, Motion Forecasting, 
    Occupancy Prediction and Planning for Autonomous Driving
    """
    def __init__(self, seg_head=None, motion_head=None, 
                 occ_head=None, planning_head=None, ...):
        # 构建各个任务头
        if seg_head: self.seg_head = build_head(seg_head)
        if motion_head: self.motion_head = build_head(motion_head)
        if occ_head: self.occ_head = build_head(occ_head)
        if planning_head: self.planning_head = build_head(planning_head)
```
**重点**: 所有任务头通过 `build_head()` 动态构建，配置文件中定义

**2. 前向入口 (Line 70-84)**
```python
def forward(self, return_loss=True, **kwargs):
    if return_loss:
        return self.forward_train(**kwargs)
    else:
        return self.forward_test(**kwargs)
```

**3. 训练前向 (Line 88-200)** ⭐⭐⭐⭐⭐
```python
def forward_train(self, img, img_metas, gt_bboxes_3d, ...):
    # Step 1: 提取BEV特征（继承自UniADTrack）
    # Step 2: 跟踪（继承自UniADTrack）
    # Step 3: 建图（seg_head）
    # Step 4: 运动预测（motion_head）
    # Step 5: 占用预测（occ_head）
    # Step 6: 规划（planning_head）
    # Step 7: 计算损失
```

**⚠️ 关键注意事项**:
1. **数据维度理解**:
   - `img`: (B, T, N, C, H, W) = (批次, 时序, 相机数, 通道, 高, 宽)
   - `bev_embed`: (B, H*W, D) = (批次, 200*200, 256)
   - `track_query`: (B, num_query, D)

2. **损失权重配置**:
```python
task_loss_weight = dict(
    track=1.0,      # 检测+跟踪
    map=1.0,        # 建图
    motion=1.0,     # 运动预测
    occ=1.0,        # 占用预测
    planning=1.0,   # 规划
)
```

---

#### 文件: `projects/mmdet3d_plugin/uniad/detectors/uniad_track.py` ⭐⭐⭐⭐

| 属性 | 内容 |
|------|------|
| 重要程度 | ⭐⭐⭐⭐⭐ |
| 阅读目标 | 理解BEV特征提取和跟踪机制 |
| 预计时间 | 1.5-2小时 |
| 前置知识 | DETR、Transformer、多目标跟踪 |

**🎯 核心代码段阅读顺序**:

**1. 类定义与配置 (Line 24-100)**
```python
@DETECTORS.register_module()
class UniADTrack(MVXTwoStageDetector):
    """UniAD tracking part"""
    def __init__(self, ...,
                 qim_args=dict(qim_type="QIMBase", ...),
                 mem_args=dict(memory_bank_type="MemoryBank", ...),
                 ...):
```

**关键配置参数**:
- `qim_args`: Query Interaction Module 配置
- `mem_args`: Memory Bank 配置
- `queue_length`: 时序帧数 (stage1=5, stage2=3)
- `num_query`: 900 (检测query数)

**2. 特征提取流程 (Line 200-250)**
```python
def extract_img_feat(self, img, img_metas, len_queue=None):
    # 1. 图像特征提取 (ResNet+DCN)
    # 2. FPN融合多尺度特征
    # 3. 返回多尺度特征列表
```

**3. BEV特征生成 (Line 350-450)**
```python
def extract_pts_feat(self, img, img_metas, img_feats, **kwargs):
    # 1. 准备BEV queries和位置编码
    # 2. 调用PerceptionTransformer
    # 3. 返回bev_embed和bev_pos
```

**4. 训练前向 (Line 500-650)**
```python
def forward_train_track(self, img, img_metas, ...):
    # Step 1: 提取图像特征
    # Step 2: 提取BEV特征
    # Step 3: 检测Decoder得到query
    # Step 4: QIM处理query
    # Step 5: Memory Bank时序融合
    # Step 6: 匈牙利匹配和损失计算
```

**⚠️ 关键注意事项**:
1. **冻结策略**:
```python
freeze_img_backbone=True   # 冻结图像骨干
freeze_img_neck=True       # 冻结FPN
freeze_bn=True             # 冻结BN
freeze_bev_encoder=True    # 冻结BEV编码器 (stage2)
```

2. **Memory Bank 工作机制**:
   - 存储历史帧的 `query`, `score`, `label`
   - `memory_bank_len=4` 表示存储4帧历史
   - 通过 `QueryInteractionModule` 与当前query交互

---

### 2.3 纯检测基线：BEVFormer

#### 文件: `projects/mmdet3d_plugin/uniad/detectors/bevformer.py` ⭐⭐⭐

| 属性 | 内容 |
|------|------|
| 重要程度 | ⭐⭐⭐ |
| 阅读目标 | 理解纯检测版本的BEVFormer |
| 预计时间 | 30min |

**🎯 阅读重点**:
- 与 `UniADTrack` 的区别：没有跟踪组件（QIM/Memory Bank）
- 用于预训练BEV特征表示
- 输出格式：`track_bboxes`, `track_scores`, `track_ids`

**⚠️ 注意事项**:
- 这是原始BEVFormer的实现，理解它有助于理解UniAD的改进
- 实际项目中主要使用 `UniADTrack` 和 `UniAD`

---

## 第三阶段：任务头详解（按执行顺序）⭐⭐⭐ 必读

### 3.1 检测+跟踪头

#### 文件: `projects/mmdet3d_plugin/uniad/dense_heads/track_head.py` ⭐⭐⭐⭐

| 属性 | 内容 |
|------|------|
| 重要程度 | ⭐⭐⭐⭐ |
| 阅读目标 | 理解3D检测和跟踪头的实现 |
| 预计时间 | 1小时 |

**🎯 核心组件**:

**1. BEVFormerTrackHead 类 (Line 24-150)**
```python
class BEVFormerTrackHead(DETRHead):
    """基于BEV特征的3D检测和跟踪头"""
    def __init__(self, ..., with_box_refine=True, ...):
```

**关键参数**:
- `with_box_refine=True`: 逐层精炼边界框
- `transformer`: PerceptionTransformer配置
- `num_query=900`: 检测query数量

**2. 前向传播 (Line 200-350)**
```python
def forward(self, mlvl_feats, img_metas, ...):
    # 1. 准备query和位置编码
    # 2. 调用Transformer Decoder (6层)
    # 3. 输出分类、边界框、跟踪特征
```

**3. 损失计算 (Line 400-550)**
```python
def loss(self, ..., gt_bboxes_3d, gt_labels_3d, ...):
    # 1. 匈牙利匹配 (HungarianAssigner3DTrack)
    # 2. 计算分类损失 (Focal Loss)
    # 3. 计算回归损失 (L1 Loss)
```

---

#### 文件: `projects/mmdet3d_plugin/uniad/dense_heads/track_head_plugin/` ⭐⭐⭐⭐

**modules.py** - QIM和Memory Bank实现
```python
class QueryInteractionModule(nn.Module):
    """Query交互模块：实现检测query间的信息传递"""
    def forward(self, queries, ...):
        # 通过Transformer层让query交互
        ...

class MemoryBank:
    """记忆库：存储历史帧跟踪信息"""
    def update(self, track_instances):
        # 更新历史query
        ...
```

**tracker.py** - 运行时跟踪器
```python
class RuntimeTrackerBase:
    """在线跟踪器，维护轨迹生命周期"""
    def update(self, track_instances):
        # 匹配当前帧与历史轨迹
        # 处理新轨迹创建、轨迹更新、轨迹删除
        ...
```

**track_instance.py** - 跟踪实例定义
```python
class Instances:
    """存储单个跟踪目标的属性"""
    # query, score, label, bbox, vel, track_id, ...
```

**⚠️ 注意事项**:
1. **Query vs Instance**:
   - `query`: Transformer中的特征向量 (256-dim)
   - `instance`: 包含query和元信息的完整跟踪目标

2. **匹配策略**:
   - 训练时：匈牙利算法与GT匹配
   - 推理时：基于IoU和分数的贪心匹配

---

### 3.2 在线建图头

#### 文件: `projects/mmdet3d_plugin/uniad/dense_heads/panseg_head.py` ⭐⭐⭐⭐

| 属性 | 内容 |
|------|------|
| 重要程度 | ⭐⭐⭐⭐ |
| 阅读目标 | 理解车道线和可行驶区域分割 |
| 预计时间 | 1小时 |

**🎯 核心概念**:

**1. Things vs Stuff**
```python
# Things: 车道线（有明确边界的实例）
thing_classes = ['divider', 'ped_crossing', 'boundary']

# Stuff: 可行驶区域（无明确边界的区域）
stuff_classes = ['drivable_area']
```

**2. 网络结构 (Line 50-200)**
```python
class PansegformerHead(nn.Module):
    def __init__(self, ...):
        # thing_transformer_head: 4层Mask Head
        # stuff_transformer_head: 6层Mask Head
        # SegDeformableTransformer: 编码器+解码器
```

**3. 输出格式**
```python
# 车道线：边界框 + Mask
thing_pred_results = dict(
    cls_scores=...,     # 分类分数
    bbox_preds=...,     # 边界框
    mask_preds=...,     # Mask预测
)

# 可行驶区域：区域框 + Mask
stuff_pred_results = dict(
    cls_scores=...,     # 分类分数
    mask_preds=...,     # Mask预测
)
```

**⚠️ 注意事项**:
1. **输入依赖**: 建图头使用 `track_query` 与 `bev_embed` 进行交叉注意力
2. **损失组合**: Focal + L1 + Dice + GIoU，多任务损失需谨慎平衡

---

### 3.3 运动预测头

#### 文件: `projects/mmdet3d_plugin/uniad/dense_heads/motion_head.py` ⭐⭐⭐⭐⭐

| 属性 | 内容 |
|------|------|
| 重要程度 | ⭐⭐⭐⭐⭐ |
| 阅读目标 | 理解多模态轨迹预测机制 |
| 预计时间 | 1.5小时 |
| 前置知识 | 注意力机制、轨迹预测、非线性优化 |

**🎯 核心代码段**:

**1. 类定义 (Line 24-100)**
```python
class MotionHead(BaseMotionHead):
    """多模态运动预测头"""
    def __init__(self, ..., 
                 num_motion_mode=6,    # 6种运动模式
                 num_future_frame=12,   # 预测未来12步(6秒)
                 ...):
```

**2. Motion Query设计** ⭐⭐⭐⭐⭐
```python
# Motion Query = Q_pos + Q_ctx

# Q_pos (位置信息): 
#   - 场景级锚点: 从全局运动统计聚类(k-means)
#   - 智能体级锚点: 从局部意图聚类
#   - 当前位置 + 预测目标点

# Q_ctx (上下文信息):
#   - 来自Track Query的agent特征
#   - 来自Map Query的地图特征
```

**3. 三种交互 (Line 200-400)**
```python
def forward(self, track_query, map_query, ...):
    # 1. Agent-Agent交互: 自注意力建模agent间关系
    # 2. Agent-Map交互: 交叉注意力关注地图信息
    # 3. Agent-Goal Point交互: 可变形注意力关注目标点
```

**4. 非线性优化 (Line 450-550)**
```python
# 使用casadi进行轨迹平滑
def nonlinear_smoother(self, traj, ...):
    # 约束: jerk, curvature, curvature_rate, 
    #       acceleration, lateral_acceleration
    ...
```

**⚠️ 关键注意事项**:
1. **锚点机制**: `motion_anchor_infos_mode6.pkl` 存储预计算的6种运动模式锚点
2. **训练 vs 推理**:
   - 训练：使用GT轨迹选择最佳模式
   - 推理：选择分类分数最高的模式
3. **minADE/minFDE**: 仅作为评估指标，不参与训练

---

### 3.4 占用预测头

#### 文件: `projects/mmdet3d_plugin/uniad/dense_heads/occ_head.py` ⭐⭐⭐⭐

| 属性 | 内容 |
|------|------|
| 重要程度 | ⭐⭐⭐⭐ |
| 阅读目标 | 理解未来占据栅格预测 |
| 预计时间 | 1小时 |

**🎯 核心概念**:

**1. 栅格配置**
```python
occflow_grid_conf = {
    'xbound': [-50.0, 50.0, 0.5],   # 100m范围, 0.5m分辨率 → 200像素
    'ybound': [-50.0, 50.0, 0.5],
    'zbound': [-10.0, 10.0, 20.0],  # 高度压缩为1层
}
```

**2. 网络结构**
```python
class OccHead(BaseModule):
    def __init__(self, ...):
        # BevFeatureSlicer: 提取BEV时序特征
        # DetrTransformerDecoder: 5层时序解码器
        # CVT_Decoder: 上采样到高分辨率
```

**3. 像素-Agent交互** ⭐⭐⭐⭐
```python
# 核心创新：将密集BEV特征作为Query
# 将实例级Agent特征作为Key/Value
# 使用注意力掩码限制关注范围
```

**4. 输出格式**
```python
# 每帧未来预测包含:
- semantic_seg: 语义分割 (车辆/背景)
- instance_seg: 实例分割 (区分不同车辆)
- centerness: 实例中心置信度
- offset: 像素到实例中心的偏移
- flow: 前向/后向流向
```

**⚠️ 注意事项**:
1. **时序建模**: `occ_n_future=4` 预测未来4帧(2秒)
2. **注意力掩码**: 限制每个像素只关注占据它的agent
3. **占用引导**: 复用mask特征而非原始agent特征

---

### 3.5 规划头

#### 文件: `projects/mmdet3d_plugin/uniad/dense_heads/planning_head.py` ⭐⭐⭐⭐⭐

| 属性 | 内容 |
|------|------|
| 重要程度 | ⭐⭐⭐⭐⭐ |
| 阅读目标 | 理解轨迹规划和碰撞优化 |
| 预计时间 | 1小时 |

**🎯 核心代码段**:

**1. 类定义 (Line 20-80)**
```python
class PlanningHeadSingleMode(nn.Module):
    """单模态轨迹规划头"""
    def __init__(self, ..., planning_steps=6, ...):
        # navi_embed: 导航指令嵌入(左转/直行/右转)
        # reg_branch: 轨迹回归MLP
```

**2. 前向传播 (Line 100-200)**
```python
def forward(self, bev_embed, navi, ...):
    # 1. 导航指令嵌入 (3类 → 256-dim)
    navi_embed = self.navi_embed(navi)  # (B, 256)
    
    # 2. 与BEV特征交叉注意力
    # 3. 回归未来6步轨迹 (6×2=12个值)
    planning_traj = self.reg_branch(query)  # (B, 6, 2)
```

**3. 碰撞优化 (推理时)** ⭐⭐⭐⭐⭐
```python
# CollisionNonlinearOptimizer
# 使用牛顿法优化轨迹
# 代价函数 = L2损失(保持原始预测) + 碰撞项(推离占用栅格)
```

**⚠️ 关键注意事项**:
1. **导航指令**: 3类编码 (0=左转, 1=直行, 2=右转)
2. **碰撞损失训练**: 三段式 (delta=0.0, 0.5, 1.0)，权重不同
3. **推理优化**: 仅推理时使用碰撞优化，训练时不使用

---

## 第四阶段：BEV Transformer模块 ⭐⭐⭐⭐ 必读

### 4.1 整体Transformer

#### 文件: `projects/mmdet3d_plugin/uniad/modules/transformer.py` ⭐⭐⭐⭐

| 属性 | 内容 |
|------|------|
| 重要程度 | ⭐⭐⭐⭐ |
| 阅读目标 | 理解PerceptionTransformer整体结构 |
| 预计时间 | 45min |

**🎯 核心结构**:
```python
class PerceptionTransformer(nn.Module):
    def __init__(self, ...):
        # encoder: BEVFormerEncoder (6层)
        # decoder: DetectionTransformerDecoder (6层)
    
    def forward(self, mlvl_feats, ...):
        # 1. 编码器：生成BEV特征 (bev_embed)
        # 2. 解码器：检测query与BEV特征交互
```

---

### 4.2 BEV编码器

#### 文件: `projects/mmdet3d_plugin/uniad/modules/encoder.py` ⭐⭐⭐⭐⭐

| 属性 | 内容 |
|------|------|
| 重要程度 | ⭐⭐⭐⭐⭐ |
| 阅读目标 | 理解时序自注意力和空间交叉注意力 |
| 预计时间 | 1.5小时 |

**🎯 核心组件**:

**1. BEVFormerEncoder (Line 50-150)**
```python
class BEVFormerEncoder(nn.Module):
    """6层Transformer编码器"""
    def __init__(self, ...):
        # layers: 每层包含两种注意力
        # - TemporalSelfAttention (时序)
        # - SpatialCrossAttention (空间)
```

**2. 时序自注意力 (temporal_self_attention.py)** ⭐⭐⭐⭐
```python
class TemporalSelfAttention(nn.Module):
    """将当前BEV与历史BEV对齐"""
    def forward(self, query, key, value, ...):
        # 1. 根据车辆运动补偿旋转和平移
        # 2. 使用可变形注意力融合时序信息
```

**⚠️ 关键细节**:
- `bev_h_=200, bev_w_=200`: BEV分辨率
- 使用车辆位姿信息进行时序对齐

**3. 空间交叉注意力 (spatial_cross_attention.py)** ⭐⭐⭐⭐⭐
```python
class SpatialCrossAttention(nn.Module):
    """图像特征 → BEV空间投影"""
    def forward(self, query, key, value, ...):
        # 1. 在BEV空间生成3D参考点
        # 2. 将3D点投影到多相机图像平面
        # 3. 使用可变形注意力采样图像特征
```

**⚠️ 关键注意事项**:
1. **3D参考点生成**: BEV网格 + 高度采样
2. **多相机投影**: 使用相机外参将3D点投影到6个相机
3. **CUDA加速**: `multi_scale_deformable_attn_function` 提供CUDA实现

---

### 4.3 解码器

#### 文件: `projects/mmdet3d_plugin/uniad/modules/decoder.py` ⭐⭐⭐

**🎯 核心内容**:
```python
class DetectionTransformerDecoder(nn.Module):
    """6层Transformer解码器"""
    def __init__(self, ...):
        # 每层包含：
        # - Self-Attention (query间交互)
        # - Cross-Attention (query与BEV特征交互)
        # - FFN
```

---

## 第五阶段：数据集与数据处理 ⭐⭐⭐ 必读

### 5.1 端到端数据集

#### 文件: `projects/mmdet3d_plugin/datasets/nuscenes_e2e_dataset.py` ⭐⭐⭐⭐

| 属性 | 内容 |
|------|------|
| 重要程度 | ⭐⭐⭐⭐ |
| 阅读目标 | 理解数据加载和标注格式 |
| 预计时间 | 1小时 |

**🎯 核心数据结构**:
```python
class NuScenesE2EDataset:
    """支持时序和未来标注的数据集"""
    
    # 关键数据项
    'img': (T, N, C, H, W)          # 时序多视角图像
    'gt_bboxes_3d': [...]            # 3D边界框
    'gt_labels_3d': [...]            # 类别标签
    'gt_inds': [...]                 # 实例ID(跟踪)
    'gt_fut_traj': (N, 12, 2)        # 目标未来轨迹(12步)
    'gt_sdc_fut_traj': (6, 2)        # 自车未来轨迹(6步)
    'command': int                   # 导航指令
    'gt_segmentation': (H, W)        # 占据栅格
    'gt_instance': (H, W)            # 实例ID
    'gt_flow': (H, W, 2)             # 流向
```

**⚠️ 注意事项**:
1. `queue_length`: 时序长度 (stage1=5, stage2=3)
2. 未来标注：用于运动预测和占用预测监督
3. 坐标系：ego坐标系，单位米

---

### 5.2 数据处理管线

#### 文件: `projects/mmdet3d_plugin/datasets/pipelines/` ⭐⭐⭐

| 文件 | 功能 | 重要程度 |
|------|------|---------|
| `loading.py` | 加载图像和标注 | ⭐⭐⭐⭐ |
| `transform_3d.py` | 3D数据增强 | ⭐⭐⭐ |
| `occflow_label.py` | 生成占用/流向标签 | ⭐⭐⭐⭐ |
| `formating.py` | 数据格式化 | ⭐⭐⭐ |

**训练管线流程**:
```
LoadMultiViewImageFromFilesInCeph  # 加载6相机图像
  → PhotoMetricDistortionMultiViewImage  # 光度增强
  → LoadAnnotations3D_E2E  # 加载完整标注
  → GenerateOccFlowLabels  # 生成占用/流向标签
  → ObjectRangeFilterTrack  # 过滤范围外目标
  → ObjectNameFilterTrack  # 过滤类别
  → NormalizeMultiviewImage  # 图像归一化
  → PadMultiViewImage  # 填充到32的倍数
  → DefaultFormatBundle3D  # 格式化
  → CustomCollect3D  # 收集指定字段
```

---

## 第六阶段：损失函数 ⭐⭐⭐ 必读

### 6.1 各任务损失

#### 文件: `projects/mmdet3d_plugin/losses/` ⭐⭐⭐⭐

| 文件 | 功能 | 用途 |
|------|------|------|
| `track_loss.py` | Focal + L1 + IoU | 检测+跟踪 |
| `dice_loss.py` | Dice Loss | 分割Mask |
| `traj_loss.py` | 分类 + NLL + minADE | 运动预测 |
| `planning_loss.py` | L2 + 碰撞损失 | 规划 |
| `occflow_loss.py` | Top-K二元分割 | 占用预测 |

**🎯 损失权重配置示例**:
```python
# track_head
loss_cls=dict(type='FocalLoss', ...)
loss_bbox=dict(type='L1Loss', ...)
loss_iou=dict(type='GIoULoss', ...)

# motion_head
loss_traj_cls=dict(type='CrossEntropyLoss', ...)
loss_traj_reg=dict(type='NLLLoss', ...)

# planning_head
loss_planning_reg=dict(type='L1Loss', ...)
loss_planning_col=dict(type='CollisionLoss', ...)
```

**⚠️ 注意事项**:
1. **损失平衡**: 各任务损失量级差异大，需要仔细调整权重
2. **匈牙利匹配**: 跟踪任务使用 `HungarianAssigner3DTrack`

---

## 第七阶段：配置文件 ⭐⭐⭐ 必读

### 7.1 配置继承关系

```
_base_/default_runtime.py          # 基础运行时配置

stage1_track_map/base_track_map.py  # 阶段1: 检测+跟踪+建图
    ├── 训练6 epoch, ~50GB显存
    └── load_from: bevformer预训练权重

stage2_e2e/base_e2e.py             # 阶段2: 端到端
    ├── 训练20 epoch, ~17GB显存
    └── load_from: stage1权重
    └── freeze_img_backbone/neck/bn/bev_encoder=True
```

### 7.2 关键配置参数

| 参数 | 阶段1 | 阶段2 | 说明 |
|------|-------|-------|------|
| `queue_length` | 5 | 3 | 时序帧数 |
| `freeze_img_backbone` | False | True | 冻结图像骨干 |
| `freeze_bev_encoder` | False | True | 冻结BEV编码器 |
| `num_query` | 900 | 900 | 检测query数 |
| `total_epochs` | 6 | 20 | 训练轮数 |

---

## 第八阶段：工具脚本 ⭐⭐ 参考

### 8.1 训练/评估脚本

| 脚本 | 功能 |
|------|------|
| `tools/train.py` | 训练入口 |
| `tools/test.py` | 评估入口 |
| `tools/uniad_dist_train.sh` | 分布式训练 |
| `tools/uniad_dist_eval.sh` | 分布式评估 |

### 8.2 可视化工具

| 脚本 | 功能 |
|------|------|
| `tools/uniad_vis_result.sh` | 启动可视化 |
| `tools/analysis_tools/visualize/run.py` | BEV/相机视角渲染 |
| `tools/analysis_tools/analyze_logs.py` | 训练曲线分析 |

---

## 📋 推荐阅读顺序总结

### 快速入门路线（2-3天）

| 天数 | 阶段 | 核心文件 |
|------|------|---------|
| Day 1 | 第1-2阶段 | README → docs/ → uniad_e2e.py → uniad_track.py |
| Day 2 | 第3阶段 | track_head.py → panseg_head.py → motion_head.py |
| Day 3 | 第3-4阶段 | occ_head.py → planning_head.py → transformer.py → encoder.py |

### 深入研究路线（1-2周）

在上述基础上，深入阅读：
- 所有 `*_plugin/` 目录下的实现细节
- 损失函数实现
- 数据处理管线
- 配置文件参数调优

---

## 🔗 关键依赖关系图

```
UniAD (uniad_e2e.py)
├── 继承: UniADTrack (uniad_track.py)
│     ├── 继承: MVXTwoStageDetector (mmdet3d)
│     ├── PerceptionTransformer (transformer.py)
│     │     ├── BEVFormerEncoder (encoder.py)
│     │     │     ├── TemporalSelfAttention
│     │     │     └── SpatialCrossAttention
│     │     └── DetectionTransformerDecoder
│     ├── BEVFormerTrackHead (track_head.py)
│     │     ├── QueryInteractionModule
│     │     └── MemoryBank
│     └── RuntimeTrackerBase (track_head_plugin/tracker.py)
│
├── PansegformerHead (panseg_head.py)
│     └── SegDeformableTransformer
│
├── MotionHead (motion_head.py)
│     ├── MotionTransformerDecoder
│     └── nonlinear_smoother (casadi)
│
├── OccHead (occ_head.py)
│     ├── BevFeatureSlicer
│     └── CVT_Decoder
│
└── PlanningHeadSingleMode (planning_head.py)
      └── CollisionNonlinearOptimizer
```

---

## 💡 调试技巧

1. **可视化中间结果**: 使用 `tools/analysis_tools/visualize/` 查看BEV特征和预测结果
2. **单步调试**: 在 `forward_train` 各阶段添加打印，观察张量形状
3. **梯度检查**: 使用 `torch.autograd.set_detect_anomaly(True)` 检测梯度异常
4. **学习率监控**: 使用 `tools/analysis_tools/analyze_logs.py` 绘制训练曲线

---

## 📖 扩展阅读

- **BEVFormer 论文**: 理解BEV特征生成的基础
- **DETR/DETR3D**: 理解query-based检测的基础
- **MOTR**: 理解跟踪query的设计
- **CasADi 文档**: 理解非线性轨迹优化

---

> **提示**: 本文档为阅读指南，详细代码分析请参考 [CODEBASE_ANALYSIS.md](CODEBASE_ANALYSIS.md)，论文细节请参考 [UniAD_Paper_Key_Info.md](UniAD_Paper_Key_Info.md)。
