# BEVFormer 在 UniAD 中的架构分析与理解

---

## 一、BEVFormer 论文核心理解

### 1.1 问题背景

自动驾驶感知系统需要从多摄像头图像中理解 3D 场景。传统方法通常分为"2D 检测→3D 投影/融合"的流水线，存在信息丢失和误差累积问题。BEVFormer 提出了一种**端到端的鸟瞰视角（BEV, Bird's Eye View）特征构建方法**，利用 Transformer 架构直接从多视角图像中学习统一的 BEV 表示。

### 1.2 核心思想

BEVFormer 的核心创新在于**将 BEV 视为一个可学习的查询空间**，通过 Transformer 的注意力机制，让 BEV 网格上的每个查询点主动从多相机图像特征中"查找"对应的视觉信息。

BEVFormer 包含三个关键注意力机制：

#### (1) BEV Queries（BEV 可学习查询）

BEV 空间被划分为 $H \times W$ 的网格（例如 $200 \times 200$），每个网格点对应一个可学习的嵌入向量（embedding）。这些向量作为 Transformer 的 query，通过注意力机制从图像特征中聚合信息。

```python
# 代码对应：BEVFormerHead._init_layers()
self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)
# 例如：40000 个 BEV 网格点，每个点 256 维特征
```

#### (2) 空间交叉注意力（Spatial Cross-Attention, SCA）

**这是 BEVFormer 中最核心的机制。** 每个 BEV query 在 3D 空间中定义一组参考点（pillar），将这些 3D 参考点投影到各个相机视角下，然后在投影位置附近采样图像特征。

关键步骤：
1. **3D 参考点生成**：每个 BEV 网格在高度方向采样 $Z$ 个点（如 4 个），形成 $(x, y, z_i)$ 的 3D 参考点
2. **相机投影**：利用相机内外参矩阵，将 3D 参考点投影到各相机平面上
3. **可变形注意力**：在投影位置附近使用可变形注意力（Deformable Attention）采样图像特征
4. **多相机聚合**：对每个 BEV 查询，聚合来自所有可见相机的特征

```python
# 代码对应：encoder.py BEVFormerEncoder.point_sampling()
# 3D 参考点 → 相机投影 → 得到 reference_points_cam 和 bev_mask
reference_points_cam, bev_mask = self.point_sampling(ref_3d, self.pc_range, img_metas)
```

**重要优化**：每个相机只与其可见的 BEV 查询交互，大幅减少计算量：

```python
# 代码对应：spatial_cross_attention.py SpatialCrossAttention.forward()
for i, mask_per_img in enumerate(bev_mask):
    index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
    # 只保留在当前相机可见的 BEV 查询
```

#### (3) 时间自注意力（Temporal Self-Attention, TSA）

BEVFormer 利用**历史帧的 BEV 特征**来增强当前帧的表征，实现时序信息融合。

关键机制：
1. **自车运动补偿**：将历史 BEV 特征根据自车的运动（平移+旋转）进行对齐
2. **可变形自注意力**：当前 BEV 查询同时关注当前位置和历史 BEV 中对齐位置的特征
3. **旋转对齐**：使用 `torchvision.transforms.functional.rotate` 对历史 BEV 进行旋转变换

```python
# 代码对应：transformer.py PerceptionTransformer.get_bev_features()
if self.rotate_prev_bev:
    for i in range(bs):
        rotation_angle = img_metas[i]['can_bus'][-1]  # 自车旋转角度
        tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle, center=self.rotate_center)
```

**时间注意力的设计特点**：
- 只使用**一帧历史 BEV**（当前帧 + 1 帧历史），`num_bev_queue = 2`
- 将历史 BEV 和当前 BEV 沿 batch 维度拼接，统一处理
- 最后通过 **mean 操作**融合历史与当前特征

```python
# 代码对应：temporal_self_attention.py TemporalSelfAttention.forward()
# 融合历史与当前 BEV
output = output.view(num_query, embed_dims, bs, self.num_bev_queue)
output = output.mean(-1)  # 平均融合
```

### 1.3 整体架构

```
多相机图像 (N×3×H×W)
     │
     ▼
┌──────────────┐
│  Backbone    │  ResNet-101 + FPN
│  + Neck      │  多尺度特征: 1/8, 1/16, 1/32, 1/64
└──────┬───────┘
       │ 多尺度图像特征
       ▼
┌──────────────────────────────────────────┐
│         BEVFormer Encoder                │
│  (共 6 层 BEVFormerLayer)                │
│                                          │
│  每层包含：                               │
│  ┌─────────────────────────────────┐     │
│  │ 1. Temporal Self-Attention      │     │
│  │    (当前BEV ←→ 历史BEV)         │     │
│  ├─────────────────────────────────┤     │
│  │ 2. Spatial Cross-Attention      │     │
│  │    (BEV Query ←→ 图像特征)      │     │
│  ├─────────────────────────────────┤     │
│  │ 3. Feed-Forward Network         │     │
│  └─────────────────────────────────┘     │
│                                          │
│  输出: BEV Embedding (H×W×C)             │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────┐
│    Detection Decoder         │
│  (共 6 层，DETR-style)       │
│                              │
│  Object Query (900个)        │
│  与 BEV Embedding 交互       │
│                              │
│  输出: 3D检测框 + 类别       │
└──────────────────────────────┘
```

### 1.4 BEVFormer 与 DETR3D 的关系

BEVFormer 可以看作是 **DETR3D 的改进版本**。DETR3D 也使用 object queries 与图像特征进行 3D 检测，但 BEVFormer 的关键改进在于：

| 特性 | DETR3D | BEVFormer |
|------|--------|-----------|
| 特征空间 | Object queries 直接与图像交互 | 先构建统一的 BEV 特征，再与 object queries 交互 |
| 时序建模 | 无 | 有（Temporal Self-Attention） |
| 可复用性 | 检测专用 | BEV 特征可被多个下游任务复用 |
| 空间先验 | 较弱 | 强（预定义的 BEV 网格） |

**BEV 特征的可复用性**是 BEVFormer 被 UniAD 选为核心模块的关键原因——同一份 BEV 特征可以同时服务于检测、跟踪、建图、运动预测、占用预测和规划等多个任务。

---

## 二、BEVFormer 在 UniAD 架构中的位置和作用

### 2.1 UniAD 整体架构回顾

UniAD 的类继承关系如下：

```
MVXTwoStageDetector (mmdet3d 基类)
    │
    └── UniADTrack        (跟踪模块)
         │
         └── UniAD        (完整多任务模型)
```

### 2.2 BEVFormer 是 UniAD 的"感知前端"

在 UniAD 中，BEVFormer 扮演着**统一感知前端**的角色，负责将多相机图像转换为统一的 BEV 特征表示。这个 BEV 特征随后被**所有下游任务共享**：

```
输入: 多帧多相机图像
        │
        ▼
┌─── BEVFormer (感知前端) ──────────────┐
│                                        │
│  BEVFormer Encoder: 图像→BEV特征       │
│  Detection Decoder:  BEV特征→检测框    │
│                                        │
│  输出:                                  │
│    - bev_embed  (BEV特征, 下游共享)     │
│    - track_query (跟踪查询)             │
│    - 3D检测框                           │
└──────────────┬─────────────────────────┘
               │
    ┌──────────┼──────────┬──────────────┬─────────────┐
    ▼          ▼          ▼              ▼             ▼
  Map       Motion      Occupancy     Planning
  (在线建图) (运动预测)  (占用预测)    (轨迹规划)
```

### 2.3 代码中的具体体现

在 [uniad_e2e.py:163-232](projects/mmdet3d_plugin/uniad/detectors/uniad_e2e.py#L163-L232) 的 `forward_train` 中可以看到这个流程：

```python
# Step 1: BEVFormer 生成 BEV 特征和跟踪结果
losses_track, outs_track = self.forward_track_train(...)

# Step 2: 提取 BEV 特征
bev_embed = outs_track["bev_embed"]  # ← BEVFormer 的输出
bev_pos   = outs_track["bev_pos"]

# Step 3-6: 下游任务共享同一份 BEV 特征
losses_seg, outs_seg = self.seg_head.forward_train(bev_embed, ...)     # 建图
ret_dict_motion = self.motion_head.forward_train(bev_embed, ...)       # 运动预测
losses_occ = self.occ_head.forward_train(bev_embed, ...)               # 占用预测
outs_planning = self.planning_head.forward_train(bev_embed, ...)       # 规划
```

### 2.4 BEVFormer 相关模块的代码组织

在 UniAD 的实现中，BEVFormer 的代码分布在以下文件中：

| 文件 | 功能 |
|------|------|
| [bevformer.py](projects/mmdet3d_plugin/uniad/detectors/bevformer.py) | BEVFormer 检测器（独立使用时） |
| [bevformer_head.py](projects/mmdet3d_plugin/uniad/dense_heads/bevformer_head.py) | BEVFormer 检测头（独立使用时） |
| [track_head.py](projects/mmdet3d_plugin/uniad/dense_heads/track_head.py) | **BEVFormerTrackHead**（UniAD 中实际使用的检测头） |
| [transformer.py](projects/mmdet3d_plugin/uniad/modules/transformer.py) | **PerceptionTransformer**（核心 Transformer 结构） |
| [encoder.py](projects/mmdet3d_plugin/uniad/modules/encoder.py) | **BEVFormerEncoder** + **BEVFormerLayer** |
| [spatial_cross_attention.py](projects/mmdet3d_plugin/uniad/modules/spatial_cross_attention.py) | **SpatialCrossAttention** + **MSDeformableAttention3D** |
| [temporal_self_attention.py](projects/mmdet3d_plugin/uniad/modules/temporal_self_attention.py) | **TemporalSelfAttention** |
| [decoder.py](projects/mmdet3d_plugin/uniad/modules/decoder.py) | **DetectionTransformerDecoder** + **CustomMSDeformableAttention** |

### 2.5 UniAD 对 BEVFormer 的适配修改

UniAD 没有直接使用原始的 BEVFormer 检测头，而是创建了 **BEVFormerTrackHead**（继承自 BEVFormerHead），主要改动：

1. **分离了 BEV 生成和检测解码**：
   - `get_bev_features()` — 只生成 BEV 特征（Encoder 部分）
   - `get_detections()` — 基于 BEV 特征进行目标检测（Decoder 部分）

   ```python
   # track_head.py
   def get_bev_features(self, mlvl_feats, img_metas, prev_bev=None):
       # 只调用 encoder，获取 BEV 特征
       bev_embed = self.transformer.get_bev_features(...)
       return bev_embed, bev_pos

   def get_detections(self, bev_embed, object_query_embeds, ref_points, img_metas):
       # 只调用 decoder，获取检测结果
       hs, init_reference, inter_references = self.transformer.get_states_and_refs(...)
       return outs
   ```

   这样设计的好处是：**跟踪模块（UniADTrack）可以在 Encoder 和 Decoder 之间插入轨迹更新逻辑**。

2. **增加了轨迹预测分支**：
   ```python
   # track_head.py: _init_layers()
   past_traj_reg_branch = nn.Sequential(...)
   self.past_traj_reg_branches = _get_clones(past_traj_reg_branch, num_pred)
   ```

3. **支持外部传入 query 和 reference points**：
   ```python
   def get_detections(self, bev_embed, object_query_embeds=None, ref_points=None, img_metas=None):
       # 允许外部传入的 track query 替换默认的 learnable query
   ```

### 2.6 训练流程中的调用链路

```
UniAD.forward_train()
  └─> UniADTrack.forward_track_train()
        └─> UniADTrack._forward_single_frame_train()  [逐帧循环]
              ├─> UniADTrack.get_bevs()
              │     └─> BEVFormerTrackHead.get_bev_features()
              │           └─> PerceptionTransformer.get_bev_features()
              │                 └─> BEVFormerEncoder.forward()  [6层 BEVFormerLayer]
              │                       └─> BEVFormerLayer.forward()
              │                             ├─> TemporalSelfAttention  [TSA]
              │                             └─> SpatialCrossAttention [SCA]
              │
              └─> BEVFormerTrackHead.get_detections()
                    └─> PerceptionTransformer.get_states_and_refs()
                          └─> DetectionTransformerDecoder.forward()  [6层 Decoder]
```

---

## 三、BEVFormer 中的知识点和注意事项

### 3.1 3D 参考点与相机投影

**知识点**：BEVFormer 的核心是将 3D 空间中的 BEV 网格点通过相机内外参投影到 2D 图像平面上。

**关键代码**：[encoder.py:91-144](projects/mmdet3d_plugin/uniad/modules/encoder.py#L91-L144) 的 `point_sampling()` 方法：

```python
# 1. 将归一化的 BEV 坐标恢复到真实世界坐标
reference_points[..., 0:1] = reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
# 2. 齐次坐标变换
reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
# 3. 通过 lidar2img 矩阵投影到相机平面
reference_points_cam = torch.matmul(lidar2img, reference_points)
# 4. 透视除法
reference_points_cam = reference_points_cam[..., 0:2] / reference_points_cam[..., 2:3]
# 5. 归一化到 [0, 1]
reference_points_cam[..., 0] /= img_shape[0][1]
reference_points_cam[..., 1] /= img_shape[0][0]
```

**注意事项**：
- `lidar2img` 矩阵是一个 $4 \times 4$ 的变换矩阵，将激光雷达坐标系下的点投影到相机像素坐标
- `bev_mask` 记录了哪些 3D 参考点在哪些相机中是可见的（投影点在前方且落在图像范围内）
- 投影时需要处理 `NaN` 和 `inf` 值（`torch.nan_to_num`）

### 3.2 BEV 坐标归一化与反归一化

**知识点**：BEV 空间中的坐标在多处使用归一化和反归一化操作。

- **归一化**：将真实世界坐标映射到 $[0, 1]$ 范围
  ```python
  # 归一化
  x_norm = (x - pc_range[0]) / (pc_range[3] - pc_range[0])
  ```
- **反归一化**：将 $[0, 1]$ 的归一化坐标恢复到真实世界坐标
  ```python
  # 反归一化
  x = x_norm * (pc_range[3] - pc_range[0]) + pc_range[0]
  ```
- **inverse_sigmoid**：由于输出是 sigmoid 后的值，需要使用 `inverse_sigmoid` 恢复
  ```python
  reference = inverse_sigmoid(reference)
  tmp[..., 0:2] += reference[..., 0:2]  # 残差加到 reference points 上
  tmp[..., 0:2] = tmp[..., 0:2].sigmoid()  # 再 sigmoid 回来
  ```

**注意事项**：
- `pc_range` 是 BEV 覆盖的范围，UniAD 中使用 `[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]`
- 每层 decoder 都使用 `inverse_sigmoid` → 残差相加 → `sigmoid` 的流程来迭代优化位置预测

### 3.3 可变形注意力（Deformable Attention）

**知识点**：BEVFormer 中的 SCA 和 TSA 都基于可变形注意力机制（来自 Deformable DETR）。

每个 query 学习预测 $K$ 个采样偏移量（sampling offsets），在 reference point 附近采样特征，而不是像标准注意力那样关注所有位置。

**关键参数**：
- `num_heads`: 8（多头注意力头数）
- `num_points`: 8（SCA 中每个 query 采样 8 个点）
- `num_levels`: 4（SCA 使用 4 个 FPN 层级）/ 1（TSA 使用 1 层 BEV 特征）

**代码实现**：[spatial_cross_attention.py:203-398](projects/mmdet3d_plugin/uniad/modules/spatial_cross_attention.py#L203-L398)

```python
# 采样偏移量预测
sampling_offsets = self.sampling_offsets(query).view(
    bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
# 注意力权重预测
attention_weights = self.attention_weights(query).view(
    bs, num_query, self.num_heads, self.num_levels * self.num_points)
attention_weights = attention_weights.softmax(-1)
# 采样位置 = 参考点 + 偏移量/归一化因子
sampling_locations = reference_points + sampling_offsets / offset_normalizer
```

**注意事项**：
- 偏移量的初始化使用三角函数生成网格化初始值，确保初始时采样点均匀分布在参考点周围
- CUDA 实现中 `dim_per_head` 应为 2 的幂次，否则效率较低（会有 warning）
- fp16 下的可变形注意力不稳定（因为涉及大量求和操作），代码中统一使用 fp32 计算

### 3.4 时序对齐（自车运动补偿）

**知识点**：在使用历史 BEV 特征时，需要补偿自车在两帧之间的运动。

**代码实现**：[transformer.py:101-155](projects/mmdet3d_plugin/uniad/modules/transformer.py#L101-L155)

```python
# 1. 计算自车在两帧之间的平移
delta_global = np.array([each['can_bus'][:3] for each in img_metas])
delta_lidar = np.linalg.inv(lidar2global_rotation) @ delta_global
shift_x = delta_lidar[:, 0] / real_w
shift_y = delta_lidar[:, 1] / real_h

# 2. 旋转历史 BEV
if self.rotate_prev_bev:
    rotation_angle = img_metas[i]['can_bus'][-1]
    tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle, center=self.rotate_center)
```

**注意事项**：
- `can_bus` 包含 18 个维度的信息，其中 `[:3]` 是自车平移，`[-1]` 是旋转角度
- 旋转角度信息来自 `can_bus`（IMU/里程计），而不是网络预测
- `rotate_center` 默认是 `[100, 100]`（对于 200×200 的 BEV 网格，中心就是 100,100）
- 平移以归一化单位的 `shift` 形式传入 BEVFormerEncoder，在 TSA 的 reference points 上加上偏移

### 3.5 CAN Bus 信息的注入

**知识点**：BEVFormer 将自车状态信息（CAN Bus 信号）通过 MLP 编码后注入到 BEV queries 中。

**代码实现**：[transformer.py:151-155](projects/mmdet3d_plugin/uniad/modules/transformer.py#L151-L155)

```python
can_bus = bev_queries.new_tensor([each['can_bus'] for each in img_metas])
can_bus = self.can_bus_mlp(can_bus)[None, :, :]  # MLP: 18 → 128 → 256
bev_queries = bev_queries + can_bus * self.use_can_bus
```

**注意事项**：
- `can_bus_mlp` 结构：`Linear(18, 128) → ReLU → Linear(128, 256) → ReLU → LayerNorm(256)`
- CAN Bus 信号被加到 BEV query 上，提供自车运动状态信息
- 通过 `use_can_bus` 标志控制是否启用（默认启用）

### 3.6 相机和层级嵌入

**知识点**：BEVFormer 为不同相机和不同 FPN 层级学习对应的嵌入向量。

```python
# 相机嵌入
self.cams_embeds = nn.Parameter(torch.Tensor(self.num_cams, self.embed_dims))
# 层级嵌入
self.level_embeds = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dims))
```

在图像特征上分别加上这两个嵌入：
```python
feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)  # 相机ID编码
feat = feat + self.level_embeds[None, None, lvl:lvl + 1, :].to(feat.dtype)  # 层级编码
```

**注意事项**：
- 相机嵌入让模型知道特征来自哪个相机视角，帮助处理多相机之间的几何关系
- 层级嵌入区分不同分辨率的特征图，帮助模型学习多尺度表示

### 3.7 GridMask 数据增强

**知识点**：BEVFormer 在训练时使用 GridMask 作为一种数据增强策略。

```python
self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
```

**注意事项**：
- GridMask 以 0.7 的概率随机遮挡图像中的网格区域
- 这是一个重要的正则化手段，防止模型过拟合特定的视觉模式
- 只在训练时使用，推理时关掉

### 3.8 历史 BEV 的获取（obtain_history_bev）

**知识点**：训练时，需要从历史帧中提取 BEV 特征。这个操作在 `torch.no_grad()` 下进行以节省显存。

```python
# bevformer.py:157-176
def obtain_history_bev(self, imgs_queue, img_metas_list):
    self.eval()  # 切换到 eval 模式
    with torch.no_grad():  # 不计算梯度
        prev_bev = None
        for i in range(len_queue):
            prev_bev = self.pts_bbox_head(..., prev_bev, only_bev=True)
    self.train()  # 切回训练模式
    return prev_bev
```

**注意事项**：
- 历史 BEV 的生成不参与梯度计算，节省显存
- 模型临时切换到 `eval()` 模式，然后切回 `train()`
- `only_bev=True` 表示只使用 encoder，不经过 decoder
- 在 UniAD 中，`UniADTrack.get_history_bev()` 实现了类似逻辑

### 3.9 推理时的时序信息管理

**知识点**：推理时，BEVFormer 维护一个 `prev_frame_info` 字典来管理跨帧的时序状态。

```python
self.prev_frame_info = {
    'prev_bev': None,       # 上一帧的 BEV 特征
    'scene_token': None,    # 场景标识
    'prev_pos': 0,          # 上一帧自车位置
    'prev_angle': 0,        # 上一帧自车角度
}
```

**推理流程**：
1. **场景切换时**：`prev_bev` 重置为 None（新场景开始时无历史信息）
2. **非视频模式**：`prev_bev` 始终为 None（不使用时序信息）
3. **时序模式**：计算自车运动增量，传入 encoder 用于 TSA 对齐

```python
# bevformer.py:252-258
tmp_pos = img_metas[0][0]['can_bus'][:3]  # 当前帧自车位置
tmp_angle = img_metas[0][0]['can_bus'][-1]  # 当前帧自车角度
if self.prev_frame_info['prev_bev'] is not None:
    img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']  # 计算平移增量
    img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']  # 计算旋转增量
```

**注意事项**：
- 场景切换时务必重置 `prev_bev`，否则会错误地使用上一场景的 BEV 特征
- 平移和旋转增量被填入 `can_bus` 中，在 `get_bev_features` 中用于运动补偿

### 3.10 BEV 位置编码

**知识点**：BEV 网格使用可学习的位置编码（Learned Positional Encoding）。

```python
# bevformer_head.py:104-105
self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)

# 配置中
positional_encoding=dict(
    type="LearnedPositionalEncoding",
    num_feats=_pos_dim_,      # 128
    row_num_embed=bev_h_,     # 200
    col_num_embed=bev_w_,     # 200
)
```

**注意事项**：
- 与标准 Transformer 使用正弦位置编码不同，BEVFormer 使用可学习的位置编码
- BEV 的位置编码在 `forward` 中通过 `self.positional_encoding(bev_mask)` 计算
- `bev_mask` 是全零的张量（所有位置都有效），实际位置信息来自可学习的嵌入

### 3.11 检测框的编码格式

**知识点**：BEVFormer 输出的检测框使用 10 维编码：

```
[cx, cy, w, l, cz, h, sin(heading), cos(heading), vx, vy]
```

其中：
- `cx, cy, cz`：3D 边界框的中心坐标
- `w, l, h`：宽、长、高
- `sin, cos`：朝向角的三角函数表示
- `vx, vy`：速度分量

**回归权重**：
```python
code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
```
速度和朝向角的权重较低（0.2），反映了这些参数的不确定性较高。

### 3.12 多层 Decoder 的迭代优化

**知识点**：BEVFormer 的 decoder 使用迭代优化的策略，每层都输出检测结果。

```python
# decoder.py:93-128
for lid, layer in enumerate(self.layers):
    output = layer(output, ...)
    if reg_branches is not None:
        tmp = reg_branches[lid](output)
        new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points[..., :2])
        new_reference_points = new_reference_points.sigmoid()
        reference_points = new_reference_points.detach()
```

**注意事项**：
- 每层 decoder 都使用独立的回归和分类分支（`self.cls_branches[lid]`, `self.reg_branches[lid]`）
- Reference points 在每层更新后 detach 后再传入下一层（`reference_points = new_reference_points.detach()`）
- 每层都计算损失（intermediate supervision），共 6 层 decoder 产生 6 组损失
- 训练时使用匈牙利匹配器（Hungarian Assigner）将预测框与 GT 框做匹配

### 3.13 显存优化技巧

**知识点**：BEVFormer 中使用了一些显存优化技巧。

1. **SCA 中的稀疏交互**：每个相机只与其可见的 BEV 查询交互
   ```python
   # 只保留可见的 BEV 查询
   index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
   ```

2. **历史 BEV 不计算梯度**：
   ```python
   with torch.no_grad():
       prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)
   ```

3. **配置中的 `queue_length` 参数**：
   ```python
   queue_length = 5  # 可调整为 3 以节省显存
   ```

### 3.14 从预训练权重加载

**知识点**：UniAD 的 stage1 训练从 BEVFormer 的预训练权重开始。

```python
# 配置中
load_from = "ckpts/bevformer_r101_dcn_24ep.pth"
```

**注意事项**：
- 预训练权重来自 BEVFormer 在 nuScenes 上的 24 个 epoch 训练
- UniAD 冻结 backbone（`freeze_img_backbone=True`），只训练 neck 和后续部分
- Backbone 使用 DCNv2（Deformable Convolution v2）增强

### 3.15 关键超参数总结

| 参数 | 值 | 说明 |
|------|-----|------|
| `bev_h`, `bev_w` | 200×200 | BEV 网格分辨率 |
| `embed_dims` | 256 | 特征维度 |
| `pc_range` | [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0] | 点云覆盖范围 |
| `num_points_in_pillar` | 4 | 每个 BEV 柱体的高度采样点 |
| `num_cams` | 6 | 相机数量（nuScenes） |
| `num_feature_levels` | 4 | FPN 层级数 |
| `num_heads` | 8 | 注意力头数 |
| `num_points` | 8 (SCA) / 4 (TSA) | 可变形注意力采样点数 |
| `num_layers` (encoder) | 6 | BEVFormerEncoder 层数 |
| `num_layers` (decoder) | 6 | Detection Decoder 层数 |
| `num_query` | 900 (+1 SDC) | Object queries 数量 |
| `num_bev_queue` | 2 | 时序 BEV 队列长度（当前+历史） |
| `queue_length` | 5 | 训练序列帧数 |
| `num_classes` | 10 | 目标类别数 |

---

## 四、总结

BEVFormer 是 UniAD 系统中的**核心感知模块**，它的主要贡献在于：

1. **统一的 BEV 特征表示**：将多相机图像转换为统一的鸟瞰视角特征，为下游多任务提供共享表示
2. **时空融合**：通过 Temporal Self-Attention 融合历史信息，提升检测的时序一致性
3. **几何感知**：通过 Spatial Cross-Attention 和 3D-2D 投影，显式建模几何关系
4. **可复用性**：BEV 特征可以被检测、跟踪、建图、运动预测、占用预测和规划等多个任务共享

在 UniAD 中，BEVFormer 被适配为 BEVFormerTrackHead，将 BEV 生成和检测解码分离，使得跟踪模块可以在两者之间插入轨迹更新逻辑，实现端到端的多目标跟踪。