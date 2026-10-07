# LESSONS — EdgeSAM / DWPose 路线工程经验（climb-color）

> 本文件收录 **climb-color 项目**（EdgeSAM 攀岩线路色识别）的工程经验，
> 对应主仓库 `LESSONS_LEARNED.md` 的第 10-13 章。
> 其它路线（VLM、MediaPipe 传统 CV、FastSAM、MobileSAM/V2）的经验见主仓库。
>
> 章节：
> - 第 10 章 环形带黑修正（核心创新）
> - 第 11 章 YOLO26 pose 替代实验（否决）
> - 第 12 章 DWPose 替代实验
> - 第 13 章 DWPose ONNX NMS 性能 bug（12x 加速）

---

## 10. 环形带黑修正：端点偏移导致黑 hold 漏切的零成本修复 (2026-08-13)

### 10.1 触发与结论一句话

用户报告 2X_01 黑线 0005s 帧：R_hand 端点判橙，但真值黑（黑 hold 被灯光打亮）。
根因是 **MediaPipe 端点位置偏了 49px**（端点 (291,672) vs 黑岩点 (250,646)），
EdgeSAM 以端点为锚切 mask 只切到"端点所在的橙 hold"，真黑 hold 在 mask 外 49px 被当背景。
实验 3 系列验证了各种"找黑岩点"方案后，最终落地的方案是一个**零额外 predict 的像素级修正**：

**mask 判色非黑时，统计 mask 外扩 20-70px 环形带的黑票占比，>= 18% 就修正判定为黑。**

- 全量 7/7 已知线路命中（基线 6/7），**修复 2X_01 黑线**（橙 12→黑 13），其余墙零破坏。
- 成本：纯 numpy 像素操作，毫秒级，不新增 EdgeSAM predict。

### 10.2 问题本质：端点偏移 49px + SAM 语义"正点锚定"

- 0005s 帧 R_hand：MediaPipe wrist (291,672)，真黑岩点中心 (250,646)，偏差 49px。
- EdgeSAM 单点分割（无论 box 调多大、候选数调多少）都不会切出 49px 外的黑岩点——
  **SAM 的语义是"切正点所在的物体"**，黑岩点在 mask 外被当背景。
- 实验数据：单点 4 候选黑岩点覆盖 0px；box_half 80→150 全部 0px；
  num_multimask_outputs 6 直接 ValueError（EdgeSAM 只支持 3-4）。

### 10.3 为什么"多锚点方案"被否掉（速度 + 破坏）

| 方案 | 成本 | 0005s 修复 | 全量破坏 |
|---|---|---|---|
| C: 8 方向偏移锚点独立分割 + 多数投票 | 9x predict | ✅ | ❌ 3O wall_03 红→蓝、line2 粉红→红 |
| 多正点（端点+左+左上，1 predict） | 0x | ✅ (R) | ❌ L_hand 黑→黄（偏移点落黄 hold） |
| 拇指方向正点（姿态自适应） | 0x | ❌ top 3橙1黑 | ❌ L_hand 破坏 |
| **环形带黑修正（落地）** | **0x** | **✅** | **✅ 7/7 零破坏** |

教训：
- **"多锚点 + 固定方向"在单帧有效，但方向猜错就破坏其他墙**——固定方向不是通用方案。
- 姿态几何（前臂方向/拇指方向）在 R_hand 上有效（拇指水平向左正好指向黑岩点），
  但 L_hand 拇指朝上，偏移点落进黄 hold 就破坏了——**方向信息本身不可靠**。
- 唯一不破坏的通用思路是：**不猜方向，直接看"端点周围是否存在一块没被切进 mask 的黑 hold"**。

### 10.4 为什么只在"非黑→黑"方向生效（不对称性）

核心：**"环带黑票高"是有信息量的信号，"环带彩色票高"基本没有信息量**。

- 黑色 hold 在彩色岩壁上稀缺：环带黑票 29%（R_hand）意味着"附近确实有一块黑 hold 没被切进 mask"——强证据。
- 彩色 hold 在环带里占多是常态（墙就是彩色 hold 拼的）：L_hand 环带黄票 58%，
  但 L_hand mask 判黑是**正确的**（黑票 1954 高置信）。若对称规则"环带 top 色≠mask 色就修正"，
  L_hand 会被立刻改成黄 → 破坏。
- 彩色→彩色修正同样危险：mask 是 EdgeSAM 精确切的 hold，判色有 mask 支撑；
  环带是"端点附近随便一块区域"，top 色只代表"旁边哪个 hold 最大"。
- 黑色分类本身更稳：低饱和 + 低明度，不依赖色相，不像橙/黄/红容易互相污染。
- **恰好命中原始场景**：灯光主要伤害暗色 hold（黑 hold 被照亮后 V 上升，mask 内被判成
  白/未知暗），但环带里黑 hold 边缘/阴影仍保持低 V → 黑票照样高。彩色 hold 受灯光影响小。

### 10.5 实现细节（已落地 edgesam_holds.py）

```python
RING_BLACK_D1 = 20          # 环形带内半径 (px)
RING_BLACK_D2 = 70          # 环形带外半径 (px)
RING_BLACK_RATIO_MIN = 0.18 # 环带黑票占比下限
RING_BLACK_MIN_PIXELS = 100 # 环带有效像素下限
```

- `segment_endpoint(..., ring_black_fix=True)`：mask 判色非黑时，
  `_ring_black_ratio()` 算环带黑票占比，>=0.18 修正为黑。
- 返回 dict 加 `ring_black: {ratio, n_pixels, fixed}` 字段；test_edge_sam.py 打印 `[ring黑修正 xx%]`。
- 阈值来源：R_hand(需修) 29.3% vs L_hand(不可修) 7.8%，0.18 取中间留余量。
- **想泛化到彩色方向**需要"稀缺颜色检测"（环带某色占比高 + 整墙占比低），
  成本上升且目前唯一验证收益场景就是黑色，暂不做。

### 10.6 全量回归（生产代码，7/7 命中）

| wall | 真值 | 生产判定 | 修正票 |
|---|---|---|---|
| 2X_01 | 黑 | **黑(13)✅**（基线橙12❌） | 9 |
| 2X_02 | 蓝 | 蓝✅ | 0 |
| 2X_03 | 粉红 | 粉红✅ | 4 |
| 2X_04 | 青 | 青✅ | 1 |
| 3O wall_03 | 红 | 红✅ | 0 |
| 3O wall_05 | 绿 | 绿✅ | 3 |

### 10.7 相关文件
- `climb-color/edgesam_holds.py` — 落地代码（`_ring_black_ratio` + `segment_endpoint` 修正）
- `climb-color/test_edge_sam.py` — 打印 ring 修正信息
- `climb-color/dbg_rhand_0005s/exp3*.py` — 实验 3 全系列（probe/multipoint/multi_anchor/directional/thumb/ring/wall_final）
- `climb-color/dbg_rhand_0005s/exp3i_visual_ringfix.png` — 判断逻辑可视化（L_hand/R_hand 对比）


## 第十一章：YOLO26 pose 替代 MediaPipe Pose 实验（否决，保持 MediaPipe）

用户提议用更好的 pose 模型（yolo26 pose）优化端点精度。实验结论：**不能优化，4/8 → 5/8 命中（MediaPipe 基线 7/7），否决**。

### 11.1 动机与假设

- 2X_01 黑线根因是 MediaPipe R_hand 端点偏 49px（0005s：端点 (291,672) vs 黑岩点 (250,646)），
  ring 黑修正兜底。若 YOLO26 pose 端点更准，可能从源头解决。

### 11.2 端点映射（COCO 17 点 vs MediaPipe 33 点）

| 端点 | MediaPipe 索引 | YOLO 索引 |
|---|---|---|
| L_hand | 15 (wrist) | 9 (left_wrist) |
| R_hand | 16 (wrist) | 10 (right_wrist) |
| L_foot | 31 (foot_index 脚尖) | **15 (ankle 脚踝)** ⚠️ |
| R_foot | 32 (foot_index 脚尖) | **16 (ankle 脚踝)** ⚠️ |

**YOLO 没有脚尖关键点，只有脚踝** — 攀岩踩点用脚尖，这是结构性缺陷。

### 11.3 单帧 0005s 端点差

| 端点 | 位置差 | 备注 |
|---|---|---|
| L_hand | 19.5px | YOLO 判黑直接命中（MP 判橙❌） |
| R_hand | 93.6px | YOLO conf=0.19 极低 |
| L_foot | 117.0px | ankle vs 脚尖 |
| R_foot | 71.3px | ankle vs 脚尖 |

### 11.4 全量回归对比（2 视频 8 个已知真值墙段）

| 墙段 | 真值 | MP 基线 | YOLO conf0.3 | YOLO conf0.05 |
|---|---|---|---|---|
| 2X_01 | 黑 | 黑✅(ring修) | 橙❌ | 橙❌ |
| 2X_02 | 蓝 | 蓝✅ | 蓝✅ | 蓝✅ |
| 2X_03 | 粉红 | 粉红✅ | 粉红✅ | 粉红✅ |
| 2X_04 | 青 | 青✅ | 青✅ | 青✅ |
| 3O wall_03 | 红 | 红✅ | 红✅ | 红✅ |
| 3O wall_04 line1 | 白 | 白✅ | 青❌ | **白✅** |
| 3O wall_04 line2 | 粉红 | 粉红✅ | 红❌ | 红❌ |
| 3O wall_05 | 绿 | 绿✅ | 黑❌ | 黑❌ |
| **命中** | | **7/7** | **4/8** | **5/8** |

### 11.5 失败根因

1. **YOLO hand 关键点置信度普遍低**：R_hand conf 常见 0.1-0.26（抓握/遮挡），
   conf=0.3 阈值大量丢票；放到 0.05 后票回来了但位置不准。
2. **无脚尖关键点**：L/R_foot 用 ankle，位置偏 71-117px，直接破坏踩点处判色
   （3O_05 绿线 R_foot 黑票暴涨 → 判黑 miss）。
3. **YOLO pose 推理非确定**：0005s R_hand 单帧实验判黑、conf0.05 全量实验判白，
   同端点多次推理结果不一致，比 FastSAM 的分割非确定性更致命。
4. 0005s L_hand YOLO 判黑成功（MP 判橙）是孤立亮点，撑不起整体命中率。

### 11.6 结论

- **保持 MediaPipe Pose**：7/7 命中 + ring 黑修正兜底已是可靠状态。
- YOLO26 pose 手部更准的假设不成立（置信度低 + 非确定），脚部结构性缺脚尖。
- 若未来想再试：yolo26m-pose / s-pose 可能改善 conf 但不解决"无脚尖 + 非确定"。
- 产物：`exp_yolo26pose_0005s.py`（单帧对比）、`exp_yolo26pose_full.py`（全量回归，
  YOLO_POSE_CONF 环境变量开关）、`yolo26pose_0005s_compare.jpg/json`、
  `yolo26pose_full_summary.txt`、`yolo26pose_full_conf005.log`、`yolo26n-pose.pt`（7.51MB）。


## 第十二章：DWPose 替代 MediaPipe Pose 实验（否决，保持 MediaPipe）

YOLO26 否决后用户又要求试 DWPose（RTMPose 系，133 点 COCO-WholeBody，含**脚尖 big_toe**）。
实验结论：**6/8 命中，优于 YOLO26（5/8）但仍不如 MediaPipe（7/7），否决**。

### 12.1 DWPose 优势假设

- 133 点含 `big_toe`（17/20）和 `heel`（19/22）——**正好补上 YOLO26 缺脚尖的结构性缺陷**，
  攀岩踩点（foot_index）有了真正对应物。
- RTMPose 架构对遮挡/抓握的手部理论上比 YOLO pose 更稳。
- 若成立，可从源头解决 MediaPipe 端点偏移（49px）问题，替代 ring 黑修正。

### 12.2 环境搭建（ONNX 路线，无 mmpose）

- 不装 mmcv/mmpose 全家桶，直接走 **onnxruntime + lllyasviel ControlNet 生态的 ONNX 模型**：
  - `yolox_l.onnx`（206.7MB，人体检测，输出 [1,8400,85]）
  - `dw-ll_ucoco_384.onnx`（128.1MB，133 点 pose，输出 simcc_x/simcc_y，输入 288x384）
- 参考 comfyui_controlnet_aux 的 `cv_ox_det.py` / `cv_ox_pose.py` 提取纯 numpy 版
  （`dwpose_onnx.py`，零 torch 依赖），只依赖 cv2+numpy+onnxruntime。
- ⚠️ **yolox_l.onnx 输出是 logits**（obj/cls 未 sigmoid）：直接 `obj*cls` person 最高 0.21
  检不出人；必须先 sigmoid（`1/(1+exp(-x))`）才正常（person top 0.39，检出主框）。
  两个源（fashn-ai / yzd-v）的 yolox_l.onnx 内容完全相同。
- ⚠️ **detect 按面积取最大人框会选到近全图假框** → pose 全废（scores 0.05-0.52）；
  改按**置信度取最高分框**后 133 点全部 >0.3（L_wrist 0.72, L_big_toe 0.50）。

### 12.3 端点映射（133 点 COCO-WholeBody vs MediaPipe 33 点）

| 端点 | MediaPipe 索引 | DWPose 索引 |
|---|---|---|
| L_hand | 15 (wrist) | 9 (left_wrist) |
| R_hand | 16 (wrist) | 10 (right_wrist) |
| L_foot | 31 (foot_index 脚尖) | **17 (big_toe 脚尖)** ✅ |
| R_foot | 32 (foot_index 脚尖) | **20 (big_toe 脚尖)** ✅ |
| body_center | 肩(11,12)+髋(23,24) | 肩(5,6)+髋(11,12) |

### 12.4 全量回归对比（2 视频 8 个已知真值墙段）

| 墙段 | 真值 | MP 基线 | YOLO26 | DWPose |
|---|---|---|---|---|
| 2X_01 | 黑 | 黑✅(ring修) | 橙❌ | **黑✅**（无 ring 修） |
| 2X_02 | 蓝 | 蓝✅ | 蓝✅ | 蓝✅ |
| 2X_03 | 粉红 | 粉红✅ | 粉红✅ | 粉红✅ |
| 2X_04 | 青 | 青✅ | 青✅ | 青✅ |
| 3O wall_03 | 红 | 红✅ | 红✅ | 红✅ |
| 3O wall_04 line1 | 白 | 白✅ | 白✅ | 黄❌ |
| 3O wall_04 line2 | 粉红 | 粉红✅ | 红❌ | 红❌ |
| 3O wall_05 | 绿 | 绿✅ | 黑❌ | **绿✅** |
| **命中** | | **7/7** | **5/8** | **6/8** |

### 12.5 关键亮点：0005s 黑线直接命中（无需 ring 修正）

- 0005s 帧（2X_01 黑线，MediaPipe R_hand 偏 49px 的老问题）：
  - MediaPipe: L_hand 判橙❌（需 ring 修）、R_hand 判黑（ring 修 23%）
  - DWPose（带 person_mask）: **L_hand 黑 + R_hand 黑直接命中**（无 ring 修正）
- 说明 DWPose 端点精度确实优于 MediaPipe——但只在 wall_01 这一个墙体现。

### 12.6 失败根因（为何 6/8 仍 < 7/7）

1. **手部/脚部关键点在多数帧置信度仍不稳**：wall_01 8 帧中 4 帧有端点
   conf<0.3 或非岩点（0000s R_hand 0.24、0010s 双 foot 0.21/0.27、0015s R_hand 0.35、0020s L_hand 0.06）。
2. **3O wall_04 line1（白）退化**：MP/YOLO26 都命中，DWPose 判黄（4/17 票 24% 弱聚合）——
   DWPose 端点落在黄 hold 上导致整墙偏黄，比 MP 的白 hold 命中更差。
3. **3O wall_04 line2（粉红）同 YOLO26 判红**：粉红/红边界色，DWPose 端点位置偏移
   把 mask 切到红色 hold（票 红5:粉红3），与 MP 的粉红命中差一档。
4. **速度成本**：yolox 检测 + 133 点推理 ~6.6s/帧（CPU），比 MediaPipe 单帧慢一个量级，
   无法用于实时/批量场景。

### 12.7 结论

- **保持 MediaPipe Pose**：7/7 命中 + ring 黑修正兜底仍是当前最优。
- DWPose > YOLO26（6/8 vs 5/8）：**脚尖 big_toe 确实救回 2X_01 黑 + 3O_05 绿**，
  证明"有脚尖"的方向是对的；但端点置信度不稳 + 白/粉红两墙退化，整体仍输 MP。
- 若未来想再试：`dw-ll_ucoco` 已是最强 DWPose onnx；可考虑 batch 检测多候选
  或调 person 检测阈值，但收益预期有限——根因是 133 点中 hand/foot 在抓握遮挡下
  本身就置信度低，换模型解决不了遮挡问题。
- 产物：`dwpose_onnx.py`（纯 numpy onnx 推理模块，可复用）、`exp_dwpose_0005s.py`（单帧对比）、
  `exp_dwpose_full.py`（全量回归，DW_POSE_CONF 环境变量开关）、`dwpose_0005s_compare.jpg/json`、
  `dwpose_full_summary.txt`、`dwpose_full_run.log`、`yolox_l.onnx` + `dw-ll_ucoco_384.onnx`（权重）。

---

## 第十三章：DWPose ONNX NMS 性能 bug —— 单类 NMS 带来 12x 加速 (2026-08-13)

### 现象

做 ablation 实验时发现 DWPose 全量回归从 8.06 s/帧 变成 **48.8 s/帧**（慢 6 倍）。
同版本代码、同机器、CPU 占用仅 13-24%（16 逻辑核不饱和）。

### 诊断链（三步定位）

| 步骤 | 脚本 | 发现 |
| --- | --- | --- |
| 1. 分阶段计时 | `debug/bench_frame.py` | `DWPose.run = 43.1 s`，而 EdgeSAM 4 端点仅 2.4 s、person mask 1.8 s、set_image 1.1 s（都正常） |
| 2. 裸 session 对照 | `debug/bench_ort.py` | onnxruntime 直接跑 yolox_l：**0.36 s** → 不是推理慢 |
| 3. detect 内部拆解 | `debug/bench_detect_stages.py` | **`_multiclass_nms = 41.93 s`**，输出 160075 个检测框 |

### 根因

`dwpose_onnx.DWPoseONNX.detect()` 调用 `_multiclass_nms(boxes, scores, nms_thr, score_thr)`，
该函数对 **80 个 COCO 类别逐个**执行：

```python
for cls_ind in range(num_classes):        # 80 次
    valid = cls_scores > score_thr        # 0.1
    keep = self._nms(vboxes, vscores, nms_thr)   # 纯 Python while 循环 NMS
```

问题在于 **阈值 0.1 太低 + yolox 输出 sigmoid 后概率普遍 > 0.1**，每类过滤后
仍剩数千框，80 类的 Python `while` NMS 累积 41.9 s，输出 16 万框。

### 修复

我们只关心 person（COCO class 0）。改为只对 class 0 做一次 NMS：

```python
cls0 = 1.0 / (1.0 + np.exp(-predictions[:, 5]))   # person logit → prob
scores0 = obj * cls0
valid = scores0 > DET_SCORE_THR
keep = self._nms(boxes_xyxy[valid], scores0[valid], DET_NMS_THR)
final_boxes = boxes_xyxy[valid][keep]
final_scores = scores0[valid][keep]
```

**结果**：`detect` 34.5 s → **3.39 s**（12x），全量 65 帧 496.5 s（7.64 s/帧），
与修复前命中率完全一致（10/10）。

### 为什么之前没发现

这个性能坑被 **sys.path 顺序 bug**（第九章后续）掩盖：DWPose 脚本此前实际加载
根目录的 `dwpose_onnx.py`，修好 sys.path 让 `dwpose_exp/` 本地副本真正生效后才暴露。

### 可复用经验

1. **性能回归先问"加载的是哪份代码"**：多份同名副本 + sys.path 顺序问题会让
   一个改动看似"引入性能退化"，实际只是加载路径变了。
2. **分阶段计时 + 裸算子对照**：`bench_frame`（阶段）→ `bench_ort`（裸 kernel）
   → `bench_detect_stages`（函数内部）三步可定位到具体函数行。
3. **通用检测器只取所需类别**：COCO 80 类的通用后处理在只需 1 类时是纯粹浪费；
   ultralytics/mmdet 内部都有"按类过滤后 NMS"的优化，手写实现时要注意。
4. **纯 Python NMS 在数千框规模上不可用**：O(n²) 的 while 循环在 2000+ 框时
   即达秒级；生产代码应使用 `cv2.dnn.NMSBoxes` 或向量化实现。
