# -*- coding: utf-8 -*-
"""追加 LESSONS 第 11 章（YOLO26 pose 实验）"""
from pathlib import Path

content = """

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
"""

p = Path(r"D:\Project\climbing_wall_search\LESSONS_LEARNED.md")
with open(p, "a", encoding="utf-8") as f:
    f.write(content)
print("appended", len(content), "chars")
