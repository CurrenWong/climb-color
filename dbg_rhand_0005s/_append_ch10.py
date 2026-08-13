# -*- coding: utf-8 -*-
"""追加 LESSONS_LEARNED.md 第 10 章"""
from pathlib import Path

p = Path(r'D:\Project\climbing_wall_search\LESSONS_LEARNED.md')
text = p.read_text(encoding='utf-8')

chapter = """

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
- `mediapipe2/edgesam_holds.py` — 落地代码（`_ring_black_ratio` + `segment_endpoint` 修正）
- `mediapipe2/test_edge_sam.py` — 打印 ring 修正信息
- `mediapipe2/dbg_rhand_0005s/exp3*.py` — 实验 3 全系列（probe/multipoint/multi_anchor/directional/thumb/ring/wall_final）
- `mediapipe2/dbg_rhand_0005s/exp3i_visual_ringfix.png` — 判断逻辑可视化（L_hand/R_hand 对比）
"""

if '## 10. 环形带黑修正' not in text:
    p.write_text(text.rstrip('\n') + '\n' + chapter, encoding='utf-8')
    print('appended chapter 10')
else:
    print('chapter 10 already exists')
