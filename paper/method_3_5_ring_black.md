# Method 3.5 — Ring-Based Black Vote Correction

## 3.5.1 Motivation (核心洞察)

**问题**：MediaPipe Pose 端点在攀岩姿态下会偏移数 10 px（典型 49 px，0005s 帧
R_hand 真值 hold 中心 (250, 646) 距端点 (291, 672) 49 px）。EdgeSAM 以端点
为 prompt anchor 切 mask 时，会切到**端点所在的另一个 hold**，而真抓的黑 hold
在 mask 外 49 px 处被当背景切不进来 → 判色错。

**两个层级的失败**：
1. *候选层*：EdgeSAM 给出的 4 候选 mask 中，含黑 hold 的候选 #3
   (score 0.733, area 9925, 黑像素 39.9%)，但 score 最高的候选 #1
   (score 0.828, area 3550, 纯橙 79.8%) 切到端点附近的橙 hold。
2. *颜色采样层*：即使切到含黑 hold 的候选 #3，颜色采样
   `_sample_color` 系统性剔除低饱和像素 (`S < WHITE_SAT_MAX = 35`)，
   黑 hold 在暖光下 S ≈ 20-30 正好被砍 → 仅剩 25% 橙色像素撑颜色 →
   判橙 HSV=(17.5, 83, 117)。

**洞察**：黑 hold 在彩色墙上稀缺。如果 mask 外扩环带里黑票占比高，是
**有信息量的信号**——说明端点附近确实存在一块没被切进 mask 的黑 hold。
彩色 hold 在环带里占多是常态（无信息量）。因此 ring 黑票修正**只做
"非黑→黑"方向**，绝不做反向。

## 3.5.2 Algorithm

```python
def ring_black_vote(image_bgr, mask_bool, ix, iy, person_mask=None,
                    ring_d1=20, ring_d2=70, ratio_min=0.18,
                    min_pixels=100):
    """
    Compute ring-based black vote for black hold correction.

    Returns: (ratio, n_pixels) — ratio of black pixels in the ring
             outside the mask. If ratio >= ratio_min, the caller
             should correct color_name from non-black to black.
    """
    h, w = mask_bool.shape
    yy, xx = np.ogrid[:h, :w]
    dist = np.hypot(xx - ix, yy - iy)

    # Build ring: outside mask, radius [ring_d1, ring_d2]
    ring = (~mask_bool) & (dist >= ring_d1) & (dist <= ring_d2)

    # Exclude person mask pixels (climber body, not wall holds)
    if person_mask is not None:
        ring = ring & (~person_mask.astype(bool))

    n_ring = ring.sum()
    if n_ring < min_pixels:
        return 0.0, 0  # not enough samples → not trustworthy

    # Black criterion: V<60 OR S<35
    bgr = image_bgr[ring]
    hsv = cv2.cvtColor(bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    black_pixels = ((hsv[:, 2] < 60) | (hsv[:, 1] < 35)).sum()
    ratio = black_pixels / n_ring
    return ratio, n_ring
```

**Threshold 18% 来自实测**：

| Endpoint | Ring black ratio | Action |
|---|---|---|
| 0005s R_hand (需修) | 29.3% | 修正为黑 |
| 0005s L_hand (不可修) | 7.8% | 保持原判色 |

中间留余量取 18%。

## 3.5.3 Direction asymmetry 论证

设 `M` = mask, `R` = 环带 (outside M), `B` = 黑票集。

* **非黑→黑修正**（我们做的）：
  - 触发条件：mask 判色 ≠ 黑 AND R∩B / |R| ≥ 18%
  - 物理含义：黑 hold 稀疏，环带黑票高 = 端点附近有黑 hold 未被切进 mask
  - 错误率上界：环带里恰好有 18%+ 黑阴影（暖光下墙黑斑）→ 误修正
    但墙阴影 V<60 S<40 的区域通常小且散，难以达 18% 阈值

* **黑→非黑修正**（我们不做）：
  - 如果对称做：mask 判色 = 黑 AND R∩彩色 ≥ 50% → 修正为彩色
  - 物理问题：彩色 hold 在彩色墙上**密集分布**，环带彩色票高是常态
    无信息量。容易误把真黑 hold（mask 内）修正为彩色
  - L_hand 反例：mask 判黑正确，但环带黄票 58%——对称规则会立刻破坏
    正确结果

**结论**：环带投票对黑 hold 是**稀缺性信号**（信息量高），对彩色 hold 是
**常态噪声**（信息量低）。方向不对称是这个修正能 work 的根本原因。

## 3.5.4 Cost Analysis

- **零额外 EdgeSAM predict**：仅一次 numpy 像素投票
- 计算复杂度：O(|ring|) = O(π × (70² − 20²)) = O(13000 pixels)
- 实际耗时：~0.001s/端点 (CPU)
- **总开销**：相对完整 EdgeSAM forward ~1.5s 几乎可忽略

## 3.5.5 Failure Modes

1. **暖光阴影误触**：墙上大面积阴影 V<60 S<40，环带黑票 ≥ 18%
   - 缓解：阈值 18% 已留余量（实际阴影 V 中位 ~80，S 中位 ~50）
   - 实测 0/10 walls 误触
2. **黑色墙 vs 黑 hold**：黑色墙本身黑票就高，mask 判色是黑就不会触发
   非黑→黑修正，但 mask 本身就是黑 hold 包含
3. **多色混合环带**：hold 紧邻时，环带可能跨多种颜色
   - 黑 hold 在彩色墙上仍稀缺，环带黑票 ≥ 18% 触发条件基本不变

## 3.5.6 Algorithm Wrapper

```python
# In segment_endpoint(), after color_name from _sample_color:
ring_black_ratio, ring_n = _ring_black_ratio(
    self.last_image_bgr, chosen, ix, iy, person_mask=person_mask)
if (color_name != "黑"
        and ring_black_ratio >= RING_BLACK_RATIO_MIN  # 0.18
        and ring_n >= RING_BLACK_MIN_PIXELS):         # 100
    color_name = "黑"
    ring_black_fixed = True
```

## 3.5.7 Visualization (Fig 3)

画法：
1. 原图作底
2. 端点红点 + 白圈
3. mask 边界按端点颜色（高对比色）
4. **环带**：仅画内外两个圆 (RING_BLACK_D1=20, D2=70)：
   - 修正命中：黄圈 + 黑边
   - 未命中：白圈（低调）
5. Header 多一行图例：`[环带黑修正 xN] 黄环=统计区(20-70px) 黑票>=18%->判黑`
6. 端点标签前缀：`[ring黑修正 29.3%]` 或 `[ring黑 7.8%<18%]`

样例：0005s R_hand 端点 (291, 672)，mask 切到橙色 hold (cand#1)，
环带黑票 29.3% → 修正为黑。R_hand 标签变为 `R_hand->黑 s=0.83 a=3550 [ring黑修正 29.3%]`。

复用素材：`docs/demo/wall_01_black_ringfix.jpg`（0005s R_hand 已带环带可视化）。
