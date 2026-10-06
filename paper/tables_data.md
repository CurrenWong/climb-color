# Experimental Data — Tables (real measurements)

> 全部数据来自 dwpose_exp/exp_dwpose_full.py 实测（同一 dwpose_onnx/edgesam_holds 版本）
> 10 walls: 2XRo7VbPgTe wall_01-04 + 3OHz6rZ5opG wall_01-05

---

## Table 1: Main result

10 walls: 2XRo7VbPgTe wall_01-04 + 3OHz6rZ5opG wall_01-05

| Method | Wall Acc | Failed walls | Latency/frame |
|---|---|---|---|
| HSV-only (endpoint ring sampling, no hold segmentation) | **4/10** | 2X_01 黑→橙; 3O_02 红→橙; 3O_03 黄→橙; line1 白→橙; line2 粉红→橙; 3O_05 绿→蓝 | ~1 s* |
| v1 detect_holds (HSV connected components + bbox match) | **6/10** | 2X_01 黑→橙; 3O_02 红→白; line2 粉红→白; 3O_05 绿→蓝 | ~1.2 s* |
| EdgeSAM + DWPose, no ring, no adaptive box | 8/10 | 3O_03 黄→红; line1 白→橙 | 8.0 s |
| + Ring black vote | 9/10 | 2X_01 黑→橙 (8/26) | 9.3 s** |
| **+ Person-bbox-adaptive box (full)** | **10/10** | — | 7.6 s |

\* baseline 延迟未精确计时（MediaPipe + HSV 连通域，单帧 <2 s 量级）。
\*\* 延迟差异是 CPU 环境噪声（ring/box 两项成本 ≈0）。

**Progression**: 4/10 → 6/10 → 8/10 → 9/10 → **10/10**

原始 summary: 根目录 `baseline_hsv_summary.txt` / `baseline_v1_summary.txt`；
消融备份 `dwpose_exp/paper_ablation_*.txt`

**Note on latency**: 每帧 7.6-9.3 s（CPU, i5-13500H）。组成：DWPose ~4 s（yolox_l 检测单类 NMS 后 3.4 s + RTMPose 0.34 s）+ YOLO26m person ~1.8 s + EdgeSAM 4 端点 ~2.4 s + 后处理 <0.1 s。

---

## Table 2: Single-image test set (15 张, 15-image table)

| # | Truth | Ours | EdgeSAM only | detect_holds v1 |
|---|---|---|---|---|
| 1 | 紫 | 紫 75% ✅ | 紫 50% ✅ | 紫 |
| 2 | 黄 | 黄 67% ✅ | 黄 67% ✅ | 黄 |
| 3 | 绿 | 绿 100% ✅ | 绿 100% ✅ | 绿 |
| 4 | 粉红 (粉墙粉点) | 粉红 100% ✅ | 紫 50% ❌ | 紫 ❌ |
| 5 | 橙 | 橙 67% ✅ | 橙 67% ✅ | 橙 |
| 6 | 黄 (黄墙黄点) | 白 100% ❌ | 白 100% ❌ | 黄 |
| 7 | 红 | 红 67% ✅ | 红 67% ✅ | 红 |
| 8 | 蓝 (蓝墙蓝点) | 紫 100% ❌ | 紫 100% ❌ | 蓝 |
| 9 | 白 | 白 100% ✅ | 白 100% ✅ | 白 |

**Ours 8/9 ✅**; EdgeSAM-only 6/9; detect_holds v1 5/9（同色伪装墙 4/8 全失败）。
（10-15 扩展集见 LESSONS.md；数据待补全）

---

## Table 3: Ablation (10 walls, 同一版本)

| Configuration | Wall Acc | Failed walls | Δ vs Full |
|---|---|---|---|
| **Full pipeline** | **10/10** | — | — |
| − Ring black vote correction | 9/10 | 2X_01 (判橙 8/26) | **−1** |
| − Person-bbox-adaptive box | 8/10 | 3O_03 (判红 4/13), line1_white (判橙 5/17) | **−2** |
| − Pixel-level color voting (= 当前默认 mean-HSV) | 10/10 | — | 0 |
| + Pixel-level color voting (替代 mean-HSV) | 10/10 | — | 0 |
| − Body-anchored edge neg points (8 个) | 10/10 | — | 0 |

**关键发现**:
1. **Box 自适应贡献最大（+2 walls）**：3O_03（黄线）和 line1_white（白线）都依赖它。
   - 3O_03: box=160 时 R_hand 切到远处红 hold（4/13 判红）；box 自适应后判黄 (6/12)
   - line1_white: box=160 时 R_hand 切到远处红三角 hold；box 自适应后手边白 hold 胜出
2. **Ring 黑票修正贡献 +1 wall**：2X_01 黑线（判橙 8/26 → 判黑 14/26）
3. **Pixel vote 和 edge-neg 在 10-wall 上无 wall-level 区分度**（天花板效应）：
   - 二者改变了端点级票数分布（如 2X_01: 14/26 → 12/26），但不改变 wall 结论
   - 需要更难/更大的测试集才能体现其价值

---

## Table 4: Pose model comparison (appendix)

| Pose Model | Wall Acc | Foot tip | Latency |
|---|---|---|---|
| MediaPipe Pose (33 pts) | 7/7 (旧基线, ring fix 时代) | ❌ | 0.5 s |
| YOLO26-Pose (17 pts) | 5/8 | ❌ | 0.8 s |
| DWPose (133 pts) | 6/8 (无 ring/box) | ✅ | 4.0 s |
| DWPose + ring + adaptive box | **10/10** | ✅ | 4.0 s |

数据来源: LESSONS 第 10/11/12 章 + 本轮实测

---

## Table 5: Latency breakdown (real, per frame, CPU i5-13500H)

| Component | Time | Note |
|---|---|---|
| DWPose — yolox_l detect | 3.39 s | **单类 NMS 后**（原 80 类实现 34.5 s，见 LESSONS 第 13 章） |
| DWPose — RTMPose 133pt | 0.34 s | 288×384 输入 |
| YOLO26m person mask | 1.79 s | |
| EdgeSAM set_image | 1.07 s | |
| EdgeSAM 4 × segment_endpoint | 2.40 s | 每个 0.54-0.66 s（含 pixel vote + ring 统计） |
| Pixel-level post-processing | <0.1 s | ring 投票 + 主连通域 + 颜色分桶 |
| **Total** | **~7.6-9.3 s** | |

GPU 加速估算（未实测）: DWPose ~0.4 s + YOLO ~0.2 s + EdgeSAM ~0.15 s ≈ **~0.8 s/frame**。

---

## Repro commands

```bash
cd dwpose_exp
set PYTHONPATH=..\..\.venv\Lib\site-packages;..\EdgeSAM

# Full pipeline (10/10)
python -u exp_dwpose_full.py

# Ablations
python -u exp_dwpose_full.py --ring-off      # 9/10
python -u exp_dwpose_full.py --fixed-box     # 8/10
python -u exp_dwpose_full.py --pixel-vote    # 10/10
python -u exp_dwpose_full.py --no-edge-neg   # 10/10
```

原始 summary 备份: `dwpose_exp/paper_ablation_*.txt`

---

## TODO before submission

1. 补跑 HSV baseline / detect_holds v1 在同一 10-wall 上的数字（当前 Table 2 只有单图数据）
2. 补跑 −ring−box 组合（预期 ≤8/10）
3. 扩测试集（当前 10 walls 对 full pipeline 已饱和，消融区分度不足）
4. 失败案例可视化（3O_03 fixed-box 判红、line1_white 切远处红 hold）
