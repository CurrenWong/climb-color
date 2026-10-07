# baselines/ — 论文对比基线

论文 Table 1 里"颜色空间"一侧的两个基线在这里；第三个对比项（通用视觉大模型
Qwen-VL）见 [`../vlm_exp/`](../vlm_exp/)。

| 文件 | 说明 | 墙段准确率 |
|---|---|---|
| `detect_holds_walls.py` | v1 基线：MediaPipe Pose + detect_holds（HSV 连通域）+ 端点投票 | 6/10 |
| `hsv_summary.txt` | 仅 HSV（端点环向采样，不做岩点分割）的逐墙结果 | 4/10 |
| `v1_summary.txt` | v1 detect_holds 的逐墙结果 | — |

`detect_holds_walls.py` 加 `--hsv-only` 即为"仅 HSV"那一档。

## 用法

从**项目根**运行（脚本用相对路径读 `img_xhs_walls/`）：

```bash
python baselines/detect_holds_walls.py          # 全部 10 墙段
python baselines/detect_holds_walls.py 2        # 前 2 个墙段（冒烟）
python baselines/detect_holds_walls.py --hsv-only
```

结果写到本目录（`hsv_summary.txt` / `v1_summary.txt`）。

> 两个 summary 是 2026-10-06 在 10 墙段视频集（65 帧）上的实测输出，
> 对应论文 Table 1 的 4/10 与 6/10。
