# tools/ — 人工核对与校准工具

不属于主流程，供人工查看/调参时使用。主流程只有 `../test_edge_sam.py`。

| 文件 | 说明 |
|---|---|
| `test_mp.py` | v1 主程序（MediaPipe Tasks API + 线路色识别），保留作对照 |
| `calibrate.py` | 实时校准工具：4 个 trackbar，调绿→紫 的边界 |
| `show_points.py` | 只标 6 个接触点（无骨架、无 header），肉眼核对端点位置 |

## 用法

从**项目根**运行（脚本按相对路径读 `img/`，并依赖根目录的 `route_color.py` 等）：

```bash
python tools/show_points.py        # 输出 points_*.jpg
python tools/calibrate.py          # 交互式：N 切图 / S 保存 / ESC 退出
python tools/test_mp.py --img img/1.jpg
```

> 关于 `test_mp.py` 末尾的 `os._exit(0)`：Windows 上 MediaPipe 1.0.0 的 TFLite
> 在解释器清理阶段可能阻塞，见 [`../debug/OS_EXIT_NOTE.md`](../debug/OS_EXIT_NOTE.md)。
