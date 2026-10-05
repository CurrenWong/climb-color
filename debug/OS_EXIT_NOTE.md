# os._exit(0) 必要性验证结论（2026-08-13）

## 问题
实验脚本里 `os._exit(0)` 是历史遗留：最初因 MediaPipe 1.0.0 + TFLite XNNPACK
在 Windows 上解释器清理阶段阻塞而加的变通（见 test_mp.py 头注释）。

DWPose / YOLO26pose 架构（onnxruntime，无 MediaPipe）是否还需要？

## 验证结果

| 场景 | 行为 | 是否需要 os._exit |
|---|---|---|
| DWPose 全流程（onnxruntime + EdgeSAM + YOLO seg） | 正常 return，exit code 0，唯一后台线程是 daemon=True 的 onnxruntime `_post` 线程 | ❌ 不需要 |
| MediaPipe close() | **42s/次**（不是无限阻塞，是极慢析构） | ✅ 保留（省 42s×N 等待） |
| MediaPipe del + gc | **42s/次** | ✅ 保留 |
| MediaPipe with 上下文退出 | **42s/次** | ✅ 保留 |

mediapipe 1.0.0 的 TFLite 析构：每次 landmarker 清理约 42 秒（不是当初以为的
无限阻塞，可能版本/环境变化，但 42s/次 仍不值得等）。

## 已改动
- `exp_dwpose_full.py`：删 `os._exit(0)` → `return 0` + `sys.exit(main())`，
  全量回归 6/8 与之前一致，进程正常退出。
- `exp_yolo26pose_full.py`：同样改法。

## 仍保留 os._exit 的脚本（含 MediaPipe，避免 42s/次 清理）
- `test_mp.py` / `test_edge_sam.py` / `calibrate.py` / `show_points.py`
- `exp_dwpose_0005s.py` / `exp_yolo26pose_0005s.py`（单帧对比，有 MediaPipe landmarker）

## 验证脚本（保留供复查）
- `test_no_os_exit_dwpose.py`：DWPose 架构无 os._exit 退出验证
- `test_no_os_exit_mp.py`：MediaPipe 无 os._exit 退出验证（意外通过）
- `test_mp_close_paths.py`：close/del/with 三种清理路径计时（各 ~42s）
