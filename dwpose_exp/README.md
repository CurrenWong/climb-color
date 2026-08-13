# dwpose_exp — DWPose 实验独立目录

**分割背景（2026-08-13）**：DWPose 实验代码与 MediaPipe 生产管线彻底分离。
本目录完全自包含（除大权重和 EdgeSAM 源码外），以后 DWPose 实验改动
**不影响**根目录 MediaPipe 生产（test_edge_sam.py / route_color.py /
edgesam_holds.py / _run_walls_agg.py），无需互相关回归。

## 目录内容

| 文件 | 说明 |
| --- | --- |
| `exp_dwpose_full.py` | DWPose 全量回归（7 wall），输出 `dwpose_full_summary.txt` + `dwpose_vis_<wall>/` |
| `exp_dwpose_0005s.py` | DWPose vs MediaPipe vs YOLO26 单帧 0005s 对比 |
| `exp_yolo26pose_full.py` | YOLO26 pose 全量回归（实验用，已否决） |
| `exp_yolo26pose_0005s.py` | YOLO26 pose 单帧对比 |
| `dbg_line2_*.py` / `dbg_pink_red_*.py` | line2 粉红/红边界诊断脚本 |
| `test_no_os_exit_dwpose.py` | os._exit 必要性验证（DWPose 无 MediaPipe，正常 return 即可） |
| `_make_dwpose_miss_grids.py` | 拼图脚本（读 `dwpose_vis_<wall>/`） |
| `route_color.py` / `edgesam_holds.py` / `yolo_person_mask.py` / `dwpose_onnx.py` | 共享模块**本地副本**（从根目录复制，各自演进） |
| `draw_results.py` | 可视化（从 test_edge_sam.draw_results 抽出，DWPose 专用副本） |

## 运行环境（关键）

```cmd
cd D:\Project\climbing_wall_search\mediapipe2\dwpose_exp
set PYTHONPATH=..\..\.venv\Lib\site-packages;..\EdgeSAM
D:\Program\Conda\python.exe -u exp_dwpose_full.py 1   # 1 = 只跑 1 个 wall 冒烟
```

- 必须 **conda 3.13**（`D:\Program\Conda\python.exe`），不能用 PATH 里的 copaw 3.12
- PYTHONPATH 挂载主仓库 `.venv\Lib\site-packages`（cv2/torch/numpy）+ 根目录 EdgeSAM
- 脚本内部 `sys.path` 已加：`dwpose_exp/`（本地模块）、根目录（edgesam_stub）、根目录 EdgeSAM
- **大权重不复制**（共 430MB+），脚本用 `../` 相对根目录引用：
  - `../yolox_l.onnx`、`../dw-ll_ucoco_384.onnx`（DWPose）
  - `../weights/edge_sam.pth`（EdgeSAM）
  - `../yolo26m-seg.pt`（person mask）、`../yolo26n-pose.pt`（YOLO26 实验）
- 输入图在 `../img_xhs_walls/`（脚本用 `Path(BASE)/".."/"img_xhs_walls"`）

## 修改约定

- 改颜色判定/采样逻辑：改 **本目录** `route_color.py` / `edgesam_holds.py`，不动根目录
- 改可视化：改 **本目录** `draw_results.py`
- 根目录 MediaPipe 生产管线改动：只动根目录文件，**不回填**到本目录
- 两套代码从复制时刻起各自演进；如果想把某个修复同步到另一套，需手动复制

## 依赖关系（本目录内）

```
exp_*.py / dbg_*.py
  ├─ route_color.py   （颜色分类）
  ├─ edgesam_holds.py （EdgeSAM 分割 + 采样 + ring 黑修正）
  ├─ yolo_person_mask.py （YOLO26 person mask）
  ├─ dwpose_onnx.py   （DWPose ONNX 推理）
  ├─ draw_results.py  （可视化，自包含）
  └─ edgesam_stub / edge_sam（根目录共享，只读）
```
