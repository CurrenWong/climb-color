"""清理 mediapipe2 项目临时产物。"""
import os
import shutil
from pathlib import Path

ROOT = Path(r"D:\Project\climbing_wall_search\mediapipe2")

# 1. .log
deleted = 0
for p in ROOT.rglob("*.log"):
    p.unlink()
    deleted += 1
print(f"删 .log: {deleted}")

# 2. *TODO.md (Coding Mode session-local)
deleted = 0
for p in ROOT.rglob("*_TODO.md"):
    p.unlink()
    deleted += 1
print(f"删 TODO.md: {deleted}")

# 3. dwpose_vis_<wall>/
deleted = 0
for p in ROOT.iterdir():
    if p.is_dir() and p.name.startswith("dwpose_vis_"):
        shutil.rmtree(p)
        deleted += 1
print(f"删 dwpose_vis_*: {deleted}")

# 4. result_edgesam_*.jpg / result_xhs_*.jpg / result_route_*.jpg / result_edgesam_old_*.jpg
patterns_4 = [
    "result_edgesam_*.jpg", "result_edgesam_old_*.jpg",
    "result_xhs_*.jpg", "result_route_*.jpg",
    "agg_*.jpg", "wall_grid_*.jpg",
    "w_2XRo7VbP_*.jpg", "w_3OHz6rZ5_*.jpg",
    "test_ringfix*.jpg", "test_agg_w3_*.jpg",
    "_check_15grid.jpg", "_dbg_*.jpg",
    "dbg_yolox_sigmoid.png", "diag_person_mask_1.jpg",
    "result_edgesam_1.jpg",  # demo 已备份
]
deleted = 0
for pat in patterns_4:
    for p in ROOT.glob(pat):
        p.unlink()
        deleted += 1
print(f"删 result/agg/wall/test_*: {deleted}")

# 5. dwpose_exp 内的输出图（保留 dwpose_exp/dbg_line1_frame02_candidates.jpg 已复制到 docs/demo）
deleted = 0
DWPOSE_EXP = ROOT / "dwpose_exp"
for pat in ["dbg_line1_frame02_candidates.jpg",
            "dwpose_grid_*.jpg", "dwpose_miss_grid_*.jpg"]:
    for p in DWPOSE_EXP.glob(pat):
        p.unlink()
        deleted += 1
print(f"删 dwpose_exp 输出图: {deleted}")

# 6. dwpose_full_summary.txt 保留（结果），其他 *_summary.txt / dwpose_*_compare.* 删
for p in DWPOSE_EXP.glob("dwpose_*_compare.*"):
    p.unlink()
for p in ROOT.glob("dwpose_*_compare.*"):
    p.unlink()
for p in DWPOSE_EXP.glob("yolo26pose_*_compare.*"):
    p.unlink()
for p in ROOT.glob("yolo26pose_*_compare.*"):
    p.unlink()
for p in ROOT.glob("yolo26pose_*_summary.txt"):
    p.unlink()
print(f"删 dwpose/yolo26 compare & summary 副本")

# 7. _ 前缀临时 txt / log 已删, 还在的 _ 前缀脚本移到 debug/
# 先看下根目录还有哪些 _ 前缀 .py / .txt
remaining_underscore = []
for p in ROOT.glob("_*.py"):
    remaining_underscore.append(p.name)
for p in ROOT.glob("_*.txt"):
    if p.exists():
        remaining_underscore.append(p.name)
print(f"剩余 _ 前缀: {remaining_underscore}")

# 8. 根目录临时 .log 检查
remaining_logs = list(ROOT.glob("*.log"))
print(f"剩余 .log: {[p.name for p in remaining_logs]}")

# 9. dwpose_exp 内 log 清理
deleted = 0
for p in DWPOSE_EXP.glob("*.log"):
    p.unlink()
    deleted += 1
print(f"删 dwpose_exp .log: {deleted}")
