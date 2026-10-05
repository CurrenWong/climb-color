"""把临时 _*.py 脚本移到 debug/，删根目录旧 exp_*/dbg_* 副本。"""
import os
import shutil
from pathlib import Path

ROOT = Path(r"D:\Project\climbing_wall_search\mediapipe2")
DEBUG = ROOT / "debug"
DEBUG.mkdir(exist_ok=True)

# 1. 移 _*.py 到 debug/（除 _clean_temp.py 自身保留运行）
moved = []
for p in sorted(ROOT.glob("_*.py")):
    if p.name == "_clean_temp.py":
        continue
    target = DEBUG / p.name
    shutil.move(str(p), str(target))
    moved.append(p.name)
print(f"移 debug/ ({len(moved)}): {moved}")

# 2. 移 _*.txt 到 debug/（除 _clean_temp 等需要保留的）
for p in sorted(ROOT.glob("_*.txt")):
    shutil.move(str(p), DEBUG / p.name)
    print(f"  移 _*.txt: {p.name}")

# 3. 删根目录旧 exp_*/dbg_* 副本（按用户决策）
# dwpose_exp/ 内已有完整版本（带 sys.path 修复 + box 自适应）
to_delete = []
for p in ROOT.glob("exp_dwpose_*.py"):
    to_delete.append(p)
for p in ROOT.glob("exp_yolo26pose_*.py"):
    to_delete.append(p)
for p in ROOT.glob("dbg_line2_*.py"):
    to_delete.append(p)
for p in ROOT.glob("dbg_pink_red_*.py"):
    to_delete.append(p)

print(f"\n将删根目录旧副本 ({len(to_delete)}):")
for p in to_delete:
    print(f"  - {p.name}")

for p in to_delete:
    p.unlink()

# 4. 其他清理
# efficientdet_lite0.tflite (早期 v1 用的，v2 不用了)
p = ROOT / "efficientdet_lite0.tflite"
if p.exists():
    p.unlink()
    print(f"  - 删 {p.name} (v1 残留)")

# test_mp_close_paths.py / test_no_os_exit_*.py — 调试脚本，移 debug/
for fname in ["test_mp_close_paths.py", "test_no_os_exit_dwpose.py", "test_no_os_exit_mp.py",
              "debug_rhand_0005s.py", "test_box_prompt_9.py", "test_box_v2_9.py",
              "OS_EXIT_NOTE.md", "_clean_temp.py"]:
    p = ROOT / fname
    if p.exists():
        shutil.move(str(p), DEBUG / fname)
        print(f"  移 {fname} -> debug/")

# 5. 大小写处理：dWpose_exp/_xhs_run.log 等也要清
import os
for p in DEBUG.iterdir():
    print(f"  debug/{p.name}: {p.stat().st_size}B")
