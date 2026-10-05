"""用 Python 重新整理 wall_04 line1/line2 文件夹。"""
import shutil
from pathlib import Path

root = Path(r'D:\Project\climbing_wall_search\mediapipe2\img_xhs_walls')
src_dir = Path(r'D:\Project\climbing_wall_search\video_frames_cache\小红书_3OHz6rZ5opG\walls\wall_04_known_IMG_20260808_193621')

line1_dir = root / '3OHz6rZ5opG_wall_04_line1_white'
line2_dir = root / '3OHz6rZ5opG_wall_04_line2_pink'

# 删旧建新
if line1_dir.exists():
    shutil.rmtree(line1_dir)
if line2_dir.exists():
    shutil.rmtree(line2_dir)
line1_dir.mkdir()
line2_dir.mkdir()

# 从原 line1_0100s_0120s 复制 5 张（0100s-0120s）
src1 = src_dir / 'line1_0100s_0120s'
for f in sorted(src1.glob('*.jpg')):
    shutil.copy2(f, line1_dir / f.name)

# 从原 line2_0125s_0140s 复制 4 张（0125s-0140s）
src2 = src_dir / 'line2_0125s_0140s'
for f in sorted(src2.glob('*.jpg')):
    shutil.copy2(f, line2_dir / f.name)

print(f"line1 ({line1_dir.name}):")
for f in sorted(line1_dir.glob('*.jpg')):
    print(f"  {f.name}")
print(f"line2 ({line2_dir.name}):")
for f in sorted(line2_dir.glob('*.jpg')):
    print(f"  {f.name}")