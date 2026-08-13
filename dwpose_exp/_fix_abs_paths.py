"""批量把 dwpose_exp 内脚本的绝对路径改为相对路径（.. 引用根目录）。"""
import os

EXP_DIR = r"D:\Project\climbing_wall_search\mediapipe2\dwpose_exp"

REPL = [
    # img_xhs_walls 引用 → ../img_xhs_walls
    (r'..\img_xhs_walls',
     r'..\img_xhs_walls'),
    # video_frames_cache 引用 → ../../video_frames_cache（相对 dwpose_exp）
    (r'..\..\video_frames_cache',
     r'..\..\video_frames_cache'),
]

for f in os.listdir(EXP_DIR):
    if not f.endswith('.py'):
        continue
    p = os.path.join(EXP_DIR, f)
    with open(p, encoding='utf-8') as fh:
        c = fh.read()
    orig = c
    for old, new in REPL:
        c = c.replace(old, new)
    if c != orig:
        with open(p, 'w', encoding='utf-8') as fh:
            fh.write(c)
        print(f"OK {f}")
