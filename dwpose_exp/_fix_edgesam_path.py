"""批量修改 dwpose_exp 内脚本的 EdgeSAM 路径引用（指向根目录共享副本）。"""
import os
import re

EXP_DIR = r"D:\Project\climbing_wall_search\mediapipe2\dwpose_exp"
FILES = [
    'dbg_line2_lhand_pink.py', 'dbg_line2_h_hist.py', 'dbg_line2_grid_debug.py',
    'dbg_pink_red_hist.py', 'dbg_pink_red_ref.py', 'test_no_os_exit_dwpose.py',
]

OLD = 'sys.path.insert(0, os.path.join(_BASE, "EdgeSAM"))'
NEW = ('# EdgeSAM 源码：引用根目录共享副本（大目录不复制，实验与生产共用即可）\n'
       '_ROOT = os.path.dirname(_BASE)\n'
       'sys.path.insert(0, os.path.join(_ROOT, "EdgeSAM"))')

for f in FILES:
    p = os.path.join(EXP_DIR, f)
    with open(p, encoding='utf-8') as fh:
        c = fh.read()
    if OLD in c:
        c = c.replace(OLD, NEW)
        with open(p, 'w', encoding='utf-8') as fh:
            fh.write(c)
        print(f"OK {f}")
    else:
        print(f"NO-MATCH {f}")
