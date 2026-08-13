"""批量把 dwpose_exp 内脚本的 sys.path 加上根目录（供 edgesam_stub import）。"""
import os

EXP_DIR = r"D:\Project\climbing_wall_search\mediapipe2\dwpose_exp"

OLD = ('_ROOT = os.path.dirname(_BASE)\n'
       'sys.path.insert(0, os.path.join(_ROOT, "EdgeSAM"))')
NEW = ('_ROOT = os.path.dirname(_BASE)\n'
       'sys.path.insert(0, _ROOT)  # 供 edgesam_stub（根目录共享）\n'
       'sys.path.insert(0, os.path.join(_ROOT, "EdgeSAM"))')

for f in os.listdir(EXP_DIR):
    if not f.endswith('.py'):
        continue
    p = os.path.join(EXP_DIR, f)
    with open(p, encoding='utf-8') as fh:
        c = fh.read()
    if OLD in c:
        c = c.replace(OLD, NEW)
        with open(p, 'w', encoding='utf-8') as fh:
            fh.write(c)
        print(f"OK {f}")
