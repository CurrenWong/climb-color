"""批量修复 dwpose_exp 内脚本的 sys.path 顺序。

问题: 之前多次 sys.path.insert(0, ...) 让最后一次插的在 sys.path[0],
     导致 _BASE 没在最前, dwpose_exp 本地副本从未被加载, DWPose 实验
     一直用的根目录共享模块。

修复: 用一次 slice insert [0:0] = [...] 保留原 sys.path, 但 _BASE 优先。
"""
import os
import re

EXP_DIR = r"D:\Project\climbing_wall_search\mediapipe2\dwpose_exp"

# 匹配从 _BASE 定义到 sys.path.insert(EdgeSAM) 结束的整段
# 用正则捕获中间所有 sys.path.insert + 注释
PATTERN = re.compile(
    r'_BASE = os\.path\.dirname\(os\.path\.abspath\(__file__\)\)\n'
    r'((?:.*?\n)*?)'                        # 捕获中间所有行（非贪婪）
    r'sys\.path\.insert\(0, os\.path\.join\(_ROOT, "EdgeSAM"\)\)'
)

NEW = (
    '# sys.path 顺序: _BASE (dwpose_exp 本地副本) 优先, 然后根目录\n'
    '# (edgesam_stub 等共享模块), 最后 EdgeSAM 源码。注意: 必须用\n'
    '# 一次 slice insert 保留原 sys.path, 多次 insert(0, ...) 会反向\n'
    '# 把 _BASE 推到后面, 导致 dwpose_exp 本地副本永远不被加载。\n'
    'sys.path[0:0] = [_BASE, _ROOT, os.path.join(_ROOT, "EdgeSAM")]'
)


def replace_in_file(p: str) -> bool:
    with open(p, encoding='utf-8') as fh:
        c = fh.read()
    m = PATTERN.search(c)
    if not m:
        return False
    # 替换整个匹配段（从 _BASE 到 EdgeSAM insert）
    new_c = c[:m.start()] + (
        '_BASE = os.path.dirname(os.path.abspath(__file__))\n'
        '_ROOT = os.path.dirname(_BASE)\n'
    ) + NEW + c[m.end():]
    with open(p, 'w', encoding='utf-8') as fh:
        fh.write(new_c)
    return True


for f in sorted(os.listdir(EXP_DIR)):
    if not f.endswith('.py'):
        continue
    p = os.path.join(EXP_DIR, f)
    if replace_in_file(p):
        print(f"OK {f}")