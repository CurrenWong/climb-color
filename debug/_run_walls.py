"""循环跑 9 个 wall，每个 wall 用独立 prefix 输出，统计端点颜色最高频。"""
import subprocess
import os
import re
import sys
from pathlib import Path
from collections import Counter

SKIP_COLORS = {'低置信度', '非岩点', '无法判断'}
PYTHON = r'D:\Program\Conda\python.exe'
ENV = {**os.environ, 'PYTHONPATH': r'..\.venv\Lib\site-packages;EdgeSAM'}

# 已知真值（从 GROUND_TRUTH.md + 推断）
TRUTH = {
    '2XRo7VbPgTe_wall_01_known_IMG_20260808_193753': '黑 (2X_01)',
    '2XRo7VbPgTe_wall_02_known_IMG_20260808_193745': '蓝 (2X_02)',
    '2XRo7VbPgTe_wall_03_known_IMG_20260808_193754': '粉红 (2X_03)',
    '2XRo7VbPgTe_wall_04_known_IMG_20260808_193724': '青 (2X_04)',
    '3OHz6rZ5opG_wall_03_known_IMG_20260808_193612': '红 (3O_02?)',
    '3OHz6rZ5opG_wall_04_known_IMG_20260808_193621': '白/粉红 (3O_04 两线)',
    '3OHz6rZ5opG_wall_05_known_IMG_20260808_193650': '绿 (3O_05)',
}

walls = sorted(Path('img_xhs_walls').iterdir())
print(f"共 {len(walls)} 个 wall\n")

results = []
for wall_dir in walls:
    wall_name = wall_dir.name
    prefix = f"w_{wall_name[:8]}"
    n_frames = sum(1 for _ in wall_dir.glob('*.jpg'))
    print(f"=== {wall_name} ({n_frames} 张) ===")

    # 跑 test_edge_sam.py
    log_path = Path(f"_wall_{wall_name}.log")
    cmd = [PYTHON, '-u', 'test_edge_sam.py',
           '--img', str(wall_dir), '--prefix', prefix]
    r = subprocess.run(cmd, env=ENV, capture_output=True, text=True, timeout=600)
    log_path.write_text(r.stdout + r.stderr, encoding='utf-8', errors='ignore')

    # 解析每端点颜色
    text = log_path.read_text(encoding='utf-8', errors='ignore')
    pattern_ep = re.compile(r'(L_hand|R_hand|L_foot|R_foot)\s*:\s*(\S+?)(?:\s+(?:area=|vis=))')
    colors = []
    for m in pattern_ep.finditer(text):
        c = m.group(2)
        if c not in SKIP_COLORS:
            colors.append(c)
    counter = Counter(colors)
    total = sum(counter.values())

    if counter:
        top, top_n = counter.most_common(1)[0]
        truth = TRUTH.get(wall_name, '?')
        hit = '✅' if truth.startswith(top) else '❌'
        print(f"  端点有效票: {total}")
        for color, n in counter.most_common(5):
            pct = n / total * 100 if total else 0
            print(f"    {color:4s}: {n:3d} ({pct:5.1f}%)")
        print(f"  >>> 线路色 = {top}  ({top_n}/{total})  真值={truth}  {hit}")
        results.append((wall_name, top, total, top_n, truth, hit))
    else:
        print(f"  ⚠️ 无有效端点颜色")
        results.append((wall_name, '-', 0, 0, '?', '?'))
    print()

# 汇总
print("\n=== 汇总 ===")
hit_count = sum(1 for r in results if r[5] == '✅')
total_walls = len(results)
print(f"命中: {hit_count}/{total_walls} (含 '黑'/真值未知不计)")
for r in results:
    print(f"  {r[0][:50]:50s}  判={r[1]:4s}  ({r[3]}/{r[2]})  真值={r[4]:25s}  {r[5]}")