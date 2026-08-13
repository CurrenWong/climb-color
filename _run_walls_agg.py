"""循环跑 10 个 wall（4 个 2XRo + 1 个 wall_03 + 2 个 line1/line2 + 1 个 wall_05 + 2 个 unknown）。

每个 wall 用 --aggregate-walls 选项跑（自动输出跨帧聚合统计）。

输出汇总到 _aggregate_summary.txt。
"""
import subprocess
import os
import re
from pathlib import Path

PYTHON = r'D:\Program\Conda\python.exe'
ENV = {**os.environ, 'PYTHONPATH': r'..\.venv\Lib\site-packages;EdgeSAM'}

# 已知真值
TRUTH = {
    '2XRo7VbPgTe_wall_01_known_IMG_20260808_193753': '黑 (2X_01)',
    '2XRo7VbPgTe_wall_02_known_IMG_20260808_193745': '蓝 (2X_02)',
    '2XRo7VbPgTe_wall_03_known_IMG_20260808_193754': '粉红 (2X_03)',
    '2XRo7VbPgTe_wall_04_known_IMG_20260808_193724': '青 (2X_04)',
    '3OHz6rZ5opG_wall_01_unknown': '?',
    '3OHz6rZ5opG_wall_02_unknown': '?',
    '3OHz6rZ5opG_wall_03_known_IMG_20260808_193612': '红 (3O_02)',
    '3OHz6rZ5opG_wall_04_line1_white': '白 (3O_04 line1)',
    '3OHz6rZ5opG_wall_04_line2_pink': '粉红 (3O_04 line2)',
    '3OHz6rZ5opG_wall_05_known_IMG_20260808_193650': '绿 (3O_05)',
}

walls = sorted(Path('img_xhs_walls').iterdir())
print(f"共 {len(walls)} 个 wall\n")

results = []
for wall_dir in walls:
    wall_name = wall_dir.name
    # 全名 prefix（不能截断：line1_white/line2_pink 截 24 字符会相同导致互相覆盖）
    prefix = f"agg_{wall_name}"
    n_frames = sum(1 for _ in wall_dir.glob('*.jpg'))
    print(f"=== {wall_name} ({n_frames} 张) ===")

    cmd = [PYTHON, '-u', 'test_edge_sam.py',
           '--img', str(wall_dir), '--prefix', prefix, '--aggregate-walls']
    r = subprocess.run(cmd, env=ENV, capture_output=True, text=True, timeout=600)

    # 从输出提取聚合统计
    text = r.stdout + r.stderr
    m = re.search(r'>>> 线路色\s*=\s*(\S+)\s+\((\d+)/(\d+)\s*=\s*(\d+)%\)', text)
    if m:
        top, top_n, total, pct = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        truth = TRUTH.get(wall_name, '?')
        # 命中判断
        if truth == '?':
            hit = '?'
        elif top in truth:
            hit = '✅'
        else:
            hit = '❌'
        print(f"  聚合: {top} {pct}% ({top_n}/{total})  真值={truth}  {hit}")
        results.append((wall_name, top, top_n, total, pct, truth, hit))
    else:
        print(f"  ⚠️ 未提取到聚合结果")
        results.append((wall_name, '?', 0, 0, 0, '?', '?'))
    print()

# 汇总
print("\n=== 汇总 ===")
hits = sum(1 for r in results if r[6] == '✅')
known = sum(1 for r in results if r[5] != '?')
print(f"命中: {hits}/{known} (已知真值 wall)")
for r in results:
    print(f"  {r[0][:55]:55s}  判={r[1]:4s} ({r[2]}/{r[3]}={r[4]}%)  真值={r[5]:25s}  {r[6]}")