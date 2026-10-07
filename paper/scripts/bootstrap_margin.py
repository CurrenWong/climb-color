"""Bootstrap 稳定性分析：从 _verify_full.log 解析逐帧端点票，重采样帧估计 wall-level 稳健性。

动机：10/10 命中里有 5 面墙 margin <= 2 票（最紧的 3O_04_line2 是 4 vs 3）。
      单次跑出来的 10/10 可能是运气，需要量化。
方法：对每面墙的帧做有放回重采样（n = 原帧数），重新聚合端点票取 plurality，
      重复 2000 次，统计"仍判对"的比例（bootstrap accuracy）。
"""
import re
import random
import json
from pathlib import Path
from collections import Counter

LOG = Path(r"D:\Project\climbing_wall_search\mediapipe2\dwpose_exp\_verify_full.log")

TRUTH = {
    '2XRo7VbPgTe_wall_01_known_IMG_20260808_193753': '黑',
    '2XRo7VbPgTe_wall_02_known_IMG_20260808_193745': '蓝',
    '2XRo7VbPgTe_wall_03_known_IMG_20260808_193754': '粉红',
    '2XRo7VbPgTe_wall_04_known_IMG_20260808_193724': '青',
    '3OHz6rZ5opG_wall_01_unknown': '蓝',
    '3OHz6rZ5opG_wall_02_unknown': '红',
    '3OHz6rZ5opG_wall_03_known_IMG_20260808_193612': '黄',
    '3OHz6rZ5opG_wall_04_line1_white': '白',
    '3OHz6rZ5opG_wall_04_line2_pink': '粉红',
    '3OHz6rZ5opG_wall_05_known_IMG_20260808_193650': '绿',
}

SHORT = {
    '2XRo7VbPgTe_wall_01_known_IMG_20260808_193753': '2X_wall_01',
    '2XRo7VbPgTe_wall_02_known_IMG_20260808_193745': '2X_wall_02',
    '2XRo7VbPgTe_wall_03_known_IMG_20260808_193754': '2X_wall_03',
    '2XRo7VbPgTe_wall_04_known_IMG_20260808_193724': '2X_wall_04',
    '3OHz6rZ5opG_wall_01_unknown': '3O_wall_01',
    '3OHz6rZ5opG_wall_02_unknown': '3O_wall_02',
    '3OHz6rZ5opG_wall_03_known_IMG_20260808_193612': '3O_wall_03',
    '3OHz6rZ5opG_wall_04_line1_white': '3O_wall_04_line1_white',
    '3OHz6rZ5opG_wall_04_line2_pink': '3O_wall_04_line2_pink',
    '3OHz6rZ5opG_wall_05_known_IMG_20260808_193650': '3O_wall_05',
}

text = LOG.read_text(encoding='utf-8', errors='replace')

# --- split into wall blocks ---
blocks = re.split(r'^=== (.+?) \(\d+ 张\) ===$', text, flags=re.M)
walls = {}
for i in range(1, len(blocks), 2):
    name = blocks[i].strip()
    body = blocks[i + 1]
    if name not in TRUTH:
        continue
    frames = []
    for m in re.finditer(r'>>> 线路色 = (\S+)\s+置信度 = \d+%\s+票数 = (\{[^}]*\})', body):
        votes = eval(m.group(2))
        if votes:
            frames.append(votes)
    walls[name] = frames

print(f"parsed {len(walls)} walls\n")

# --- exact result + margin analysis ---
print(f"{'wall':<32} {'truth':<5} {'pred':<5} {'votes':>6} {'share':>6} {'runner-up':>12} {'margin':>7}")
print('-' * 90)
rows = []
for name, frames in walls.items():
    total = Counter()
    for fv in frames:
        total.update(fv)
    pred, n = total.most_common(1)[0]
    overall = sum(total.values())
    second = total.most_common(2)[1] if len(total) > 1 else ('-', 0)
    margin = n - second[1]
    ok = pred == TRUTH[name]
    rows.append((SHORT[name], TRUTH[name], pred, n, overall, second, margin, ok))
    print(f"{SHORT[name]:<32} {TRUTH[name]:<5} {pred:<5} {n:>3}/{overall:<3} "
          f"{n/overall*100:>5.1f}% {second[0]:>6}({second[1]:>2}) {margin:>5}票 {'OK' if ok else 'MISS'}")

# --- bootstrap ---
random.seed(0)
B = 2000
print(f"\n=== Bootstrap ({B} resamples of frames, per wall) ===")
print(f"{'wall':<32} {'acc':>7} {'miss-cases (pred)':<30}")
print('-' * 90)
wall_accs = []
for name, frames in walls.items():
    hit = 0
    miss_pred = Counter()
    n_frames = len(frames)
    for _ in range(B):
        resampled = [random.choice(frames) for _ in range(n_frames)]
        total = Counter()
        for fv in resampled:
            total.update(fv)
        p = total.most_common(1)[0][0]
        if p == TRUTH[name]:
            hit += 1
        else:
            miss_pred[p] += 1
    acc = hit / B
    wall_accs.append(acc)
    miss_str = ', '.join(f"{k}({v})" for k, v in miss_pred.most_common(3)) or '-'
    print(f"{SHORT[name]:<32} {acc:>6.1%} {miss_str:<30}")

# --- joint: all-walls-correct in a single resample ---
random.seed(0)
joint = 0
for _ in range(B):
    ok_all = True
    for name, frames in walls.items():
        resampled = [random.choice(frames) for _ in range(len(frames))]
        total = Counter()
        for fv in resampled:
            total.update(fv)
        if total.most_common(1)[0][0] != TRUTH[name]:
            ok_all = False
            break
    if ok_all:
        joint += 1

per_wall_mean = sum(wall_accs) / len(wall_accs)
print(f"\nper-wall bootstrap accuracy (mean over 10 walls): {per_wall_mean:.1%}")
print(f"expected #correct walls per run (sum of per-wall acc): {sum(wall_accs):.2f} / 10")
print(f"joint P(all 10 walls correct in one run): {joint/B:.1%}")

out = {
    'per_wall': {SHORT[n]: {'truth': TRUTH[n],
                            'pred': rows[i][2],
                            'votes': f"{rows[i][3]}/{rows[i][4]}",
                            'margin': rows[i][6],
                            'bootstrap_acc': wall_accs[i]}
                 for i, n in enumerate(walls)},
    'mean_bootstrap_acc': per_wall_mean,
    'expected_correct': sum(wall_accs),
    'joint_all_correct': joint / B,
    'B': B,
}
Path(r'D:\Project\climbing_wall_search\mediapipe2\paper\data\bootstrap_results.json').write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
print("\nsaved paper/data/bootstrap_results.json")
