"""视频聚合层（方案 A）：

从 _xhs_run3.log 解析每帧每端点的颜色名（不是每帧投票后的线路色），
跨帧统计所有端点颜色出现次数，取最高频 = 线路色。
"""
import re
from pathlib import Path
from collections import Counter

log = Path('_xhs_run3.log').read_text(encoding='utf-8', errors='ignore')

# 解析 [N] filename ... L_hand : <color> ...
pattern_block = re.compile(
    r'\[(\d+)\]\s+(\S+\.jpg)\s*\n'
    r'(?:[^\n]*\n)*?'  # 中间任意行（包括 "EdgeSAM 命中 N/4 端点"）
    r'(?P<samples>(?:\s+(?:L_hand|R_hand|L_foot|R_foot)\s*:[^\n]*\n)+)',
)

# 单行端点格式: "    L_hand  : <color>   area=... score=..."
pattern_ep = re.compile(
    r'(L_hand|R_hand|L_foot|R_foot)\s*:\s*(\S+?)(?:\s+(?:area=|vis=))'
)

# 跳过非投票的颜色
SKIP_COLORS = {'低置信度', '非岩点', '无法判断'}

video_colors = {'2XRo': Counter(), '3OHz': Counter()}
video_frames = {'2XRo': 0, '3OHz': 0}

for m in pattern_block.finditer(log):
    idx, fname, samples_block = m.group(1), m.group(2), m.group('samples')

    # 识别视频
    if fname.startswith('2XRo'):
        vname = '2XRo'
    elif fname.startswith('3OHz'):
        vname = '3OHz'
    else:
        continue
    video_frames[vname] += 1

    for em in pattern_ep.finditer(samples_block):
        ep_name, color = em.group(1), em.group(2)
        if color in SKIP_COLORS:
            continue
        video_colors[vname][color] += 1

# 输出
for vname in ('2XRo', '3OHz'):
    counter = video_colors[vname]
    total = sum(counter.values())
    print(f"\n=== {vname} 视频 ({video_frames[vname]} 帧) ===")
    print(f"有效端点颜色投票（跳过 低置信度/非岩点/无法判断）: {total} 票")
    for color, n in counter.most_common():
        pct = n / total * 100 if total else 0
        print(f"  {color:4s}: {n:3d} 票  ({pct:5.1f}%)")
    if counter:
        top_color, top_n = counter.most_common(1)[0]
        print(f"  >>> 线路色 = {top_color}  ({top_n}/{total} = {top_n/total*100:.0f}%)")