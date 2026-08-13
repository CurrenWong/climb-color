"""汇总 mediapipe2/_xhs_run3.log 里两个视频的线路色时间序列。

文件名按字典序：2XRo_frame_* (1-29) + 3OHz_frame_* (30-65)
"""
import re
from pathlib import Path
from collections import Counter

log = Path('_xhs_run3.log').read_text(encoding='utf-8', errors='ignore')

# 解析 filename ↔ 线路色 映射
# log 里 [N] <filename> 紧跟 ... >>> 线路色 = ...
pattern = re.compile(
    r'\[(\d+)\]\s+(\S+\.jpg)\s*\n[\s\S]*?线路色\s*=\s*(\S+)\s+置信度\s*=\s*(\d+)%',
)
results = []
for m in pattern.finditer(log):
    idx, fname, color, conf = int(m.group(1)), m.group(2), m.group(3), int(m.group(4))
    results.append((idx, fname, color, conf))

# 按视频前缀分组
videos = {'2XRo': [], '3OHz': []}
for idx, fname, color, conf in results:
    if fname.startswith('2XRo'):
        videos['2XRo'].append((idx, fname, color, conf))
    elif fname.startswith('3OHz'):
        videos['3OHz'].append((idx, fname, color, conf))

for vname, entries in videos.items():
    print(f"\n=== {vname} 视频：{len(entries)} 帧 ===")
    # 输出每帧的颜色
    colors = [e[2] for e in entries]
    counter = Counter(colors)
    print(f"颜色分布: {dict(counter)}")
    print(f"按时间序列（每 5 秒一帧）:")
    for idx, fname, color, conf in entries:
        sec = re.search(r'(\d+)s', fname).group(1)
        print(f"  {fname:30s}  {color:4s} {conf}%")