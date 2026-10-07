from pathlib import Path

p = Path(r'D:\Project\climbing_wall_search\mediapipe2\PIPELINE.md')
t = p.read_text(encoding='utf-8')

pairs = [
    ('## v1 pipeline（`test_mp.py`，长期存在）',
     '## v1 pipeline（`tools/test_mp.py`，保留作对照）'),
    ('| `test_mp.py` | v1 main entry | 长期存在，不动 |',
     '| `tools/test_mp.py` | v1 main entry | 保留作对照，不动 |'),
    ('calibrate.py / show_points.py',
     'tools/calibrate.py / tools/show_points.py'),
]

for a, b in pairs:
    n = t.count(a)
    print(f'hit {n}: {a[:50]}')
    t = t.replace(a, b)

p.write_text(t, encoding='utf-8')
print('saved PIPELINE.md')
