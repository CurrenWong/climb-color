from pathlib import Path
import re

for name in ['dwpose_exp/_verify_full.log', 'dwpose_exp/dense_sampling.log']:
    p = Path(name)
    if not p.exists():
        print(name, 'MISSING')
        continue
    raw = p.read_bytes()
    print(f'--- {name}  ({len(raw)} bytes) ---')
    print('  head bytes:', raw[:60])
    for enc in ('utf-8', 'gbk', 'gb18030'):
        try:
            t = raw.decode(enc)
        except UnicodeDecodeError as e:
            print(f'  {enc:8s} FAIL {e}')
            continue
        m = re.findall(r'^===.*$', t, re.M)[:2]
        has_timer = '⏱' in t
        print(f'  {enc:8s} OK   timer_char={has_timer}  walls={m}')
