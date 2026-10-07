"""拼 paper Fig 4: adaptive box 对比（box=160 fixed vs box=128 person-adaptive）。

素材: debug/fig4_src_box160.jpg / debug/fig4_src_box128.jpg（FIXED_BOX=0/1 两次运行）
     （dbg_line1_frame02_candidates.py 在 FIXED_BOX=0/1 两次运行）
场景: 3O wall_04 line1_white frame02 (0105s)，真值 = 白
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parents[2]      # 项目根
EXP = BASE / "debug"      # 素材已移到 debug/

ROWS = [
    ('fig4_src_box160.jpg',
     '(a)  box = 160 px  (fixed)',
     'R_hand candidates: 3 white + 1 distant RED hold (score 0.806, area 14574)\n'
     '-> highest score wins -> predicts RED            (WRONG)',
     (255, 160, 160)),
    ('fig4_src_box128.jpg',
     '(b)  box = 128 px  (person-bbox-adaptive, 0.6 x short side)',
     'R_hand candidates: white hold now scores highest (0.881)\n'
     '-> predicts WHITE                                 (CORRECT)',
     (160, 240, 160)),
]

MAX_W = 1180
imgs = []
for fname, title, note, col in ROWS:
    p = EXP / fname
    if not p.exists():
        print(f"missing {p}")
        continue
    im = Image.open(p)
    scale = MAX_W / im.size[0]
    im = im.resize((MAX_W, int(im.size[1] * scale)), Image.Resampling.LANCZOS)
    imgs.append((im, title, note, col))

if not imgs:
    raise SystemExit("no inputs")

TITLE_H = 44
NOTE_H = 62
HEADER_H = 76
H = HEADER_H + sum(TITLE_H + im.size[1] + NOTE_H for im, *_ in imgs)
canvas = Image.new('RGB', (MAX_W, H), (30, 30, 30))
draw = ImageDraw.Draw(canvas)

try:
    f_h = ImageFont.truetype("arial.ttf", 30)
    f_t = ImageFont.truetype("arial.ttf", 25)
    f_n = ImageFont.truetype("arial.ttf", 21)
except OSError:
    f_h = f_t = f_n = ImageFont.load_default()

draw.text((14, 14),
          "Adaptive box prevents selecting a distant, higher-scoring hold "
          "(3O wall_04 line1, frame 0105s, truth = white)",
          fill=(255, 255, 255), font=f_h)

y = HEADER_H
for im, title, note, col in imgs:
    draw.text((12, y + 6), title, fill=col, font=f_t)
    y += TITLE_H
    canvas.paste(im, (0, y))
    y += im.size[1]
    for i, line in enumerate(note.split('\n')):
        draw.text((14, y + 6 + i * 26), line, fill=(210, 210, 210), font=f_n)
    y += NOTE_H

out = BASE / "paper" / "figures" / "fig4_box_adaptive.jpg"
canvas.save(out, quality=90, subsampling=0)
print(f"saved {out}  ({canvas.size[0]}x{canvas.size[1]})")
