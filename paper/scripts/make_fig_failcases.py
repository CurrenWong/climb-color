"""拼 paper 失败案例图 (Fig 5): 6.jpg / 8.jpg 两张失败案例横向拼接 + 标注。"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parents[2]      # 项目根

CASES = [
    ('debug/failcase_img6_pred_white.jpg', 'img/6.jpg', 'yellow', 'white 100%',
     'EdgeSAM cannot segment the\nyellow hold (far / small scale)'),
    ('debug/failcase_img8_pred_purple.jpg', 'img/8.jpg', 'blue', 'purple 100%',
     'blue hold swallowed by\nperson mask (climber overlaps)'),
]

THUMB_W, THUMB_H = 760, 1350
LABEL_H = 120
HEADER_H = 90

imgs = []
for fname, src, truth, pred, note in CASES:
    p = BASE / fname
    if not p.exists():
        print(f"missing: {p}")
        continue
    im = Image.open(p)
    im.thumbnail((THUMB_W, THUMB_H), Image.Resampling.LANCZOS)
    imgs.append((im, src, truth, pred, note))

if not imgs:
    raise SystemExit("no images")

cols = len(imgs)
W = THUMB_W * cols
H = HEADER_H + THUMB_H + LABEL_H
canvas = Image.new('RGB', (W, H), (28, 28, 28))
draw = ImageDraw.Draw(canvas)

try:
    f_title = ImageFont.truetype("arial.ttf", 34)
    f_sub = ImageFont.truetype("arial.ttf", 24)
    f_note = ImageFont.truetype("arial.ttf", 21)
except OSError:
    f_title = f_sub = f_note = ImageFont.load_default()

draw.text((18, 14), "Failure cases: two remaining single-image errors (Ours 8/9 -> 2 misses shown)",
          fill=(255, 255, 255), font=f_title)

for i, (im, src, truth, pred, note) in enumerate(imgs):
    x0 = i * THUMB_W
    canvas.paste(im, (x0, HEADER_H))
    y = HEADER_H + im.size[1] + 8
    draw.text((x0 + 10, y), f"{src}   truth = {truth}   pred = {pred}",
              fill=(255, 210, 120), font=f_sub)
    draw.text((x0 + 10, y + 34), note, fill=(200, 200, 200), font=f_note)

out = BASE / "paper" / "figures" / "fig5_failure_cases.jpg"
canvas.save(out, quality=92, subsampling=0)
print(f"saved {out}  ({canvas.size[0]}x{canvas.size[1]})")
