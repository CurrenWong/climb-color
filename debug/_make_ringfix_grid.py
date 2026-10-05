# -*- coding: utf-8 -*-
"""把 test_ringfix_*.jpg 拼成 2X_01 整墙大图（带时间标签）"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

SRC = Path(r'D:\Project\climbing_wall_search\mediapipe2')
OUT = SRC / 'wall_grid_2XRo_wall_01_RINGFIX.jpg'

files = sorted(SRC.glob('test_ringfix_*.jpg'),
               key=lambda p: int(p.stem.split('_')[-1]))
imgs = [Image.open(f) for f in files]
w, h = imgs[0].size

# 每帧加时间标签（来自文件名 frame_xxxxs.jpg 对应的 index）
labels = ['0000s', '0005s', '0010s', '0015s', '0020s', '0025s', '0030s', '0035s']

pad = 8
label_h = 36
cols = 2
rows = (len(imgs) + cols - 1) // cols
grid_w = cols * w + (cols + 1) * pad
grid_h = rows * (h + label_h) + (rows + 1) * pad

canvas = Image.new('RGB', (grid_w, grid_h), (40, 40, 40))
draw = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 28)
except Exception:
    font = ImageFont.load_default()

for i, (im, lab) in enumerate(zip(imgs, labels)):
    r, c = divmod(i, cols)
    x = pad + c * (w + pad)
    y = pad + r * (h + label_h + pad)
    canvas.paste(im, (x, y))
    draw.text((x + 6, y + h + 4), lab, fill=(255, 255, 0), font=font)

canvas.save(OUT, quality=95)
print(OUT)
print(f"size={grid_w}x{grid_h}")
