# 生成 193753/193754 确认对比图（缩略 + 并排）
from PIL import Image, ImageDraw
import os

img_dir = r"D:\Project\climbing_wall_search\img"
out_dir = r"D:\Project\climbing_wall_search\mediapipe2\debug\photo_confirm"
os.makedirs(out_dir, exist_ok=True)


def shrink(p, maxw=900):
    im = Image.open(p).convert("RGB")
    w, h = im.size
    if w > maxw:
        im = im.resize((maxw, int(h * maxw / w)), Image.LANCZOS)
    return im


a = shrink(os.path.join(img_dir, "IMG_20260808_193753.jpg"))
b = shrink(os.path.join(img_dir, "IMG_20260808_193754.jpg"))
a.save(os.path.join(out_dir, "193753_2Xwall01_GT.jpg"), quality=88)
b.save(os.path.join(out_dir, "193754_2Xwall03_GT.jpg"), quality=88)

pad, cap = 20, 44
W = a.width + b.width + pad * 3
H = max(a.height, b.height) + cap * 2
canvas = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(canvas)
canvas.paste(a, (pad, cap))
canvas.paste(b, (pad * 2 + a.width, cap))
d.text((pad, 10), "IMG_193753  (GT: 2X_wall_01  black)", fill="black")
d.text((pad * 2 + a.width, 10), "IMG_193754  (GT: 2X_wall_03  pink)", fill="black")
canvas.save(os.path.join(out_dir, "compare_193753_193754.jpg"), quality=88)
print("saved", os.path.join(out_dir, "compare_193753_193754.jpg"), canvas.size)
