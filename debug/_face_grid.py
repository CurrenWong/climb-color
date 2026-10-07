from PIL import Image, ImageDraw

im = Image.open('docs/demo/wall_blue.jpg').convert('RGB')
box = (240, 340, 420, 540)
crop = im.crop(box)
s = 4
big = crop.resize((crop.width * s, crop.height * s), Image.LANCZOS)
d = ImageDraw.Draw(big)
for x in range(box[0], box[2] + 1, 20):
    px = (x - box[0]) * s
    d.line([(px, 0), (px, big.height)], fill=(255, 0, 0), width=1)
    d.text((px + 3, 3), str(x), fill=(255, 0, 0))
for y in range(box[1], box[3] + 1, 20):
    py = (y - box[1]) * s
    d.line([(0, py), (big.width, py)], fill=(255, 0, 0), width=1)
    d.text((3, py + 3), str(y), fill=(255, 0, 0))
big.save('debug/face_grid.jpg', quality=92)
print('saved', big.size)
