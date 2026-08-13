"""通用 DWPose 标注拼图脚本：任意 wall 列表 → 完整标注拼图。

用法: python _make_dwpose_grid.py
输入: dwpose_vis_<wall>/frame_*.jpg（exp_dwpose_full.py 生成）
输出: dwpose_grid_<wall>.jpg
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import re
import sys

# 用法: python _make_dwpose_grid.py "wall_name|真值|判定" ["wall_name|真值|判定" ...]
# 不传参数则用默认列表
WALLS_DEFAULT = [
    ('3OHz6rZ5opG_wall_01_unknown', '?', '蓝'),
    ('3OHz6rZ5opG_wall_02_unknown', '?', '红'),
]

def parse_args():
    if len(sys.argv) > 1:
        out = []
        for arg in sys.argv[1:]:
            parts = arg.split('|')
            if len(parts) == 3:
                out.append((parts[0], parts[1], parts[2]))
        return out
    return WALLS_DEFAULT

def safe_name(name: str) -> str:
    return re.sub(r'[^\w]', '_', name)[:60]

def make_grid(wall_name: str, truth: str, pred: str):
    vis_dir = Path(f'dwpose_vis_{wall_name}')
    frames = sorted(vis_dir.glob('frame_*.jpg'))
    if not frames:
        print(f"⚠️ {wall_name}: 无帧，跳过")
        return
    print(f"拼图: {wall_name}  ({len(frames)} 帧)  真值={truth}  DWPose判={pred}")

    thumb_w, thumb_h = 700, 1245
    label_h = 34
    images = []
    for p in frames:
        img = Image.open(p)
        img.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        images.append(img)

    cols = 2
    rows = (len(images) + cols - 1) // cols
    header_h = 120
    grid_w = thumb_w * cols
    grid_h = (thumb_h + label_h) * rows + header_h
    canvas = Image.new('RGB', (grid_w, grid_h), (25, 25, 25))
    draw = ImageDraw.Draw(canvas)

    try:
        font_title = ImageFont.truetype("arial.ttf", 42)
        font_sub = ImageFont.truetype("arial.ttf", 30)
        font_time = ImageFont.truetype("arial.ttf", 28)
    except OSError:
        font_title = font_sub = font_time = ImageFont.load_default()

    draw.text((20, 12), f"DWPose: {wall_name}", fill=(255, 255, 255), font=font_title)
    draw.text((20, 66), f"真值: {truth}     DWPose 判: {pred}",
              fill=(255, 220, 160), font=font_sub)

    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        x = c * thumb_w
        y = r * (thumb_h + label_h) + header_h
        canvas.paste(img, (x, y))
        m = re.search(r'frame_(\d+s)\.jpg', frames[i].name)
        label = m.group(1) if m else frames[i].stem
        draw.text((x + 8, y + thumb_h + 6), label, fill=(255, 220, 120), font=font_time)

    out_path = f"dwpose_grid_{safe_name(wall_name)}.jpg"
    canvas.save(out_path, quality=95, subsampling=0)
    print(f"  💾 {out_path}  ({canvas.size[0]}x{canvas.size[1]})")

if __name__ == '__main__':
    for wall_name, truth, pred in parse_args():
        make_grid(wall_name, truth, pred)
    print("\n完成。")
