"""DWPose 回归两个 miss 墙的完整标注拼图。

背景：exp_dwpose_full.py 全量回归 6/8，两个 miss 都是 3O wall_04：
  - line1_white: 真值=白, DWPose 判=黄 ❌
  - line2_pink:  真值=粉红, DWPose 判=红 ❌

输入: dwpose_vis_<wall>/frame_*.jpg（exp_dwpose_full.py 生成，含 mask+端点+header）
输出: dwpose_miss_grid_<wall>.jpg
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import re

WALLS = [
    ('3OHz6rZ5opG_wall_04_line1_white', '白', '橙'),
    ('3OHz6rZ5opG_wall_04_line2_pink',  '粉红', '粉红(已修复)'),
]

def safe_name(name: str) -> str:
    return re.sub(r'[^\w]', '_', name)[:60]

for wall_name, truth, pred in WALLS:
    vis_dir = Path(f'dwpose_vis_{wall_name}')
    frames = sorted(vis_dir.glob('frame_*.jpg'))
    if not frames:
        print(f"⚠️ {wall_name}: 无帧，跳过")
        continue
    print(f"拼图: {wall_name}  ({len(frames)} 帧)  真值={truth}  DWPose判={pred}")

    # 加载帧
    thumb_w, thumb_h = 700, 1245
    label_h = 34
    images = []
    for p in frames:
        img = Image.open(p)
        img.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        images.append(img)

    # 2 列布局（5 帧 → 2x3，第 3 行留空也行）
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

    draw.text((20, 12), f"DWPose miss: {wall_name}", fill=(255, 255, 255), font=font_title)
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

    out_path = f"dwpose_miss_grid_{safe_name(wall_name)}.jpg"
    canvas.save(out_path, quality=95, subsampling=0)
    print(f"  💾 {out_path}  ({canvas.size[0]}x{canvas.size[1]})")

print("\n完成。")
