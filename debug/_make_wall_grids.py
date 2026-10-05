"""把每个 wall 的所有结果图拼成一张大图，方便用户确认。

输出: wall_<safe_name>_grid.jpg
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import re

walls = sorted(Path('img_xhs_walls').iterdir())
print(f"共 {len(walls)} 个 wall\n")

# 已知真值（来自 _run_walls_agg.py 汇总）
TRUTH_AND_PRED = {
    '2XRo7VbPgTe_wall_01_known_IMG_20260808_193753': ('黑 (2X_01)', '橙 50%'),
    '2XRo7VbPgTe_wall_02_known_IMG_20260808_193745': ('蓝 (2X_02)', '蓝 48% ✅'),
    '2XRo7VbPgTe_wall_03_known_IMG_20260808_193754': ('粉红 (2X_03)', '粉红 32% ✅'),
    '2XRo7VbPgTe_wall_04_known_IMG_20260808_193724': ('青 (2X_04)', '青 55% ✅'),
    '3OHz6rZ5opG_wall_01_unknown': ('?', '蓝 39%'),
    '3OHz6rZ5opG_wall_02_unknown': ('?', '橙 25%'),
    '3OHz6rZ5opG_wall_03_known_IMG_20260808_193612': ('红 (3O_02)', '红 27% ✅'),
    '3OHz6rZ5opG_wall_04_line1_white': ('白 (3O_04 L1)', '橙 36%'),
    '3OHz6rZ5opG_wall_04_line2_pink': ('粉红 (3O_04 L2)', '粉红 50% ✅'),
    '3OHz6rZ5opG_wall_05_known_IMG_20260808_193650': ('绿 (3O_05)', '绿 44% ✅'),
}

def safe_name(name: str) -> str:
    return re.sub(r'[^\w]', '_', name)[:50]

for wall_dir in walls:
    wall_name = wall_dir.name
    # 全名 prefix（不能截断：line1_white/line2_pink 截 24 字符会相同）
    prefix = f"agg_{wall_name}"
    # 找 agg_*.jpg 结果图
    result_imgs = sorted(Path('.').glob(f"{prefix}_*.jpg"))
    if not result_imgs:
        print(f"⚠️ {wall_name}: 无结果图，跳过")
        continue

    truth, pred = TRUTH_AND_PRED.get(wall_name, ('?', '?'))
    print(f"拼图: {wall_name}  ({len(result_imgs)} 张)  真值={truth}  判={pred}")

    # 源帧按字母序 = agg_*.jpg 的编号顺序（test_edge_sam.py 输出 agg_{prefix}_{i}.jpg）
    src_frames = sorted(p.name for p in Path(wall_dir).glob('*.jpg'))

    # 加载 + resize（原图 720x1280，尽量保持全尺寸，别缩太小）
    cols = 4
    rows = (len(result_imgs) + cols - 1) // cols
    thumb_w, thumb_h = 700, 1245  # 接近原图尺寸，保证清晰
    label_h = 30  # 每帧下方时间标签高度
    images = []
    for p in result_imgs:
        img = Image.open(p)
        img.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        images.append(img)

    # 网格布局
    header_h = 120
    grid_w = thumb_w * cols
    grid_h = (thumb_h + label_h) * rows + header_h
    canvas = Image.new('RGB', (grid_w, grid_h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    # 字体（用默认）
    try:
        font_title = ImageFont.truetype("arial.ttf", 40)
        font_sub = ImageFont.truetype("arial.ttf", 28)
        font_time = ImageFont.truetype("arial.ttf", 26)
    except OSError:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()
        font_time = ImageFont.load_default()

    # Header
    draw.text((20, 12), wall_name, fill=(255, 255, 255), font=font_title)
    draw.text((20, 62), f"真值: {truth}    判: {pred}", fill=(200, 230, 255), font=font_sub)

    # 拼图
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        x = c * thumb_w
        y = r * (thumb_h + label_h) + header_h
        canvas.paste(img, (x, y))
        # 每帧下方标时间（frame_0100s.jpg → 0100s）
        if i < len(src_frames):
            time_label = re.search(r'frame_(\d+s)\.jpg', src_frames[i])
            label = time_label.group(1) if time_label else src_frames[i]
        else:
            label = f"#{i+1}"
        draw.text((x + 8, y + thumb_h + 4), label, fill=(255, 220, 120), font=font_time)

    out_path = f"wall_grid_{safe_name(wall_name)}.jpg"
    canvas.save(out_path, quality=95, subsampling=0)  # 高画质无色彩抽样
    print(f"  💾 {out_path}  ({canvas.size[0]}x{canvas.size[1]})")

print("\n所有 wall 拼图完成。")