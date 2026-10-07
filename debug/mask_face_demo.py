"""Mask the climber's face in a README demo image.

Usage:
    python debug/mask_face_demo.py                        # apply to docs/demo/wall_blue.jpg
    python debug/mask_face_demo.py --box 285,385,365,470  # custom region (l,t,r,b)
    python debug/mask_face_demo.py --preview              # write a side-by-side preview only

The demo images come from public platforms (Xiaohongshu / Baidu Images), so the
climber's face is pixelated before the image is used in README / paper material.
Pixelation (mosaic) is used on purpose: it is unmistakably a privacy treatment.
"""

from __future__ import annotations

import argparse
import io
import os
import sys

from PIL import Image, ImageDraw, ImageFilter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TARGET = os.path.join(BASE, "docs", "demo", "wall_blue.jpg")
# Head region of the climber in docs/demo/wall_blue.jpg (720x1280), l,t,r,b.
DEFAULT_BOX = (285, 385, 365, 470)


def _feathered_ellipse_mask(size: tuple[int, int], feather: int = 3) -> Image.Image:
    """A rounded rectangle mask with soft edges.

    A rectangle is used (rather than an ellipse) so the whole head — hair, face
    and jaw — is covered without the corners of the region leaking through.
    """
    w, h = size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((1, 1, w - 2, h - 2), radius=8, fill=255)
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(feather))
    return mask


def pixelate(region: Image.Image, blocks: int = 8, blur: float = 0.6) -> Image.Image:
    """Mosaic the region: shrink with NEAREST, blow back up, then soften slightly.

    Kept deliberately crisp (small blur) so it reads as an intentional privacy
    treatment rather than an out-of-focus area.
    """
    w, h = region.size
    small = region.resize(
        (max(1, blocks), max(1, round(blocks * h / w))), Image.NEAREST
    )
    out = small.resize((w, h), Image.NEAREST)
    if blur > 0:
        out = out.filter(ImageFilter.GaussianBlur(blur))
    return out


def apply_mosaic(img: Image.Image, box: tuple[int, int, int, int],
                 blocks: int = 8) -> Image.Image:
    l, t, r, b = box
    region = img.crop(box)
    mosaic = pixelate(region, blocks=blocks)
    mask = _feathered_ellipse_mask(region.size)
    img.paste(mosaic, (l, t), mask)
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=DEFAULT_TARGET)
    ap.add_argument("--box", default=",".join(str(v) for v in DEFAULT_BOX),
                    help="l,t,r,b of the face region")
    ap.add_argument("--blocks", type=int, default=9,
                    help="mosaic block count across the region")
    ap.add_argument("--preview", action="store_true",
                    help="write debug/face_mask_preview.jpg instead of editing the target")
    args = ap.parse_args()

    box = tuple(int(v) for v in args.box.split(","))
    if len(box) != 4:
        print("ERROR: --box needs 4 comma-separated ints")
        return 2

    img = Image.open(args.target).convert("RGB")
    print(f"target : {args.target}  ({img.width}x{img.height})")
    print(f"region : {box}  (w={box[2]-box[0]}, h={box[3]-box[1]})")

    if args.preview:
        before = img.copy()
        after = apply_mosaic(img.copy(), box, args.blocks)
        canvas = Image.new("RGB", (before.width * 2 + 16, before.height), (255, 255, 255))
        canvas.paste(before, (0, 0))
        canvas.paste(after, (before.width + 16, 0))
        # zoom strip of the treated region, for eyeballing quality
        pad = 40
        zbox = (max(0, box[0] - pad), max(0, box[1] - pad),
                min(before.width, box[2] + pad), min(before.height, box[3] + pad))
        zw, zh = zbox[2] - zbox[0], zbox[3] - zbox[1]
        scale = 3
        strip = Image.new("RGB", (zw * scale, zh * scale * 2 + 8), (255, 255, 255))
        strip.paste(before.crop(zbox).resize((zw * scale, zh * scale), Image.LANCZOS), (0, 0))
        strip.paste(after.crop(zbox).resize((zw * scale, zh * scale), Image.LANCZOS),
                    (0, zh * scale + 8))
        out = os.path.join(BASE, "debug", "face_mask_preview.jpg")
        canvas.save(out, quality=88)
        strip.save(os.path.join(BASE, "debug", "face_mask_zoom.jpg"), quality=88)
        print(f"preview: {out}")
        print(f"zoom   : {os.path.join(BASE, 'debug', 'face_mask_zoom.jpg')}")
        return 0

    out = apply_mosaic(img, box, args.blocks)
    out.save(args.target, quality=95)
    print(f"written: {args.target}  ({os.path.getsize(args.target)} bytes)")
    return 0


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    raise SystemExit(main())
