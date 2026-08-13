"""
逐步复现 _sample_color 的网格采样，看每个 patch 的色相均值，
找出 frame_0130s L_hand mask 为何网格均值是 0.2（红）而全像素均值是 172.5（粉红）。
"""
import os
import sys
from pathlib import Path

_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BASE)
# EdgeSAM 源码：引用根目录共享副本（大目录不复制，实验与生产共用即可）
_ROOT = os.path.dirname(_BASE)
sys.path.insert(0, _ROOT)  # 供 edgesam_stub（根目录共享）
sys.path.insert(0, os.path.join(_ROOT, "EdgeSAM"))

import cv2
import numpy as np

import edgesam_stub  # noqa: F401
import edgesam_holds

from dwpose_onnx import DWPoseONNX
from edgesam_holds import EdgeSamHoldSegmenter
from yolo_person_mask import PersonMasker

WALL = r"..\img_xhs_walls\3OHz6rZ5opG_wall_04_line2_pink"

dw = DWPoseONNX('../yolox_l.onnx', '../dw-ll_ucoco_384.onnx')
segmenter = EdgeSamHoldSegmenter('../weights/edge_sam.pth', device='cpu')
person_masker = PersonMasker(model_name="../yolo26m-seg.pt", device='cpu')

GRID_N = edgesam_holds.GRID_N
WHITE_SAT_MAX = edgesam_holds.WHITE_SAT_MAX
COLOR_VAL_MIN = edgesam_holds.COLOR_VAL_MIN

fname = 'frame_0130s.jpg'
img_path = Path(WALL) / fname
image_bgr = cv2.imread(str(img_path))
boxes, kps, scores = dw.run(image_bgr)
k, s = kps[0], scores[0]
body_pts = k[[5, 6, 11, 12]]
body_cx = float(body_pts[:, 0].mean())
body_cy = float(body_pts[:, 1].mean())
person_mask = person_masker.segment(image_bgr)
segmenter.set_image(image_bgr)

px, py = float(k[9, 0]), float(k[9, 1])
res = segmenter.segment_endpoint(px, py, body_cx, body_cy, person_mask=person_mask)
mask = res['mask_bool']
hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

ys, xs = np.where(mask)
bx0, bx1 = xs.min(), xs.max()
by0, by1 = ys.min(), ys.max()
bw = max(1, bx1 - bx0)
bh = max(1, by1 - by0)
print(f"mask bbox: x[{bx0},{bx1}] y[{by0},{by1}]  bw={bw} bh={bh}")
print(f"mask 像素数: {len(xs)}")

hues, sats, vals = [], [], []
patch_details = []
for gy in range(GRID_N):
    for gx in range(GRID_N):
        cx_g = bx0 + (gx + 0.5) * bw / GRID_N
        cy_g = by0 + (gy + 0.5) * bh / GRID_N
        ix_p = int(round(cx_g))
        iy_p = int(round(cy_g))
        if not (2 <= ix_p < hsv.shape[1] - 2 and 2 <= iy_p < hsv.shape[0] - 2):
            patch_details.append((gx, gy, ix_p, iy_p, 'out_of_range', None))
            continue
        patch = hsv[iy_p - 2:iy_p + 3, ix_p - 2:ix_p + 3].reshape(-1, 3)
        # patch 内哪些像素在 mask 内？
        mask_patch = mask[iy_p - 2:iy_p + 3, ix_p - 2:ix_p + 3].reshape(-1)
        keep = (patch[:, 1] >= WHITE_SAT_MAX) & (patch[:, 2] >= COLOR_VAL_MIN)
        in_mask_keep = keep & mask_patch
        if keep.sum() < 3:
            patch_details.append((gx, gy, ix_p, iy_p, f'skip keep={keep.sum()}', None))
            continue
        pix = patch[keep]
        hr = pix[:, 0].astype(np.float64) * (2 * np.pi / 180.0)
        h_mean = float(np.degrees(np.arctan2(
            np.mean(np.sin(hr)), np.mean(np.cos(hr))
        )) / 2.0) % 180.0
        s_med = float(np.median(pix[:, 1]))
        v_med = float(np.median(pix[:, 2]))
        hues.append(h_mean)
        sats.append(s_med)
        vals.append(v_med)
        # patch 内 mask 像素色相
        mp = patch[mask_patch]
        if len(mp):
            mp_h = float(np.median(mp[:, 0]))
        else:
            mp_h = float('nan')
        patch_details.append((gx, gy, ix_p, iy_p,
                              f'keep={keep.sum()}(inmask={in_mask_keep.sum()}) h={h_mean:5.1f} s={s_med:5.1f} v={v_med:5.1f} maskHmed={mp_h:.0f}',
                              h_mean))

print(f"\n有效 patch 数: {len(hues)}")
for d in patch_details:
    gx, gy, ix, iy, msg, h = d
    print(f"  g=({gx},{gy}) center=({ix},{iy}) {msg}")

if hues:
    hr2 = np.array(hues) * (2 * np.pi / 180.0)
    h_final = float(np.degrees(np.arctan2(
        np.mean(np.sin(hr2)), np.mean(np.cos(hr2))
    )) / 2.0) % 180.0
    print(f"\n网格圆形均值 H = {h_final:.1f}  S中位={np.median(sats):.1f}  V中位={np.median(vals):.1f}")
    print(f"网格普通均值 H = {np.mean(hues):.1f}")
    print(f"网格 H 中位 = {np.median(hues):.1f}")

# 全像素圆形均值（H 全部，S>=35 & V>=40 过滤后）
full = hsv[ys, xs]
full_keep = (full[:, 1] >= WHITE_SAT_MAX) & (full[:, 2] >= COLOR_VAL_MIN)
full_k = full[full_keep]
if len(full_k):
    hr = full_k[:, 0].astype(np.float64) * (2 * np.pi / 180.0)
    h_full = float(np.degrees(np.arctan2(
        np.mean(np.sin(hr)), np.mean(np.cos(hr))
    )) / 2.0) % 180.0
    print(f"\n全像素(过滤后 n={len(full_k)}) 圆形均值 H = {h_full:.1f}  S中位={np.median(full_k[:,1]):.1f}  V中位={np.median(full_k[:,2]):.1f}")
