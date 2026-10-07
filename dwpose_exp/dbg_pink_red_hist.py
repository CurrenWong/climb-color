"""
对比 line2_pink (真粉红) vs 3O wall_03 (真红) 端点 mask 的 H/S 分布。
决定 (170,180) 高饱和区间内粉红/红如何区分。
"""
import os
import sys
from pathlib import Path

_BASE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_BASE)
# sys.path 顺序: _BASE (dwpose_exp 本地副本) 优先, 然后根目录
# (edgesam_stub 等共享模块), 最后 EdgeSAM 源码。注意: 必须用
# 一次 slice insert 保留原 sys.path, 多次 insert(0, ...) 会反向
# 把 _BASE 推到后面, 导致 dwpose_exp 本地副本永远不被加载。
sys.path[0:0] = [_BASE, _ROOT, os.path.join(_ROOT, "EdgeSAM")]

import cv2
import numpy as np

import edgesam_stub  # noqa: F401

from dwpose_onnx import DWPoseONNX
from edgesam_holds import EdgeSamHoldSegmenter
from yolo_person_mask import PersonMasker

# (label, wall_dir, frame, endpoint_kidx)
CASES = [
    ('line2_pink 0130s L_hand', r"..\img_xhs_walls\3OHz6rZ5opG_wall_04_line2_pink\frame_0130s.jpg", 9),
    ('wall03_red 0090s R_hand', r"..\img_xhs_walls\3OHz6rZ5opG_wall_03_known_IMG_20260808_193612\frame_0090s.jpg", 10),
    ('2X03_pink 0105s L_hand', r"..\img_xhs_walls\2XRo7VbPgTe_wall_03_known_IMG_20260808_193754\frame_0105s.jpg", 9),
]

dw = DWPoseONNX('../weights/yolox_l.onnx', '../weights/dw-ll_ucoco_384.onnx')
segmenter = EdgeSamHoldSegmenter('../weights/edge_sam.pth', device='cpu')
person_masker = PersonMasker(model_name="../weights/yolo26m-seg.pt", device='cpu')

for label, img_path_s, kidx in CASES:
    image_bgr = cv2.imread(img_path_s)
    boxes, kps, scores = dw.run(image_bgr)
    k, s = kps[0], scores[0]
    body_pts = k[[5, 6, 11, 12]]
    body_cx = float(body_pts[:, 0].mean())
    body_cy = float(body_pts[:, 1].mean())
    person_mask = person_masker.segment(image_bgr)
    segmenter.set_image(image_bgr)

    px, py = float(k[kidx, 0]), float(k[kidx, 1])
    res = segmenter.segment_endpoint(px, py, body_cx, body_cy, person_mask=person_mask)
    mask = res['mask_bool']
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    pix = hsv[mask]

    print(f"\n=== {label}  (res={res['color_name']} H={res['h']:.1f} S={res['s']:.1f}) n={len(pix)} ===")
    h_all = pix[:, 0].astype(float)
    s_all = pix[:, 1].astype(float)
    v_all = pix[:, 2].astype(float)

    hist, edges = np.histogram(h_all, bins=18, range=(0, 180))
    for i in range(18):
        lo, hi = int(edges[i]), int(edges[i + 1])
        bar = '#' * min(hist[i] // max(1, int(hist.max() / 40)), 40)
        print(f"  [{lo:3d}-{hi:3d}): {hist[i]:6d} {bar}")

    # 高饱和像素 (>180) 的 H 分布
    hi_sat = s_all > 180
    print(f"\n  S>180 像素: {hi_sat.sum()} ({hi_sat.sum()/len(pix)*100:.0f}%)")
    if hi_sat.sum() > 0:
        h_hi = h_all[hi_sat]
        hh, ee = np.histogram(h_hi, bins=18, range=(0, 180))
        for i in range(18):
            if hh[i]:
                print(f"    H[{int(ee[i]):3d}-{int(ee[i+1]):3d}): {hh[i]}")
