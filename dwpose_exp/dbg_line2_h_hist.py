"""
分析 3O line2_pink frame2/frame3 L_hand mask 内像素的色相分布。
确认: 粉色 hold 的 H 集中在哪个区间（150-180? 170+?），
以及圆形均值为何算到 0/5（红）。
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

WALL = r"..\img_xhs_walls\3OHz6rZ5opG_wall_04_line2_pink"

dw = DWPoseONNX('../yolox_l.onnx', '../dw-ll_ucoco_384.onnx')
segmenter = EdgeSamHoldSegmenter('../weights/edge_sam.pth', device='cpu')
person_masker = PersonMasker(model_name="../yolo26m-seg.pt", device='cpu')

for idx, fname in [(1, 'frame_0130s.jpg'), (2, 'frame_0135s.jpg')]:
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
    mask_bool = res['mask_bool']
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    pix = hsv[mask_bool]

    print(f"\n=== {fname} L_hand mask n={len(pix)} ===")
    h_all = pix[:, 0].astype(float)
    s_all = pix[:, 1].astype(float)
    v_all = pix[:, 2].astype(float)

    # 全像素直方图（H bin 10°）
    hist, edges = np.histogram(h_all, bins=18, range=(0, 180))
    print("H 直方图 (0-180, 10° bin):")
    for i in range(18):
        lo, hi = int(edges[i]), int(edges[i + 1])
        bar = '#' * min(hist[i] // max(1, int(hist.max() / 40)), 40)
        print(f"  [{lo:3d}-{hi:3d}): {hist[i]:6d} {bar}")

    # 关注 H>=140 的像素（粉红/红/紫候选）
    mask_hi = h_all >= 140
    print(f"\nH>=140 像素: {mask_hi.sum()} ({mask_hi.sum()/len(pix)*100:.0f}%)")
    if mask_hi.sum() > 0:
        print(f"  其中 H>=150: {(h_all>=150).sum()}  H>=160: {(h_all>=160).sum()}  H>=170: {(h_all>=170).sum()}")
        print(f"  H>=140 像素的 S 分布: 中位={np.median(s_all[mask_hi]):.0f} "
              f"P25={np.percentile(s_all[mask_hi],25):.0f} P75={np.percentile(s_all[mask_hi],75):.0f}")
        print(f"  H>=140 像素的 V 分布: 中位={np.median(v_all[mask_hi]):.0f}")

    # 色相圆形均值（复现 _sample_color 的算法）
    hr = h_all * (2 * np.pi / 180.0)
    h_circ = float(np.degrees(np.arctan2(np.mean(np.sin(hr)), np.mean(np.cos(hr)))) / 2.0) % 180.0
    print(f"\n全像素圆形均值 H = {h_circ:.1f}  (S中位={np.median(s_all):.0f}, V中位={np.median(v_all):.0f})")
    print(f"全像素普通均值 H = {np.mean(h_all):.1f}")
    print(f"全像素 H 中位 = {np.median(h_all):.1f}")
