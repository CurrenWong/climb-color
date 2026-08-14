"""
采样 2X_03 (粉红) / 3O wall_03 (红) 端点 mask 的 HSV，确定粉红/红边界。
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

WALLS = {
    '2X_03 粉红': r"..\img_xhs_walls\2XRo7VbPgTe_wall_03_known_IMG_20260808_193754",
    '3O wall_03 红': r"..\img_xhs_walls\3OHz6rZ5opG_wall_03_known_IMG_20260808_193612",
}

dw = DWPoseONNX('../yolox_l.onnx', '../dw-ll_ucoco_384.onnx')
segmenter = EdgeSamHoldSegmenter('../weights/edge_sam.pth', device='cpu')
person_masker = PersonMasker(model_name="../yolo26m-seg.pt", device='cpu')

for label, wall_dir in WALLS.items():
    frames = sorted(Path(wall_dir).glob('*.jpg'))
    print(f"\n{'='*60}\n{label} ({wall_dir.split(chr(92))[-1]}) {len(frames)} 帧")
    for fp in frames[:6]:
        image_bgr = cv2.imread(str(fp))
        boxes, kps, scores = dw.run(image_bgr)
        if kps is None:
            print(f"  {fp.name}: 无人体")
            continue
        k, s = kps[0], scores[0]
        body_pts = k[[5, 6, 11, 12]]
        body_cx = float(body_pts[:, 0].mean())
        body_cy = float(body_pts[:, 1].mean())
        person_mask = person_masker.segment(image_bgr)
        segmenter.set_image(image_bgr)

        parts = []
        for ep_name, kidx in [('L_hand', 9), ('R_hand', 10)]:
            px, py = float(k[kidx, 0]), float(k[kidx, 1])
            if float(s[kidx]) < 0.3:
                parts.append(f"{ep_name}:低conf")
                continue
            res = segmenter.segment_endpoint(px, py, body_cx, body_cy, person_mask=person_mask)
            if res is None or res['area'] < 100:
                parts.append(f"{ep_name}:非岩点")
                continue
            h, ss, v = res['h'], res['s'], res['v']
            parts.append(f"{ep_name}:{res['color_name']}(H={h:.0f},S={ss:.0f},V={v:.0f})")
        print(f"  {fp.name}: {' | '.join(parts)}")
