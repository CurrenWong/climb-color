"""分阶段 CPU profiling —— 为论文 Table 5 (latency breakdown) 提供实测数据。

分阶段: DWPose detect / DWPose pose / YOLO26m person mask /
        EdgeSAM set_image / EdgeSAM 4x segment_endpoint
第 1 帧作为 warmup 不计入统计。

用法: python dbg_profile_frame.py
输出: profile_stages.json
"""
import os
import sys
import time
import json
from pathlib import Path

_BASE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_BASE)
sys.path[0:0] = [_BASE, _ROOT, os.path.join(_ROOT, "EdgeSAM")]

import cv2
import numpy as np
import edgesam_stub  # noqa: F401

from dwpose_onnx import DWPoseONNX
from edgesam_holds import EdgeSamHoldSegmenter
from yolo_person_mask import PersonMasker

WALL_FRAMES = (Path(_BASE) / ".." / "img_xhs_walls"
               / "2XRo7VbPgTe_wall_01_known_IMG_20260808_193753")
DW_ENDPOINTS = [('L_hand', 9), ('R_hand', 10), ('L_foot', 17), ('R_foot', 20)]
DW_SHOULDERS = (5, 6)
DW_HIPS = (11, 12)
CONF_THRESHOLD = 0.30
N_TIMED = 3

KEYS = ('detect', 'pose', 'person_mask', 'edge_encode', 'endpoints', 'total')


def main():
    files = sorted(WALL_FRAMES.glob("*.jpg"))[:N_TIMED + 1]
    print(f"frames found: {len(files)} (frame 0 = warmup, excluded)")

    print("loading models (untimed) ...")
    dw = DWPoseONNX('../weights/yolox_l.onnx', '../weights/dw-ll_ucoco_384.onnx')
    segmenter = EdgeSamHoldSegmenter('../weights/edge_sam.pth', device='cpu')
    pm = PersonMasker(model_name="../weights/yolo26m-seg.pt", device='cpu')

    acc = []
    for i, fp in enumerate(files):
        img = cv2.imread(str(fp))
        r = {}

        t = time.perf_counter()
        boxes = dw.detect(img)
        r['detect'] = time.perf_counter() - t

        t = time.perf_counter()
        kps, scores = dw.pose(img, boxes[:1])
        r['pose'] = time.perf_counter() - t

        t = time.perf_counter()
        person_mask = pm.segment(img)
        r['person_mask'] = time.perf_counter() - t

        t = time.perf_counter()
        segmenter.set_image(img)
        r['edge_encode'] = time.perf_counter() - t

        n_seg, n_cand = 0, 0
        t = time.perf_counter()
        if kps is not None:
            k, s = kps[0], scores[0]
            pts = k[list(DW_SHOULDERS) + list(DW_HIPS)]
            bcx, bcy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
            for ep, kidx in DW_ENDPOINTS:
                if float(s[kidx]) < CONF_THRESHOLD:
                    continue
                x, y = float(k[kidx][0]), float(k[kidx][1])
                try:
                    res = segmenter.segment_endpoint(x, y, bcx, bcy,
                                                     person_mask=person_mask)
                    if res is not None:
                        n_cand += 1
                        if res.get('mask') is not None:
                            n_seg += 1
                except Exception as e:
                    print(f"  seg fail {ep}: {e}")
        r['endpoints'] = time.perf_counter() - t
        r['n_seg'] = n_seg
        r['n_cand'] = n_cand
        r['total'] = sum(r[k] for k in KEYS[:-1])

        tag = " [warmup]" if i == 0 else ""
        print(f"  frame {i} (seg={n_seg}): " +
              "  ".join(f"{k}={r[k]:.2f}" for k in KEYS) + tag)
        if i > 0:
            acc.append(r)

    mean = {k: float(np.mean([a[k] for a in acc])) for k in KEYS}
    mean['n_seg'] = float(np.mean([a['n_seg'] for a in acc]))
    print(f"\n=== mean over {len(acc)} timed frames ===")
    for k in KEYS:
        print(f"  {k:<14} {mean[k]:>6.2f} s")
    print(f"  {'segments/frame':<14} {mean['n_seg']:>6.1f}")

    out = os.path.join(_BASE, "profile_stages.json")
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'mean': mean, 'raw': acc, 'n_timed': len(acc),
                   'wall': WALL_FRAMES.name,
                   'note': 'CPU Intel Core i5-13500H; warmup frame excluded; '
                           'total = sum of stages (excludes I/O and drawing)'},
                  f, ensure_ascii=False, indent=2)
    print("saved", out)


if __name__ == '__main__':
    main()
