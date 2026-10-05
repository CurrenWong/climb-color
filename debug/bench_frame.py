"""单帧耗时分解 benchmark：DWPose / person mask / EdgeSAM 各阶段。"""
import os
import sys
import time

_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path[0:0] = [_BASE, os.path.dirname(_BASE), os.path.join(os.path.dirname(_BASE), "EdgeSAM")]

import cv2
import numpy as np
import edgesam_stub  # noqa: F401
from dwpose_onnx import DWPoseONNX
from edgesam_holds import EdgeSamHoldSegmenter
from yolo_person_mask import PersonMasker

os.chdir(_BASE)
IMG = r"..\img_xhs_walls\2XRo7VbPgTe_wall_01_known_IMG_20260808_193753\frame_0005s.jpg"

def tic(): return time.time()

img = cv2.imread(IMG)
print(f"image shape: {img.shape}")

t = tic(); dw = DWPoseONNX('../yolox_l.onnx', '../dw-ll_ucoco_384.onnx'); print(f"load DWPose      : {tic()-t:6.2f}s")
t = tic(); seg = EdgeSamHoldSegmenter('../weights/edge_sam.pth', device='cpu'); print(f"load EdgeSAM     : {tic()-t:6.2f}s")
t = tic(); pm = PersonMasker(model_name="../yolo26m-seg.pt", device='cpu'); print(f"load PersonMasker: {tic()-t:6.2f}s")

# warmup
_ = dw.run(img)
_ = pm.segment(img)
seg.set_image(img)

print("--- per-frame stages ---")
t = tic(); boxes, kps, scores = dw.run(img); print(f"DWPose.run       : {tic()-t:6.2f}s")
t = tic(); person = pm.segment(img);       print(f"person.segment   : {tic()-t:6.2f}s")
t = tic(); seg.set_image(img);             print(f"EdgeSAM.set_image: {tic()-t:6.2f}s")

k = kps[0]
pts = k[[5, 6, 11, 12]]
body_cx, body_cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())

for ep_name, kidx in [('L_hand', 9), ('R_hand', 10), ('L_foot', 17), ('R_foot', 20)]:
    px, py = float(k[kidx, 0]), float(k[kidx, 1])
    t = tic()
    res = seg.segment_endpoint(px, py, body_cx, body_cy, person_mask=person)
    print(f"  segment_endpoint {ep_name:7s}: {tic()-t:6.2f}s  area={res.get('area')}")
print("done")
