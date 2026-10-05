"""只加载 DWPose（不加载 torch 模型），测 run 耗时。"""
import os
import sys
import time

_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path[0:0] = [_BASE, os.path.dirname(_BASE), os.path.join(os.path.dirname(_BASE), "EdgeSAM")]
os.chdir(_BASE)

import cv2
import edgesam_stub  # noqa: F401
from dwpose_onnx import DWPoseONNX

img = cv2.imread(r"..\img_xhs_walls\2XRo7VbPgTe_wall_01_known_IMG_20260808_193753\frame_0005s.jpg")
dw = DWPoseONNX('../yolox_l.onnx', '../dw-ll_ucoco_384.onnx')

for i in range(3):
    t = time.time()
    boxes, kps, scores = dw.run(img)
    print(f"dw.run #{i}: {time.time()-t:.2f}s  boxes={len(boxes) if boxes is not None else 0}")

# 分阶段
t = time.time(); boxes = dw.detect(img); print(f"detect only: {time.time()-t:.2f}s")
t = time.time(); kp, sc = dw.pose(img, boxes[:1]); print(f"pose only  : {time.time()-t:.2f}s")
