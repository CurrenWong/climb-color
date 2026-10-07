"""分解 detect() 内部各步骤耗时。"""
import os
import sys
import time

_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path[0:0] = [_BASE, os.path.dirname(_BASE), os.path.join(os.path.dirname(_BASE), "EdgeSAM")]
os.chdir(_BASE)

import cv2
import numpy as np
import edgesam_stub  # noqa: F401
from dwpose_onnx import DWPoseONNX, DET_INPUT_SIZE, DET_NMS_THR, DET_SCORE_THR

img = cv2.imread(r"..\img_xhs_walls\2XRo7VbPgTe_wall_01_known_IMG_20260808_193753\frame_0005s.jpg")
dw = DWPoseONNX('../weights/yolox_l.onnx', '../weights/dw-ll_ucoco_384.onnx')

# warmup
_ = dw.detect(img)

t = time.time()
pre, ratio = dw._det_preprocess(img, DET_INPUT_SIZE)
inp = pre[None, :, :, :].astype(np.float32)
print(f"1. preprocess      : {time.time()-t:6.2f}s  input={inp.shape}")

name = dw.det_sess.get_inputs()[0].name
t = time.time()
output = dw.det_sess.run(None, {name: inp})[0]
print(f"2. session.run     : {time.time()-t:6.2f}s  out={output.shape}")

t = time.time()
pred = dw._demo_postprocess(output, DET_INPUT_SIZE)[0]
print(f"3. postprocess     : {time.time()-t:6.2f}s  pred={pred.shape}")

t = time.time()
obj = 1.0 / (1.0 + np.exp(-pred[:, 4]))
cls = 1.0 / (1.0 + np.exp(-pred[:, 5:]))
scores = obj[:, None] * cls
print(f"4. sigmoid         : {time.time()-t:6.2f}s  scores={scores.shape}")

boxes = pred[:, :4]
bxyxy = np.ones_like(boxes)
bxyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
bxyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
bxyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
bxyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
bxyxy /= ratio

t = time.time()
dets = dw._multiclass_nms(bxyxy, scores, DET_NMS_THR, DET_SCORE_THR)
print(f"5. multiclass_nms  : {time.time()-t:6.2f}s  dets={None if dets is None else dets.shape}")
