"""
验证：DWPose 架构（onnxruntime + EdgeSAM + YOLO seg，无 MediaPipe）是否还需要 os._exit(0)。

测试方法：模拟 exp_dwpose_full.py 的组件加载，正常 return 退出（不调 os._exit），
看解释器能否干净退出（exit code 0 且不卡死），并打印退出前活跃线程。
"""
import os
import sys
import threading
import time

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


def list_threads(tag):
    print(f"\n[{tag}] 活跃线程 ({threading.active_count()}):")
    for t in threading.enumerate():
        print(f"  - {t.name} (daemon={t.daemon})")


def main():
    IMG = r"..\..\video_frames_cache\小红书_2XRo7VbPgTe\walls\wall_01_known_IMG_20260808_193753\frame_0005s.jpg"
    image_bgr = cv2.imread(IMG)

    # ---- 按 exp_dwpose_full.py 顺序加载组件 ----
    t0 = time.time()
    dw = DWPoseONNX('../weights/yolox_l.onnx', '../weights/dw-ll_ucoco_384.onnx')
    segmenter = EdgeSamHoldSegmenter('../weights/edge_sam.pth', device='cpu')
    person_masker = PersonMasker(model_name="../weights/yolo26m-seg.pt", device='cpu')
    print(f"组件加载完成: {time.time()-t0:.1f}s")

    list_threads("加载后")

    # ---- 跑一帧完整流程 ----
    t0 = time.time()
    boxes, kps, scores = dw.run(image_bgr)
    print(f"DWPose 推理: {time.time()-t0:.1f}s, kps={None if kps is None else kps.shape}")

    person_mask = person_masker.segment(image_bgr)
    segmenter.set_image(image_bgr)

    if kps is not None:
        k, s = kps[0], scores[0]
        body_cx = float(k[[5, 6, 11, 12], 0].mean())
        body_cy = float(k[[5, 6, 11, 12], 1].mean())
        for ep, kidx in [('L_hand', 9), ('R_hand', 10), ('L_foot', 17), ('R_foot', 20)]:
            res = segmenter.segment_endpoint(float(k[kidx, 0]), float(k[kidx, 1]),
                                             body_cx, body_cy, person_mask=person_mask)
            name = res['color_name'] if res else 'None'
            print(f"  {ep}: {name}")
    print(f"一帧全流程: {time.time()-t0:.1f}s")

    list_threads("推理后")

    print("\n✅ 主流程完成，正常 return（不调 os._exit）")
    return 0


if __name__ == '__main__':
    rc = main()
    print(f"exit code = {rc}")
    # 不调用 os._exit，让解释器自然清理退出
    sys.exit(rc)
