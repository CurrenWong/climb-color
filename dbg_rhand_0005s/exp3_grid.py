"""实验3 网格锚点多数投票：端点周围网格锚点各自判 top 色，多数派决定。"""
import os, sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "EdgeSAM"))
import edgesam_holds

IMG = r"D:\Project\climbing_wall_search\mediapipe2\img_xhs_walls\2XRo7VbPgTe_wall_01_known_IMG_20260808_193753\frame_0005s.jpg"
image_bgr = cv2.imread(IMG)
h, w = image_bgr.shape[:2]

import mediapipe as mp
from test_edge_sam import detect_pose
from yolo_person_mask import PersonMasker

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
landmarker = PoseLandmarker.create_from_options(
    PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=r"D:\Project\climbing_wall_search\mediapipe2\pose_landmarker_full.task"),
        running_mode=VisionRunningMode.IMAGE, num_poses=1,
        min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3))

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
landmarks = detect_pose(image_rgb, landmarker, w, h, flip_ensemble=True)
person_masker = PersonMasker(model_name="yolo26m-seg.pt", device="cpu")
person_mask = person_masker.segment(image_bgr)

hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
yy, xx = np.ogrid[:h, :w]


def anchor_top(cx, cy, r=25):
    """单锚点圆盘 top 色（排除 person mask），像素太少返回 None"""
    cx, cy = int(cx), int(cy)
    disk = (xx - cx)**2 + (yy - cy)**2 <= r*r
    disk = disk & ~person_mask.astype(bool)
    if disk.sum() < 20:
        return None
    pix = hsv[disk]
    votes: dict[str, float] = {}
    for hh, ss, vv in pix:
        name = edgesam_holds._classify_pixel(float(hh), float(ss), float(vv))
        votes[name] = votes.get(name, 0) + 1
    return max(votes, key=votes.get)


def grid_majority(px, py, half=60, step=30, r=25):
    """端点周围网格锚点, 每个取 top 色, 多数投票"""
    counts: dict[str, int] = {}
    total = 0
    for ox in range(-half, half + 1, step):
        for oy in range(-half, half + 1, step):
            top = anchor_top(px + ox, py + oy, r=r)
            if top is None:
                continue
            counts[top] = counts.get(top, 0) + 1
            total += 1
    if not counts:
        return None
    winner = max(counts, key=counts.get)
    return winner, counts, total


for ep_name, idx in [('R_hand', 16), ('L_hand', 15), ('R_foot', 32), ('L_foot', 31)]:
    lm = landmarks[idx]
    px, py = lm.x * w, lm.y * h
    if lm.visibility < 0.3:
        print(f"\n{ep_name}: 低置信度")
        continue
    print(f"\n{ep_name} ({px:.0f},{py:.0f})")

    for half, step in [(60, 30), (80, 40), (90, 30)]:
        res = grid_majority(px, py, half=half, step=step, r=25)
        if res:
            winner, counts, total = res
            print(f"  网格 half={half} step={step}: 多数={winner:4s} ({total} 锚点) 分布={ {k: v for k, v in sorted(counts.items(), key=lambda x: -x[1])} }")
        else:
            print(f"  网格 half={half} step={step}: 无有效锚点")
