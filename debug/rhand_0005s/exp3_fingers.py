"""实验3 手指关节采样：用食指/拇指/小指指尖 landmark 替代手腕采样。"""
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
        base_options=BaseOptions(model_asset_path=r"D:\Project\climbing_wall_search\mediapipe2\weights\pose_landmarker_full.task"),
        running_mode=VisionRunningMode.IMAGE, num_poses=1,
        min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3))

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
landmarks = detect_pose(image_rgb, landmarker, w, h, flip_ensemble=True)
person_masker = PersonMasker(model_name="weights/yolo26m-seg.pt", device="cpu")
person_mask = person_masker.segment(image_bgr)

hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
yy, xx = np.ogrid[:h, :w]

# PoseLandmarker 33 点索引
# 15=L_wrist 16=R_wrist 17=L_pinky 18=R_pinky 19=L_index 20=R_index 21=L_thumb 22=R_thumb

JOINT_NAMES = {
    15: 'L_wrist', 16: 'R_wrist',
    17: 'L_pinky', 18: 'R_pinky',
    19: 'L_index', 20: 'R_index',
    21: 'L_thumb', 22: 'R_thumb',
}


def anchor_top(cx, cy, r=25, use_mask=True):
    cx, cy = int(cx), int(cy)
    disk = (xx - cx)**2 + (yy - cy)**2 <= r*r
    if use_mask:
        disk = disk & ~person_mask.astype(bool)
    if disk.sum() < 20:
        return None
    pix = hsv[disk]
    votes: dict[str, float] = {}
    for hh, ss, vv in pix:
        name = edgesam_holds._classify_pixel(float(hh), float(ss), float(vv))
        votes[name] = votes.get(name, 0) + 1
    total = sum(votes.values())
    return max(votes, key=votes.get), votes, total


# 手部关节组
hand_groups = {
    'R': [16, 18, 20, 22],   # wrist + pinky + index + thumb
    'L': [15, 17, 19, 21],
}

for side, idxs in hand_groups.items():
    print(f"\n===== {side}_hand 各关节采样 =====")
    for idx in idxs:
        lm = landmarks[idx]
        if lm.visibility < 0.3:
            print(f"  {JOINT_NAMES[idx]} ({idx}): 低置信度")
            continue
        px, py = lm.x * w, lm.y * h
        res = anchor_top(px, py, r=25)
        if res:
            top, votes, total = res
            print(f"  {JOINT_NAMES[idx]} ({idx}) ({px:.0f},{py:.0f}): top={top:4s} "
                  f"黑={votes.get('黑',0)/total*100:.1f}% 橙={votes.get('橙',0)/total*100:.1f}% "
                  f"白={votes.get('白',0)/total*100:.1f}%")

    # 组合投票: wrist + index + thumb (+pinky) 合并像素
    print(f"  --- {side}_hand 关节合并投票 ---")
    merged: dict[str, float] = {}
    total_pix = 0
    for idx in idxs:
        lm = landmarks[idx]
        if lm.visibility < 0.3:
            continue
        px, py = lm.x * w, lm.y * h
        res = anchor_top(px, py, r=25)
        if not res:
            continue
        _, votes, t = res
        total_pix += t
        for k, v in votes.items():
            merged[k] = merged.get(k, 0) + v
    if merged:
        top = max(merged, key=merged.get)
        print(f"  合并 {total_pix} px: top={top:4s} "
              f"黑={merged.get('黑',0)/total_pix*100:.1f}% 橙={merged.get('橙',0)/total_pix*100:.1f}% "
              f"白={merged.get('白',0)/total_pix*100:.1f}%")

    # 关节各自 top 色多数投票
    print(f"  --- {side}_hand 关节 top 色多数 ---")
    counts: dict[str, int] = {}
    for idx in idxs:
        lm = landmarks[idx]
        if lm.visibility < 0.3:
            continue
        px, py = lm.x * w, lm.y * h
        res = anchor_top(px, py, r=25)
        if not res:
            continue
        top, _, _ = res
        counts[top] = counts.get(top, 0) + 1
    if counts:
        print(f"  关节 top 色分布: {counts}")
