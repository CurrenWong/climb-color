"""实验3h：0005s 单帧验证环形带黑修正（对齐 exp3g 阈值 0.18）"""
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

print("加载 EdgeSAM…")
segmenter = edgesam_holds.EdgeSamHoldSegmenter(
    r"D:\Project\climbing_wall_search\mediapipe2\weights\edge_sam.pth", device="cpu")
segmenter.set_image(image_bgr)

sh_l, sh_r, hp_l, hp_r = landmarks[11], landmarks[12], landmarks[23], landmarks[24]
body_cx = (sh_l.x + sh_r.x + hp_l.x + hp_r.x) * w / 4
body_cy = (sh_l.y + sh_r.y + hp_l.y + hp_r.y) * h / 4

for ep_name, idx in [('R_hand', 16), ('L_hand', 15)]:
    lm = landmarks[idx]
    px, py = lm.x * w, lm.y * h
    res = segmenter.segment_endpoint(px, py, body_cx, body_cy, person_mask=person_mask)
    mask = res['mask_bool']
    ix, iy = int(round(px)), int(round(py))
    yy, xx = np.ogrid[:h, :w]
    dist = np.hypot(xx - ix, yy - iy)
    ring = (dist >= 20) & (dist <= 70) & ~mask & ~person_mask.astype(bool)
    pix = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)[ring]
    black = 0
    for hh, ss, vv in pix:
        if edgesam_holds._classify_pixel(float(hh), float(ss), float(vv)) == '黑':
            black += 1
    ratio = black / len(pix)
    fixed = '黑' if (res['color_name'] != '黑' and ratio >= 0.18) else res['color_name']
    print(f"{ep_name} ({ix},{iy}): 基线={res['color_name']} area={res['area']} "
          f"ring黑票={ratio*100:.1f}% → 修正后={fixed}")
