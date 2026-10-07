"""实验2 单元验证：segment_endpoint 的 fusion 字段在 0005s R_hand 的结果。"""
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

segmenter = edgesam_holds.EdgeSamHoldSegmenter(
    r"D:\Project\climbing_wall_search\mediapipe2\weights\edge_sam.pth", device="cpu")
person_masker = PersonMasker(model_name="weights/yolo26m-seg.pt", device="cpu")

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
sh_l, sh_r, hp_l, hp_r = landmarks[11], landmarks[12], landmarks[23], landmarks[24]
body_cx = (sh_l.x + sh_r.x + hp_l.x + hp_r.x) * w / 4
body_cy = (sh_l.y + sh_r.y + hp_l.y + hp_r.y) * h / 4
person_mask = person_masker.segment(image_bgr)
segmenter.set_image(image_bgr)

for ep_name, idx in [('R_hand', 16), ('L_hand', 15)]:
    lm = landmarks[idx]
    px, py = lm.x * w, lm.y * h
    res = segmenter.segment_endpoint(px, py, body_cx, body_cy, person_mask=person_mask)

    print(f"\n{'='*66}\n{ep_name} ({px:.0f},{py:.0f})")
    print(f"  现有判定: {res['color_name']}  HSV=({res['h']:.1f},{res['s']:.1f},{res['v']:.1f})  area={res['area']}")
    print(f"  实验1投票: {res['color_vote']['top_color']} (ratio={res['color_vote']['top_ratio']:.1%})  "
          f"votes={ {k: round(v) for k, v in res['color_vote']['votes'].items()} }")

    fu = res['fusion']
    print(f"  实验2融合:")
    for p in fu['per_candidate']:
        print(f"    cand: score={p['score']:.3f} area={p['area']} top={p['top_color']} "
              f"(ratio={p['top_ratio']:.1%})")
    for mode, col in fu['results'].items():
        if mode == 'pixel_union_votes':
            continue
        print(f"    [{mode}] -> {col}")
