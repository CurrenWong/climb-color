"""实验1 单元验证：_pixel_color_vote 在 0005s cand1/cand3 上的投票结果。"""
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
R_hand = landmarks[16]
px, py = R_hand.x * w, R_hand.y * h
person_mask = person_masker.segment(image_bgr)

segmenter.set_image(image_bgr)
eff_box_half = max(edgesam_holds.BOX_HALF_MIN,
                   min(h // edgesam_holds.BOX_HALF_HEIGHT_DIV, edgesam_holds.BOX_HALF_MAX))
ix = max(0, min(int(round(px)), w - 1))
iy = max(0, min(int(round(py)), h - 1))
bx1, by1 = max(0, ix - eff_box_half), max(0, iy - eff_box_half)
bx2, by2 = min(w - 1, ix + eff_box_half), min(h - 1, iy + eff_box_half)
box = np.array([bx1, by1, bx2, by2], dtype=np.float64)

neg = edgesam_holds.EdgeSamHoldSegmenter._make_negative_prompts(px, py, body_cx, body_cy)
edge_pts = edgesam_holds._sample_person_edge_points(
    person_mask, n=edgesam_holds.PERSON_EDGE_NEG_N,
    max_radius=edgesam_holds.PERSON_EDGE_MAX_RADIUS)
if len(edge_pts) > 0:
    neg = np.vstack([neg, edge_pts])
point_coords = np.vstack([[[px, py]], neg])
point_labels = np.array([1] + [0] * len(neg), dtype=np.int64)
masks, scores, _ = segmenter.predictor.predict(
    box=box, point_coords=point_coords, point_labels=point_labels,
    num_multimask_outputs=4, use_stability_score=True)
if masks.ndim == 2:
    masks = masks[None, :, :]
    scores = np.array([float(scores)])

# 端点距离权重图（实验2 会用到，先看等权）
yy, xx = np.ogrid[:h, :w]
dist_map = np.hypot(xx - ix, yy - iy)
weight_near = np.where(dist_map <= 40, 2.0, 1.0)  # r<=40 权重×2

print("=" * 70)
for i in [0, 1, 2, 3]:
    m = masks[i].astype(bool)
    # 减 person → 主连通域（对齐正式流程）
    m2 = m & ~person_mask.astype(bool)
    cc = edgesam_holds._keep_main_component(m2.copy(), ix, iy)

    # 现有判定
    hh, ss, vv = edgesam_holds._sample_color(image_bgr, cc)
    if vv < edgesam_holds.DARK_V_MAX and ss < edgesam_holds.DARK_S_MAX:
        cn = "未知(暗)"
    elif ss < edgesam_holds.WHITE_SAT_MAX:
        cn = "白" if vv >= edgesam_holds.WHITE_VAL_MIN else "黑"
    else:
        cn = edgesam_holds.route_color.hue_to_color_name(hh, ss)

    # 实验1：等权投票
    vote = edgesam_holds._pixel_color_vote(image_bgr, cc)
    # 实验1+2 预演：近端点加权投票
    vote_w = edgesam_holds._pixel_color_vote(image_bgr, cc, weight_map=weight_near)

    print(f"cand#{i} score={scores[i]:.3f} area={int(m.sum())}")
    print(f"  现有判定: HSV=({hh:.1f},{ss:.1f},{vv:.1f}) -> {cn}")
    print(f"  实验1等权: {vote['top_color']} (ratio={vote['top_ratio']:.1%})  votes={ {k: round(v) for k, v in vote['votes'].items()} }")
    print(f"  实验1加权: {vote_w['top_color']} (ratio={vote_w['top_ratio']:.1%})  votes={ {k: round(v) for k, v in vote_w['votes'].items()} }")
    print()
