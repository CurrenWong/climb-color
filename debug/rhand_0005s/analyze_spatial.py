"""分析 cand3 mask 内黑像素空间分布 + 端点圆盘内颜色构成，为通用方案提供依据。"""
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
person_masker = PersonMasker(model_name="yolo26m-seg.pt", device="cpu")

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

hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

for i in [1, 3]:
    m = masks[i].astype(bool)
    print(f"\n{'='*60}\nCAND#{i}  score={scores[i]:.3f}  area={int(m.sum())}")

    # 黑像素在 mask 内的空间分布
    mask_hsv = hsv[m]
    black = (mask_hsv[:, 1] < 35) & (mask_hsv[:, 2] < 140)
    white = (mask_hsv[:, 1] < 35) & (mask_hsv[:, 2] >= 140)
    orange = (mask_hsv[:, 0] >= 8) & (mask_hsv[:, 0] <= 30) & (mask_hsv[:, 1] > 60) & (mask_hsv[:, 2] > 100)
    ys, xs = np.where(m)
    print(f"mask bbox: x=[{xs.min()},{xs.max()}] y=[{ys.min()},{ys.max()}]")

    for nm, cond in [("黑", black), ("白", white), ("橙", orange)]:
        idx = np.where(m)
        sub_ys, sub_xs = idx[0][cond], idx[1][cond]
        if len(sub_xs) == 0:
            print(f"  {nm}: 0 px")
            continue
        dist = np.hypot(sub_xs - px, sub_ys - py)
        print(f"  {nm}: {len(sub_xs):5d} px  bbox=({sub_xs.min()},{sub_ys.min()})-({sub_xs.max()},{sub_ys.max()})  "
              f"中心=({sub_xs.mean():.0f},{sub_ys.mean():.0f})  距端点中位={np.median(dist):.0f}px  max={dist.max():.0f}px")

    # 端点不同半径圆盘内（mask 交集）颜色构成
    print(f"  端点圆盘 ∩ mask 颜色构成:")
    for r in [20, 30, 40, 50, 60, 80]:
        disk = np.zeros_like(m)
        yy, xx = np.ogrid[:h, :w]
        disk[(xx - ix)**2 + (yy - iy)**2 <= r*r] = True
        inter = disk & m
        if inter.sum() < 20:
            print(f"    r={r:3d}: {inter.sum():5d} px")
            continue
        pix = hsv[inter]
        n_black = ((pix[:,1] < 35) & (pix[:,2] < 140)).sum()
        n_white = ((pix[:,1] < 35) & (pix[:,2] >= 140)).sum()
        n_orange = ((pix[:,0] >= 8) & (pix[:,0] <= 30) & (pix[:,1] > 60) & (pix[:,2] > 100)).sum()
        print(f"    r={r:3d}: {inter.sum():5d} px  黑={n_black:5d}({n_black/inter.sum()*100:4.1f}%)  "
              f"白={n_white:4d}({n_white/inter.sum()*100:4.1f}%)  橙={n_orange:5d}({n_orange/inter.sum()*100:4.1f}%)")

    # 若只用 mask 而不受圆盘限制，黑像素占比
    print(f"  全 mask 黑占比={black.sum()/m.sum()*100:.1f}%")
