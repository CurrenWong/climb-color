"""实验3e：腕→拇指方向 多正点（姿态自适应方向）。

R_hand 拇指方向水平向左命中黑岩点；验证 L_hand 是否被破坏。
"""
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

print("加载 EdgeSAM…")
segmenter = edgesam_holds.EdgeSamHoldSegmenter(
    r"D:\Project\climbing_wall_search\mediapipe2\weights\edge_sam.pth", device="cpu")
segmenter.set_image(image_bgr)

sh_l, sh_r, hp_l, hp_r = landmarks[11], landmarks[12], landmarks[23], landmarks[24]
body_cx = (sh_l.x + sh_r.x + hp_l.x + hp_r.x) * w / 4
body_cy = (sh_l.y + sh_r.y + hp_l.y + hp_r.y) * h / 4


def mask_covers(mask_bool, pt, radius=20):
    yy, xx = np.ogrid[:h, :w]
    disk = (xx - pt[0])**2 + (yy - pt[1])**2 <= radius*radius
    return int((mask_bool & disk).sum())


def run_mp(px, py, extra_pts, label):
    ix, iy = int(round(px)), int(round(py))
    eff_box_half = max(edgesam_holds.BOX_HALF_MIN, h // edgesam_holds.BOX_HALF_HEIGHT_DIV)
    eff_box_half = min(eff_box_half, edgesam_holds.BOX_HALF_MAX)
    box = np.array([max(0, ix-eff_box_half), max(0, iy-eff_box_half),
                    min(w-1, ix+eff_box_half), min(h-1, iy+eff_box_half)], dtype=np.float64)
    neg = segmenter._make_negative_prompts(px, py, body_cx, body_cy)
    anchors = [np.array([px, py])] + [np.array(p) for p in extra_pts]
    point_coords = np.vstack(anchors + [neg])
    point_labels = np.array([1]*len(anchors) + [0]*len(neg), dtype=np.int64)
    masks, scores, _ = segmenter.predictor.predict(
        box=box, point_coords=point_coords, point_labels=point_labels,
        num_multimask_outputs=4, use_stability_score=True)
    if masks.ndim == 2:
        masks = masks[None, :, :]
        scores = np.array([float(scores)])
    print(f"\n{label}: 正点={[(round(a[0]), round(a[1])) for a in anchors]}")
    tops = []
    for i, m in enumerate(masks):
        m = m & ~person_mask.astype(bool)
        cc = edgesam_holds._keep_main_component(m.copy(), ix, iy)
        if cc.sum() < 20:
            continue
        vote = edgesam_holds._pixel_color_vote(image_bgr, cc)
        cov = mask_covers(cc, (250, 646))
        total = sum(vote['votes'].values())
        black_pct = vote['votes'].get('黑', 0) / total * 100
        tops.append(vote['top_color'])
        mark = " ⬅️黑岩点" if cov > 100 else ""
        print(f"  mask{i} score={scores[i]:.3f} area={int(cc.sum())} cov={cov}px "
              f"top={vote['top_color']} 黑={black_pct:.1f}%{mark}")
    from collections import Counter
    cnt = Counter(tops)
    print(f"  → top色多数: {dict(cnt)} → {cnt.most_common(1)[0][0] if cnt else '?'}")


# R_hand: 腕→拇指方向
R = landmarks[16]
rx, ry = R.x * w, R.y * h
R_thb = landmarks[22]
tx, ty = R_thb.x * w, R_thb.y * h
ux, uy = tx - rx, ty - ry
n = np.hypot(ux, uy) or 1
ux, uy = ux / n, uy / n
print(f"R_hand wrist=({rx:.0f},{ry:.0f}) thumb=({tx:.0f},{ty:.0f}) 方向=({ux:.2f},{uy:.2f})")
run_mp(rx, ry, [(rx + ux*40, ry + uy*40)], "R_hand 端点+拇指40")

# L_hand: 腕→拇指方向
L = landmarks[15]
lx, ly = L.x * w, L.y * h
L_thb = landmarks[21]
tx, ty = L_thb.x * w, L_thb.y * h
ux, uy = tx - lx, ty - ly
n = np.hypot(ux, uy) or 1
ux, uy = ux / n, uy / n
print(f"L_hand wrist=({lx:.0f},{ly:.0f}) thumb=({tx:.0f},{ty:.0f}) 方向=({ux:.2f},{uy:.2f})")
run_mp(lx, ly, [(lx + ux*40, ly + uy*40)], "L_hand 端点+拇指40")
run_mp(lx, ly, [], "L_hand 基线(单点)")
