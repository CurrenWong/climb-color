"""实验3b：不改 predict 次数，只调 box 大小 / 候选数，看能否切出黑岩点。

思路：0005s 单点分割黑岩点覆盖 0px，怀疑是 box prompt 太小。
改 box half（60/80/100/120）+ 候选数（4/6/8），零额外 predict 成本。
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

BLACK_CENTER = (250, 646)

sh_l, sh_r, hp_l, hp_r = landmarks[11], landmarks[12], landmarks[23], landmarks[24]
body_cx = (sh_l.x + sh_r.x + hp_l.x + hp_r.x) * w / 4
body_cy = (sh_l.y + sh_r.y + hp_l.y + hp_r.y) * h / 4


def mask_covers(mask_bool, pt, radius=20):
    yy, xx = np.ogrid[:h, :w]
    disk = (xx - pt[0])**2 + (yy - pt[1])**2 <= radius*radius
    return int((mask_bool & disk).sum())


def run_box(px, py, box_half, num_masks):
    ix, iy = int(round(px)), int(round(py))
    box = np.array([max(0, ix-box_half), max(0, iy-box_half),
                    min(w-1, ix+box_half), min(h-1, iy+box_half)], dtype=np.float64)
    neg = segmenter._make_negative_prompts(px, py, body_cx, body_cy)
    point_coords = np.vstack([[[px, py]], neg])
    point_labels = np.array([1] + [0]*len(neg), dtype=np.int64)
    masks, scores, _ = segmenter.predictor.predict(
        box=box, point_coords=point_coords, point_labels=point_labels,
        num_multimask_outputs=num_masks, use_stability_score=True)
    if masks.ndim == 2:
        masks = masks[None, :, :]
        scores = np.array([float(scores)])
    print(f"\nbox_half={box_half}  num_masks={num_masks}")
    best_cov = 0
    for i, m in enumerate(masks):
        m = m & ~person_mask.astype(bool)
        cc = edgesam_holds._keep_main_component(m.copy(), ix, iy)
        if cc.sum() < 20:
            continue
        vote = edgesam_holds._pixel_color_vote(image_bgr, cc)
        cov = mask_covers(cc, BLACK_CENTER)
        best_cov = max(best_cov, cov)
        total = sum(vote['votes'].values())
        black_pct = vote['votes'].get('黑', 0) / total * 100
        mark = " ⬅️黑岩点" if cov > 100 else ""
        print(f"  mask{i} score={scores[i]:.3f} area={int(cc.sum())} cov={cov}px "
              f"top={vote['top_color']} 黑={black_pct:.1f}%{mark}")
    print(f"  → 最大黑岩点覆盖: {best_cov}px")


R = landmarks[16]
rx, ry = R.x * w, R.y * h
print(f"R_hand ({rx:.0f},{ry:.0f})  黑岩点 ({BLACK_CENTER[0]},{BLACK_CENTER[1]})")

# 当前默认 box（segment_endpoint 逻辑）
eff_box_half = max(edgesam_holds.BOX_HALF_MIN, h // edgesam_holds.BOX_HALF_HEIGHT_DIV)
eff_box_half = min(eff_box_half, edgesam_holds.BOX_HALF_MAX)
print(f"\n>>> 当前默认 box_half = {eff_box_half} (BOX_HALF_MIN={edgesam_holds.BOX_HALF_MIN}, "
      f"DIV={edgesam_holds.BOX_HALF_HEIGHT_DIV}, MAX={edgesam_holds.BOX_HALF_MAX})")

run_box(rx, ry, eff_box_half, 4)
for bh in [80, 100, 120, 150]:
    run_box(rx, ry, bh, 4)
run_box(rx, ry, eff_box_half, 6)
run_box(rx, ry, 100, 6)
