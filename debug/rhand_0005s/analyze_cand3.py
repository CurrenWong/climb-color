"""分析 cand3 (mask#3)：几何 + 颜色分布 + 如果选中会判什么颜色。"""
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

# 手工复现 predict，拿 4 个原始候选
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
    m = masks[i]
    print(f"\n{'='*60}\nCAND#{i}  score={scores[i]:.3f}  area={int(m.sum())}")
    ys, xs = np.where(m)
    print(f"bbox: x=[{xs.min()},{xs.max()}] y=[{ys.min()},{ys.max()}]  (宽 {xs.max()-xs.min()}, 高 {ys.max()-ys.min()})")
    print(f"mask 中心=({xs.mean():.0f},{ys.mean():.0f})  端点=({px:.0f},{py:.0f})")

    pix = hsv[ys, xs]
    print(f"mask 内颜色分布:")
    for name, cond in [
        ("白  (S<35,V>140)", (pix[:,1] < 35) & (pix[:,2] > 140)),
        ("黑  (S<35,V<140)", (pix[:,1] < 35) & (pix[:,2] <= 140)),
        ("低饱 (S<35)", pix[:,1] < 35),
        ("橙   (H 8-30,S>60,V>100)", (pix[:,0] >= 8) & (pix[:,0] <= 30) & (pix[:,1] > 60) & (pix[:,2] > 100)),
        ("高饱 (S>100)", pix[:,1] > 100),
        ("亮 (V>150)", pix[:,2] > 150),
    ]:
        n = cond.sum()
        print(f"  {name:28s}: {n:5d}  ({n/len(pix)*100:5.1f}%)")
    print(f"  S 中位数={np.median(pix[:,1]):.0f}  V 中位数={np.median(pix[:,2]):.0f}")

    # 正式流程会怎么判
    # 减 person
    m2 = m & ~person_mask.astype(bool)
    print(f"减 person: {m.sum()} -> {m2.sum()}")
    if m2.sum() < edgesam_holds.COLOR_MIN_PIXELS:
        print("  -> 减 person 后 < 50 px, 判非岩点")
        continue
    cc = edgesam_holds._keep_main_component(m2.copy(), ix, iy)
    print(f"主连通域: {m2.sum()} -> {cc.sum()}")
    hh, ss, vv = edgesam_holds._sample_color(image_bgr, cc)
    if vv < edgesam_holds.DARK_V_MAX and ss < edgesam_holds.DARK_S_MAX:
        cn = "未知(暗)"
    elif ss < edgesam_holds.WHITE_SAT_MAX:
        cn = "白" if vv >= edgesam_holds.WHITE_VAL_MIN else "黑"
    else:
        cn = edgesam_holds.route_color.hue_to_color_name(hh, ss)
    print(f"颜色: HSV=({hh:.1f},{ss:.1f},{vv:.1f}) -> {cn}")

    # 端点局部 patch
    patch = hsv[max(0,int(py)-20):int(py)+20, max(0,int(px)-20):int(px)+20].reshape(-1,3)
    white_p = (patch[:,1] < 35) & (patch[:,2] > 140)
    dark_p = (patch[:,1] < 35) & (patch[:,2] <= 140)
    print(f"端点 40x40: S 中位={np.median(patch[:,1]):.0f} V 中位={np.median(patch[:,2]):.0f} 白={white_p.sum()/len(patch)*100:.1f}% 黑={dark_p.sum()/len(patch)*100:.1f}%")
