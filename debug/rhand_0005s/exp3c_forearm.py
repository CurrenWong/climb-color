"""实验3c：用姿态几何（肘→腕延长方向）替代固定偏移，找手指方向的 hold。

攀岩手抓 hold 时，hold 在手指方向 ≈ 前臂延伸方向（肘→腕→指尖）。
用 MediaPipe elbow(13/14) -> wrist(15/16) 方向，延长 30-50px 作为锚点。
零猜方向，自适应每帧姿态。
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

hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
yy, xx = np.ogrid[:h, :w]

sh_l, sh_r, hp_l, hp_r = landmarks[11], landmarks[12], landmarks[23], landmarks[24]
body_cx = (sh_l.x + sh_r.x + hp_l.x + hp_r.x) * w / 4
body_cy = (sh_l.y + sh_r.y + hp_l.y + hp_r.y) * h / 4


def disk_top(cx, cy, r=25):
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
    total = sum(votes.values())
    top = max(votes, key=votes.get)
    return top, votes.get('黑', 0) / total * 100


def mask_covers(mask_bool, pt, radius=20):
    yy, xx = np.ogrid[:h, :w]
    disk = (xx - pt[0])**2 + (yy - pt[1])**2 <= radius*radius
    return int((mask_bool & disk).sum())


def run_multi_point(px, py, extra_pts, label):
    """端点 + 额外正点 一次 predict"""
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
    chosen = None
    for i, m in enumerate(masks):
        m = m & ~person_mask.astype(bool)
        cc = edgesam_holds._keep_main_component(m.copy(), ix, iy)
        if cc.sum() < 20:
            continue
        vote = edgesam_holds._pixel_color_vote(image_bgr, cc)
        cov = mask_covers(cc, (250, 646))
        total = sum(vote['votes'].values())
        black_pct = vote['votes'].get('黑', 0) / total * 100
        mark = " ⬅️黑岩点" if cov > 100 else ""
        print(f"  mask{i} score={scores[i]:.3f} area={int(cc.sum())} cov={cov}px "
              f"top={vote['top_color']} 黑={black_pct:.1f}%{mark}")
        if chosen is None or scores[i] > chosen[0]:
            chosen = (float(scores[i]), vote['top_color'])
    print(f"  → score最高选: {chosen[1]} (score={chosen[0]:.3f})")


# ---- 姿态几何方向 ----
for side, (elb_idx, wri_idx, idx_idx, thb_idx) in {
    'L': (13, 15, 19, 21),
    'R': (14, 16, 20, 22),
}.items():
    elb = landmarks[elb_idx]
    wri = landmarks[wri_idx]
    idx = landmarks[idx_idx]
    thb = landmarks[thb_idx]
    ex, ey = elb.x * w, elb.y * h
    wx, wy = wri.x * w, wri.y * h
    ixx, ixy = idx.x * w, idx.y * h
    txx, txy = thb.x * w, thb.y * h
    print(f"\n===== {side}_hand =====")
    print(f"  elbow=({ex:.0f},{ey:.0f})  wrist=({wx:.0f},{wy:.0f})  index=({ixx:.0f},{ixy:.0f})  thumb=({txx:.0f},{txy:.0f})")

    # 肘→腕方向（前臂）
    ux, uy = wx - ex, wy - ey
    n = np.hypot(ux, uy) or 1
    ux, uy = ux / n, uy / n
    print(f"  前臂方向(肘→腕)=({ux:.2f},{uy:.2f})")

    for off in [30, 40, 50]:
        ax, ay = wx + ux * off, wy + uy * off
        t, bp = disk_top(ax, ay)
        print(f"  前臂延长{off}px ({ax:.0f},{ay:.0f}): 圆盘top={t} 黑={bp:.1f}%")

    # 腕→食指方向
    ux2, uy2 = ixx - wx, ixy - wy
    n2 = np.hypot(ux2, uy2) or 1
    ux2, uy2 = ux2 / n2, uy2 / n2
    print(f"  腕→食指方向=({ux2:.2f},{uy2:.2f})")
    for off in [30, 40]:
        ax, ay = wx + ux2 * off, wy + uy2 * off
        t, bp = disk_top(ax, ay)
        print(f"  腕→食指延长{off}px ({ax:.0f},{ay:.0f}): 圆盘top={t} 黑={bp:.1f}%")

    # 腕→拇指方向
    ux3, uy3 = txx - wx, txy - wy
    n3 = np.hypot(ux3, uy3) or 1
    ux3, uy3 = ux3 / n3, uy3 / n3
    print(f"  腕→拇指方向=({ux3:.2f},{uy3:.2f})")
    for off in [30, 40]:
        ax, ay = wx + ux3 * off, wy + uy3 * off
        t, bp = disk_top(ax, ay)
        print(f"  腕→拇指延长{off}px ({ax:.0f},{ay:.0f}): 圆盘top={t} 黑={bp:.1f}%")


# ---- 用前臂方向做多正点 ----
R = landmarks[16]
rx, ry = R.x * w, R.y * h
R_elb = landmarks[14]
rex, rey = R_elb.x * w, R_elb.y * h
ux, uy = rx - rex, ry - rey
n = np.hypot(ux, uy) or 1
ux, uy = ux / n, uy / n
run_multi_point(rx, ry, [(rx + ux*40, ry + uy*40)], "R_hand 端点+前臂40")

L = landmarks[15]
lx, ly = L.x * w, L.y * h
L_elb = landmarks[13]
lex, ley = L_elb.x * w, L_elb.y * h
ux, uy = lx - lex, ly - ley
n = np.hypot(ux, uy) or 1
ux, uy = ux / n, uy / n
run_multi_point(lx, ly, [(lx + ux*40, ly + uy*40)], "L_hand 端点+前臂40")
