"""实验3f：mask 外扩环形带投票 —— 零额外 predict 的方向自适应方案。

核心思想：
- EdgeSAM 单点分割只切出"端点所在 hold"（R_hand 是橙 hold，黑岩点在 49px 外被当背景）
- 但像素级扫描不需要 SAM：在端点 mask 外扩环形带采样颜色，
  如果检测到与 mask 判色不同的"第二颜色簇"（如黑岩点），修正判定。
- 验证：R_hand 应判黑（黑岩点），L_hand 应保持黑（基线正确）。
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

hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
yy, xx = np.ogrid[:h, :w]
sh_l, sh_r, hp_l, hp_r = landmarks[11], landmarks[12], landmarks[23], landmarks[24]
body_cx = (sh_l.x + sh_r.x + hp_l.x + hp_r.x) * w / 4
body_cy = (sh_l.y + sh_r.y + hp_l.y + hp_r.y) * h / 4

BLACK_CENTER = (250, 646)


def vote_pixels(mask_bool):
    pix = hsv[mask_bool]
    if len(pix) == 0:
        return None
    votes: dict[str, int] = {}
    for hh, ss, vv in pix:
        name = edgesam_holds._classify_pixel(float(hh), float(ss), float(vv))
        votes[name] = votes.get(name, 0) + 1
    total = sum(votes.values())
    top = max(votes, key=votes.get)
    return top, votes, total


def analyze(ep_name, idx):
    lm = landmarks[idx]
    px, py = lm.x * w, lm.y * h
    ix, iy = int(round(px)), int(round(py))
    print(f"\n{'='*64}\n{ep_name} ({ix},{iy})")

    res = segmenter.segment_endpoint(px, py, body_cx, body_cy, person_mask=person_mask)
    mask = res['mask'].astype(bool)
    cvotes = res.get('color_vote', {})
    if isinstance(cvotes, dict) and 'votes' in cvotes:
        cvotes = cvotes['votes']
    print(f"  基线: color={res['color_name']} area={res['area']} "
          f"黑票={cvotes.get('黑',0):.0f} 橙票={cvotes.get('橙',0):.0f}")

    # 1) mask 外扩环形带投票（排除 person）
    print("  --- mask 外扩环形带 ---")
    for d1, d2 in [(20, 50), (30, 60), (40, 70), (50, 90), (20, 90)]:
        ring = np.zeros((h, w), dtype=bool)
        dist = np.sqrt((yy - iy)**2 + (xx - ix)**2)
        ring = (dist >= d1) & (dist <= d2)
        ring = ring & ~mask & ~person_mask.astype(bool)
        if ring.sum() < 50:
            print(f"  ring {d1}-{d2}px: 像素太少({ring.sum()})")
            continue
        top, votes, total = vote_pixels(ring)
        print(f"  ring {d1}-{d2}px: top={top:4s} 黑={votes.get('黑',0)/total*100:5.1f}% "
              f"橙={votes.get('橙',0)/total*100:5.1f}% 黄={votes.get('黄',0)/total*100:5.1f}% "
              f"票={total}")

    # 2) 网格扫描: 黑票密度峰值位置
    print("  --- 网格扫描(10px步长, r=25) 黑票top3 ---")
    bests = []
    for ox in range(-90, 91, 10):
        for oy in range(-90, 91, 10):
            cx, cy = ix + ox, iy + oy
            if cx < 0 or cy < 0 or cx >= w or cy >= h:
                continue
            disk = (xx - cx)**2 + (yy - cy)**2 <= 25*25
            disk = disk & ~person_mask.astype(bool)
            if disk.sum() < 20:
                continue
            pix = hsv[disk]
            black = 0
            for hh, ss, vv in pix:
                if edgesam_holds._classify_pixel(float(hh), float(ss), float(vv)) == '黑':
                    black += 1
            ratio = black / disk.sum()
            bests.append((ratio, black, cx, cy))
    bests.sort(reverse=True)
    for ratio, black, cx, cy in bests[:3]:
        dist_bc = np.hypot(cx - BLACK_CENTER[0], cy - BLACK_CENTER[1])
        print(f"  黑票密度={ratio*100:4.1f}% 位置=({cx},{cy}) 偏移=({cx-ix},{cy-iy}) "
              f"距黑岩点中心={dist_bc:.0f}px")

    # 3) 融合判定：mask 颜色 + 环形带颜色
    ring50_90 = (np.sqrt((yy - iy)**2 + (xx - ix)**2) >= 20) & \
                (np.sqrt((yy - iy)**2 + (xx - ix)**2) <= 90) & \
                ~mask & ~person_mask.astype(bool)
    if ring50_90.sum() > 50:
        top, votes, total = vote_pixels(ring50_90)
        print(f"  --- 融合(20-90px环): mask={res['color_name']} 环top={top} → ", end='')
        if top == '黑':
            print("修正为 黑 ✅")
        elif top == res['color_name']:
            print(f"一致 {top} ✅")
        else:
            print(f"不一致但非黑，保持 {res['color_name']}")


analyze('R_hand', 16)
analyze('L_hand', 15)
