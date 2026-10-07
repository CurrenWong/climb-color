"""实验3 方向性偏移：沿 body->endpoint 方向偏移锚点采样。"""
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

sh_l, sh_r, hp_l, hp_r = landmarks[11], landmarks[12], landmarks[23], landmarks[24]
body_cx = (sh_l.x + sh_r.x + hp_l.x + hp_r.x) * w / 4
body_cy = (sh_l.y + sh_r.y + hp_l.y + hp_r.y) * h / 4
print(f"body=({body_cx:.0f},{body_cy:.0f})")

hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
yy, xx = np.ogrid[:h, :w]


def anchor_vote(cx, cy, r=30):
    disk = (xx - cx)**2 + (yy - cy)**2 <= r*r
    disk = disk & ~person_mask.astype(bool)
    if disk.sum() < 20:
        return None
    pix = hsv[disk]
    votes: dict[str, float] = {}
    for hh, ss, vv in pix:
        name = edgesam_holds._classify_pixel(float(hh), float(ss), float(vv))
        votes[name] = votes.get(name, 0) + 1
    return votes


def summarize(votes, label):
    if votes is None:
        print(f"  {label}: 无像素")
        return None
    total = sum(votes.values())
    top = max(votes, key=votes.get)
    print(f"  {label}: top={top:4s} 黑={votes.get('黑',0)/total*100:.1f}% 橙={votes.get('橙',0)/total*100:.1f}%  "
          f"白={votes.get('白',0)/total*100:.1f}%  票数前3={ {k: int(v) for k, v in sorted(votes.items(), key=lambda x: -x[1])[:3]} }")
    return top


def merge_votes(votes_list):
    out: dict[str, float] = {}
    for v in votes_list:
        if v is None:
            continue
        for k, val in v.items():
            out[k] = out.get(k, 0) + val
    return out


for ep_name, idx in [('R_hand', 16), ('L_hand', 15), ('R_foot', 32), ('L_foot', 31)]:
    lm = landmarks[idx]
    px, py = lm.x * w, lm.y * h
    if lm.visibility < 0.3:
        print(f"\n{ep_name}: 低置信度")
        continue

    # 方向：body -> endpoint，继续延伸
    ux, uy = px - body_cx, py - body_cy
    norm = np.hypot(ux, uy) or 1.0
    ux, uy = ux / norm, uy / norm

    print(f"\n{ep_name} ({px:.0f},{py:.0f})  方向=({ux:.2f},{uy:.2f})")
    va = anchor_vote(int(px), int(py), r=30)
    summarize(va, "A. 端点 r=30")

    # 沿方向偏移 20/30/40/50px 单锚点
    for off in [20, 30, 40, 50]:
        ax, ay = px + ux * off, py + uy * off
        v = anchor_vote(int(ax), int(ay), r=30)
        summarize(v, f"   偏移{off}px ({ax:.0f},{ay:.0f}) r=30")

    # 合并：端点 + 方向偏移 30px + 40px
    v_merge = merge_votes([
        va,
        anchor_vote(int(px + ux * 30), int(py + uy * 30), r=30),
        anchor_vote(int(px + ux * 40), int(py + uy * 40), r=30),
    ])
    summarize(v_merge, "M1. 端点+30+40 合并")

    # 合并：端点 + 方向偏移 20 + 30 + 40
    v_merge2 = merge_votes([
        va,
        anchor_vote(int(px + ux * 20), int(py + uy * 20), r=30),
        anchor_vote(int(px + ux * 30), int(py + uy * 30), r=30),
        anchor_vote(int(px + ux * 40), int(py + uy * 40), r=30),
    ])
    summarize(v_merge2, "M2. 端点+20+30+40 合并")
