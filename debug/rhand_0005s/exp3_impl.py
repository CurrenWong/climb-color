"""实验3 实现验证：多锚点环形偏移采样，验证 0005s R_hand + L_hand 对照组。"""
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

hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
yy, xx = np.ogrid[:h, :w]


def anchor_vote(cx, cy, r=30):
    """单锚点圆盘颜色投票（排除 person mask）"""
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


def multi_anchor_vote(cx, cy, offsets, r=30):
    """多个偏移锚点圆盘合并投票"""
    all_votes: dict[str, float] = {}
    for ox, oy in offsets:
        v = anchor_vote(cx + ox, cy + oy, r=r)
        if v is None:
            continue
        for k, val in v.items():
            all_votes[k] = all_votes.get(k, 0) + val
    total = sum(all_votes.values())
    if total <= 0:
        return None
    top = max(all_votes, key=all_votes.get)
    return top, all_votes, total


# 环形偏移锚点集（8 方向 × 30px + 端点本身）
RING_OFFSETS = [(0, 0)] + [
    (round(30 * np.cos(t)), round(30 * np.sin(t)))
    for t in np.linspace(0, 2 * np.pi, 8, endpoint=False)
]

print("=" * 70)
for ep_name, idx in [('R_hand', 16), ('L_hand', 15)]:
    lm = landmarks[idx]
    px, py = lm.x * w, lm.y * h
    print(f"\n{ep_name} ({px:.0f},{py:.0f})")

    # A. 单端点（r=30）——现有思路的圆盘版
    va = anchor_vote(int(px), int(py), r=30)
    if va:
        ta = sum(va.values())
        top_a = max(va, key=va.get)
        print(f"  A. 单端点 r=30     : top={top_a:4s} 黑={va.get('黑',0)/ta*100:.1f}% 橙={va.get('橙',0)/ta*100:.1f}%")

    # B. 环形偏移 8 锚点合并（r=30）
    vb = multi_anchor_vote(int(px), int(py), RING_OFFSETS, r=30)
    if vb:
        top_b, votes_b, total_b = vb
        print(f"  B. 环形8锚点 r=30  : top={top_b:4s} 黑={votes_b.get('黑',0)/total_b*100:.1f}% "
              f"橙={votes_b.get('橙',0)/total_b*100:.1f}% 票数={ {k: int(v) for k, v in sorted(votes_b.items(), key=lambda x: -x[1])[:5]} }")

    # C. 环形偏移 8 锚点合并（r=25 更紧）
    vc = multi_anchor_vote(int(px), int(py), RING_OFFSETS, r=25)
    if vc:
        top_c, votes_c, total_c = vc
        print(f"  C. 环形8锚点 r=25  : top={top_c:4s} 黑={votes_c.get('黑',0)/total_c*100:.1f}% "
              f"橙={votes_c.get('橙',0)/total_c*100:.1f}%")

    # D. 多半径环形（r=25 + r=40 双环合并）
    offsets_wide = [(0, 0)] + [
        (round(30 * np.cos(t)), round(30 * np.sin(t)))
        for t in np.linspace(0, 2 * np.pi, 8, endpoint=False)
    ] + [
        (round(45 * np.cos(t)), round(45 * np.sin(t)))
        for t in np.linspace(0, 2 * np.pi, 8, endpoint=False)
    ]
    vd = multi_anchor_vote(int(px), int(py), offsets_wide, r=28)
    if vd:
        top_d, votes_d, total_d = vd
        print(f"  D. 双环17锚点 r=28 : top={top_d:4s} 黑={votes_d.get('黑',0)/total_d*100:.1f}% "
              f"橙={votes_d.get('橙',0)/total_d*100:.1f}%")
