"""实验3 探测：端点周围偏移锚点的圆盘颜色分布，找黑岩点覆盖范围。"""
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
R_hand = landmarks[16]
px, py = R_hand.x * w, R_hand.y * h

hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
yy, xx = np.ogrid[:h, :w]

# 黑岩点中心 (从之前分析: 250,646)
black_center = (250, 646)

def disk_vote(cx, cy, r=30):
    """以 (cx,cy) 为圆心 r 半径圆盘内颜色投票"""
    disk = (xx - cx)**2 + (yy - cy)**2 <= r*r
    pix = hsv[disk]
    votes: dict[str, int] = {}
    for hh, ss, vv in pix:
        name = edgesam_holds._classify_pixel(float(hh), float(ss), float(vv))
        votes[name] = votes.get(name, 0) + 1
    total = sum(votes.values())
    top = max(votes, key=votes.get) if votes else '?'
    return top, total, votes

print(f"端点=({px:.0f},{py:.0f})  黑岩点中心={black_center}\n")

# 1) 主端点 + 几个方向偏移锚点
anchors = {
    'endpoint(主)': (px, py),
    'left-30': (px-30, py),
    'left-40': (px-40, py),
    'left-50': (px-50, py),
    'up-30': (px, py-30),
    'up-40': (px, py-40),
    'up-50': (px, py-50),
    'upleft-30': (px-30, py-30),
    'upleft-40': (px-40, py-40),
    'upleft-50': (px-50, py-50),
    'black_center(真值)': black_center,
}
print("--- 各锚点 r=30 圆盘投票 ---")
for name, (cx, cy) in anchors.items():
    cx, cy = int(cx), int(cy)
    top, total, votes = disk_vote(cx, cy, r=30)
    black_pct = votes.get('黑', 0) / total * 100 if total else 0
    orange_pct = votes.get('橙', 0) / total * 100 if total else 0
    print(f"  {name:22s} ({cx:3d},{cy:3d}): top={top:4s} 黑={black_pct:5.1f}% 橙={orange_pct:5.1f}%  "
          f"votes={ {k: v for k, v in sorted(votes.items(), key=lambda x: -x[1])[:3]} }")

# 2) 偏移锚点汇总投票：多个锚点圆盘合并
print("\n--- 多锚点合并投票 (端点 + 左/上偏移各 r=30) ---")
all_pix = []
for (cx, cy) in [(px, py), (px-30, py), (px-40, py-20), (px-20, py-30), (px-30, py-30)]:
    cx, cy = int(cx), int(cy)
    disk = (xx - cx)**2 + (yy - cy)**2 <= 30*30
    all_pix.append(hsv[disk])
merged = np.vstack(all_pix)
votes: dict[str, int] = {}
for hh, ss, vv in merged:
    name = edgesam_holds._classify_pixel(float(hh), float(ss), float(vv))
    votes[name] = votes.get(name, 0) + 1
total = sum(votes.values())
top = max(votes, key=votes.get)
print(f"  合并 {len(all_pix)} 个锚点圆盘: top={top}  黑={votes.get('黑',0)/total*100:.1f}% 橙={votes.get('橙',0)/total*100:.1f}%")
print(f"  全部票数: { {k: v for k, v in sorted(votes.items(), key=lambda x: -x[1])} }")

# 3) 不同半径的端点圆盘
print("\n--- 端点圆盘不同半径 ---")
for r in [30, 50, 70, 90]:
    top, total, votes = disk_vote(int(px), int(py), r=r)
    black_pct = votes.get('黑', 0) / total * 100 if total else 0
    print(f"  r={r}: top={top} 黑={black_pct:.1f}%")

# 4) 找"哪个锚点位置"能最大化黑票
print("\n--- 网格搜索: 黑票最大化锚点 ---")
best = (0, 0, 0)  # (黑票数, cx, cy)
for ox in range(-80, 41, 10):
    for oy in range(-80, 41, 10):
        cx, cy = int(px) + ox, int(py) + oy
        if cx < 0 or cy < 0 or cx >= w or cy >= h:
            continue
        top, total, votes = disk_vote(cx, cy, r=25)
        black = votes.get('黑', 0)
        if black > best[0]:
            best = (black, cx, cy)
print(f"  黑票最大锚点: 偏移=({best[1]-px:.0f},{best[2]-py:.0f}) 位置=({best[1]},{best[2]}) 黑票={best[0]}")
