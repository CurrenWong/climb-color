"""实验3 可视化：0005s R_hand 多锚点校正对比图。

左：基线（端点 mask 判橙）
右：实验3（4 锚点独立 mask + 多数投票判黑）
下方列出每个锚点的 mask 颜色。
"""
import os, sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "EdgeSAM"))
import edgesam_holds

IMG = r"D:\Project\climbing_wall_search\mediapipe2\img_xhs_walls\2XRo7VbPgTe_wall_01_known_IMG_20260808_193753\frame_0005s.jpg"
OUT = r"D:\Project\climbing_wall_search\mediapipe2\dbg_rhand_0005s\exp3_visual.jpg"
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

R = landmarks[16]
rx, ry = R.x * w, R.y * h

ANCHORS = [
    (0, 0, '端点(基线)', (0, 165, 255)),
    (-40, 0, '左40', (0, 255, 255)),
    (-35, -25, '左上', (255, 0, 255)),
    (0, -40, '上40', (255, 255, 0)),
]

COLOR_BGR = {
    '黑': (40, 40, 40), '白': (235, 235, 235), '红': (60, 60, 255),
    '橙': (0, 140, 255), '黄': (0, 230, 255), '绿': (80, 220, 80),
    '青': (220, 220, 80), '蓝': (220, 120, 30), '紫': (220, 80, 160),
    '粉红': (180, 120, 255), '未知': (128, 128, 128),
}

# 逐锚点分割
results = []
for ox, oy, name, _ in ANCHORS:
    ax, ay = rx + ox, ry + oy
    res = segmenter.segment_endpoint(ax, ay, body_cx, body_cy, person_mask=person_mask)
    v = res['color_vote']
    results.append({
        'name': name, 'pos': (ax, ay), 'top': v['top_color'],
        'votes': v['votes'], 'mask': res['mask_bool'],
        'area': res['area'], 'score': res['score'],
    })
    total = sum(v['votes'].values())
    black_pct = v['votes'].get('黑', 0) / total * 100
    print(f"  {name} ({ax:.0f},{ay:.0f}): top={v['top_color']} 黑={black_pct:.1f}% area={res['area']}")

# 多数投票
from collections import Counter
tops = [r['top'] for r in results]
cnt = Counter(tops)
winner, _ = cnt.most_common(1)[0]
print(f"多数投票: {dict(cnt)} → {winner}")

# 画图：左侧 = 基线锚点(端点)，右侧 = 全部锚点
disp = image_bgr.copy()
for r in results:
    col = COLOR_BGR.get(r['top'], (128, 128, 128))
    # mask 半透明
    mask_u8 = (r['mask'].astype(np.uint8)) * 255
    overlay = disp.copy()
    overlay[mask_u8 > 0] = col
    cv2.addWeighted(overlay, 0.45, disp, 0.55, 0, disp)
    # 锚点
    ax, ay = int(r['pos'][0]), int(r['pos'][1])
    cv2.circle(disp, (ax, ay), 8, (0, 0, 0), -1)
    cv2.circle(disp, (ax, ay), 6, col, -1)
    cv2.putText(disp, f"{r['name']}:{r['top']}", (ax + 12, ay),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(disp, f"{r['name']}:{r['top']}", (ax + 12, ay),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)

# header
cv2.rectangle(disp, (0, 0), (w, 78), (20, 20, 20), -1)
cv2.putText(disp, f"frame_0005s R_hand 实验3: 4锚点独立分割", (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(disp, f"多数投票 = {winner}   (基线端点判橙)", (10, 56),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 255) if winner == '黑' else (255, 255, 255), 2, cv2.LINE_AA)

# 黑岩点真值标记
cv2.circle(disp, (250, 646), 14, (0, 255, 0), 2)
cv2.putText(disp, "黑岩点(真值)", (260, 640), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

cv2.imwrite(OUT, disp)
print(f"💾 {OUT}")
