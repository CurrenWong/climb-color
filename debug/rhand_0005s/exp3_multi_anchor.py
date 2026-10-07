"""实验3 多锚点独立分割 + top色多数投票。

每个锚点单独调 segment_endpoint（独立 mask），各锚点 top 色多数投票。
锚点集：端点 + 左/左上/上 偏移（不合并正点，避免 mask 扩散污染）。
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


def mask_covers(mask_bool, pt, radius=20):
    yy, xx = np.ogrid[:h, :w]
    disk = (xx - pt[0])**2 + (yy - pt[1])**2 <= radius*radius
    return int((mask_bool & disk).sum())


def one_anchor(px, py, ax, ay, label):
    """在锚点 (ax,ay) 单独分割，返回 top 色"""
    body_cx = (landmarks[11].x + landmarks[12].x + landmarks[23].x + landmarks[24].x) * w / 4
    body_cy = (landmarks[11].y + landmarks[12].y + landmarks[23].y + landmarks[24].y) * h / 4
    res = segmenter.segment_endpoint(ax, ay, body_cx, body_cy,
                                     person_mask=person_mask, num_multimask_outputs=4)
    if res is None or res['mask_bool'] is None:
        print(f"  {label} ({ax:.0f},{ay:.0f}): 空")
        return None, None
    vote = res['color_vote']
    cov = mask_covers(res['mask_bool'], BLACK_CENTER)
    top = vote['top_color']
    total = sum(vote['votes'].values())
    black_pct = vote['votes'].get('黑', 0) / total * 100 if total else 0
    print(f"  {label} ({ax:.0f},{ay:.0f}): top={top} area={res['area']} 黑岩点覆盖={cov}px 黑={black_pct:.1f}% "
          f"score={res['score']:.3f}")
    return top, res['mask_bool']


def multi_anchor(px, py, offsets, label):
    print(f"\n===== {label} =====")
    tops = []
    union = np.zeros((h, w), dtype=bool)
    for (ox, oy, name) in offsets:
        top, m = one_anchor(px, py, px + ox, py + oy, name)
        if top:
            tops.append((name, top))
        if m is not None:
            union |= m
    print(f"  top色序列: {tops}")

    from collections import Counter
    cnt = Counter(t for _, t in tops)
    print(f"  多数投票: {dict(cnt)} → {cnt.most_common(1)[0][0] if cnt else '无'}")

    uv = edgesam_holds._pixel_color_vote(image_bgr, union)
    ut = sum(uv['votes'].values())
    print(f"  并集投票: top={uv['top_color']} 黑={uv['votes'].get('黑',0)/ut*100:.1f}% "
          f"橙={uv['votes'].get('橙',0)/ut*100:.1f}% 白={uv['votes'].get('白',0)/ut*100:.1f}%")


R = landmarks[16]
rx, ry = R.x * w, R.y * h
multi_anchor(rx, ry, [
    (0, 0, '端点'),
    (-40, 0, '左40'),
    (-35, -25, '左上'),
    (0, -40, '上40'),
], "R_hand 4锚点")

multi_anchor(rx, ry, [
    (0, 0, '端点'),
    (-40, 0, '左40'),
    (-35, -25, '左上'),
], "R_hand 3锚点(无上)")

L = landmarks[15]
lx, ly = L.x * w, L.y * h
multi_anchor(lx, ly, [
    (0, 0, '端点'),
    (-40, 0, '左40'),
    (-35, -25, '左上'),
], "L_hand 3锚点")
