"""实验3g 墙级回归：mask 判色 + 环形带黑票修正（零额外 predict）vs 基线。

规则：segment_endpoint 结果中，
  若 mask 判色 != '黑'，且 mask 外扩环形带(20-70px) top='黑'（且黑票占比>阈值）
  → 修正判定为 '黑'。
其余不变。对比墙级命中率。
"""
import os, sys
import re
from pathlib import Path
from collections import Counter

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "EdgeSAM"))
import edgesam_holds

import mediapipe as mp
from test_edge_sam import detect_pose
from yolo_person_mask import PersonMasker

WALLS_ROOT = Path(r'D:\Project\climbing_wall_search\mediapipe2\img_xhs_walls')

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

RING_D1, RING_D2 = 20, 70
BLACK_RATIO_MIN = 0.18   # 环形带黑票占比下限（相对票数）

TRUTH = {
    '2XRo7VbPgTe_wall_01_known_IMG_20260808_193753': '黑',
    '2XRo7VbPgTe_wall_02_known_IMG_20260808_193745': '蓝',
    '2XRo7VbPgTe_wall_03_known_IMG_20260808_193754': '粉红',
    '2XRo7VbPgTe_wall_04_known_IMG_20260808_193724': '青',
    '3OHz6rZ5opG_wall_03_known_IMG_20260808_193612': '红',
    '3OHz6rZ5opG_wall_04_known_IMG_20260808_193621': '白',
    '3OHz6rZ5opG_wall_05_known_IMG_20260808_193650': '绿',
}


def ring_black_vote(image_bgr, mask_bool, ix, iy, person_mask, d1=RING_D1, d2=RING_D2):
    """mask 外扩环形带黑票占比（纯像素，排除 person + mask）"""
    h, w = image_bgr.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    dist = np.hypot(xx - ix, yy - iy)
    ring = (dist >= d1) & (dist <= d2) & ~mask_bool
    if person_mask is not None and person_mask.any():
        ring = ring & ~person_mask.astype(bool)
    if ring.sum() < 100:
        return 0.0, 0
    pix = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)[ring]
    if len(pix) == 0:
        return 0.0, 0
    black = 0
    for hh, ss, vv in pix:
        if edgesam_holds._classify_pixel(float(hh), float(ss), float(vv)) == '黑':
            black += 1
    return black / len(pix), len(pix)


def corrected_color(res, image_bgr, person_mask):
    """应用环形带黑修正；返回 (color_name, was_corrected)"""
    if res is None:
        return None, False
    color = res['color_name']
    if color == '黑':
        return color, False
    mask = res['mask_bool'] if 'mask_bool' in res else (res['mask'] > 0)
    ix, iy = int(round(res['endpoint'][0])), int(round(res['endpoint'][1]))
    ratio, n = ring_black_vote(image_bgr, mask, ix, iy, person_mask)
    if ratio >= BLACK_RATIO_MIN:
        return '黑', True
    return color, False


def main():
    print("加载模型…")
    segmenter = edgesam_holds.EdgeSamHoldSegmenter(
        r"D:\Project\climbing_wall_search\mediapipe2\weights\edge_sam.pth", device="cpu")
    person_masker = PersonMasker(model_name="weights/yolo26m-seg.pt", device="cpu")
    landmarker = PoseLandmarker.create_from_options(
        PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=r"D:\Project\climbing_wall_search\mediapipe2\weights\pose_landmarker_full.task"),
            running_mode=VisionRunningMode.IMAGE, num_poses=1,
            min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.3,
            min_tracking_confidence=0.3))

    walls = sorted(WALLS_ROOT.iterdir())
    print(f"{'wall':48s} {'基线':22s} {'修正后':22s}  真值")

    for wall_dir in walls:
        wall_name = wall_dir.name
        files = sorted([f for f in wall_dir.glob('*.jpg')])

        votes_base = Counter()
        votes_fix = Counter()
        n_corrected = 0

        for name in files:
            image_bgr = cv2.imread(str(name))
            if image_bgr is None:
                continue
            h, w = image_bgr.shape[:2]
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            landmarks = detect_pose(image_rgb, landmarker, w, h, flip_ensemble=True)
            if landmarks is None:
                continue
            person_mask = person_masker.segment(image_bgr)
            segmenter.set_image(image_bgr)

            sh_l, sh_r, hp_l, hp_r = landmarks[11], landmarks[12], landmarks[23], landmarks[24]
            body_cx = (sh_l.x + sh_r.x + hp_l.x + hp_r.x) * w / 4
            body_cy = (sh_l.y + sh_r.y + hp_l.y + hp_r.y) * h / 4

            for ep_name, idx in [('L_hand', 15), ('R_hand', 16), ('L_foot', 31), ('R_foot', 32)]:
                lm = landmarks[idx]
                px, py = lm.x * w, lm.y * h
                if lm.visibility < 0.3:
                    continue
                res = segmenter.segment_endpoint(px, py, body_cx, body_cy, person_mask=person_mask)
                if res is None or res['area'] < 100:
                    continue
                bn = res['color_name']
                cn, was = corrected_color(res, image_bgr, person_mask)
                if was:
                    n_corrected += 1
                votes_base[bn] += 1
                votes_fix[cn] += 1

        b_top, b_n = votes_base.most_common(1)[0] if votes_base else ('-', 0)
        f_top, f_n = votes_fix.most_common(1)[0] if votes_fix else ('-', 0)
        truth = TRUTH.get(wall_name, '?')
        b_hit = '✅' if truth != '?' and b_top == truth else ('❌' if truth != '?' else '?')
        f_hit = '✅' if truth != '?' and f_top == truth else ('❌' if truth != '?' else '?')
        print(f"{wall_name[:48]:48s} {b_top}({b_n}){b_hit:2s}        {f_top}({f_n}){f_hit:2s}    {truth}  [修正{n_corrected}票]")


if __name__ == '__main__':
    main()
