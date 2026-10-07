"""实验3j 最终墙级回归：生产代码 segment_endpoint（默认 ring_black_fix=True）。"""
import os, sys
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

TRUTH = {
    '2XRo7VbPgTe_wall_01_known_IMG_20260808_193753': '黑',
    '2XRo7VbPgTe_wall_02_known_IMG_20260808_193745': '蓝',
    '2XRo7VbPgTe_wall_03_known_IMG_20260808_193754': '粉红',
    '2XRo7VbPgTe_wall_04_known_IMG_20260808_193724': '青',
    '3OHz6rZ5opG_wall_03_known_IMG_20260808_193612': '红',
    '3OHz6rZ5opG_wall_04_known_IMG_20260808_193621': '白',
    '3OHz6rZ5opG_wall_05_known_IMG_20260808_193650': '绿',
}

COLOR_NAMES = {'红','橙','黄','绿','青','蓝','紫','粉红','白','黑'}


def main():
    print("加载模型…")
    segmenter = edgesam_holds.EdgeSamHoldSegmenter(
        r"D:\Project\climbing_wall_search\mediapipe2\weights\edge_sam.pth", device="cpu")
    person_masker = PersonMasker(model_name="yolo26m-seg.pt", device="cpu")
    landmarker = PoseLandmarker.create_from_options(
        PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=r"D:\Project\climbing_wall_search\mediapipe2\pose_landmarker_full.task"),
            running_mode=VisionRunningMode.IMAGE, num_poses=1,
            min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.3,
            min_tracking_confidence=0.3))

    walls = sorted(WALLS_ROOT.iterdir())
    print(f"{'wall':48s} {'基线(参考)':20s} {'生产(含修正)':20s}  真值  修正票数")

    for wall_dir in walls:
        wall_name = wall_dir.name
        files = sorted([f for f in wall_dir.glob('*.jpg')])

        votes = Counter()
        n_fixed = 0
        fixed_detail = Counter()

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
                if res is None or res['area'] < 100 or res['color_name'] == '非岩点':
                    continue
                votes[res['color_name']] += 1
                rb = res.get('ring_black') or {}
                if rb.get('fixed'):
                    n_fixed += 1
                    fixed_detail[ep_name] += 1

        top, n = votes.most_common(1)[0] if votes else ('-', 0)
        truth = TRUTH.get(wall_name, '?')
        hit = '✅' if truth != '?' and top == truth else ('❌' if truth != '?' else '?')
        detail = ', '.join(f"{k}={v}" for k, v in fixed_detail.items()) if fixed_detail else '-'
        print(f"{wall_name[:48]:48s} {'-':20s} {top}({n}){hit:2s}    {truth}  {n_fixed} [{detail}]")


if __name__ == '__main__':
    main()
