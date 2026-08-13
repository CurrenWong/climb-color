"""实验3 全量对比回归：基线 vs 多锚点独立分割+多数投票。

对 img_xhs_walls 下每个 wall 的每帧 4 端点：
- 基线: segment_endpoint(端点) 的 color_name
- 实验3: 端点 + 左40 + 左上(35,25) + 上40 各自独立分割，top色多数投票
对比哪些帧被改变、是否更接近真值。
"""
import os, sys, re, json
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

PYTHON = r'D:\Program\Conda\python.exe'
WALLS_ROOT = Path(r'D:\Project\climbing_wall_search\mediapipe2\img_xhs_walls')

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# 锚点偏移（实验3 方案）
ANCHOR_OFFSETS = [
    (0, 0, '端点'),
    (-40, 0, '左40'),
    (-35, -25, '左上'),
    (0, -40, '上40'),
]

# 真值（_run_walls.py 同源）
TRUTH = {
    '2XRo7VbPgTe_wall_01_known_IMG_20260808_193753': '黑',
    '2XRo7VbPgTe_wall_02_known_IMG_20260808_193745': '蓝',
    '2XRo7VbPgTe_wall_03_known_IMG_20260808_193754': '粉红',
    '2XRo7VbPgTe_wall_04_known_IMG_20260808_193724': '青',
    '3OHz6rZ5opG_wall_03_known_IMG_20260808_193612': '红',
    '3OHz6rZ5opG_wall_04_known_IMG_20260808_193621': '白',
    '3OHz6rZ5opG_wall_05_known_IMG_20260808_193650': '绿',
}


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
    diff_rows = []
    all_rows = []

    for wall_dir in walls:
        wall_name = wall_dir.name
        files = sorted([f for f in wall_dir.glob('*.jpg')])
        print(f"\n{'='*70}\n=== {wall_name} ({len(files)} 张) ===")

        wall_truth = TRUTH.get(wall_name)
        wall_votes_base = Counter()
        wall_votes_exp3 = Counter()

        for name in files:
            image_bgr = cv2.imread(str(name))
            if image_bgr is None:
                continue
            h, w = image_bgr.shape[:2]
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            landmarks = detect_pose(image_rgb, landmarker, w, h, flip_ensemble=True)
            if landmarks is None:
                print(f"  ⚠️ 无 pose: {name.name}")
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

                # 基线
                try:
                    res = segmenter.segment_endpoint(px, py, body_cx, body_cy, person_mask=person_mask)
                    base_name = res['color_name'] if (res and res['area'] >= 100) else '非岩点'
                except Exception:
                    base_name = '失败'

                # 实验3：多锚点独立分割 + top色多数（仅手部启用）
                if ep_name.endswith('_hand'):
                    tops = []
                    for ox, oy, aname in ANCHOR_OFFSETS:
                        ax, ay = px + ox, py + oy
                        if ax < 0 or ay < 0 or ax >= w or ay >= h:
                            continue
                        try:
                            r = segmenter.segment_endpoint(ax, ay, body_cx, body_cy, person_mask=person_mask)
                            if r and r['area'] >= 100:
                                v = r['color_vote']
                                if v and v['top_color']:
                                    tops.append(v['top_color'])
                        except Exception:
                            pass
                    if tops:
                        cnt = Counter(tops)
                        exp3_name, _ = cnt.most_common(1)[0]
                    else:
                        exp3_name = '无有效'
                else:
                    exp3_name = base_name

                if base_name != '非岩点' and base_name != '失败' and base_name != '低置信度':
                    wall_votes_base[base_name] += 1
                if exp3_name != '非岩点' and exp3_name != '无有效':
                    wall_votes_exp3[exp3_name] += 1

                if base_name != exp3_name:
                    diff_rows.append({
                        'wall': wall_name[:20], 'frame': name.name,
                        'ep': ep_name, 'px': round(px), 'py': round(py),
                        'base': base_name, 'exp3': exp3_name,
                        'anchor_tops': tops,
                    })
                all_rows.append({
                    'wall': wall_name[:20], 'frame': name.name, 'ep': ep_name,
                    'base': base_name, 'exp3': exp3_name,
                })

        b_top, b_n = wall_votes_base.most_common(1)[0] if wall_votes_base else ('-', 0)
        e_top, e_n = wall_votes_exp3.most_common(1)[0] if wall_votes_exp3 else ('-', 0)
        b_hit = '✅' if wall_truth and b_top == wall_truth else ('❌' if wall_truth else '?')
        e_hit = '✅' if wall_truth and e_top == wall_truth else ('❌' if wall_truth else '?')
        print(f"  基线: {b_top} ({b_n}/{sum(wall_votes_base.values())})  真值={wall_truth}  {b_hit}")
        print(f"  实验3: {e_top} ({e_n}/{sum(wall_votes_exp3.values())})  真值={wall_truth}  {e_hit}")

    # 汇总
    print(f"\n{'='*70}\n=== 汇总 ===")
    print(f"总共帧端点判定: {len(all_rows)}")
    print(f"基线 vs 实验3 不同的: {len(diff_rows)}")
    for d in diff_rows:
        print(f"  {d['wall']} {d['frame']} {d['ep']} ({d['px']},{d['py']}): "
              f"基线={d['base']} → 实验3={d['exp3']}  anchors={d['anchor_tops']}")


if __name__ == '__main__':
    main()
