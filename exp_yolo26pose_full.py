"""
YOLO26 pose 全量回归实验：用 YOLO26 pose 替代 MediaPipe Pose 跑 10 个 wall。

对比基准：test_edge_sam.py (MediaPipe) 当前全量 7/7 已知命中（ring 黑修正落地后）。

YOLO COCO 17 点映射:
  L_hand = 9 (left_wrist) / R_hand = 10 (right_wrist)
  L_foot = 15 (left_ankle) / R_foot = 16 (right_ankle)   ← 无脚尖，只有脚踝
  body center = 肩(5,6) + 髋(11,12) 平均

用法：
  python exp_yolo26pose_full.py
输出：
  yolo26pose_full_summary.txt   （每 wall 聚合结果 + 命中判定）
  yolo26pose_vis_<wall>/        （每帧可视化，简化版：不画骨架）
"""
import os
import re
import sys
from collections import Counter
from pathlib import Path

_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BASE)
sys.path.insert(0, os.path.join(_BASE, "EdgeSAM"))

import cv2
import numpy as np

import edgesam_stub  # noqa: F401
from ultralytics import YOLO

from edgesam_holds import EdgeSamHoldSegmenter
from yolo_person_mask import PersonMasker

# 复用 test_edge_sam 的 draw_results（画 mask + 端点 + header，不画骨架）
from test_edge_sam import draw_results

BASE = r"D:\Project\climbing_wall_search\mediapipe2"
WALLS_DIR = Path(BASE) / "img_xhs_walls"

TRUTH = {
    '2XRo7VbPgTe_wall_01_known_IMG_20260808_193753': '黑',
    '2XRo7VbPgTe_wall_02_known_IMG_20260808_193745': '蓝',
    '2XRo7VbPgTe_wall_03_known_IMG_20260808_193754': '粉红',
    '2XRo7VbPgTe_wall_04_known_IMG_20260808_193724': '青',
    '3OHz6rZ5opG_wall_01_unknown': None,
    '3OHz6rZ5opG_wall_02_unknown': None,
    '3OHz6rZ5opG_wall_03_known_IMG_20260808_193612': '红',
    '3OHz6rZ5opG_wall_04_line1_white': '白',
    '3OHz6rZ5opG_wall_04_line2_pink': '粉红',
    '3OHz6rZ5opG_wall_05_known_IMG_20260808_193650': '绿',
}

# 端点 -> (YOLO 关键点索引, 名称)
YOLO_ENDPOINTS = [
    ('L_hand', 9),
    ('R_hand', 10),
    ('L_foot', 15),
    ('R_foot', 16),
]
YOLO_SHOULDERS = (5, 6)
YOLO_HIPS = (11, 12)

SKIP_COLORS = {'低置信度', '非岩点', '无法判断', '未知'}

# 关键点置信度阈值（环境变量可调，A/B 用）
CONF_THRESHOLD = float(os.environ.get('YOLO_POSE_CONF', '0.3'))


def analyze_frame(image_bgr, yolo, segmenter, person_masker):
    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # ---------- YOLO26 pose ----------
    res = yolo.predict(image_rgb, conf=0.25, verbose=False)[0]
    if res.keypoints is None or len(res.keypoints) == 0:
        return None
    kps = res.keypoints[0]
    kps_xy = kps.xy[0].cpu().numpy()
    kps_conf = kps.conf[0].cpu().numpy() if kps.conf is not None else np.ones(17)

    # body center（YOLO 肩髋）
    pts = kps_xy[list(YOLO_SHOULDERS) + list(YOLO_HIPS)]
    body_cx = float(pts[:, 0].mean())
    body_cy = float(pts[:, 1].mean())

    # ---------- person mask ----------
    person_mask = person_masker.segment(image_bgr)

    # ---------- EdgeSAM ----------
    segmenter.set_image(image_bgr)
    samples = []
    color_names = {'红', '橙', '黄', '绿', '青', '蓝', '紫', '粉红', '白', '黑', '未知'}

    for ep_name, kidx in YOLO_ENDPOINTS:
        x, y = kps_xy[kidx]
        conf = float(kps_conf[kidx])
        px, py = float(x), float(y)

        if conf < CONF_THRESHOLD:
            samples.append({
                'point': ep_name, 'name': '低置信度',
                'px': (px, py), 'visibility': conf,
                'mask_uint8': None, 'mask_bool': None,
                'area': 0, 'score': 0.0,
            })
            continue

        try:
            res_seg = segmenter.segment_endpoint(px, py, body_cx, body_cy,
                                                 person_mask=person_mask)
        except Exception as e:
            print(f"  ⚠️ EdgeSAM 失败 ({ep_name}): {e}")
            res_seg = None

        if res_seg is None or res_seg['area'] < 100 or res_seg['color_name'] == '非岩点':
            samples.append({
                'point': ep_name, 'name': '非岩点',
                'px': (px, py), 'visibility': conf,
                'mask_uint8': None, 'mask_bool': None,
                'area': 0, 'score': 0.0,
            })
            continue

        samples.append({
            'point': ep_name,
            'name': res_seg['color_name'],
            'h': res_seg['h'], 's': res_seg['s'], 'v': res_seg['v'],
            'area': res_seg['area'],
            'score': res_seg['score'],
            'px': (px, py),
            'mask_uint8': res_seg['mask'],
            'mask_bool': res_seg['mask_bool'],
            'visibility': conf,
            'ring_black': res_seg.get('ring_black', {}),
        })

    # ---------- 投票 ----------
    valid = [s for s in samples if s['name'] in color_names]
    if not valid:
        route_color, votes, conf = '无法判断', {}, 0.0
    else:
        votes = Counter(s['name'] for s in valid)
        route_color, top = votes.most_common(1)[0]
        conf = top / len(valid)
        votes = dict(votes)

    n_segmented = sum(1 for s in samples if s.get('mask_uint8') is not None)
    print(f"  ✅ YOLO26-pose 命中 {n_segmented}/4 端点")
    for s in samples:
        if s.get('mask_uint8') is None:
            print(f"    {s['point']:8s}: {s['name']:6s}  conf={s.get('visibility', 0):.2f}")
        else:
            ring_tag = ""
            rb = s.get('ring_black') or {}
            if rb.get('fixed'):
                ring_tag = f"  [ring黑修正 {rb.get('ratio', 0)*100:.1f}%]"
            print(f"    {s['point']:8s}: {s['name']:6s}  "
                  f"area={s['area']:5d} score={s.get('score', 0):.2f}  "
                  f"HSV=({s['h']:5.1f},{s['s']:5.1f},{s['v']:5.1f}){ring_tag}")
    print(f"  >>> 线路色 = {route_color}  置信度 = {conf:.0%}  票数 = {votes}")

    return {
        'route_color': route_color,
        'confidence': conf,
        'votes': votes,
        'samples': samples,
    }


def main():
    os.chdir(BASE)
    yolo = YOLO('yolo26n-pose.pt')
    segmenter = EdgeSamHoldSegmenter('weights/edge_sam.pth', device='cpu')
    person_masker = PersonMasker(model_name="yolo26m-seg.pt", device='cpu')

    walls = sorted(WALLS_DIR.iterdir())
    print(f"共 {len(walls)} 个 wall\n")

    lines = []
    all_results = []

    for wall_dir in walls:
        wall_name = wall_dir.name
        n_frames = len(list(wall_dir.glob('*.jpg')))
        print(f"=== {wall_name} ({n_frames} 张) ===")
        vis_dir = Path(BASE) / f"yolo26pose_vis_{wall_name}"
        vis_dir.mkdir(exist_ok=True)

        files = sorted(wall_dir.glob('*.jpg'),
                       key=lambda f: (int(re.sub(r'\D', '', f.name)) if re.sub(r'\D', '', f.name) else 99999, f.name))

        aggregate_samples = []
        for i, fp in enumerate(files, 1):
            image_bgr = cv2.imread(str(fp))
            if image_bgr is None:
                continue
            print(f"\n[{i}] {fp.name}")
            result = analyze_frame(image_bgr, yolo, segmenter, person_masker)
            if result is None:
                print("  ⚠️ 未检测到人体")
                continue
            aggregate_samples.append((fp.name, result['samples']))
            # 可视化（简化版：无骨架）
            display = image_bgr.copy()
            draw_results(display, result['samples'], result['route_color'],
                         result['confidence'], result['votes'],
                         sum(1 for s in result['samples'] if s.get('mask_uint8') is not None),
                         person_mask=None)
            out_path = vis_dir / f"frame_{i:02d}.jpg"
            cv2.imwrite(str(out_path), display)

        # ---------- 跨帧聚合 ----------
        all_colors: Counter = Counter()
        n_frames_out = len(aggregate_samples)
        for name, samples in aggregate_samples:
            for s in samples:
                c = s.get('name', '')
                if c in SKIP_COLORS:
                    continue
                all_colors[c] += 1
        total = sum(all_colors.values())
        print("=" * 60)
        print(f"  📊 跨帧聚合: {n_frames_out} 帧, {total} 有效端点票")
        print("=" * 60)
        if total:
            for color, n in all_colors.most_common():
                pct = n / total * 100
                print(f"    {color:5s}: {n:3d}  ({pct:5.1f}%)")
            top_color, top_n = all_colors.most_common(1)[0]
            print(f"  >>> 线路色 = {top_color}  ({top_n}/{total} = {top_n/total*100:.0f}%)")
        else:
            top_color, top_n, total = '?', 0, 0

        truth = TRUTH.get(wall_name)
        if truth is None:
            hit = '?'
        elif top_color == truth:
            hit = '✅'
        else:
            hit = '❌'
        print(f"  真值={truth or '?'}  {hit}")
        lines.append(f"{wall_name[:55]:55s}  判={top_color:4s} ({top_n}/{total})  真值={truth or '?':6s}  {hit}")
        all_results.append((wall_name, top_color, top_n, total, truth, hit))
        print()

    print("\n=== 汇总 (YOLO26-pose) ===")
    hits = sum(1 for r in all_results if r[5] == '✅')
    known = sum(1 for r in all_results if r[4] is not None)
    print(f"命中: {hits}/{known} (已知真值 wall)")
    for l in lines:
        print(f"  {l}")

    with open('yolo26pose_full_summary.txt', 'w', encoding='utf-8') as f:
        f.write(f"YOLO26-pose 全量回归: 命中 {hits}/{known}\n")
        for l in lines:
            f.write(l + "\n")
    print("\n已保存 yolo26pose_full_summary.txt")

    # 本脚本无 MediaPipe，不需要 os._exit；正常 return 即可
    return 0


if __name__ == '__main__':
    sys.exit(main())
