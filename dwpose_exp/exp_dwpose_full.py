"""
DWPose 全量回归实验：用 DWPose ONNX 替代 MediaPipe Pose 跑 10 个 wall。

对比基准：test_edge_sam.py (MediaPipe) 当前全量 7/7 已知命中（ring 黑修正落地后）。

DWPose 133 点 COCO-WholeBody 映射:
  L_hand = 9 (left_wrist) / R_hand = 10 (right_wrist)
  L_foot = 17 (left_big_toe) / R_foot = 20 (right_big_toe)   ← 脚尖！
  body center = 肩(5,6) + 髋(11,12) 平均

用法：
  python exp_dwpose_full.py
输出：
  dwpose_full_summary.txt   （每 wall 聚合结果 + 命中判定）
  dwpose_vis_<wall>/        （每帧可视化，简化版：不画骨架）
"""
import os
import re
import sys
from collections import Counter
from pathlib import Path

_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BASE)
# EdgeSAM 源码：引用根目录共享副本（大目录不复制，实验与生产共用即可）
_ROOT = os.path.dirname(_BASE)
sys.path.insert(0, _ROOT)  # 供 edgesam_stub（根目录共享）
sys.path.insert(0, os.path.join(_ROOT, "EdgeSAM"))

import cv2
import numpy as np

import edgesam_stub  # noqa: F401

from dwpose_onnx import DWPoseONNX
from edgesam_holds import EdgeSamHoldSegmenter
from yolo_person_mask import PersonMasker

# 复用可视化（dwpose_exp 本地副本，与 MediaPipe 生产 test_edge_sam 分离）
from draw_results import draw_results

BASE = os.path.dirname(os.path.abspath(__file__))
WALLS_DIR = Path(BASE) / ".." / "img_xhs_walls"

TRUTH = {
    '2XRo7VbPgTe_wall_01_known_IMG_20260808_193753': '黑',
    '2XRo7VbPgTe_wall_02_known_IMG_20260808_193745': '蓝',
    '2XRo7VbPgTe_wall_03_known_IMG_20260808_193754': '粉红',
    '2XRo7VbPgTe_wall_04_known_IMG_20260808_193724': '青',
    '3OHz6rZ5opG_wall_01_unknown': '蓝',
    '3OHz6rZ5opG_wall_02_unknown': '红',
    '3OHz6rZ5opG_wall_03_known_IMG_20260808_193612': '红',
    '3OHz6rZ5opG_wall_04_line1_white': '白',
    '3OHz6rZ5opG_wall_04_line2_pink': '粉红',
    '3OHz6rZ5opG_wall_05_known_IMG_20260808_193650': '绿',
}

# 端点 -> (DWPose 关键点索引, 名称)
DW_ENDPOINTS = [
    ('L_hand', 9),
    ('R_hand', 10),
    ('L_foot', 17),   # big_toe 脚尖
    ('R_foot', 20),   # big_toe 脚尖
]
DW_SHOULDERS = (5, 6)
DW_HIPS = (11, 12)

SKIP_COLORS = {'低置信度', '非岩点', '无法判断', '未知'}

# 关键点置信度阈值
CONF_THRESHOLD = float(os.environ.get('DW_POSE_CONF', '0.3'))


def analyze_frame(image_bgr, dw, segmenter, person_masker):
    h, w = image_bgr.shape[:2]

    # ---------- DWPose ----------
    boxes, kps, scores = dw.run(image_bgr)
    if kps is None:
        return None
    k = kps[0]
    s = scores[0]

    # body center（DWPose 肩髋）
    pts = k[list(DW_SHOULDERS) + list(DW_HIPS)]
    body_cx = float(pts[:, 0].mean())
    body_cy = float(pts[:, 1].mean())

    # ---------- person mask ----------
    person_mask = person_masker.segment(image_bgr)

    # ---------- EdgeSAM ----------
    segmenter.set_image(image_bgr)
    samples = []
    color_names = {'红', '橙', '黄', '绿', '青', '蓝', '紫', '粉红', '白', '黑', '未知'}

    for ep_name, kidx in DW_ENDPOINTS:
        x, y = k[kidx]
        conf = float(s[kidx])
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
    print(f"  ✅ DWPose 命中 {n_segmented}/4 端点")
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
        'person_mask': person_mask,
    }


def main():
    os.chdir(BASE)
    # 大权重在根目录（不复制），用 ../ 引用
    dw = DWPoseONNX('../yolox_l.onnx', '../dw-ll_ucoco_384.onnx')
    segmenter = EdgeSamHoldSegmenter('../weights/edge_sam.pth', device='cpu')
    person_masker = PersonMasker(model_name="../yolo26m-seg.pt", device='cpu')

    walls = sorted(WALLS_DIR.iterdir())
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        walls = walls[:int(sys.argv[1])]
    print(f"共 {len(walls)} 个 wall\n")

    lines = []
    all_results = []

    for wall_dir in walls:
        wall_name = wall_dir.name
        n_frames = len(list(wall_dir.glob('*.jpg')))
        print(f"=== {wall_name} ({n_frames} 张) ===")
        vis_dir = Path(BASE) / f"dwpose_vis_{wall_name}"
        vis_dir.mkdir(exist_ok=True)

        files = sorted(wall_dir.glob('*.jpg'),
                       key=lambda f: (int(re.sub(r'\D', '', f.name)) if re.sub(r'\D', '', f.name) else 99999, f.name))

        aggregate_samples = []
        for i, fp in enumerate(files, 1):
            image_bgr = cv2.imread(str(fp))
            if image_bgr is None:
                continue
            print(f"\n[{i}] {fp.name}")
            result = analyze_frame(image_bgr, dw, segmenter, person_masker)
            if result is None:
                print("  ⚠️ 未检测到人体")
                continue
            aggregate_samples.append((fp.name, result['samples']))
            # 可视化（简化版：无骨架）
            display = image_bgr.copy()
            draw_results(display, result['samples'], result['route_color'],
                         result['confidence'], result['votes'],
                         sum(1 for s in result['samples'] if s.get('mask_uint8') is not None),
                         person_mask=result['person_mask'])
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

    print("\n=== 汇总 (DWPose) ===")
    hits = sum(1 for r in all_results if r[5] == '✅')
    known = sum(1 for r in all_results if r[4] is not None)
    print(f"命中: {hits}/{known} (已知真值 wall)")
    for l in lines:
        print(f"  {l}")

    with open('dwpose_full_summary.txt', 'w', encoding='utf-8') as f:
        f.write(f"DWPose 全量回归: 命中 {hits}/{known}\n")
        for l in lines:
            f.write(l + "\n")
    print("\n已保存 dwpose_full_summary.txt")

    # 本脚本无 MediaPipe（onnxruntime + EdgeSAM + YOLO seg），
    # 不需要 os._exit；正常 return 即可（已验证 test_no_os_exit_dwpose.py）
    return 0


if __name__ == '__main__':
    sys.exit(main())
