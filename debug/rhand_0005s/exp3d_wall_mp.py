"""实验3d 墙级回归：多正点方案（1 predict，零额外成本）vs 基线。

方案：segment_endpoint 的 predict 用 端点+偏移锚点 作为多个正点
（替代原来的单正点），其余逻辑（候选筛选/颜色统计）不变。
对比墙级命中率（_run_walls 口径）。
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

# 多正点方案锚点偏移
ANCHOR_OFFSETS = [(0, 0), (-40, 0), (-35, -25)]

TRUTH = {
    '2XRo7VbPgTe_wall_01_known_IMG_20260808_193753': '黑',
    '2XRo7VbPgTe_wall_02_known_IMG_20260808_193745': '蓝',
    '2XRo7VbPgTe_wall_03_known_IMG_20260808_193754': '粉红',
    '2XRo7VbPgTe_wall_04_known_IMG_20260808_193724': '青',
    '3OHz6rZ5opG_wall_03_known_IMG_20260808_193612': '红',
    '3OHz6rZ5opG_wall_04_known_IMG_20260808_193621': '白',
    '3OHz6rZ5opG_wall_05_known_IMG_20260808_193650': '绿',
}


def segment_multipoint(segmenter, px, py, body_cx, body_cy, person_mask):
    """端点+偏移锚点多正点 predict，复用 segment_endpoint 的主体逻辑"""
    h, w = segmenter.last_h, segmenter.last_w
    eff_box_half = max(edgesam_holds.BOX_HALF_MIN, h // edgesam_holds.BOX_HALF_HEIGHT_DIV)
    eff_box_half = min(eff_box_half, edgesam_holds.BOX_HALF_MAX)
    ix = max(0, min(int(round(px)), w - 1))
    iy = max(0, min(int(round(py)), h - 1))
    bx1 = max(0, ix - eff_box_half)
    by1 = max(0, iy - eff_box_half)
    bx2 = min(w - 1, ix + eff_box_half)
    by2 = min(h - 1, iy + eff_box_half)
    box = np.array([bx1, by1, bx2, by2], dtype=np.float64)

    neg = segmenter._make_negative_prompts(px, py, body_cx, body_cy)
    if person_mask is not None and person_mask.any():
        edge_pts = edgesam_holds._sample_person_edge_points(
            person_mask, n=edgesam_holds.PERSON_EDGE_NEG_N,
            max_radius=edgesam_holds.PERSON_EDGE_MAX_RADIUS)
        if len(edge_pts) > 0:
            neg = np.vstack([neg, edge_pts])

    anchors = []
    for ox, oy in ANCHOR_OFFSETS:
        ax, ay = px + ox, py + oy
        if 0 <= ax < w and 0 <= ay < h:
            anchors.append([ax, ay])
    point_coords = np.vstack([anchors, neg])
    point_labels = np.array([1] * len(anchors) + [0] * len(neg), dtype=np.int64)

    masks, scores, _ = segmenter.predictor.predict(
        box=box, point_coords=point_coords, point_labels=point_labels,
        num_multimask_outputs=4, use_stability_score=True)
    if masks.ndim == 2:
        masks = masks[None, :, :]
        scores = np.array([float(scores)])

    img_area = w * h
    area_cap = int(img_area * edgesam_holds.AREA_MAX_RATIO)
    candidates = []
    for i, m in enumerate(masks):
        area = int(m.sum())
        if area < edgesam_holds.AREA_MIN or area > area_cap:
            continue
        if float(scores[i]) < edgesam_holds.SCORE_MIN:
            continue
        ys_m, xs_m = np.where(m)
        if len(xs_m) == 0:
            continue
        dist_to_ep = float(np.min(np.hypot(xs_m - ix, ys_m - iy)))
        if dist_to_ep > edgesam_holds.DIST_MAX:
            continue
        candidates.append((float(scores[i]), dist_to_ep, area, m))

    if not candidates:
        return None
    candidates.sort(key=lambda c: (-c[0], c[1], -c[2]))
    _, _, _, chosen = candidates[0]
    if person_mask is not None:
        chosen = chosen & ~person_mask.astype(bool)
    if chosen.sum() < edgesam_holds.COLOR_MIN_PIXELS:
        return None
    chosen = edgesam_holds._keep_main_component(chosen, ix, iy)
    if chosen.sum() < 100:
        return None
    vote = edgesam_holds._pixel_color_vote(segmenter.last_image_bgr, chosen)
    return vote['top_color']


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
    print(f"{'wall':48s} {'基线':22s} {'多正点':22s}  真值")

    for wall_dir in walls:
        wall_name = wall_dir.name
        files = sorted([f for f in wall_dir.glob('*.jpg')])

        votes_base = Counter()
        votes_mp = Counter()

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
                # 基线
                try:
                    res = segmenter.segment_endpoint(px, py, body_cx, body_cy, person_mask=person_mask)
                    bn = res['color_name'] if (res and res['area'] >= 100) else None
                except Exception:
                    bn = None
                # 多正点
                try:
                    mn = segment_multipoint(segmenter, px, py, body_cx, body_cy, person_mask)
                except Exception:
                    mn = None
                if bn in edgesam_holds.COLOR_NAMES if hasattr(edgesam_holds, 'COLOR_NAMES') else bn in {'红','橙','黄','绿','青','蓝','紫','粉红','白','黑'}:
                    votes_base[bn] += 1
                if mn in {'红','橙','黄','绿','青','蓝','紫','粉红','白','黑'}:
                    votes_mp[mn] += 1

        b_top, b_n = votes_base.most_common(1)[0] if votes_base else ('-', 0)
        m_top, m_n = votes_mp.most_common(1)[0] if votes_mp else ('-', 0)
        truth = TRUTH.get(wall_name, '?')
        b_hit = '✅' if truth != '?' and b_top == truth else ('❌' if truth != '?' else '?')
        m_hit = '✅' if truth != '?' and m_top == truth else ('❌' if truth != '?' else '?')
        print(f"{wall_name[:48]:48s} {b_top}({b_n}){b_hit:2s}        {m_top}({m_n}){m_hit:2s}    {truth}")


if __name__ == '__main__':
    main()
