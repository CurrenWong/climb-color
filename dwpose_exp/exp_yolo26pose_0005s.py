"""
YOLO26 pose vs MediaPipe Pose 端点对比实验（单帧 0005s 黑线案例）

目标：验证 YOLO26 pose 能否提供更准的端点（尤其 R_hand 黑线案例：
MediaPipe 端点偏了导致漏切黑色岩点，之前用 ring 修正补救）。

YOLO26 pose = COCO 17 关键点:
  0 nose, 1-2 eyes, 3-4 ears, 5-6 shoulders, 7-8 elbows,
  9 left_wrist, 10 right_wrist, 11-12 hips, 13-14 knees, 15-16 ankles

MediaPipe 33 点映射（本项目端点）:
  L_hand = 15 (L_wrist)   -> YOLO 9  (left_wrist)
  R_hand = 16 (R_wrist)   -> YOLO 10 (right_wrist)
  L_foot = 31 (L_foot_index) -> YOLO 15 (left_ankle)  ← 注意：YOLO 没有脚尖，只有脚踝
  R_foot = 32 (R_foot_index) -> YOLO 16 (right_ankle) ← 注意：YOLO 没有脚尖，只有脚踝

⚠️ 关键差异：MediaPipe 的 L_foot/R_foot 是"脚尖"（foot_index，踩点处），
YOLO pose 只有 ankle（脚踝）。攀岩脚踩岩点通常是脚尖踩，所以 foot 端点
位置差异可能很大——这是实验重点观察项。

用法：
  python exp_yolo26pose_0005s.py
输出：
  yolo26pose_0005s_compare.jpg   （左右对比图）
  yolo26pose_0005s_compare.json
"""
import json
import os
import sys

_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BASE)
# EdgeSAM 源码：引用根目录共享副本（大目录不复制，实验与生产共用即可）
_ROOT = os.path.dirname(_BASE)
sys.path.insert(0, _ROOT)  # 供 edgesam_stub（根目录共享）
sys.path.insert(0, os.path.join(_ROOT, "EdgeSAM"))

import cv2
import numpy as np
import mediapipe as mp

import edgesam_stub  # noqa: F401
from ultralytics import YOLO

from edgesam_holds import EdgeSamHoldSegmenter
from route_color import merge_flip_landmarks, _LM, _FLIP_LR_PAIR

BASE = os.path.dirname(os.path.abspath(__file__))
IMG = r"..\..\video_frames_cache\小红书_2XRo7VbPgTe\walls\wall_01_known_IMG_20260808_193753\frame_0005s.jpg"

# YOLO COCO 17 点索引
YOLO_LEFT_WRIST, YOLO_RIGHT_WRIST = 9, 10
YOLO_LEFT_ANKLE, YOLO_RIGHT_ANKLE = 15, 16

# MediaPipe 33 点索引
MP_L_WRIST, MP_R_WRIST = 15, 16
MP_L_FOOT, MP_R_FOOT = 31, 32


def mp_detect_pose(image_rgb, landmarker, w, h, flip_ensemble=True):
    """同 test_edge_sam.detect_pose 的逻辑"""
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = landmarker.detect(mp_img)
    if not result.pose_landmarks:
        if flip_ensemble:
            flip_rgb = cv2.cvtColor(cv2.flip(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), 1), cv2.COLOR_BGR2RGB)
            mp_flip = mp.Image(image_format=mp.ImageFormat.SRGB, data=flip_rgb)
            result_flip = landmarker.detect(mp_flip)
            if result_flip.pose_landmarks:
                merged = []
                for i in range(33):
                    pf = result_flip.pose_landmarks[0][_FLIP_LR_PAIR.get(i, i)]
                    merged.append(_LM(1.0 - pf.x, pf.y, pf.z, pf.visibility, pf.presence))
                return merged
        return None
    if flip_ensemble:
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        flip_rgb = cv2.cvtColor(cv2.flip(image_bgr, 1), cv2.COLOR_BGR2RGB)
        mp_flip = mp.Image(image_format=mp.ImageFormat.SRGB, data=flip_rgb)
        result_flip = landmarker.detect(mp_flip)
        if result_flip.pose_landmarks and result.pose_landmarks:
            return merge_flip_landmarks(
                result.pose_landmarks[0], result_flip.pose_landmarks[0], w
            )
    return result.pose_landmarks[0]


def main():
    os.chdir(BASE)
    image_bgr = cv2.imread(IMG)
    if image_bgr is None:
        sys.exit(f"无法读取 {IMG}")
    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # ---------- 1. MediaPipe ----------
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    landmarker = PoseLandmarker.create_from_options(
        PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='pose_landmarker_full.task'),
            running_mode=VisionRunningMode.IMAGE,
            num_poses=1,
        )
    )
    mp_lms = mp_detect_pose(image_rgb, landmarker, w, h, flip_ensemble=True)
    mp_pts = {
        'L_hand': (mp_lms[MP_L_WRIST].x * w, mp_lms[MP_L_WRIST].y * h) if mp_lms else None,
        'R_hand': (mp_lms[MP_R_WRIST].x * w, mp_lms[MP_R_WRIST].y * h) if mp_lms else None,
        'L_foot': (mp_lms[MP_L_FOOT].x * w, mp_lms[MP_L_FOOT].y * h) if mp_lms else None,
        'R_foot': (mp_lms[MP_R_FOOT].x * w, mp_lms[MP_R_FOOT].y * h) if mp_lms else None,
    }
    print("MediaPipe 端点:")
    for k, v in mp_pts.items():
        print(f"  {k}: {v}")

    # ---------- 2. YOLO26 pose ----------
    yolo = YOLO('../yolo26n-pose.pt')
    res = yolo.predict(image_rgb, conf=0.25, verbose=False)[0]
    yolo_pts = {}
    print("\nYOLO26 pose 检测: person 数量 =", len(res.keypoints) if res.keypoints is not None else 0)
    if res.keypoints is not None and len(res.keypoints) > 0:
        kps = res.keypoints[0]  # 第一个人
        kps_xy = kps.xy[0].cpu().numpy()  # (17, 2)
        kps_conf = kps.conf[0].cpu().numpy() if kps.conf is not None else None
        def _pt(idx):
            x, y = kps_xy[idx]
            c = kps_conf[idx] if kps_conf is not None else 1.0
            return (float(x), float(y), float(c))
        yolo_pts = {
            'L_hand': _pt(YOLO_LEFT_WRIST),
            'R_hand': _pt(YOLO_RIGHT_WRIST),
            'L_foot': _pt(YOLO_LEFT_ANKLE),   # 注意：是 ankle 不是脚尖
            'R_foot': _pt(YOLO_RIGHT_ANKLE),
        }
        print("YOLO26 端点 (L/R_foot = ankle 脚踝, 非脚尖):")
        for k, v in yolo_pts.items():
            print(f"  {k}: ({v[0]:.1f}, {v[1]:.1f}) conf={v[2]:.2f}")

    # ---------- 3. 端点差值 ----------
    print("\n端点位置差 (YOLO - MediaPipe):")
    dists = {}
    for k in mp_pts:
        if mp_pts[k] and k in yolo_pts:
            dx = yolo_pts[k][0] - mp_pts[k][0]
            dy = yolo_pts[k][1] - mp_pts[k][1]
            d = np.hypot(dx, dy)
            dists[k] = d
            print(f"  {k}: dx={dx:+.1f} dy={dy:+.1f} dist={d:.1f}px")

    # ---------- 4. EdgeSAM segment_endpoint 对比 ----------
    segmenter = EdgeSamHoldSegmenter('../weights/edge_sam.pth', device='cpu')
    segmenter.set_image(image_bgr)

    # body center 用 MediaPipe 肩髋（两边都有就用各自的）
    def _body_center(lms):
        if lms is None:
            return w / 2, h / 2
        sl, sr = lms[11], lms[12]
        hl, hr = lms[23], lms[24]
        return ((sl.x + sr.x + hl.x + hr.x) * w / 4,
                (sl.y + sr.y + hl.y + hr.y) * h / 4)

    mp_bc = _body_center(mp_lms)
    print(f"\nbody_center (MediaPipe): {mp_bc}")

    results = {}
    for ep, idx_mp in [('L_hand', 15), ('R_hand', 16), ('L_foot', 31), ('R_foot', 32)]:
        px, py = mp_pts[ep]
        res_mp = segmenter.segment_endpoint(px, py, mp_bc[0], mp_bc[1])
        res_mp_name = res_mp['color_name'] if res_mp and res_mp.get('color_name') else ('非岩点' if res_mp is None else '?')
        res_mp_area = res_mp['area'] if res_mp else 0
        ring_mp = res_mp.get('ring_black', {}) if res_mp else {}
        print(f"[MP]  {ep} @({px:.0f},{py:.0f}) -> {res_mp_name}  area={res_mp_area}  ring={ring_mp.get('ratio', 0):.1%} fixed={ring_mp.get('fixed', False)}")

        res_yolo = None
        if ep in yolo_pts:
            yx, yy, _ = yolo_pts[ep]
            res_yolo = segmenter.segment_endpoint(yx, yy, mp_bc[0], mp_bc[1])
            res_yolo_name = res_yolo['color_name'] if res_yolo and res_yolo.get('color_name') else '非岩点'
            res_yolo_area = res_yolo['area'] if res_yolo else 0
            ring_y = res_yolo.get('ring_black', {}) if res_yolo else {}
            print(f"[YOLO]{ep} @({yx:.0f},{yy:.0f}) -> {res_yolo_name}  area={res_yolo_area}  ring={ring_y.get('ratio', 0):.1%} fixed={ring_y.get('fixed', False)}")

        results[ep] = {
            'mp': {'pt': mp_pts[ep], 'color': res_mp_name if res_mp else None, 'area': res_mp_area,
                   'ring_ratio': (res_mp.get('ring_black', {}) or {}).get('ratio', 0) if res_mp else None,
                   'ring_fixed': (res_mp.get('ring_black', {}) or {}).get('fixed', False) if res_mp else None},
            'yolo': {'pt': yolo_pts.get(ep), 'color': res_yolo['color_name'] if res_yolo else None,
                     'area': res_yolo_area,
                     'ring_ratio': (res_yolo.get('ring_black', {}) or {}).get('ratio', 0) if res_yolo else None,
                     'ring_fixed': (res_yolo.get('ring_black', {}) or {}).get('fixed', False) if res_yolo else None},
            'dist': dists.get(ep),
        }

    with open('yolo26pose_0005s_compare.json', 'w', encoding='utf-8') as f:
        json.dump({'img': IMG, 'dists': {k: round(v, 1) for k, v in dists.items()},
                   'results': results}, f, ensure_ascii=False, indent=2)
    print("\n已保存 yolo26pose_0005s_compare.json")

    # ---------- 5. 可视化对比图 ----------
    # 左 = MediaPipe，右 = YOLO26 pose
    vis = np.zeros((h, w * 2, 3), dtype=np.uint8)
    vis[:, :w] = image_bgr
    vis[:, w:] = image_bgr

    COLORS = {'L_hand': (0, 215, 255), 'R_hand': (255, 215, 0), 'L_foot': (255, 0, 255), 'R_foot': (0, 165, 255)}

    def _draw(offset_x, pts, tag, yolo_mode=False):
        # 画端点 + 标签
        for ep, pt in pts.items():
            if pt is None:
                continue
            px, py = pt[0] + offset_x, pt[1]
            color = COLORS[ep]
            cv2.circle(vis, (int(px), int(py)), 6, (255, 255, 255), -1)
            cv2.circle(vis, (int(px), int(py)), 4, color, -1)
            label = f"{ep}"
            if yolo_mode and ep in yolo_pts:
                label += f" c={yolo_pts[ep][2]:.2f}"
            cv2.putText(vis, label, (int(px) + 8, int(py) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        cv2.putText(vis, tag, (offset_x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2, cv2.LINE_AA)

    _draw(0, {k: v for k, v in mp_pts.items()}, "MediaPipe (full.task)")
    _draw(w, {k: (v[0], v[1]) for k, v in yolo_pts.items()}, "YOLO26-pose", yolo_mode=True)

    cv2.imwrite('yolo26pose_0005s_compare.jpg', vis)
    print("已保存 yolo26pose_0005s_compare.jpg")

    os._exit(0)


if __name__ == '__main__':
    main()
