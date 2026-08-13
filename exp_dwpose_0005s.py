"""
DWPose vs MediaPipe vs YOLO26 pose 端点对比实验（单帧 0005s 黑线案例）

DWPose = 133 点 COCO-WholeBody，含脚尖 big_toe（17/20），正好补 YOLO26 缺脚尖。
MediaPipe 33 点: L_hand=15(R_wrist? 不) L_hand=15? 见下
  MP: L_wrist=15, R_wrist=16, L_foot_index=31, R_foot_index=32
YOLO26: 9=left_wrist, 10=right_wrist, 15/16=ankle（无脚尖）
DWPose: 9=left_wrist, 10=right_wrist, 17/20=big_toe（脚尖）, 15/16=ankle

输出:
  dwpose_0005s_compare.jpg   （三栏对比图）
  dwpose_0005s_compare.json
"""
import json
import os
import sys

_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BASE)
sys.path.insert(0, os.path.join(_BASE, "EdgeSAM"))

import cv2
import numpy as np
import mediapipe as mp

import edgesam_stub  # noqa: F401
from ultralytics import YOLO

from dwpose_onnx import DWPoseONNX
from edgesam_holds import EdgeSamHoldSegmenter
from route_color import merge_flip_landmarks, _LM, _FLIP_LR_PAIR

BASE = r"D:\Project\climbing_wall_search\mediapipe2"
IMG = r"D:\Project\climbing_wall_search\video_frames_cache\小红书_2XRo7VbPgTe\walls\wall_01_known_IMG_20260808_193753\frame_0005s.jpg"

# 索引
MP_L_WRIST, MP_R_WRIST = 15, 16
MP_L_FOOT, MP_R_FOOT = 31, 32
YOLO_LEFT_WRIST, YOLO_RIGHT_WRIST = 9, 10
YOLO_LEFT_ANKLE, YOLO_RIGHT_ANKLE = 15, 16
DW_L_WRIST, DW_R_WRIST = 9, 10
DW_L_BIG_TOE, DW_R_BIG_TOE = 17, 20
DW_L_ANKLE, DW_R_ANKLE = 15, 16


def mp_detect_pose(image_rgb, landmarker, w, h, flip_ensemble=True):
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
    yolo = YOLO('yolo26n-pose.pt')
    res = yolo.predict(image_rgb, conf=0.25, verbose=False)[0]
    yolo_pts = {}
    print("\nYOLO26 pose 检测: person 数量 =", len(res.keypoints) if res.keypoints is not None else 0)
    if res.keypoints is not None and len(res.keypoints) > 0:
        kps = res.keypoints[0]
        kps_xy = kps.xy[0].cpu().numpy()
        kps_conf = kps.conf[0].cpu().numpy() if kps.conf is not None else None
        def _pt(idx):
            x, y = kps_xy[idx]
            c = kps_conf[idx] if kps_conf is not None else 1.0
            return (float(x), float(y), float(c))
        yolo_pts = {
            'L_hand': _pt(YOLO_LEFT_WRIST),
            'R_hand': _pt(YOLO_RIGHT_WRIST),
            'L_foot': _pt(YOLO_LEFT_ANKLE),
            'R_foot': _pt(YOLO_RIGHT_ANKLE),
        }
        print("YOLO26 端点 (L/R_foot = ankle 脚踝, 非脚尖):")
        for k, v in yolo_pts.items():
            print(f"  {k}: ({v[0]:.1f}, {v[1]:.1f}) conf={v[2]:.2f}")

    # ---------- 3. DWPose ----------
    dw = DWPoseONNX(os.path.join(BASE, 'yolox_l.onnx'), os.path.join(BASE, 'dw-ll_ucoco_384.onnx'))
    dw_boxes, dw_kps, dw_scores = dw.run(image_bgr)
    dw_pts = {}
    print("\nDWPose 检测 box:", None if dw_boxes is None else dw_boxes[0])
    if dw_kps is not None:
        k = dw_kps[0]
        s = dw_scores[0]
        def _pt(idx, tag):
            if s[idx] < 0.3:
                return None
            return (float(k[idx, 0]), float(k[idx, 1]), float(s[idx]))
        dw_pts = {
            'L_hand': _pt(DW_L_WRIST, 'L_wrist'),
            'R_hand': _pt(DW_R_WRIST, 'R_wrist'),
            'L_foot': _pt(DW_L_BIG_TOE, 'L_big_toe'),   # 脚尖！
            'R_foot': _pt(DW_R_BIG_TOE, 'R_big_toe'),   # 脚尖！
        }
        print("DWPose 端点 (L/R_foot = big_toe 脚尖):")
        for kk, v in dw_pts.items():
            print(f"  {kk}: {v}")

    # ---------- 4. EdgeSAM segment_endpoint 对比 ----------
    segmenter = EdgeSamHoldSegmenter('weights/edge_sam.pth', device='cpu')
    segmenter.set_image(image_bgr)

    def _body_center(lms):
        if lms is None:
            return w / 2, h / 2
        sl, sr = lms[11], lms[12]
        hl, hr = lms[23], lms[24]
        return ((sl.x + sr.x + hl.x + hr.x) * w / 4,
                (sl.y + sr.y + hl.y + hr.y) * h / 4)

    mp_bc = _body_center(mp_lms)
    print(f"\nbody_center (MediaPipe): {mp_bc}")

    # DWPose 的 body center 用肩膀+髋（DWPose: 5,6 shoulders; 11,12 hips）
    dw_bc = mp_bc
    if dw_kps is not None:
        k = dw_kps[0]
        s = dw_scores[0]
        if all(s[i] > 0.3 for i in [5, 6, 11, 12]):
            dw_bc = ((k[5,0]+k[6,0]+k[11,0]+k[12,0])/4, (k[5,1]+k[6,1]+k[11,1]+k[12,1])/4)
    print(f"body_center (DWPose): {dw_bc}")

    results = {}
    for ep in ['L_hand', 'R_hand', 'L_foot', 'R_foot']:
        row = {}
        # MP
        px, py = mp_pts[ep]
        res_mp = segmenter.segment_endpoint(px, py, mp_bc[0], mp_bc[1])
        row['mp'] = {
            'pt': mp_pts[ep],
            'color': res_mp['color_name'] if res_mp and res_mp.get('color_name') else ('非岩点' if res_mp is None else '?'),
            'area': res_mp['area'] if res_mp else 0,
            'ring_ratio': (res_mp.get('ring_black', {}) or {}).get('ratio', 0) if res_mp else None,
            'ring_fixed': (res_mp.get('ring_black', {}) or {}).get('fixed', False) if res_mp else None,
        }
        print(f"[MP]  {ep} @({px:.0f},{py:.0f}) -> {row['mp']['color']}  area={row['mp']['area']}  ring={row['mp']['ring_ratio']:.1%} fixed={row['mp']['ring_fixed']}")

        # YOLO26
        if ep in yolo_pts:
            yx, yy, _ = yolo_pts[ep]
            res_y = segmenter.segment_endpoint(yx, yy, mp_bc[0], mp_bc[1])
            row['yolo'] = {
                'pt': (yolo_pts[ep][0], yolo_pts[ep][1]),
                'color': res_y['color_name'] if res_y and res_y.get('color_name') else '非岩点',
                'area': res_y['area'] if res_y else 0,
                'ring_ratio': (res_y.get('ring_black', {}) or {}).get('ratio', 0) if res_y else None,
                'ring_fixed': (res_y.get('ring_black', {}) or {}).get('fixed', False) if res_y else None,
            }
            print(f"[YOLO]{ep} @({yx:.0f},{yy:.0f}) -> {row['yolo']['color']}  area={row['yolo']['area']}  ring={row['yolo']['ring_ratio']:.1%} fixed={row['yolo']['ring_fixed']}")

        # DWPose
        if ep in dw_pts and dw_pts[ep]:
            dx, dy, _ = dw_pts[ep]
            res_dw = segmenter.segment_endpoint(dx, dy, dw_bc[0], dw_bc[1])
            row['dwpose'] = {
                'pt': (dw_pts[ep][0], dw_pts[ep][1]),
                'color': res_dw['color_name'] if res_dw and res_dw.get('color_name') else '非岩点',
                'area': res_dw['area'] if res_dw else 0,
                'ring_ratio': (res_dw.get('ring_black', {}) or {}).get('ratio', 0) if res_dw else None,
                'ring_fixed': (res_dw.get('ring_black', {}) or {}).get('fixed', False) if res_dw else None,
            }
            print(f"[DW]  {ep} @({dx:.0f},{dy:.0f}) -> {row['dwpose']['color']}  area={row['dwpose']['area']}  ring={row['dwpose']['ring_ratio']:.1%} fixed={row['dwpose']['ring_fixed']}")

        results[ep] = row

    with open('dwpose_0005s_compare.json', 'w', encoding='utf-8') as f:
        json.dump({'img': IMG, 'results': results}, f, ensure_ascii=False, indent=2)
    print("\n已保存 dwpose_0005s_compare.json")

    # ---------- 5. 可视化三栏对比图 ----------
    vis = np.zeros((h, w * 3, 3), dtype=np.uint8)
    vis[:, :w] = image_bgr
    vis[:, w:2*w] = image_bgr
    vis[:, 2*w:] = image_bgr

    COLORS = {'L_hand': (0, 215, 255), 'R_hand': (255, 215, 0), 'L_foot': (255, 0, 255), 'R_foot': (0, 165, 255)}

    def _draw(offset_x, pts, tag, extra=None):
        for ep, pt in pts.items():
            if pt is None:
                continue
            px, py = pt[0] + offset_x, pt[1]
            color = COLORS[ep]
            cv2.circle(vis, (int(px), int(py)), 6, (255, 255, 255), -1)
            cv2.circle(vis, (int(px), int(py)), 4, color, -1)
            label = f"{ep}"
            if extra and ep in extra:
                label += f" {extra[ep]}"
            cv2.putText(vis, label, (int(px) + 8, int(py) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)
        cv2.putText(vis, tag, (offset_x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2, cv2.LINE_AA)

    mp_2d = {k: (v[0], v[1]) for k, v in mp_pts.items() if v}
    yolo_2d = {k: (v[0], v[1]) for k, v in yolo_pts.items()}
    dw_2d = {k: (v[0], v[1]) for k, v in dw_pts.items() if v}

    _draw(0, mp_2d, "MediaPipe")
    _draw(w, yolo_2d, "YOLO26-pose (ankle)")
    _draw(2*w, dw_2d, "DWPose (big_toe)", extra={k: f"{v[2]:.2f}" for k, v in dw_pts.items() if v})

    cv2.imwrite('dwpose_0005s_compare.jpg', vis)
    print("已保存 dwpose_0005s_compare.jpg")

    os._exit(0)


if __name__ == '__main__':
    main()
