"""
基于 EdgeSAM 的攀岩线路色识别 demo。

流程：
  1. MediaPipe Pose 拿 4 个端点（L_hand/R_hand/L_foot/R_foot），flip ensemble
  2. 对每个端点位置作为正点 prompt（+ 4 个负点）调 EdgeSAM 分割 hold
  3. 取 mask 内像素 HSV 主色投票
  4. 可视化：原图 + mask 半透明覆盖 + 端点 + 头部信息
  5. 输出 result_edgesam_<N>.jpg

用法：
  python test_edge_sam.py                        # 默认 edge_sam.pth
  python test_edge_sam.py --checkpoint edge_sam_3x.pth
  python test_edge_sam.py --img img --alpha 0.45

依赖：
  weights/edge_sam.pth (or edge_sam_3x.pth) — 见 README
"""
import argparse
import os
import re
import sys
from collections import Counter

import cv2
import numpy as np

import edgesam_stub  # noqa: F401 必须在 edge_sam 之前
import mediapipe as mp

import edgesam_holds
from route_color import merge_flip_landmarks
from yolo_person_mask import PersonMasker, dilate_mask


# ============================================================
# 姿态检测（复用 test_mp.py 写法）
# ============================================================
def detect_pose(image_rgb: np.ndarray, landmarker, w: int, h: int,
                flip_ensemble: bool = True):
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = landmarker.detect(mp_img)
    if not result.pose_landmarks:
        if flip_ensemble:
            flip_rgb = cv2.cvtColor(cv2.flip(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), 1), cv2.COLOR_BGR2RGB)
            mp_flip = mp.Image(image_format=mp.ImageFormat.SRGB, data=flip_rgb)
            result_flip = landmarker.detect(mp_flip)
            if result_flip.pose_landmarks:
                return merge_flip_landmarks(
                    result.pose_landmarks[0], result_flip.pose_landmarks[0], w
                )
        return None
    if flip_ensemble:
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        flip_rgb = cv2.cvtColor(cv2.flip(image_bgr, 1), cv2.COLOR_BGR2RGB)
        mp_flip = mp.Image(image_format=mp.ImageFormat.SRGB, data=flip_rgb)
        result_flip = landmarker.detect(mp_flip)
        if result_flip.pose_landmarks:
            return merge_flip_landmarks(
                result.pose_landmarks[0], result_flip.pose_landmarks[0], w
            )
    return result.pose_landmarks[0]


# ============================================================
# 可视化
# ============================================================
def draw_results(image_bgr: np.ndarray, samples: list[dict], route_color: str,
                 confidence: float, votes: dict, n_segmented: int,
                 person_mask: np.ndarray | None = None):
    h_img, w_img = image_bgr.shape[:2]

    # ---------- Header（中等暗度，不再压成 25% 透明） ----------
    header_h = 80
    overlay = image_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (w_img, header_h), (60, 60, 60), -1)
    cv2.addWeighted(overlay, 0.55, image_bgr, 0.45, 0, image_bgr)
    person_pct = (person_mask.mean() * 100) if person_mask is not None else 0
    cv2.putText(image_bgr, f"EdgeSAM route  segmented={n_segmented}  person={person_pct:.1f}%",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(image_bgr, f"Color: {route_color}    Confidence: {confidence:.0%}",
                (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    vote_str = "  ".join(f"{c}={n}" for c, n in sorted(votes.items(), key=lambda x: -x[1]))
    cv2.putText(image_bgr, vote_str, (10, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 230, 255), 1, cv2.LINE_AA)

    # ---------- 人体 mask：很淡的紫色填充 + 描边（不进 hold 颜色统计） ----------
    if person_mask is not None and person_mask.any():
        pm_u8 = person_mask.astype(np.uint8) * 255
        # 淡紫半透明填充（让"人体"区域一眼可见，但不抢 hold mask 的视觉）
        person_overlay = image_bgr.copy()
        person_overlay[pm_u8 > 0] = (255, 180, 180)  # 淡紫
        cv2.addWeighted(person_overlay, 0.18, image_bgr, 0.82, 0, image_bgr)
        # 紫描边 2px（与 hold mask 的亮青色边界明显区分）
        pm_contours, _ = cv2.findContours(pm_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image_bgr, pm_contours, -1, (180, 130, 255), 2)  # 淡紫

    LABEL_DX = {'L_hand': -1, 'R_hand': +1, 'L_foot': -1, 'R_foot': +1}
    # 四肢端点 mask 边界颜色（BGR，高对比，与人体淡紫描边区分）：
    # L_hand 亮青 / R_hand 亮黄 / L_foot 洋红 / R_foot 亮橙
    MASK_COLORS = {
        'L_hand': (255, 215, 0),
        'R_hand': (0, 255, 255),
        'L_foot': (255, 0, 255),
        'R_foot': (0, 128, 255),
    }
    for s in samples:
        px, py = int(s['px'][0]), int(s['px'][1])

        # ---------- Mask：只在 mask 区域内 blend（之前 addWeighted 全图变暗是 bug） ----------
        if s.get('mask_uint8') is not None:
            mask_bool = s['mask_bool']

            # 可视化层连通域过滤：SAM 的 mask 可以包含多个不相连的区域
            #（如 1.jpg R_hand：手 + 远处黄色 hold + 碎片共 5 个域），显示出来
            # 很困惑。这里只显示"包含端点或面积最大的连通域"（hold 主体），
            # 丢弃远处碎片。注意：只影响显示，颜色判定仍用完整 mask。
            disp_mask = mask_bool
            n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(
                mask_bool.astype(np.uint8), 8)
            if n_cc > 2:
                iy, ix = int(px), int(py)
                iy = max(0, min(iy, mask_bool.shape[0] - 1))
                ix = max(0, min(ix, mask_bool.shape[1] - 1))
                ep_label = labels[iy, ix]
                if ep_label > 0 and stats[ep_label, cv2.CC_STAT_AREA] >= 40:
                    disp_mask = (labels == ep_label)
                else:
                    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                    disp_mask = (labels == best)
            mask_bool = disp_mask

            color = np.array(_hsv_to_bgr_(s['h'], s['s'], s['v']), dtype=np.uint8)
            # mask 颜色加饱和/亮（避免色相分布后颜色发暗发灰）
            vivid_color = _vivid_color(s['h'], s['s'], s['v'])

            alpha = 0.50
            idx = mask_bool
            blend = (image_bgr[idx].astype(np.float32) * (1 - alpha)
                     + vivid_color.astype(np.float32) * alpha)
            image_bgr[idx] = np.clip(blend, 0, 255).astype(np.uint8)

            # mask 边界线：按四肢端点用不同颜色（BGR），外层黑色增强对比。
            # 不随 hold 颜色变 — 白色 hold 的 vivid 色是灰白，和人体淡紫描边、
            # 白墙都分不清，且不同端点要能一眼区分。
            mask_u8 = (mask_bool.astype(np.uint8)) * 255
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            border_color = MASK_COLORS.get(s['point'], (255, 215, 0))
            cv2.drawContours(image_bgr, contours, -1, (0, 0, 0), 6)                # 黑色外环
            cv2.drawContours(image_bgr, contours, -1, border_color, 3)             # 端点专属色 3px

        # ---------- 端点：红色实心点 + 白圈（不变） ----------
        cv2.circle(image_bgr, (px, py), 5, (255, 255, 255), -1)
        cv2.circle(image_bgr, (px, py), 3, (0, 0, 255), -1)

        # ---------- 标签（字号放大、加描边、可读性更好） ----------
        dx = LABEL_DX.get(s['point'], +1)
        if s.get('score') is not None and s.get('name') not in ('低置信度', '非岩点'):
            label = f"{s['point']}->{s['name']} s={s['score']:.2f} a={s['area']}"
        elif s['name'] not in ('低置信度', '非岩点'):
            label = f"{s['point']}->{s['name']}"
        else:
            label = f"{s['point']}({s['name']})"
        font_scale = 0.5
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        if dx > 0:
            bx, by = px + 40, py - 14
        else:
            bx, by = px - 40 - tw, py - 14
        bx = max(2, min(bx, w_img - tw - 4))
        by = max(th + 4, min(by, h_img - 4))
        cv2.line(image_bgr, (px, py), (bx + tw // 2, by - th // 2), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(image_bgr, (bx - 3, by - th - 3), (bx + tw + 3, by + 3), (0, 0, 0), -1)
        # 描边（黑色 2px）+ 文字（白色 1px）
        cv2.putText(image_bgr, label, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(image_bgr, label, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)


def _vivid_color(h: float, s: float, v: float) -> np.ndarray:
    """把 hold 的 HSV 色彩拉到最大饱和/亮（避免 mask 平均后发灰变暗）。"""
    s_new = float(np.clip(max(s, 100), 100, 255))
    v_new = float(np.clip(max(v, 150), 150, 255))
    color = np.array([[[int(h), int(s_new), int(v_new)]]], dtype=np.uint8)
    bgr = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)[0, 0]
    return bgr.astype(np.uint8)


# expose route_color 的 hsv→bgr 工具
def _hsv_to_bgr_(h, s, v):
    color = np.array([[[int(h), int(s), int(v)]]], dtype=np.uint8)
    return tuple(int(c) for c in cv2.cvtColor(color, cv2.COLOR_HSV2BGR)[0, 0])

edgesam_holds._hsv_to_bgr_ = _hsv_to_bgr_  # 让模块内部调用 ok（draw 实际在主文件用，不在子模块）


# ============================================================
# 主流程
# ============================================================
def analyze_image(image_path: str, output_path: str, args, segmenter, person_masker,
                  landmarker,
                  PoseLandmarksConnections, DrawingUtils, DrawingStyles) -> dict | None:
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"  ❌ 无法读取: {image_path}")
        return None

    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # 1. 姿态：flip ensemble
    landmarks = detect_pose(image_rgb, landmarker, w, h, flip_ensemble=True)
    if landmarks is None:
        print(f"  ⚠️ 未检测到人体")
        cv2.imwrite(output_path, image_bgr)
        return None

    # 2. 身体中心
    shoulder_l = landmarks[11]
    shoulder_r = landmarks[12]
    hip_l = landmarks[23]
    hip_r = landmarks[24]
    body_cx = (shoulder_l.x + shoulder_r.x + hip_l.x + hip_r.x) * w / 4
    body_cy = (shoulder_l.y + shoulder_r.y + hip_l.y + hip_r.y) * h / 4

    # 3. YOLO26 person mask（一次 inference，用于 4 个端点共用）
    person_mask = person_masker.segment(image_bgr) if person_masker is not None else None
    if person_mask is not None:
        # 边缘膨胀 6 px：让"贴近身体的碎片"也归为人
        person_mask_d = dilate_mask(person_mask, radius=6)
    else:
        person_mask_d = None

    # 3. 端点级 EdgeSAM 分割
    segmenter.set_image(image_bgr)

    samples = []
    color_names = {'红', '橙', '黄', '绿', '青', '蓝', '紫', '粉红', '白', '未知'}

    endpoints = [('L_hand', 15), ('R_hand', 16), ('L_foot', 31), ('R_foot', 32)]
    for ep_name, idx in endpoints:
        lm = landmarks[idx]
        px, py = lm.x * w, lm.y * h

        if lm.visibility < 0.3:
            samples.append({
                'point': ep_name, 'name': '低置信度',
                'px': (px, py), 'visibility': float(lm.visibility),
                'mask_uint8': None, 'mask_bool': None,
                'area': 0, 'score': 0.0,
            })
            continue

        try:
            res = segmenter.segment_endpoint(
                px, py, body_cx, body_cy,
                person_mask=person_mask_d,
            )
        except Exception as e:
            print(f"  ⚠️ EdgeSAM 失败 ({ep_name}): {e}")
            res = None

        if res is None or res['area'] < 100 or res['color_name'] == '非岩点':
            samples.append({
                'point': ep_name, 'name': '非岩点',
                'px': (px, py), 'visibility': float(lm.visibility),
                'mask_uint8': None, 'mask_bool': None,
                'area': 0, 'score': 0.0,
            })
            continue

        samples.append({
            'point': ep_name,
            'name': res['color_name'],
            'h': res['h'], 's': res['s'], 'v': res['v'],
            'area': res['area'],
            'score': res['score'],
            'px': (px, py),
            'mask_uint8': res['mask'],
            'mask_bool': res['mask_bool'],
            'visibility': float(lm.visibility),
        })

    # 4. 投票
    valid = [s for s in samples if s['name'] in color_names]
    if not valid:
        route_color, votes, conf = '无法判断', {}, 0.0
    else:
        votes = Counter(s['name'] for s in valid)
        route_color, top = votes.most_common(1)[0]
        conf = top / len(valid)
        votes = dict(votes)

    n_segmented = sum(1 for s in samples if s.get('mask_uint8') is not None)
    print(f"  ✅ EdgeSAM 命中 {n_segmented}/4 端点")
    for s in samples:
        if s.get('mask_uint8') is None:
            print(f"    {s['point']:8s}: {s['name']:6s}  vis={s.get('visibility', 0):.2f}")
        else:
            print(f"    {s['point']:8s}: {s['name']:6s}  "
                  f"area={s['area']:5d} score={s.get('score', 0):.2f}  "
                  f"HSV=({s['h']:5.1f},{s['s']:5.1f},{s['v']:5.1f})")
    print(f"  >>> 线路色 = {route_color}  置信度 = {conf:.0%}  票数 = {votes}")

    # 5. 可视化（在原图上画 mask + 端点 + header；先 mask 再骨架）
    display = image_bgr.copy()
    draw_results(display, samples, route_color, conf, votes, n_segmented,
                 person_mask=person_mask)
    DrawingUtils.draw_landmarks(
        image=display,
        landmark_list=landmarks,
        connections=PoseLandmarksConnections.POSE_LANDMARKS,
        landmark_drawing_spec=DrawingStyles.get_default_pose_landmarks_style(),
    )
    cv2.imwrite(output_path, display)
    print(f"  💾 {output_path}")
    return {
        'route_color': route_color,
        'confidence': conf,
        'votes': votes,
        'samples': samples,
    }


def main():
    ap = argparse.ArgumentParser(description="EdgeSAM-based climbing route color detector")
    ap.add_argument('--checkpoint', default='weights/edge_sam.pth',
                    help='Path to EdgeSAM .pth checkpoint')
    ap.add_argument('--img', default='img', help='Input image dir')
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    ap.add_argument('--pose_model', default='pose_landmarker_full.task')
    ap.add_argument('--prefix', default='result_edgesam',
                    help='Output filename prefix')
    args = ap.parse_args()

    if not os.path.isfile(args.checkpoint):
        sys.exit(f"❌ checkpoint not found: {args.checkpoint}\n"
                 f"   下载：https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/main/weights/{os.path.basename(args.checkpoint)}\n"
                 f"   放进 {os.path.abspath(os.path.dirname(args.checkpoint))}/ 后重跑。")

    # EdgeSAM segmenter
    print(f"加载 EdgeSAM: {args.checkpoint}")
    segmenter = edgesam_holds.EdgeSamHoldSegmenter(args.checkpoint, device=args.device)
    print("  ✅ EdgeSAM 加载完成")

    # YOLO26 person masker（算法与可视化统一用 yolo26m）
    # yolo26n (6.4MB) 在 640 下 mask 边缘粗糙，把手脚紧贴的岩点吞进人体
    #（图1 紫色 hold 被吞 97% → subtract 扣光 → 判黄）。yolo26m (52MB)
    # mask 更精准（紫色 hold 仅被吞 2%），且保证算法与显示用同一个 mask。
    print("加载 YOLO26 person masker (yolo26m-seg.pt)…")
    person_masker = PersonMasker(model_name="yolo26m-seg.pt", device=args.device)
    print("  ✅ YOLO26 person masker 加载完成")

    # MediaPipe pose
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    PoseLandmarksConnections = mp.tasks.vision.PoseLandmarksConnections
    DrawingUtils = mp.tasks.vision.drawing_utils
    DrawingStyles = mp.tasks.vision.drawing_styles

    landmarker = PoseLandmarker.create_from_options(
        PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=args.pose_model),
            running_mode=VisionRunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.3,
            min_pose_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
    )

    # 图片列表
    img_dir = args.img
    files = [f for f in os.listdir(img_dir)
             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    files.sort(key=lambda f: (int(re.sub(r'\D', '', f)) if re.sub(r'\D', '', f) else 99999, f))

    print("=" * 60)
    print(f"  EdgeSAM 攀岩线路色识别  ({len(files)} 张图, ckpt={os.path.basename(args.checkpoint)})")
    print("=" * 60)
    for i, name in enumerate(files, 1):
        print(f"\n[{i}] {name}")
        analyze_image(
            image_path=os.path.join(img_dir, name),
            output_path=f'{args.prefix}_{i}.jpg',
            args=args,
            segmenter=segmenter,
            person_masker=person_masker,
            landmarker=landmarker,
            PoseLandmarksConnections=PoseLandmarksConnections,
            DrawingUtils=DrawingUtils,
            DrawingStyles=DrawingStyles,
        )

    # EdgeSAM torch 没有 close() 卡死问题，但 mediapipe 有，绕开清理
    os._exit(0)


if __name__ == '__main__':
    main()
