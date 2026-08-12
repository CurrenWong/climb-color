"""
9.jpg 端点 box prompt 测试 (v2)：不走 detect_holds

策略：
  对每个端点，画 box = 端点周围固定半径（默认 80 px）的方形区域
  EdgeSAM 在 box 内切物体（多 mask 候选）
  选最大包含端点的 mask (完整物体 shape)
  HSV 统计剔除 person_mask 内的身体像素 → 颜色投票

不动 production 代码 — 仅 exploratory 测试。生成 _dbg_9_box_v2.jpg
"""
import os
import cv2
import numpy as np

import edgesam_stub  # noqa
import edgesam_holds
from yolo_person_mask import PersonMasker, dilate_mask
import mediapipe as mp
from route_color import merge_flip_landmarks


def detect_pose(image_rgb, landmarker, w, h, flip_ensemble=True):
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = landmarker.detect(mp_img)
    if not result.pose_landmarks:
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


def find_top_dirs(arr, body_center):
    """unused but kept for ref"""
    return None


def hue_to_color_name(h, s):
    if s < 35 and h < 35:  # near white/gray
        return '白(可能)'
    if h < 10 or h >= 170:
        return '红'
    if h < 20:
        return '橙'
    if h < 30:
        return '黄'
    if h < 80:
        return '绿'
    if h < 95:
        return '青'
    if h < 118:
        return '蓝'
    if h < 150:
        return '紫'
    if h < 170:
        return '粉红'
    return '白(可能)'


def main():
    img_path = 'img/9.jpg'
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    print(f'image: {w}x{h}')

    # EdgeSAM + YOLO26 + MediaPipe
    seg = edgesam_holds.EdgeSamHoldSegmenter('weights/edge_sam.pth', 'cpu')
    masker = PersonMasker('yolo26n-seg.pt', 'cpu')
    pm = masker.segment(img)
    pm_d15 = dilate_mask(pm, 15)

    B = mp.tasks.BaseOptions
    P = mp.tasks.vision.PoseLandmarker
    O = mp.tasks.vision.PoseLandmarkerOptions
    R = mp.tasks.vision.RunningMode
    opts = O(base_options=B(model_asset_path='pose_landmarker_full.task'),
             running_mode=R.IMAGE, num_poses=1,
             min_pose_detection_confidence=0.3,
             min_pose_presence_confidence=0.3,
             min_tracking_confidence=0.3)
    lk = P.create_from_options(opts)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    landmarks = detect_pose(rgb, lk, w, h)
    assert landmarks is not None

    seg.set_image(img)

    # 试多个 box 大小，挑最好的
    BOX_HALF = 80  # px, half side length
    endpoints = [('L_hand', 15, (0, 255, 255)),
                 ('R_hand', 16, (255, 0, 255)),
                 ('L_foot', 31, (0, 200, 255)),
                 ('R_foot', 32, (255, 100, 100))]

    results = []
    print()
    print('=' * 70)
    print('Strategy: box=端点周围 {}px,no detect_holds,选最大含端点 mask'.format(BOX_HALF * 2))
    print('=' * 70)

    for ep_name, idx, color in endpoints:
        lm = landmarks[idx]
        px, py = lm.x * w, lm.y * h
        vis = lm.visibility

        # box prompt：以端点为中心
        bx1 = max(0, int(px) - BOX_HALF)
        by1 = max(0, int(py) - BOX_HALF)
        bx2 = min(w - 1, int(px) + BOX_HALF)
        by2 = min(h - 1, int(py) + BOX_HALF)
        box = np.array([bx1, by1, bx2, by2], dtype=np.float64)

        # EdgeSAM box prompt + multimask=4
        masks, scores, _ = seg.predictor.predict(
            box=box,
            num_multimask_outputs=4,
            use_stability_score=True,
        )

        # 选"包含端点 + 包含 person's d15 外" 的最大 mask
        ix, iy = int(round(px)), int(round(py))
        candidates = []
        for i, m in enumerate(masks):
            if not m[iy, ix]:
                continue
            area = int(m.sum())
            if area < 50:
                continue
            # mask 是否有像素在 person_mask_d15 外（超出身体扩张 15px）
            out_d15 = int((m & ~pm_d15).sum())
            in_body = int((m & pm).sum())
            candidates.append({
                'idx': i, 'm': m, 'score': float(scores[i]),
                'area': area, 'in_body': in_body,
                'in_body_ratio': in_body / max(1, area),
                'out_d15': out_d15,
            })

        if not candidates:
            print(f'{ep_name} ({px:.0f},{py:.0f}): NO containing mask')
            continue

        # 选择规则：mask 应该"超出身体"才有 hold — 选 (out_d15 / area) 最大的
        # 但 page-rank by score 也 — 折衷：score 0.4+ 才参与
        candidates.sort(
            key=lambda c: (-c['out_d15'], -c['score']),
        )
        best = candidates[0]

        # 颜色统计：mask 内 person_mask_d15 外的像素 (用 broader exclusion)
        m_bool = best['m'].astype(bool)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # 排除 15 px 扩张 (把鞋/护腕排除)
        all_pix = hsv[m_bool]
        keep = (all_pix[:, 1] >= 35) & (all_pix[:, 2] >= 40)
        if keep.sum() >= 3:
            pix = all_pix[keep]
            hr = pix[:, 0].astype(np.float64) * (2 * np.pi / 180)
            h_mean = float(np.degrees(np.arctan2(
                np.mean(np.sin(hr)), np.mean(np.cos(hr))
            )) / 2.0) % 180
            s_mean = float(np.median(pix[:, 1]))
        else:
            # mask 全是低饱 (白)
            v_mean = float(np.mean(all_pix[:, 2]))
            h_mean = 0
            s_mean = 0
        v_mean = float(np.mean(all_pix[:, 2]))
        color = hue_to_color_name(h_mean, s_mean)

        print(f'{ep_name} ({px:.0f},{py:.0f}) vis={vis:.2f}:')
        print(f'  box=({bx1},{by1},{bx2},{by2})')
        print(f'  best mask idx={best["idx"]} area={best["area"]} score={best["score"]:.2f}')
        print(f'  in_body={best["in_body_ratio"]:.1%}  out_d15={best["out_d15"]}')
        print(f'  color={color} HSV=({h_mean:.1f},{s_mean:.1f},{v_mean:.1f})')
        results.append({
            'name': ep_name, 'px': px, 'py': py, 'vis': vis,
            'box': (bx1, by1, bx2, by2),
            'mask': best['m'], 'color': color, 'hsv': (h_mean, s_mean, v_mean),
            'score': best['score'], 'area': best['area'],
            'in_body': best['in_body_ratio'], 'out_d15': best['out_d15'],
        })

    # 可视化
    display = img.copy()

    # person_mask 半透明
    pm_layer = display.copy()
    pm_layer[pm] = (200, 130, 255)
    display = cv2.addWeighted(pm_layer, 0.20, display, 0.80, 0)

    for res in results:
        bx1, by1, bx2, by2 = res['box']
        col_idx = {n: i for i, (n, _, _) in enumerate(endpoints)}
        c = endpoints[col_idx[res['name']]][2]
        # mask 半透明填充
        m_bool = res['mask'].astype(bool)
        overlay = display.copy()
        overlay[m_bool] = c
        display = cv2.addWeighted(overlay, 0.32, display, 0.68, 0)
        # mask 边界
        contours, _ = cv2.findContours(
            (m_bool.astype(np.uint8)) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(display, contours, -1, c, 2)
        # box
        cv2.rectangle(display, (bx1, by1), (bx2, by2), c, 2)
        # 端点
        cv2.circle(display, (int(res['px']), int(res['py'])), 10, c, -1)
        cv2.circle(display, (int(res['px']), int(res['py'])), 14, c, 1)
        # label
        text = '{}->{} area={}'.format(res['name'], res['color'], res['area'])
        cv2.putText(display, text, (bx1 + 5, by1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)

    # legend
    legend_y = h - 110
    ovl = display.copy()
    cv2.rectangle(ovl, (5, legend_y), (560, h - 5), (50, 50, 50), -1)
    display = cv2.addWeighted(ovl, 0.55, display, 0.45, 0)
    legend = [
        ('Box = 端点 + ({}px 半径). NO detect_holds.'.format(BOX_HALF), (200,200,200)),
        ('Mask fill = EdgeSAM box-prompt result (multimask=4, biggest wins)', (200,200,200)),
        ('YELLOW=L_hand, MAGENTA=R_hand, CYAN=L_foot, SALMON=R_foot', (200,200,200)),
        ('Purple fill: YOLO26 person_mask', (200,130,255)),
        ('Color stats EXCLUDE person_mask dilate 15px (shoe excluded)', (200,200,200)),
    ]
    for txt, c in legend:
        legend_y += 16
        cv2.putText(display, txt, (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)

    out = '_dbg_9_box_v2.jpg'
    cv2.imwrite(out, display)
    print(f'\nsaved {out} ({os.path.getsize(out)} bytes)')


if __name__ == '__main__':
    main()
