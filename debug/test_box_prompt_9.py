"""
9.jpg box prompt 测试：用 detect_holds (HSV) 找候选 hold bbox，
给每个端点最近的 bbox 当 box prompt 让 EdgeSAM refine。

不动生产代码 — 只是 isolated 探索性脚本，输出 _dbg_9_boxprompt.jpg
让用户眼看一下 box prompt 切出来的 mask。
"""
import cv2
import numpy as np

import edgesam_stub  # noqa
import edgesam_holds
from yolo_person_mask import PersonMasker, dilate_mask
import mediapipe as mp
from route_color import detect_holds, merge_flip_landmarks


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


def find_nearest_hold(holds, px, py):
    """找离 (px, py) 最近的 hold，按中心点距离。"""
    if not holds:
        return None
    best, best_d = None, float('inf')
    for ho in holds:
        cx, cy = ho['center']
        d = np.hypot(cx - px, cy - py)
        if d < best_d:
            best_d, best = d, ho
    return best, best_d


def expand_bbox(bbox, expand=1.5):
    """扩大 bbox：让 box 比 hold 大一圈，给 EdgeSAM 留余量。"""
    bx, by, bw, bh = bbox
    cx, cy = bx + bw / 2, by + bh / 2
    nbw, nbh = bw * expand, bh * expand
    nbx, nby = int(cx - nbw / 2), int(cy - nbh / 2)
    nbw_i, nbh_i = int(nbw), int(nbh)
    return nbx, nby, nbx + nbw_i, nby + nbh_i  # x1,y1,x2,y2


def expand_box_with_endpoint(box_xyxy, px, py, w, h, min_size=80):
    """
    让 box 包含 (px, py)：把 box 和 (px,py) 取外接矩形，再扩张 10%，
    并保证 box 长宽 >= min_size。
    """
    bx1, by1, bx2, by2 = box_xyxy
    # 取并集
    ubx1 = min(bx1, int(px))
    uby1 = min(by1, int(py))
    ubx2 = max(bx2, int(px))
    uby2 = max(by2, int(py))
    # 扩张 10%
    cx = (ubx1 + ubx2) / 2
    cy = (uby1 + uby2) / 2
    bw_b = ubx2 - ubx1
    bh_b = uby2 - uby1
    bw_b = max(bw_b, min_size)
    bh_b = max(bh_b, min_size)
    ex1 = int(cx - bw_b * 0.6)
    ey1 = int(cy - bh_b * 0.6)
    ex2 = int(cx + bw_b * 0.6)
    ey2 = int(cy + bh_b * 0.6)
    ex1 = max(0, ex1); ey1 = max(0, ey1)
    ex2 = min(w - 1, ex2); ey2 = min(h - 1, ey2)
    return [ex1, ey1, ex2, ey2]


def main():
    img_path = 'img/9.jpg'
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    print(f'image: {w}x{h}')

    # 1. detect_holds (HSV) 找候选 hold
    holds, labels = detect_holds(
        img,
        min_area=80, max_area=20000,
        min_sat=20, min_val=40,        # 降低 min_sat 让白色 (S<60) hold 也能检出
    )
    print(f'\ndetect_holds found {len(holds)} candidates:')
    for ho in holds[:20]:
        bx, by, bw, bh = ho['bbox']
        cx, cy = ho['center']
        print(f"  id={ho['id']:3d} center=({cx:.0f},{cy:.0f}) bbox=({bx},{by},{bw},{bh}) "
              f"area={ho['area']:5d} name={ho['name']:4s} HSV=({ho['h']:.0f},{ho['s']:.0f},{ho['v']:.0f})")

    # 2. 加载 EdgeSAM + YOLO26
    seg = edgesam_holds.EdgeSamHoldSegmenter('weights/edge_sam.pth', 'cpu')
    masker = PersonMasker('weights/yolo26n-seg.pt', 'cpu')
    pm = masker.segment(img)

    # 3. 加载 MediaPipe pose + flip ensemble
    B = mp.tasks.BaseOptions
    P = mp.tasks.vision.PoseLandmarker
    O = mp.tasks.vision.PoseLandmarkerOptions
    R = mp.tasks.vision.RunningMode
    opts = O(base_options=B(model_asset_path='weights/pose_landmarker_full.task'),
             running_mode=R.IMAGE, num_poses=1,
             min_pose_detection_confidence=0.3,
             min_pose_presence_confidence=0.3,
             min_tracking_confidence=0.3)
    lk = P.create_from_options(opts)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    landmarks = detect_pose(rgb, lk, w, h)
    if landmarks is None:
        print('no pose!')
        return

    # 4. 准备 EdgeSAM encoder
    seg.set_image(img)

    # 5. 每个端点：找最近 hold bbox → box prompt → EdgeSAM
    endpoints = [('L_hand', 15), ('R_hand', 16), ('L_foot', 31), ('R_foot', 32)]
    endpoint_results = []
    print()
    for ep_name, idx in endpoints:
        lm = landmarks[idx]
        px, py = lm.x * w, lm.y * h
        vis = lm.visibility

        # 找最近 hold
        nearest, dist = find_nearest_hold(holds, px, py)
        if nearest is None:
            print(f'{ep_name}: no hold candidate found')
            continue

        # box prompt = bbox 扩张 1.5x，然后强制并入端点位置，再扩张 20% 让 EdgeSAM 有上下文
        bx1, by1, bx2, by2 = expand_bbox(nearest['bbox'], expand=1.5)
        box = expand_box_with_endpoint(
            [bx1, by1, bx2, by2], px, py, w, h, min_size=100,
        )
        box = np.array(box, dtype=np.float64)
        bx1, by1, bx2, by2 = [int(v) for v in box]

        # 用 box prompt 跑 EdgeSAM (multimask=4，看所有候选)
        masks, scores, _ = seg.predictor.predict(
            box=box,
            num_multimask_outputs=4,
            use_stability_score=True,
        )
        if masks.ndim == 2:
            masks = masks[None, :, :]

        # 选最大包含端点的 mask (完整 hold)
        best = None
        best_score = -1
        for i, m in enumerate(masks):
            if not m[int(round(py)), int(round(px))]:
                continue
            area = int(m.sum())
            if area < 100:
                continue
            score = float(scores[i])
            if score > best_score:
                best = (i, m, score, area)
        if best is None:
            print(f'{ep_name} ({px},{py}): nearest hold id={nearest["id"]} dist={dist:.0f}, '
                  f'box=({bx1},{by1},{bx2},{by2}), NO containing mask')
            continue

        i, m, score, area = best
        # mask 内 HSV 统计
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        in_body = int((m & pm).sum())
        all_pix = hsv[m]
        if (all_pix[:, 1] >= 35).sum() >= 3:
            keep = all_pix[:, 1] >= 35
            pix = all_pix[keep]
            hr = pix[:, 0].astype(np.float64) * (2 * np.pi / 180)
            h_mean = float(np.degrees(np.arctan2(
                np.mean(np.sin(hr)), np.mean(np.cos(hr))
            )) / 2.0) % 180
        else:
            h_mean = float(np.mean(all_pix[:, 0]))

        # 颜色名
        s_mean = float(np.median(all_pix[all_pix[:, 1] >= 35][:, 1])) if (all_pix[:, 1] >= 35).sum() >= 3 else 0
        v_mean = float(np.mean(all_pix[:, 2]))
        if v_mean < 60 and s_mean < 40:
            color = '未知(暗)'
        elif s_mean < 35 and v_mean > 140:
            color = '白'
        else:
            color = {
                (0, 10): '红', (170, 180): '红', (10, 20): '橙',
                (20, 30): '黄', (30, 80): '绿', (80, 95): '青',
                (95, 118): '蓝', (118, 150): '紫', (150, 170): '粉红',
            }.get(_h_bin(h_mean), '未知')

        endpoint_results.append({
            'name': ep_name, 'px': px, 'py': py, 'vis': vis,
            'nearest': nearest, 'dist': dist,
            'box': (bx1, by1, bx2, by2),
            'mask_idx': i, 'mask': m, 'score': score, 'area': area,
            'in_body': in_body, 'in_body_ratio': in_body/max(1, area),
            'color': color, 'hsv': (h_mean, s_mean, v_mean),
        })

        print(f'\n{ep_name} ({px},{py}) vis={vis:.2f}: nearest hold id={nearest["id"]} '
              f'name={nearest["name"]} dist={dist:.0f}')
        print(f'  box=({bx1},{by1},{bx2},{by2})  nearest hold HSV=({nearest["h"]:.0f},{nearest["s"]:.0f},{nearest["v"]:.0f})')
        print(f'  best mask: idx={i} area={area} score={score:.2f} in_body={in_body/max(1,area):.1%}')
        print(f'  mask->color={color} HSV=({h_mean:.1f},{s_mean:.1f},{v_mean:.1f})')

    # 6. 可视化
    display = img.copy()

    # 6a. 把每个 hold bbox 标出来（白/淡色），方便对比 box 位置
    for ho in holds:
        bx, by, bw, bh = ho['bbox']
        # 按 name 染色
        cv2.rectangle(display, (bx, by), (bx + bw, by + bh), (200, 200, 200), 1)

    # 6b. 端点 box（扩张过的）— 画每个端点的 box prompt 框
    COLORS = [(0, 255, 255), (255, 0, 255), (0, 200, 255), (255, 100, 100)]
    LABELS = ['L_hand', 'R_hand', 'L_foot', 'R_foot']
    for i, (ep_name, idx) in enumerate(endpoints):
        lm = landmarks[idx]
        px, py = int(lm.x * w), int(lm.y * h)
        cv2.circle(display, (px, py), 10, COLORS[i], -1)
        cv2.circle(display, (px, py), 14, COLORS[i], 1)

    # 6c. 在每个端点的 box prompt 上画结果 mask (filled 半透明)
    for res in endpoint_results:
        m = res['mask']
        m_bool = m.astype(bool)
        # 用端点颜色作为 mask fill (半透明)
        col_idx = ['L_hand', 'R_hand', 'L_foot', 'R_foot'].index(res['name'])
        c = COLORS[col_idx]
        overlay = display.copy()
        overlay[m_bool] = c
        display = cv2.addWeighted(overlay, 0.30, display, 0.70, 0)
        # 边界
        contours, _ = cv2.findContours(
            (m_bool.astype(np.uint8)) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(display, contours, -1, c, 2)
        # box prompt 框
        bx1, by1, bx2, by2 = res['box']
        cv2.rectangle(display, (bx1, by1), (bx2, by2), c, 2)
        # label
        cx_t = (bx1 + bx2) // 2 - 50
        cy_t = (by1 + by2) // 2
        text = '{} box→{}'.format(res['name'], res['nearest']['name'])
        cv2.putText(display, text, (cx_t, cy_t), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)

    # 6d. YOLO26 person_mask 半透明叠一层
    pm_layer = display.copy()
    pm_layer[pm] = (200, 130, 255)
    display = cv2.addWeighted(pm_layer, 0.20, display, 0.80, 0)
    pm_contour, _ = cv2.findContours(
        pm.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(display, pm_contour, -1, (255, 0, 200), 1)

    # 6e. legend
    legend_y = h - 130
    ovl = display.copy()
    cv2.rectangle(ovl, (5, legend_y), (560, h - 5), (50, 50, 50), -1)
    display = cv2.addWeighted(ovl, 0.55, display, 0.45, 0)
    legend = [
        ('Thin gray rects: detect_holds() HSV candidates', (200, 200, 200)),
        ('Colored dots: 4 endpoints (Y=L_hand M=R_hand C=L_foot SALMON=R_foot)', (255, 255, 255)),
        ('Colored thick rects: box prompt for each endpoint (= nearest hold bbox expanded 1.5x)', (200, 200, 200)),
        ('Filled tinted area = EdgeSAM mask (box prompt, multimask=4, biggest wins)', (200, 200, 200)),
        ('Purple: YOLO26 person_mask (faint)', (200, 130, 255)),
    ]
    for txt, c in legend:
        legend_y += 16
        cv2.putText(display, txt, (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)

    out = '_dbg_9_boxprompt.jpg'
    cv2.imwrite(out, display)
    print(f'\nsaved {out}')
    print(f'  file size: {os.path.getsize(out)} bytes')


def _h_bin(h):
    for (lo, hi), name in [((0, 10), '红'), ((170, 180), '红'), ((10, 20), '橙'),
                            ((20, 30), '黄'), ((30, 80), '绿'), ((80, 95), '青'),
                            ((95, 118), '蓝'), ((118, 150), '紫'), ((150, 170), '粉红')]:
        if lo <= h < hi:
            return name
    return '未知'


if __name__ == '__main__':
    main()
