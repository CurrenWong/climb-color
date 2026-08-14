"""
line1_white frame02 EdgeSAM mask 候选选择过程可视化。

背景：DWPose 全量 3O wall_04 line1_white 判橙 (真值白)。
用户报告: 手在白色岩点上但选到别的岩点去了。
本脚本展示 EdgeSAM 4 个候选 mask + 排序 + 选中逻辑, 让用户看清楚:

  - 端点实际位置 vs box 范围
  - 4 个候选 mask 分别覆盖什么 (半透明不同颜色)
  - 每个候选的 score / area / 距端点距离 / 颜色判定 / 是否过筛选
  - 按 (-score, +dist, -area) 排序后选中的那个

用法:
  python dbg_line1_frame02_candidates.py [frame_idx]
  frame_idx: 0=0100s, 1=0105s(frame02), 2=0110s, 3=0115s, 4=0120s (默认 1)
"""
import os
import sys
import cv2
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_BASE)
# sys.path 顺序: _BASE (dwpose_exp 本地副本) 优先, 然后根目录
# (edgesam_stub 等共享模块), 最后 EdgeSAM 源码。注意: 必须用
# 一次 slice insert 保留原 sys.path, 多次 insert(0, ...) 会反向
# 把 _BASE 推到后面, 导致 dwpose_exp 本地副本永远不被加载。
sys.path[0:0] = [_BASE, _ROOT, os.path.join(_ROOT, "EdgeSAM")]

import edgesam_stub  # noqa: F401

from dwpose_onnx import DWPoseONNX
from edgesam_holds import (
    EdgeSamHoldSegmenter,
    BOX_HALF_MIN, BOX_HALF_HEIGHT_DIV, BOX_HALF_MAX,
    AREA_MIN, AREA_MAX_RATIO, SCORE_MIN, DIST_MAX, COLOR_MIN_PIXELS,
    _sample_person_edge_points,
    _sample_color, _keep_main_component,
    compute_box_half,
)
from yolo_person_mask import PersonMasker
import route_color

WALL_DIR = r"D:\Project\climbing_wall_search\mediapipe2\img_xhs_walls\3OHz6rZ5opG_wall_04_line1_white"
DW_ENDPOINTS = [('L_hand', 9), ('R_hand', 10)]
DW_SHOULDERS = (5, 6)
DW_HIPS = (11, 12)


def make_neg_prompts(segmenter, px, py, body_cx, body_cy, person_mask):
    neg = segmenter._make_negative_prompts(px, py, body_cx, body_cy)
    if person_mask is not None and person_mask.any():
        edge_pts = _sample_person_edge_points(person_mask, n=8, max_radius=60)
        if len(edge_pts) > 0:
            neg = np.vstack([neg, edge_pts])
    return neg


def analyze_endpoint(segmenter, px, py, body_cx, body_cy, person_mask):
    h, w = segmenter.last_h, segmenter.last_w
    ix = max(0, min(int(round(px)), w - 1))
    iy = max(0, min(int(round(py)), h - 1))
    eff_box_half = compute_box_half(h, person_mask)
    box = np.array([max(0, ix - eff_box_half), max(0, iy - eff_box_half),
                    min(w - 1, ix + eff_box_half), min(h - 1, iy + eff_box_half)],
                   dtype=np.float64)

    neg = make_neg_prompts(segmenter, px, py, body_cx, body_cy, person_mask)
    point_coords = np.vstack([[[px, py]], neg])
    point_labels = np.array([1] + [0] * len(neg), dtype=np.int64)

    masks, scores, _ = segmenter.predictor.predict(
        box=box,
        point_coords=point_coords,
        point_labels=point_labels,
        num_multimask_outputs=4,
        use_stability_score=True,
    )
    if masks.ndim == 2:
        masks = masks[None, :, :]
        scores = np.array([float(scores)])

    img_area = w * h
    area_cap = int(img_area * AREA_MAX_RATIO)
    info_list = []
    for i, m in enumerate(masks):
        area = int(m.sum())
        score = float(scores[i])
        ys_m, xs_m = np.where(m)
        dist_to_ep = float(np.min(np.hypot(xs_m - ix, ys_m - iy))) if len(xs_m) else 1e9

        # 过滤
        pass_screen = (area >= AREA_MIN and area <= area_cap
                       and score >= SCORE_MIN and dist_to_ep <= DIST_MAX)

        # 减 person + 主连通域
        m_after = m.astype(bool)
        if person_mask is not None:
            m_after = m_after & ~person_mask.astype(bool)
        if m_after.sum() >= COLOR_MIN_PIXELS:
            m_after = _keep_main_component(m_after, ix, iy)
        else:
            m_after = np.zeros_like(m_after)

        # 颜色
        if m_after.sum() >= COLOR_MIN_PIXELS:
            h_, s_, v_ = _sample_color(segmenter.last_image_bgr, m_after)
            if v_ < 60 and s_ < 40:
                cname = "未知(暗)"
            elif s_ < 35:
                cname = "白" if v_ >= 160 else "黑"
            else:
                cname = route_color.hue_to_color_name(h_, s_)
        else:
            h_, s_, v_, cname = 0, 0, 0, "非岩点"

        info_list.append({
            'idx': i, 'mask_raw': m, 'mask_after': m_after,
            'score': score, 'area_raw': area, 'area_after': int(m_after.sum()),
            'dist_to_ep': dist_to_ep,
            'pass_screen': pass_screen,
            'h': h_, 's': s_, 'v': v_, 'cname': cname,
            'box': box, 'eff_box_half': eff_box_half,
        })
    return info_list


def visualize_endpoint(image_bgr, ep_name, px, py, info_list, person_mask):
    """画一张图: 原图 + 4 候选 mask (不同颜色) + 端点 + box + 选中标注"""
    img = image_bgr.copy()
    h, w = img.shape[:2]

    # person mask 淡紫 (跟 draw_results 一致)
    if person_mask is not None and person_mask.any():
        pm_u8 = person_mask.astype(np.uint8) * 255
        overlay = img.copy()
        overlay[pm_u8 > 0] = (255, 180, 180)
        cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)

    # 4 候选颜色 (BGR) - 高对比互不相同
    CAND_COLORS = [
        (0, 255, 0),     # 绿 - 候选 0
        (255, 128, 0),   # 蓝 - 候选 1 (OpenCV BGR 中蓝色是 (255,0,0))
        (0, 200, 255),   # 黄
        (255, 0, 255),   # 洋红
    ]
    # 修正: 上面写错, BGR 颜色重新分配
    CAND_COLORS = [
        (0, 255, 0),     # 候选 0 = 绿
        (255, 0, 0),     # 候选 1 = 蓝 (BGR)
        (0, 255, 255),   # 候选 2 = 黄 (BGR)
        (255, 0, 255),   # 候选 3 = 洋红 (BGR)
    ]

    # box 范围 (浅灰半透明填充)
    box = info_list[0]['box']
    bx1, by1, bx2, by2 = [int(v) for v in box]
    cv2.rectangle(img, (bx1, by1), (bx2, by2), (200, 200, 200), 2)
    cv2.putText(img, f"box {bx2-bx1}x{by2-by1}", (bx1, max(20, by1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # 排序: 跟 segment_endpoint 一致 (-score, +dist, -area)
    # 但只在过了筛选的里排
    valid = [c for c in info_list if c['pass_screen']]
    valid.sort(key=lambda c: (-c['score'], c['dist_to_ep'], -c['area_raw']))
    chosen_idx = valid[0]['idx'] if valid else None

    # 画每个候选 (raw mask, 不减 person, 完整展示 EdgeSAM 输出)
    for i, c in enumerate(info_list):
        color = CAND_COLORS[i]
        m_bool = c['mask_raw']
        if not m_bool.any():
            continue
        # 半透明覆盖
        alpha = 0.35
        overlay = img.copy()
        overlay[m_bool] = color
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        # 描边
        m_u8 = m_bool.astype(np.uint8) * 255
        contours, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        thick = 4 if i == chosen_idx else 2
        cv2.drawContours(img, contours, -1, (0, 0, 0), thick + 2)  # 黑外环
        cv2.drawContours(img, contours, -1, color, thick)

    # 负点 (小灰点)
    seg = info_list[0]
    # 重新构造 neg (从 raw box 数据不可知, 重新调一次)
    # 跳过显示负点避免重复计算

    # 端点: 红点 + 白圈
    px_i, py_i = int(px), int(py)
    cv2.circle(img, (px_i, py_i), 7, (255, 255, 255), -1)
    cv2.circle(img, (px_i, py_i), 5, (0, 0, 255), -1)

    # Header: 端点 + box 信息
    header_h = 36
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_h), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    cv2.putText(img, f"{ep_name} endpoint=({px:.0f},{py:.0f})  "
                     f"box={bx2-bx1}x{by2-by1} (half={info_list[0]['eff_box_half']})",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    # 4 个候选的小标注 (左下角堆叠)
    label_x, label_y = 8, h - 22 * 4 - 12
    for i, c in enumerate(info_list):
        marker = "★ 选中" if i == chosen_idx else f"  候选{i}"
        check = "✓" if c['pass_screen'] else "✗"
        text = (f"{marker} [{check}] score={c['score']:.3f}  "
                f"area={c['area_raw']:5d}  dist={c['dist_to_ep']:5.1f}  "
                f"HSV=({c['h']:5.1f},{c['s']:5.1f},{c['v']:5.1f}) -> {c['cname']}")
        # 背景
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (label_x - 2, label_y + i * 22 - th - 2),
                      (label_x + tw + 4, label_y + i * 22 + 4),
                      (0, 0, 0), -1)
        color = CAND_COLORS[i] if i == chosen_idx else (180, 180, 180)
        cv2.putText(img, text, (label_x, label_y + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return img, chosen_idx


def main():
    frame_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 1  # 默认 frame02 = 0105s
    frames = sorted(os.listdir(WALL_DIR))
    frame_path = os.path.join(WALL_DIR, frames[frame_idx])
    print(f"frame {frame_idx:02d}: {frames[frame_idx]}  ({frame_path})")

    image_bgr = cv2.imread(frame_path)
    h, w = image_bgr.shape[:2]

    # DWPose
    dw = DWPoseONNX('../yolox_l.onnx', '../dw-ll_ucoco_384.onnx')
    boxes, kps, scores = dw.run(image_bgr)
    if kps is None:
        print("❌ DWPose 未检测到人")
        return
    k = kps[0]
    pts = k[list(DW_SHOULDERS) + list(DW_HIPS)]
    body_cx = float(pts[:, 0].mean())
    body_cy = float(pts[:, 1].mean())

    # Person mask
    pm = PersonMasker(model_name="../yolo26m-seg.pt", device='cpu')
    person_mask = pm.segment(image_bgr)

    # EdgeSAM
    seg = EdgeSamHoldSegmenter('../weights/edge_sam.pth', device='cpu')
    seg.set_image(image_bgr)

    vis_imgs = []
    for ep_name, kidx in DW_ENDPOINTS:
        px, py = float(k[kidx, 0]), float(k[kidx, 1])
        conf = float(scores[0, kidx])
        print(f"\n=== {ep_name} ({px:.0f},{py:.0f}) conf={conf:.2f} ===")

        info_list = analyze_endpoint(seg, px, py, body_cx, body_cy, person_mask)
        for c in info_list:
            print(f"  候选{c['idx']}: pass={c['pass_screen']} score={c['score']:.3f} "
                  f"area_raw={c['area_raw']:5d} area_after={c['area_after']:5d} "
                  f"dist={c['dist_to_ep']:5.1f} HSV=({c['h']:5.1f},{c['s']:5.1f},{c['v']:5.1f}) "
                  f"-> {c['cname']}")

        vis, chosen_idx = visualize_endpoint(image_bgr, ep_name, px, py, info_list, person_mask)
        vis_imgs.append((ep_name, vis))

    # 拼接左右手 (横向)
    if vis_imgs:
        # 等高对齐
        th_h = max(v.shape[0] for _, v in vis_imgs)
        th_w = max(v.shape[1] for _, v in vis_imgs)
        combined = np.full((th_h + 50, th_w * len(vis_imgs), 3), (30, 30, 30), dtype=np.uint8)
        for i, (name, v) in enumerate(vis_imgs):
            combined[:v.shape[0], i * th_w:(i * th_w + v.shape[1])] = v
            cv2.putText(combined, name, (i * th_w + 10, th_h + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        out = f"dbg_line1_frame{frame_idx + 1:02d}_candidates.jpg"
        cv2.imwrite(out, combined, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"\n💾 {out}  ({combined.shape[1]}x{combined.shape[0]})")


if __name__ == '__main__':
    main()