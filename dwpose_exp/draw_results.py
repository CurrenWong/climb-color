"""
DWPose 实验专用可视化：从 test_edge_sam.py 抽取的 draw_results。

分割背景（2026-08-13）：
  DWPose 实验代码与 MediaPipe 生产代码分离，dwpose_exp/ 完全自包含。
  此文件是 test_edge_sam.draw_results 的副本，后续 DWPose 实验可自由修改，
  不影响 MediaPipe 生产管线（test_edge_sam.py 保持原样）。

依赖：edgesam_holds（RING_BLACK 常量）、route_color（内部用 _hsv_to_bgr_）
"""
import cv2
import numpy as np

from edgesam_holds import RING_BLACK_D1, RING_BLACK_D2, RING_BLACK_RATIO_MIN


def _vivid_color(h: float, s: float, v: float) -> np.ndarray:
    """把 hold 的 HSV 色彩拉到最大饱和/亮（避免 mask 平均后发灰变暗）。"""
    s_new = float(np.clip(max(s, 100), 100, 255))
    v_new = float(np.clip(max(v, 150), 150, 255))
    color = np.array([[[int(h), int(s_new), int(v_new)]]], dtype=np.uint8)
    bgr = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)[0, 0]
    return bgr.astype(np.uint8)


def _hsv_to_bgr_(h, s, v):
    color = np.array([[[int(h), int(s), int(v)]]], dtype=np.uint8)
    return tuple(int(c) for c in cv2.cvtColor(color, cv2.COLOR_HSV2BGR)[0, 0])


def draw_results(image_bgr: np.ndarray, samples: list[dict], route_color: str,
                 confidence: float, votes: dict, n_segmented: int,
                 person_mask: np.ndarray | None = None):
    h_img, w_img = image_bgr.shape[:2]

    # ---------- Header（中等暗度，不再压成 25% 透明） ----------
    # 有 ring 修正端点时多一行图例说明
    n_fixed = sum(1 for s in samples
                  if (s.get('ring_black') or {}).get('fixed'))
    header_h = 80 + (24 if n_fixed else 0)
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
    if n_fixed:
        cv2.putText(image_bgr,
                    f"[环带黑修正 x{n_fixed}] 黄环=统计区({RING_BLACK_D1}-{RING_BLACK_D2}px) 黑票>={RING_BLACK_RATIO_MIN:.0%}->判黑",
                    (10, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

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

        # ---------- 环形带黑修正可视化（mask 之后、端点之前，避免被标签盖住） ----------
        rb = s.get('ring_black') or {}
        if rb.get('ratio', 0) > 0:
            # 只画内外边界两个圆（不填充），避免遮挡 mask/端点
            def _ring_circle(rr):
                theta = np.linspace(0, 2 * np.pi, 72)
                xs = px + rr * np.cos(theta)
                ys = py + rr * np.sin(theta)
                return np.column_stack([xs, ys]).astype(np.int32)

            if rb.get('fixed'):
                # 修正命中：黄色外圈 + 黑边（醒目但不遮挡）
                for rr in (RING_BLACK_D1, RING_BLACK_D2):
                    cv2.polylines(image_bgr, [_ring_circle(rr)], True, (0, 0, 0), 3)
                    cv2.polylines(image_bgr, [_ring_circle(rr)], True, (0, 255, 255), 1)
                tag_color = (0, 255, 255)
                tag_prefix = f"[ring黑修正 {rb['ratio']*100:.1f}%]"
            else:
                # 检查过但没命中：细白圆（低调）
                for rr in (RING_BLACK_D1, RING_BLACK_D2):
                    cv2.polylines(image_bgr, [_ring_circle(rr)], True, (255, 255, 255), 1)
                tag_color = (220, 220, 220)
                tag_prefix = f"[ring黑 {rb['ratio']*100:.1f}%<{RING_BLACK_RATIO_MIN:.0%}]"

            # 环带标签：画在端点上方偏移，避免和主标签重叠
            (tag_w, tag_h), _ = cv2.getTextSize(tag_prefix, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            tag_x = max(2, min(px - tag_w // 2, w_img - tag_w - 4))
            tag_y = max(16, py - RING_BLACK_D2 - 8)
            cv2.rectangle(image_bgr, (tag_x - 3, tag_y - tag_h - 3),
                          (tag_x + tag_w + 3, tag_y + 3), (0, 0, 0), -1)
            cv2.putText(image_bgr, tag_prefix, (tag_x, tag_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, tag_color, 1, cv2.LINE_AA)

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
