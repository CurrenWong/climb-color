"""
EdgeSAM 端点级 hold 分割模块（简化通用版）。

核心思路：
  对每个 MediaPipe 姿态端点（L/R 手腕、L/R 脚尖）作为正点 prompt，
  让 EdgeSAM 输出"该端点所在岩点"的二值 mask。
  再从 mask 像素里取 HSV 直方图主色，按色投票判定线路色。

为啥不用现有 HSV 过滤 detect_holds 那套：
  - 同色伪装墙（墙=线路色）颜色空间硬限制
  - 白色岩点 S<60 全被过滤掉
  EdgeSAM 是语义感知，绕过以上两个限制；输出是精确 mask 不是 bbox。

简化要点（去掉"针对特定图调优"的 hack）：
  - 无保护圈 / 伪块检测：mask 减 person 后不保留端点周围圆盘
  - 单 person mask：负点采样 / subtract 用同一个 mask（统一一个膨胀版）
  - 无白路线模式：统一参数，不分 white_route_mode
  - 无相对白路径 B：只用绝对白判定（S<35 AND V>140 过半）
  - 无黑色 hold 检测：dead code，砍掉
  - 无高饱和亮色域优先：主连通域只取"含端点优先 / 否则最大"
  - 候选筛选单层：score / 面积 / 距端点近 = 一个一致标准
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

import edgesam_stub  # noqa: F401 必须在 from edge_sam 之前注入 stub
import route_color  # 色相 → 色名
from edge_sam import sam_model_registry, SamPredictor  # noqa: E402


DEFAULT_CHECKPOINT = "weights/edge_sam.pth"

# ============================================================
# 候选 mask 筛选阈值
# ============================================================
SCORE_MIN = 0.4          # EdgeSAM 置信度门槛
AREA_MIN = 100           # mask 最小像素（噪声剔除）
AREA_MAX_RATIO = 0.30    # mask 最大占比（图面积比例，太大是墙不是 hold）
DIST_MAX = 60            # mask 最近像素到端点的最大距离（hold 在端点附近）

# mask 减 person 后剩多少像素才算有效 hold（太少 = mask 主体被人吞了）
COLOR_MIN_PIXELS = 50

# ============================================================
# box prompt 自适应（高分辨率图 box 太小切不到 hold）
# ============================================================
# box_half = max(BOX_HALF_DEFAULT, h // BOX_HALF_HEIGHT_DIV)，上限 BOX_HALF_MAX
# 这是通用适配（不是单图调优）：保证 box 至少能容纳 hold 的尺度
BOX_HALF_DEFAULT = 80
BOX_HALF_MIN = 80
BOX_HALF_MAX = 250
BOX_HALF_HEIGHT_DIV = 20

# ============================================================
# 颜色判定阈值
# ============================================================
# 白色 hold 阈值
# 路径 A（绝对白）：mask 内过半像素 S<WHITE_SAT_MAX AND V>WHITE_VAL_MIN → 白
# 路径 B（相对白）：mask S 中位数显著低于环境环 → 白
#   （暖光/阴影下白 hold S 常 35-75，纯阈值判不出来；
#    但白 hold 仍是"墙上最不饱和的物体"——环境环是通用区分依据）
WHITE_SAT_MAX = 35
WHITE_VAL_MIN = 140
WHITE_RATIO = 0.50
WHITE_REL_SAT_MAX = 75          # 相对白：mask S 中位数上限
WHITE_REL_RATIO = 0.60          # 相对白：mask/env S 比 < 此值 → 白

# 网格采样：剔除近黑像素
COLOR_VAL_MIN = 40
GRID_N = 5

# 颜色名兜底：暗（V<60 AND S<40）→ 未知(暗)；否则 hue_to_color_name
DARK_V_MAX = 60
DARK_S_MAX = 40

# person mask 边缘外采样负点用
PERSON_EDGE_NEG_N = 8
PERSON_EDGE_MAX_RADIUS = 60

# 负点（手工部分）参数
NEG_BODY_OFFSET = 30      # 身体方向负点偏移 (px)
NEG_RING_RADIUS = 35      # 均匀围绕负点半径
NEG_RING_N = 5            # 均匀围绕负点数量


class EdgeSamHoldSegmenter:
    """EdgeSAM 包装：一次 set_image，多次 predict 端点 mask。"""

    def __init__(
        self,
        checkpoint_path: str = DEFAULT_CHECKPOINT,
        device: str | torch.device = "cpu",
    ):
        self.device = torch.device(device)
        build_fn = sam_model_registry["edge_sam"]
        self.model = build_fn(checkpoint=checkpoint_path)
        self.model = self.model.to(self.device).eval()
        self.predictor = SamPredictor(self.model)

    def set_image(self, image_bgr: np.ndarray) -> None:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb, image_format="RGB")
        self.last_h, self.last_w = image_bgr.shape[:2]
        self.last_image_bgr = image_bgr

    # ------------------------------------------------------------
    # 端点级 hold 分割
    # ------------------------------------------------------------
    def segment_endpoint(
        self,
        px: float, py: float,
        body_cx: float | None = None, body_cy: float | None = None,
        person_mask: np.ndarray | None = None,
        num_multimask_outputs: int = 4,
    ) -> dict:
        """
        对 (px, py) 这个端点做 hold 分割。

        person_mask: HxW bool，YOLO26 攀岩者人体 mask。若提供：
          - 边缘外的点作为 negative prompt（让 EdgeSAM 把端点附近的身体
            和 hold 区分开）
          - 选定的 mask 与 person_mask 取差集（剥除人体像素，避免把手/
            脚/衣服等嵌在身体里的像素算到 hold 颜色上）

        返回 dict: mask(255 uint8) / mask_bool / score / h,s,v /
        color_name / area / endpoint
        """
        if not self.predictor.is_image_set:
            raise RuntimeError("先调用 set_image() 再 segment_endpoint()")

        # box prompt：端点周围自适应 box
        eff_box_half = max(BOX_HALF_MIN, self.last_h // BOX_HALF_HEIGHT_DIV)
        eff_box_half = min(eff_box_half, BOX_HALF_MAX)
        ix = max(0, min(int(round(px)), self.last_w - 1))
        iy = max(0, min(int(round(py)), self.last_h - 1))
        bx1 = max(0, ix - eff_box_half)
        by1 = max(0, iy - eff_box_half)
        bx2 = min(self.last_w - 1, ix + eff_box_half)
        by2 = min(self.last_h - 1, iy + eff_box_half)
        box = np.array([bx1, by1, bx2, by2], dtype=np.float64)

        # 负点：1 个身体方向 + NEG_RING_N 个均匀围绕 + person mask 边缘外
        neg = self._make_negative_prompts(px, py, body_cx, body_cy)
        if person_mask is not None and person_mask.any():
            edge_pts = _sample_person_edge_points(
                person_mask, n=PERSON_EDGE_NEG_N, max_radius=PERSON_EDGE_MAX_RADIUS
            )
            if len(edge_pts) > 0:
                neg = np.vstack([neg, edge_pts])

        point_coords = np.vstack([[[px, py]], neg])
        point_labels = np.array([1] + [0] * len(neg), dtype=np.int64)

        # EdgeSAM predict
        masks, scores, _ = self.predictor.predict(
            box=box,
            point_coords=point_coords,
            point_labels=point_labels,
            num_multimask_outputs=num_multimask_outputs,
            use_stability_score=True,
        )
        if masks.ndim == 2:
            masks = masks[None, :, :]
            scores = np.array([float(scores)])

        img_area = self.last_w * self.last_h
        area_cap = int(img_area * AREA_MAX_RATIO)

        # 候选筛选（单层）
        candidates = []
        for i, m in enumerate(masks):
            area = int(m.sum())
            if area < AREA_MIN or area > area_cap:
                continue
            if float(scores[i]) < SCORE_MIN:
                continue
            ys_m, xs_m = np.where(m)
            if len(xs_m) == 0:
                continue
            dist_to_ep = float(np.min(np.hypot(xs_m - ix, ys_m - iy)))
            if dist_to_ep > DIST_MAX:
                continue
            candidates.append((float(scores[i]), dist_to_ep, area, m))

        if not candidates:
            return self._empty_result(px, py, scores)

        # 排序：score 最高 → 距端点最近 → 面积大
        candidates.sort(key=lambda c: (-c[0], c[1], -c[2]))
        chosen_score, _, _, chosen = candidates[0]

        # 减 person（统一用同一个 mask）
        if person_mask is not None:
            chosen = chosen & ~person_mask.astype(bool)

        # 减 person 后像素太少 = mask 主体在人身上
        if chosen.sum() < COLOR_MIN_PIXELS:
            return self._empty_result(px, py, scores, score=chosen_score)

        # 主连通域
        chosen = _keep_main_component(chosen, ix, iy)

        # 颜色统计
        h, s, v = _sample_color(self.last_image_bgr, chosen)

        # 颜色名
        if v < DARK_V_MAX and s < DARK_S_MAX:
            color_name = "未知(暗)"
        elif s < WHITE_SAT_MAX:
            # 低饱和：V 高 → 白；V 中低 → 黑（黑色 hold 被灯光打亮时
            # S 仍≈0 但 V 只升到 ~60-140，真白 hold V 一般 ≥160）
            color_name = "白" if v >= WHITE_VAL_MIN else "黑"
        else:
            color_name = route_color.hue_to_color_name(h, s)

        return {
            "mask": chosen.astype(np.uint8) * 255,
            "mask_bool": chosen.astype(bool),
            "score": float(chosen_score),
            "h": float(h), "s": float(s), "v": float(v),
            "color_name": color_name,
            "area": int(chosen.sum()),
            "endpoint": (px, py),
        }

    # ------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------
    @staticmethod
    def _make_negative_prompts(
        px: float, py: float,
        body_cx: float | None, body_cy: float | None,
    ) -> np.ndarray:
        """
        负点：1 个身体方向 + 5 个均匀围绕端点。

        身体方向：端点往身体中心偏 NEG_BODY_OFFSET px（让 SAM 把
        "端点附近的身体"剔除）
        均匀围绕：以端点为圆心 NEG_RING_RADIUS px 的 NEG_RING_N 个点
        """
        pts: list[tuple[float, float]] = []
        if body_cx is not None and body_cy is not None:
            ux, uy = body_cx - px, body_cy - py
            norm = np.hypot(ux, uy) or 1.0
            ux, uy = ux / norm, uy / norm
            pts.append((px + NEG_BODY_OFFSET * ux, py + NEG_BODY_OFFSET * uy))
        for theta in np.linspace(0, 2 * np.pi, NEG_RING_N, endpoint=False):
            pts.append((px + NEG_RING_RADIUS * np.cos(theta),
                        py + NEG_RING_RADIUS * np.sin(theta)))
        return np.array(pts, dtype=np.float64)

    def _empty_result(self, px, py, scores, score=None):
        return {
            "mask": np.zeros((self.last_h, self.last_w), dtype=np.uint8),
            "mask_bool": np.zeros((self.last_h, self.last_w), dtype=bool),
            "score": float(score) if score is not None else (
                float(np.max(scores)) if len(scores) else 0.0
            ),
            "h": 0.0, "s": 0.0, "v": 0.0,
            "color_name": "非岩点",
            "area": 0,
            "endpoint": (px, py),
        }


# ============================================================
# 模块级工具函数（纯函数，方便测试和复用）
# ============================================================
def _keep_main_component(mask: np.ndarray, ix: int, iy: int,
                         min_area: int = 40) -> np.ndarray:
    """
    主连通域：含端点像素的连通域（攀岩者只接触一个 hold，端点代表
    正在接触的 hold）；端点不在 mask 内 → 取最大连通域。
    """
    if mask.sum() < 50:
        return mask
    n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8)
    if n_lbl <= 2:
        return mask

    iy = max(0, min(iy, mask.shape[0] - 1))
    ix = max(0, min(ix, mask.shape[1] - 1))
    ep_label = labels[iy, ix]
    if ep_label > 0 and stats[ep_label, cv2.CC_STAT_AREA] >= min_area:
        return (labels == ep_label)

    # 端点不在 mask：取最大连通域
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == best)


def _sample_person_edge_points(person_mask: np.ndarray,
                               n: int = 8, max_radius: int = 60) -> np.ndarray:
    """
    在 person mask 边缘外 n 个 negative prompt points（让 SAM 更清楚
    "端点附近的攀岩者身体"边界）。

    简化版（与之前 sample_edge_points 类似但参数收紧）：固定 n=8，
    max_radius=60（之前是 12 + 80，太多了没必要）。
    """
    if person_mask.sum() < 50:
        return np.zeros((0, 2), dtype=np.float64)
    pm_u8 = (person_mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(pm_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros((0, 2), dtype=np.float64)
    all_pts = np.vstack(contours).reshape(-1, 2)
    if len(all_pts) < n:
        idx = np.arange(len(all_pts))
    else:
        idx = np.linspace(0, len(all_pts) - 1, n).astype(int)
    samples = all_pts[idx].astype(np.float64)

    # 向外推（粗略 2D 法向）
    ys, xs = np.where(person_mask)
    cx, cy = float(xs.mean()), float(ys.mean())
    ux = samples[:, 0] - cx
    uy = samples[:, 1] - cy
    norm = np.hypot(ux, uy)
    norm[norm < 1] = 1.0
    ux, uy = ux / norm, uy / norm
    return np.column_stack([
        samples[:, 0] + max_radius * 0.4 * ux,
        samples[:, 1] + max_radius * 0.4 * uy,
    ])


def _sample_color(image_bgr: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    """
    mask 内 HSV 取色：

      1) 绝对白判定：过半像素 S<WHITE_SAT_MAX AND V>WHITE_VAL_MIN → 白
         （阴影/暖光下白 hold V 可能 120-140，由 WHITE_VAL_MIN=140 卡掉；
         这种情况由 hue_to_color_name 兜底判"白"通过 S<35）

      2) 否则：mask bbox 内 GRID_N×GRID_N 网格采样，每点 5×5 patch 中
         位色相聚合（圆形 hue 均值），剔除低饱/近黑像素。

    返回 (h, s, v)。
    """
    ys, xs = np.where(mask)
    if len(xs) < 20:
        return (0.0, 0.0, 0.0)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask_pixels = hsv[ys, xs]

    # 绝对白判定
    white_mask = (mask_pixels[:, 1] < WHITE_SAT_MAX) & (mask_pixels[:, 2] > WHITE_VAL_MIN)
    if white_mask.sum() / len(mask_pixels) > WHITE_RATIO:
        mean_v = float(np.mean(mask_pixels[white_mask, 2]))
        return (0.0, 0.0, mean_v)

    # 相对白：mask 内 S 中位数显著低于"环境环 S 中位数" → 白色 hold
    # （通用：白 hold 在暖光/阴影下 S 可能 35-75，但仍比墙色饱和度低）
    h_img_full, w_img_full = hsv.shape[:2]
    kernel = np.ones((20, 20), np.uint8)
    dil = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
    ring = dil & ~mask
    if ring.sum() > 50:
        ring_pix = hsv[ring]
        r_s = float(np.median(ring_pix[:, 1]))
        m_s = float(np.median(mask_pixels[:, 1]))
        if r_s > 20 and m_s < WHITE_REL_SAT_MAX and m_s / r_s < WHITE_REL_RATIO:
            return (0.0, 0.0, float(np.mean(mask_pixels[:, 2])))

    # 网格采样（5x5 = 25 点 × 5x5 patch）
    bx0, bx1 = xs.min(), xs.max()
    by0, by1 = ys.min(), ys.max()
    bw = max(1, bx1 - bx0)
    bh = max(1, by1 - by0)
    h_img, w_img = mask.shape
    hues, sats, vals = [], [], []
    for gy in range(GRID_N):
        for gx in range(GRID_N):
            cx_g = bx0 + (gx + 0.5) * bw / GRID_N
            cy_g = by0 + (gy + 0.5) * bh / GRID_N
            ix_p = int(round(cx_g))
            iy_p = int(round(cy_g))
            if 2 <= ix_p < w_img - 2 and 2 <= iy_p < h_img - 2:
                patch = hsv[iy_p - 2:iy_p + 3, ix_p - 2:ix_p + 3].reshape(-1, 3)
                keep = (patch[:, 1] >= WHITE_SAT_MAX) & (patch[:, 2] >= COLOR_VAL_MIN)
                if keep.sum() < 3:
                    continue
                pix = patch[keep]
                hr = pix[:, 0].astype(np.float64) * (2 * np.pi / 180.0)
                h_mean = float(np.degrees(np.arctan2(
                    np.mean(np.sin(hr)), np.mean(np.cos(hr))
                )) / 2.0) % 180.0
                hues.append(h_mean)
                sats.append(float(np.median(pix[:, 1])))
                vals.append(float(np.median(pix[:, 2])))

    if not hues:
        # 兜底：mask 全部像素中位
        return (0.0, float(np.median(mask_pixels[:, 1])),
                float(np.median(mask_pixels[:, 2])))

    hr2 = np.array(hues) * (2 * np.pi / 180.0)
    h_final = float(np.degrees(np.arctan2(
        np.mean(np.sin(hr2)), np.mean(np.cos(hr2))
    )) / 2.0) % 180.0
    return (h_final, float(np.median(sats)), float(np.median(vals)))