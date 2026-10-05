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

import os

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
# box_half = max(BOX_HALF_MIN, h // BOX_HALF_HEIGHT_DIV)
#   上限：参考 person_mask bbox 短边 × BOX_PERSON_SHORT_RATIO
# 这是通用适配（不是单图调优）：保证 box 至少能容纳 hold 的尺度，
# 同时不会比攀岩者人体还大太多（避免切到远处无关 hold）
BOX_HALF_DEFAULT = 80
BOX_HALF_MIN = 50            # 最小（远景小人保护）
BOX_HALF_MAX = 250
BOX_HALF_HEIGHT_DIV = 20
BOX_PERSON_SHORT_RATIO = 0.6 # box half ≤ 人体 bbox 短边 × 此比例

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

# ============================================================
# 论文 ablation 开关（环境变量控制，默认 = 生产配置）
# ============================================================
# USE_PIXEL_VOTE=1
#   用像素级颜色投票（color_vote['top_color']）替代 _sample_color
#   均值采样来判定 color_name。
#   默认 0 ⇒ 当前生产判定仍是 mean-HSV（论文贡献 3 待落地验证）。
# DISABLE_PERSON_EDGE_NEG=1
#   关闭 8 个 person mask 边缘负点（论文 ablation
#   "− Body-anchored negative points"）。
# 用法:
#   set USE_PIXEL_VOTE=1 && python exp_dwpose_full.py
USE_PIXEL_VOTE = os.environ.get("USE_PIXEL_VOTE", "0").strip() == "1"
DISABLE_PERSON_EDGE_NEG = os.environ.get("DISABLE_PERSON_EDGE_NEG", "0").strip() == "1"
# DISABLE_RING_BLACK=1 : 关闭环形带黑票修正（论文 ablation "− Ring black vote"）
DISABLE_RING_BLACK = os.environ.get("DISABLE_RING_BLACK", "0").strip() == "1"
# FIXED_BOX=1 : 关闭 person-bbox 自适应，回退固定 box_half = max(80, h//20)
#               （论文 ablation "− Person-bbox-adaptive box"）
FIXED_BOX = os.environ.get("FIXED_BOX", "0").strip() == "1"

# 负点（手工部分）参数
NEG_BODY_OFFSET = 30      # 身体方向负点偏移 (px)
NEG_RING_RADIUS = 35      # 均匀围绕负点半径
NEG_RING_N = 5            # 均匀围绕负点数量

# ============================================================
# 环形带黑修正（实验3 落地）
# ============================================================
# 背景：MediaPipe 端点位置可能偏移几十 px（如 2X_01 黑线 0005s 帧
#   R_hand 端点偏 49px），EdgeSAM 以端点为锚切 mask 会切到"端点所在
#   的另一个 hold"（橙 hold），真正抓的黑色 hold 在 mask 外 49px 处被
#   当背景切不进来 → 判色错。
# 方案：不额外 predict（零成本），在 mask 外扩一个 20-70px 环形带做
#   纯像素颜色投票。黑色 hold 在彩色岩壁上稀缺，环带黑票占比高是有
#   信息量的信号（说明端点附近确实存在一块黑 hold 没被切进 mask）；
#   而彩色 hold 在环带里占多是常态（无信息量），所以只做"非黑→黑"
#   方向的修正，绝不做反向（L_hand 反例：mask 判黑正确、环带黄票 58%，
#   对称规则会立刻破坏正确结果）。
# 阈值/范围来自 0005s 实测：R_hand(需修) 黑票 29.3% vs
#   L_hand(不可修) 黑票 7.8%，0.18 取中间留余量。
RING_BLACK_D1 = 20          # 环形带内半径 (px)
RING_BLACK_D2 = 70          # 环形带外半径 (px)
RING_BLACK_RATIO_MIN = 0.18 # 环带黑票占比下限（超过才修正）
RING_BLACK_MIN_PIXELS = 100 # 环带有效像素下限（太少不算）


def compute_box_half(img_h: int, person_mask: np.ndarray | None = None) -> int:
    """
    自适应 EdgeSAM box prompt 半边长。

    基础值: max(BOX_HALF_MIN, img_h // BOX_HALF_HEIGHT_DIV)
      — 保证 box 至少能容纳 hold 的尺度（高分辨率图自动放大）。

    上限: person_mask bbox 短边 × BOX_PERSON_SHORT_RATIO
      — 攀岩者一只手/脚约占人体短边 40-60%，box 比人体大太多会
        切到无关远处 hold（line1_white frame02 R_hand 实例：box 160
        切到远处红色三角 hold 14574px，挤压掉手边白色 hold 候选）。

    无 person_mask 或人体 bbox 太小时回退到基础值（不强行缩小）。
    """
    eff = max(BOX_HALF_MIN, img_h // BOX_HALF_HEIGHT_DIV)
    if FIXED_BOX:
        # ablation: 关闭 person-bbox 自适应（回退固定 box，min 用历史值 80）
        return min(max(80, img_h // BOX_HALF_HEIGHT_DIV), BOX_HALF_MAX)
    if person_mask is not None and person_mask.any():
        ys, xs = np.where(person_mask)
        if len(ys) > 0:
            person_short = min(int(ys.max() - ys.min()), int(xs.max() - xs.min()))
            if person_short > 0:
                person_cap = max(BOX_HALF_MIN, int(person_short * BOX_PERSON_SHORT_RATIO))
                eff = min(eff, person_cap)
    eff = max(BOX_HALF_MIN, min(eff, BOX_HALF_MAX))
    return eff


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
        ring_black_fix: bool = True,
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

        # box prompt：端点周围自适应 box（参考人体 bbox 短边限制）
        eff_box_half = compute_box_half(self.last_h, person_mask)
        ix = max(0, min(int(round(px)), self.last_w - 1))
        iy = max(0, min(int(round(py)), self.last_h - 1))
        bx1 = max(0, ix - eff_box_half)
        by1 = max(0, iy - eff_box_half)
        bx2 = min(self.last_w - 1, ix + eff_box_half)
        by2 = min(self.last_h - 1, iy + eff_box_half)
        box = np.array([bx1, by1, bx2, by2], dtype=np.float64)

        # 负点：1 个身体方向 + NEG_RING_N 个均匀围绕 + person mask 边缘外
        neg = self._make_negative_prompts(px, py, body_cx, body_cy)
        if (person_mask is not None and person_mask.any()
                and not DISABLE_PERSON_EDGE_NEG):
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

        # 【实验 2】多候选融合（并行，不改现有判定；weight_map=端点近区权重）
        yy, xx = np.ogrid[:self.last_h, :self.last_w]
        dist_map = np.hypot(xx - ix, yy - iy)
        weight_near = np.where(dist_map <= 40, 2.0, 1.0)
        fusion = _fuse_candidates(
            self.last_image_bgr, candidates, ix, iy,
            person_mask=person_mask, weight_map=weight_near,
        )

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

        # 颜色统计（mean-HSV，基线路径）
        h, s, v = _sample_color(self.last_image_bgr, chosen)

        # 【实验 1】像素级颜色投票（论文贡献 3）
        #   USE_PIXEL_VOTE=1 时由它接管 color_name 判定；
        #   默认 0 = 保持 mean-HSV 基线（与历史结果一致）。
        color_vote = _pixel_color_vote(self.last_image_bgr, chosen)

        if USE_PIXEL_VOTE:
            color_name = color_vote["top_color"]
        else:
            # 颜色名（mean-HSV 基线）
            if v < DARK_V_MAX and s < DARK_S_MAX:
                color_name = "未知(暗)"
            elif s < WHITE_SAT_MAX:
                # 低饱和：V 高 → 白；V 中低 → 黑（黑色 hold 被灯光打亮时
                # S 仍≈0 但 V 只升到 ~60-140，真白 hold V 一般 ≥160）
                color_name = "白" if v >= WHITE_VAL_MIN else "黑"
            else:
                color_name = route_color.hue_to_color_name(h, s)

        # 【实验 3】环形带黑修正（落地，默认开）
        #   仅在 mask 判色非黑时检查：mask 外扩环带黑票占比高
        #   （>= RING_BLACK_RATIO_MIN）→ 端点附近存在一块没被切进
        #   mask 的黑 hold，把判定修正为黑。零额外 predict。
        ring_black_ratio = 0.0
        ring_black_n = 0
        ring_black_fixed = False
        if ring_black_fix and not DISABLE_RING_BLACK and color_name != "黑":
            ring_black_ratio, ring_black_n = _ring_black_ratio(
                self.last_image_bgr, chosen, ix, iy, person_mask=person_mask)
            if ring_black_ratio >= RING_BLACK_RATIO_MIN:
                color_name = "黑"
                ring_black_fixed = True

        return {
            "mask": chosen.astype(np.uint8) * 255,
            "mask_bool": chosen.astype(bool),
            "score": float(chosen_score),
            "h": float(h), "s": float(s), "v": float(v),
            "color_name": color_name,
            "color_vote": color_vote,
            "fusion": fusion,
            "area": int(chosen.sum()),
            "endpoint": (px, py),
            "ring_black": {
                "ratio": float(ring_black_ratio),
                "n_pixels": int(ring_black_n),
                "fixed": bool(ring_black_fixed),
            },
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
            "color_vote": {"top_color": "非岩点", "top_ratio": 0.0,
                           "votes": {}, "n_pixels": 0},
            "fusion": {"mode": "none", "top_color": "非岩点",
                       "per_candidate": [], "votes": {}},
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


def _classify_pixel(h: float, s: float, v: float) -> str:
    """
    单像素 → 颜色桶（实验 1 用）。
    与现有判定逻辑对齐：
      S<35  → 按 V 分黑/白（V>=WHITE_VAL_MIN 白，否则黑）
      S>=35 → 按 H 走 hue_to_color_name（含青/绿饱和度分流、粉红判定）
    """
    if s < WHITE_SAT_MAX:
        return '白' if v >= WHITE_VAL_MIN else '黑'
    return route_color.hue_to_color_name(h, s)


def _ring_black_ratio(image_bgr: np.ndarray, mask_bool: np.ndarray,
                      ix: int, iy: int,
                      person_mask: np.ndarray | None = None,
                      d1: int = RING_BLACK_D1, d2: int = RING_BLACK_D2,
                      min_pixels: int = RING_BLACK_MIN_PIXELS) -> tuple[float, int]:
    """
    【实验 3】mask 外扩环形带黑票占比（零 EdgeSAM predict，纯像素）。

    以端点 (ix,iy) 为圆心，取 d1-d2 px 圆环，排除 mask 本身和 person
    mask，对剩余像素逐像素颜色分类，统计 '黑' 票占比。

    返回 (black_ratio, n_pixels)。像素不足 min_pixels 时返回 (0.0, 0)。
    """
    h, w = image_bgr.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    dist = np.hypot(xx - ix, yy - iy)
    ring = (dist >= d1) & (dist <= d2) & ~mask_bool
    if person_mask is not None and person_mask.any():
        ring = ring & ~person_mask.astype(bool)
    if ring.sum() < min_pixels:
        return 0.0, 0
    pix = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)[ring]
    if len(pix) == 0:
        return 0.0, 0
    black = 0
    for hh, ss, vv in pix:
        if _classify_pixel(float(hh), float(ss), float(vv)) == '黑':
            black += 1
    return black / len(pix), len(pix)


def _pixel_color_vote(image_bgr: np.ndarray, mask: np.ndarray,
                      weight_map: np.ndarray | None = None,
                      min_pixels: int = 20) -> dict:
    """
    【实验 1】像素级颜色分类投票（替代 _sample_color 均值采样）。

    核心差异：_sample_color 在网格采样时丢弃 S<35 的像素（黑白岩点天然
    吃亏，黑岩点在暖光下 S 只有 20-30）；这里 mask 内**每个像素**都
    分桶投票，低饱和像素正常参与。

    weight_map: 可选 HxW float，与 mask 同尺寸，逐像素权重（如端点
    距离权重）。None = 等权。

    返回:
      {
        'top_color': 最高票颜色名,
        'top_ratio': 最高票占比 (top / 总票),
        'votes': dict[颜色名 -> 加权票数],
        'n_pixels': 参与投票像素数,
      }
    """
    ys, xs = np.where(mask)
    if len(xs) < min_pixels:
        return {'top_color': '非岩点', 'top_ratio': 0.0, 'votes': {}, 'n_pixels': 0}

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    pix = hsv[ys, xs]
    if weight_map is None:
        wts = np.ones(len(xs), dtype=np.float64)
    else:
        wts = weight_map[ys, xs].astype(np.float64)
        # 权重全 0 → 退化为等权
        if wts.sum() <= 0:
            wts = np.ones(len(xs), dtype=np.float64)

    votes: dict[str, float] = {}
    for hh, ss, vv, ww in zip(pix[:, 0], pix[:, 1], pix[:, 2], wts):
        name = _classify_pixel(float(hh), float(ss), float(vv))
        votes[name] = votes.get(name, 0.0) + ww

    total = float(sum(votes.values()))
    if total <= 0:
        return {'top_color': '非岩点', 'top_ratio': 0.0, 'votes': {}, 'n_pixels': len(xs)}

    top_color = max(votes, key=votes.get)
    return {
        'top_color': top_color,
        'top_ratio': votes[top_color] / total,
        'votes': votes,
        'n_pixels': len(xs),
    }


def _fuse_candidates(image_bgr: np.ndarray,
                     candidates: list[tuple[float, float, int, np.ndarray]],
                     ix: int, iy: int,
                     person_mask: np.ndarray | None = None,
                     weight_map: np.ndarray | None = None) -> dict:
    """
    【实验 2】多候选融合投票（替代"只取 score 最高"）。

    对每个通过筛选的候选：减 person → 主连通域 → 像素级颜色投票，
    然后按不同模式融合：
      mode='score_weighted' : 各候选 top_color 按 score 加权投票
      mode='count'          : 各候选 top_color 简单计数
      mode='pixel_union'    : 所有候选 mask 并集，逐像素等权投票
                              （weight_map 可传端点近区权重）

    返回:
      {
        'mode': str,
        'top_color': 融合结果,
        'per_candidate': [{score, top_color, top_ratio, votes}, ...],
        'votes': 融合票数,
      }
    """
    per = []
    union_mask = None
    for score, dist, area, m in candidates:
        m2 = m & ~person_mask.astype(bool) if person_mask is not None else m
        if m2.sum() < COLOR_MIN_PIXELS:
            continue
        cc = _keep_main_component(m2.copy(), ix, iy)
        v = _pixel_color_vote(image_bgr, cc)
        per.append({
            'score': score, 'dist': dist, 'area': int(cc.sum()),
            'top_color': v['top_color'], 'top_ratio': v['top_ratio'],
            'votes': v['votes'], 'n_pixels': v['n_pixels'],
        })
        union_mask = cc if union_mask is None else (union_mask | cc)

    if not per:
        return {'mode': 'none', 'top_color': '非岩点', 'per_candidate': [], 'votes': {}}

    results = {}

    # mode 1: score 加权 top_color 投票
    sw: dict[str, float] = {}
    for p in per:
        sw[p['top_color']] = sw.get(p['top_color'], 0.0) + p['score']
    results['score_weighted'] = max(sw, key=sw.get)

    # mode 2: 简单计数
    cnt: dict[str, int] = {}
    for p in per:
        cnt[p['top_color']] = cnt.get(p['top_color'], 0) + 1
    results['count'] = max(cnt, key=cnt.get)

    # mode 3: 像素级并集投票
    if union_mask is not None and union_mask.sum() >= 20:
        uv = _pixel_color_vote(image_bgr, union_mask, weight_map=weight_map)
        results['pixel_union'] = uv['top_color']
        results['pixel_union_votes'] = uv['votes']
        results['pixel_union_ratio'] = uv['top_ratio']
    else:
        results['pixel_union'] = '非岩点'

    return {
        'mode': 'all',
        'results': results,
        'top_color': results.get('pixel_union', results.get('score_weighted', '非岩点')),
        'per_candidate': per,
    }


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
                # 关键：只统计 mask 内的像素，避免 patch 中心落在 mask 外时
                # 采到墙/邻接 hold 的颜色污染（实测 3O line2_pink L_hand 被
                # 墙橙红像素 H≈10 拉偏，圆形均值坍缩到 0° → 误判红）
                mask_patch = mask[iy_p - 2:iy_p + 3, ix_p - 2:ix_p + 3].reshape(-1)
                keep = (patch[:, 1] >= WHITE_SAT_MAX) & (patch[:, 2] >= COLOR_VAL_MIN) & mask_patch
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