"""
EdgeSAM 端点级 hold 分割模块。

核心思路：
  对每个 MediaPipe 姿态端点（L/R 手腕、L/R 脚尖）作为正点 prompt，
  让 EdgeSAM 输出"该端点所在岩点"的二值 mask。
  再从 mask 像素里取 HSV 直方图主色，按色投票判定线路色。

为啥不用现有 HSV 过滤 detect_holds 那套：
  - 同色伪装墙（墙=线路色）颜色空间硬限制
  - 白色岩点 S<60 全被过滤掉
  EdgeSAM 是语义感知，绕过以上两个限制；输出是精确 mask 不是 bbox。

颜色主色：
  直接复用 route_color._robust_hold_hsv 思想 —— 在 mask 内采多点 5x5 patch 取色，
  hue 圆形均值；色彩到中文色名复用 route_color.hue_to_color_name。
  （导入 route_color 而不是复制，避免循环依赖 & 颜色标准同步。）

负点（negative prompts）：
  端点靠近身体时，单独的 +点会让 EdgeSAM 把"端点附近的身体+hold"一起 mask。
  给 4 个 background 点（端点向身体侧偏移 30px 周围 + 远处墙色）可以让 SAM
  更精确把"端点所在的物体"切出来。
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

import edgesam_stub  # noqa: F401 必须在 from edge_sam 之前注入 stub
import route_color  # 色相 → 色名、皮肤过滤、墙色常量
from edge_sam import sam_model_registry, SamPredictor  # noqa: E402
from yolo_person_mask import sample_edge_points  # noqa: E402  人体 mask → 边缘外 negative prompt


# 默认 checkpoint / 设备
DEFAULT_CHECKPOINT = "weights/edge_sam.pth"
DEVICE_CPU = torch.device("cpu")

# mask_subtract_person=True 时，端点周围保护圈半径 (px)。
# 扣掉 person_mask 时保留端点附近这个小圆盘，避免"端点本身在人体内"导致
# mask 不含端点而全部被过滤。圆圈内 = 人体像素也留着（hold 就在手指/脚附近）。
PERSON_MASK_PROTECT_RADIUS = 18


class EdgeSamHoldSegmenter:
    """EdgeSAM 包装：一次 set_image，多次 predict 端点 mask。"""

    def __init__(
        self,
        checkpoint_path: str = DEFAULT_CHECKPOINT,
        device: str | torch.device = "cpu",
    ):
        """
        checkpoint_path: .pth 路径
        device: "cpu" / "cuda" / torch.device
        """
        self.device = torch.device(device)
        # EdgeSAM 注册的 key 是 "edge_sam" / "default"
        build_fn = sam_model_registry["edge_sam"]
        self.model = build_fn(checkpoint=checkpoint_path)
        self.model = self.model.to(self.device).eval()
        self.predictor = SamPredictor(self.model)
        # image_size 缓存（被 SamPredictor.set_image / ResizeLongestSide 用）
        self.img_size = self.model.image_encoder.img_size

    def set_image(self, image_bgr: np.ndarray) -> None:
        """一次性算 image embedding。需要 RGB uint8 HxWx3。"""
        # SamPredictor.set_image 接受 RGB 或 BGR，按 image_format 参数
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb, image_format="RGB")
        self.last_h, self.last_w = image_bgr.shape[:2]
        self.last_image_bgr = image_bgr  # 给后续 mask 颜色用

    @staticmethod
    def _make_negative_prompts(
        px: float, py: float,
        body_cx: float | None, body_cy: float | None,
        n: int = 6, radius: float = 35.0,
        white_route: bool = False,
    ) -> np.ndarray:
        """
        在端点周围制造 negative points，把"端点附近的身体 + 背景墙"从 hold mask 剔除。

        white_route=True 时加强"亮高光"方向上的负点 — 防止 EdgeSAM
        把镁粉/天花板反光误当作 hold mask。

        返回 shape (n, 2) 的 array，labels 由调用方填 0。
        """
        pts: list[tuple[float, float]] = []
        if body_cx is not None and body_cy is not None:
            ux, uy = body_cx - px, body_cy - py
            norm = np.hypot(ux, uy) or 1.0
            ux, uy = ux / norm, uy / norm
            # 身体侧一点（端点往身体方向偏 30px）
            inward_x = px + 30 * ux
            inward_y = py + 30 * uy
            pts.append((inward_x, inward_y))
            # 4 个垂直端点方向的"墙/背景"
            for theta in np.linspace(0, 2 * np.pi, 5, endpoint=False):
                rx = px + radius * (np.cos(theta))
                ry = py + radius * (np.sin(theta))
                pts.append((rx, ry))
            if white_route:
                # 加大远端斜上 60-90px 的负点（白岩点常见 = 镁粉在端点正上方/侧上）
                for theta_off in (np.pi / 4, 3 * np.pi / 4, -3 * np.pi / 4, -np.pi / 4):
                    ux_off = ux * np.cos(theta_off) - uy * np.sin(theta_off)
                    uy_off = ux * np.sin(theta_off) + uy * np.cos(theta_off)
                    pts.append((px + 60 * ux_off, py + 60 * uy_off))
        else:
            for theta in np.linspace(0, 2 * np.pi, n, endpoint=False):
                pts.append((px + radius * np.cos(theta),
                            py + radius * np.sin(theta)))
        return np.array(pts, dtype=np.float64)

    def segment_endpoint(
        self,
        px: float, py: float,
        body_cx: float | None = None, body_cy: float | None = None,
        num_multimask_outputs: int = 4,
        use_box_prompt: bool = True,
        box_half: int = 80,
        use_stability_score: bool = True,
        white_route_mode: bool = False,
        person_mask: np.ndarray | None = None,
        exclude_mask: np.ndarray | None = None,
        mask_subtract_person: bool = True,
        subtract_person_mask: np.ndarray | None = None,
    ) -> dict:
        """
        对 (px, py) 这个端点做 hold 分割。

        white_route_mode: True 时加强负点密度 + 用更严的面积上限 + 降低 score 门槛。
        用于白岩点这类容易跟墙颜色重叠的场景。

        use_box_prompt (默认 True)：
          - True: 用 box prompt（端点周围 box_half*2 px 正方形）+ 1 个 center positive
            point + 远处 negative points。让 EdgeSAM 在 box 内切"端点附近的物体"，
            **不依赖 detect_holds**（HSV 颜色阈值体系）。
          - False: 退回纯 point prompt（1+16 个 negative）。

        box_half: box 半边长 (px)。默认 80 → 160x160 box。
          太大贪心合并，太小涵盖不到完整 hold。

        num_multimask_outputs (默认 4)：
          EdgeSAM 候选 mask 数。1/3/4，越多越能选到"完整 hold shape" (m[2])。

        person_mask: HxW bool，YOLO26 攀岩者人体 mask。若提供：
          - 把 person_mask **边缘外**的点作为 negative prompt（让 EdgeSAM 把
            "端点附近的身体"区分开）
          - mask_subtract_person=True 时再额外把 SAM 输出 mask 与 person_mask 取差集
            （剥除手腕/衣服等"嵌在身体里的碎片"）

        exclude_mask: HxW bool 可选 — 用于颜色统计时**排除**的像素范围。
          通常应比 person_mask 大（戴 6-15 px 膨胀），把鞋/手套/护腕等"贴在身体
          上但露出 person_mask 边界"的附件也算作身体。
          9.jpg R_foot 攀岩鞋，鞋尖/鞋面常超出 person_mask 边界——不扩大 exclude
          范围会把鞋的黄色错误地计入颜色统计。

        mask_subtract_person (默认 False)：
          - False（推荐）：只给 negative prompt，让 EdgeSAM 自己处理手表等碎片
          - True: 把 hold mask 里的 person_mask 像素减掉（更激进，可能误伤）
            → 适合 hold 与身体重叠程度已知很小的场景

        subtract_person_mask: HxW bool 可选 — 用于 mask_subtract_person 的 mask。
          默认 = person_mask（膨胀版，负点+subtract 用同一个）。
          传未膨胀的原始 person mask 可以让"脚下方/手旁边的 hold"不被吞掉
          （5.jpg L_foot: 膨胀6px 把脚下橙色岩点吞进 person → 取色区域=0
          判非岩点；用未膨胀版则保留岩点）。

        返回 dict:
          mask: HxW bool（端点所在 hold 的 mask，原图坐标系）
          mask_bool: HxW bool（同上，保留别名）
          score: float（mask 置信度）
          h, s, v: float（mask 内颜色统计）
          color_name: str（中文色名）
          area: int（mask 像素数）
          endpoint: (px, py)
        """
        if not self.predictor.is_image_set:
            raise RuntimeError("先调用 set_image() 再 segment_endpoint()")

        # box prompt: 端点周围 box_half px 的方框
        box = None
        if use_box_prompt:
            bx1 = max(0, int(round(px)) - box_half)
            by1 = max(0, int(round(py)) - box_half)
            bx2 = min(self.last_w - 1, int(round(px)) + box_half)
            by2 = min(self.last_h - 1, int(round(py)) + box_half)
            box = np.array([bx1, by1, bx2, by2], dtype=np.float64)

        # 正点（端点）+ 负点（白路线模式更激进）
        neg = self._make_negative_prompts(
            px, py, body_cx, body_cy,
            white_route=white_route_mode,
        )

        # YOLO26 提供的 person mask：边缘外的点作为更精确的 negative prompt
        if person_mask is not None and person_mask.any():
            edge_pts = sample_edge_points(person_mask, n=12, max_radius=80)
            if len(edge_pts) > 0:
                neg = np.vstack([neg, edge_pts])

        point_coords = np.vstack([
            [[px, py]],
            neg,
        ])
        point_labels = np.array([1] + [0] * len(neg), dtype=np.int64)

        # EdgeSAM: box prompt（如果启用）+ 1 个 positive point + 多个 negative
        # box + point 同时给是 SAM 支持的，与本 mask 选择策略兼容
        masks, scores, _ = self.predictor.predict(
            box=box,
            point_coords=point_coords,
            point_labels=point_labels,
            num_multimask_outputs=num_multimask_outputs,
            use_stability_score=use_stability_score,
        )

        if masks.ndim == 2:  # num_multimask_outputs=1 时
            masks = masks[None, :, :]
            scores = np.array([float(scores)])

        ix, iy = int(round(px)), int(round(py))
        ix = max(0, min(ix, self.last_w - 1))
        iy = max(0, min(iy, self.last_h - 1))
        img_area = self.last_w * self.last_h

        # 面积上限：白路线模式更激进（白 hold 一般更小）
        area_cap = img_area * (0.20 if white_route_mode else 0.30)

        # 过滤：必须包含端点、score ≥ 阈值、面积 < 阈值
        score_min_main = 0.35 if white_route_mode else 0.4
        candidates = []
        for i, m in enumerate(masks):
            # mask_subtract_person=True：把 mask 中 person_mask 像素剥除，
            # 但保留端点周围 PERSON_MASK_PROTECT_RADIUS 的圆盘（否则端点被扣光，
            # m_work[iy, ix] 恒 False → 全被过滤）。
            # 核心价值：让"岩点 mask"与"人体"不重合 — 鞋/手/腿/衣服的像素被扣掉。
            m_work = m.copy()
            if mask_subtract_person and person_mask is not None:
                # 负点用 person_mask（膨胀版，避开身体边缘），
                # subtract 用 subtract_person_mask（可传未膨胀版，避免吞掉
                # 手/脚旁的 hold —— 5.jpg L_foot 脚下岩点被膨胀6px吞掉）
                sub_mask = subtract_person_mask if subtract_person_mask is not None else person_mask
                # 端点保护圈：mask 内保留以 (ix,iy) 为圆心、protect_radius 内的 person 像素
                protect = sub_mask.copy()
                Y, X = np.ogrid[:self.last_h, :self.last_w]
                dist = np.hypot(X - ix, Y - iy)
                protect[(dist > PERSON_MASK_PROTECT_RADIUS) | (m == 0)] = False
                # 保护圈内的人体像素先"摘出来"再减
                keep_ring = m & protect
                m_work[sub_mask] = False
                m_work |= keep_ring
            area = int(m_work.sum())
            if area < 100:
                continue
            if area > area_cap:
                continue
            # 必须包含端点 OR（端点本身在 person 内时）mask 距端点足够近。
            # 脚的端点落在脚上（person 内），而 hold 在脚下方（person 外）——
            # 此时 mask 不包含端点像素（5.jpg R_foot），但不能因此全过滤。
            contains_ep = bool(m_work[iy, ix])
            if not contains_ep and person_mask is not None and person_mask[iy, ix]:
                # 端点被 person 覆盖：接受"距端点 < PROTECT_RADIUS*2 内非空"的 mask
                ep_region = m_work[max(0, iy - 40):iy + 41, max(0, ix - 40):ix + 41]
                if ep_region.sum() < 30:
                    continue
            elif not contains_ep:
                continue
            if float(scores[i]) < score_min_main:
                continue
            candidates.append((float(scores[i]), area, m_work))

        # 兜底：放低 score 门槛
        if not candidates:
            soft_score_min = 0.25 if white_route_mode else 0.3
            soft_area_cap = img_area * 0.45
            soft = []
            for i, m in enumerate(masks):
                m_work = m.copy()
                if mask_subtract_person and person_mask is not None:
                    sub_mask = subtract_person_mask if subtract_person_mask is not None else person_mask
                    protect = sub_mask.copy()
                    Y, X = np.ogrid[:self.last_h, :self.last_w]
                    dist = np.hypot(X - ix, Y - iy)
                    protect[(dist > PERSON_MASK_PROTECT_RADIUS) | (m == 0)] = False
                    keep_ring = m & protect
                    m_work[sub_mask] = False
                    m_work |= keep_ring
                area = int(m_work.sum())
                if area < 50 or area > soft_area_cap:
                    continue
                contains_ep = bool(m_work[iy, ix])
                if not contains_ep and person_mask is not None and person_mask[iy, ix]:
                    ep_region = m_work[max(0, iy - 40):iy + 41, max(0, ix - 40):ix + 41]
                    if ep_region.sum() < 30:
                        continue
                elif not contains_ep:
                    continue
                if float(scores[i]) < soft_score_min:
                    continue
                soft.append((float(scores[i]), area, m_work))
            if soft:
                candidates = soft

        if not candidates:
            # 真没有 hold 形状 → 返回低置信空 mask（让调用方判定"非岩点"）
            return {
                "mask": np.zeros((self.last_h, self.last_w), dtype=np.uint8),
                "mask_bool": np.zeros((self.last_h, self.last_w), dtype=bool),
                "score": float(np.max(scores)) if len(scores) else 0.0,
                "h": 0.0, "s": 0.0, "v": 0.0,
                "color_name": "未知",
                "area": 0,
                "endpoint": (px, py),
            }

        # 在合格候选里选"包含端点的最小 mask"——hold 一般小，墙+hold 混合的 mask 大
        # 评面积比 score 重：score 差不多时选最小的
        # 排序：stability_score ↓ 优先，同分时 area ↓（大mask优先）
        candidates.sort(key=lambda c: (-c[0], -c[1]))
        chosen_score, _, chosen = candidates[0]

        # 主连通域过滤（判断与可视化一致）：EdgeSAM 的 mask 可含多个不相连
        # 区域（1.jpg R_hand: 含端点的紫色 hold 3485px + 远处黄色 hold 2136px
        # + 碎片共 10 域）。攀岩者只接触一个 hold，取**包含端点的连通域**
        # （端点代表正在接触的 hold）。若端点不在 mask 内，取最大连通域。
        chosen = _keep_main_component(chosen, px, py, self.last_image_bgr)

        # 颜色聚类剔除：EdgeSAM 的 mask 有时把端点附近的"异色岩点/杂质"圈进来
        # （如 9.jpg R_hand：紫色岩点 + 白色 hold 共用一个 mask）。做 k-means 聚类，
        # 只保留"端点像素所属的颜色簇"（hold 的真实颜色簇），剔除异色大块。
        chosen = _keep_endpoint_cluster(
            self.last_image_bgr, chosen, px, py,
            min_area=60, n_clusters=3,
        )
        if chosen.sum() < 50:
            # 聚类把 mask 削得太小 → 说明 mask 本身颜色均匀，回退原 mask
            chosen = candidates[0][2]

        # 取色区域占比检查：mask 主体不能在 person 内（9.jpg L_hand/R_hand 的
        # mask 100% 在 person mask 内 —— 保护圈保住的其实是身体/手本身，不是
        # hold，却以 mask 形式投票）。若取色区域 (mask & ~person) < 20%，
        # 判定为"非岩点"（被身体覆盖，无真实 hold 像素）。
        # 注意：与 subtract 用同一个 mask（subtract_person_mask 优先）——
        # 5.jpg L_foot 用未膨胀版 subtract 保留了脚下岩点，这里也必须用未膨胀版
        # 否则膨胀版 person mask 又把岩点算进"person 内"→ 误判非岩点。
        if person_mask is not None:
            _region_check_mask = subtract_person_mask if subtract_person_mask is not None else person_mask
            _color_region = chosen & ~_region_check_mask.astype(bool)
            _region_ratio = float(_color_region.sum()) / max(1.0, float(chosen.sum()))
            if _region_ratio < 0.20:
                return {
                    "mask": chosen.astype(np.uint8) * 255,
                    "mask_bool": chosen.astype(bool),
                    "score": chosen_score,
                    "h": 0.0, "s": 0.0, "v": 0.0,
                    "color_name": "非岩点",
                    "area": int(chosen.sum()),
                    "endpoint": (px, py),
                }

        # 用 mask 取 hold 颜色（person_mask 内的身体/衣服像素不进颜色统计）
        h, s, v = _robust_hsv_from_mask(
            self.last_image_bgr, chosen,
            exclude_mask=subtract_person_mask if subtract_person_mask is not None else person_mask,
        )
        # 颜色名判定优先级（避免 hue bin 把黑/白误判成"红"等）：
        #   1. V<60 AND S<40: 真暗（衣服/手套/手表）→ "未知(暗)"（不投 hold）
        #   2. S<35: 低饱和（白/灰 hold）→ "白"。不要求 V>140 —— 白色 hold
        #      在阴影/暖光下 V 可能只有 120-140（9.jpg R_foot: 相对白判定
        #      返回 (0,0,136)，V=136<140 会落入 hue_to_color_name(0,0) 判"红"）
        #   3. 否则: hue_to_color_name（深蓝 hold：V<60 但 S>=60 也走这里）
        if v < 60 and s < 40:
            color_name = "未知(暗)"
        elif s < 35:
            color_name = "白"
        else:
            color_name = route_color.hue_to_color_name(h, s)

        return {
            "mask": chosen.astype(np.uint8) * 255,  # uint8 for 可视化
            "mask_bool": chosen.astype(bool),
            "score": chosen_score,
            "h": float(h), "s": float(s), "v": float(v),
            "color_name": color_name,
            "area": int(chosen.sum()),
            "endpoint": (px, py),
        }


# ============================================================
# mask → 主连通域过滤（判断与可视化一致：只保留含端点的连通域）
# ============================================================
def _keep_main_component(mask: np.ndarray, px: float, py: float,
                         image_bgr: np.ndarray | None = None,
                         min_area: int = 40) -> np.ndarray:
    """
    EdgeSAM 的 mask 可含多个不相连区域（1.jpg R_hand：含端点紫色 hold 3485px
    + 远处黄色 hold 2136px + 碎片共 10 域）。攀岩者只接触一个 hold，
    只保留**包含端点的连通域**（端点代表正在接触的 hold）。

    若端点不在 mask 内（被 person 扣掉，5.jpg R_foot 脚在 person 内而
    hold 在脚下 person 外），取"距端点最近 + 高饱和亮色"的连通域 ——
    脚/鞋在端点附近但暗（S 低），真正的 hold 距端点略远但高饱和亮色。
    不能取"面积最大"：5.jpg 候选3 的最大连通域 (658px) 是左侧的绳/衣物。
    """
    if mask.sum() < 50:
        return mask
    n_lbl, labels, stats, cent = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8)
    if n_lbl <= 2:
        return mask

    iy, ix = int(round(py)), int(round(px))
    iy = max(0, min(iy, mask.shape[0] - 1))
    ix = max(0, min(ix, mask.shape[1] - 1))
    ep_label = labels[iy, ix]

    if ep_label > 0 and stats[ep_label, cv2.CC_STAT_AREA] >= min_area:
        return (labels == ep_label)

    # 端点不在 mask 内：取"距端点最近 + 高饱和亮色"的连通域。
    # 5.jpg R_foot 候选3 有 3 个橙色域：
    #   域5 (距9px,  S=70)  = 脚/鞋本身（低饱和暗色，不是 hold）
    #   域7 (距45px, S=248) = 脚下岩点（高饱和亮色，正确）✓
    #   域3 (距127px,S=252) = 左侧绳/衣物（高饱和但更远）
    # 优先"高饱和 (S>100) 亮色 (V>80)"的连通域，按距离排序选最近者；
    # 若都低饱和（白色/暗色 hold），退化为按距离选。
    if image_bgr is not None:
        hsv_now = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    else:
        hsv_now = None
    best_j, best_d = 0, 1e18
    fallback_j, fallback_d = 0, 1e18
    for j in range(1, n_lbl):
        a = stats[j, cv2.CC_STAT_AREA]
        if a < min_area:
            continue
        cx, cy = cent[j]
        d = float(np.hypot(cx - ix, cy - iy))
        # 连通域平均饱和/亮度
        if hsv_now is not None:
            ys_c, xs_c = np.where(labels == j)
            s_med = float(np.median(hsv_now[ys_c, xs_c, 1]))
            v_med = float(np.median(hsv_now[ys_c, xs_c, 2]))
        else:
            s_med, v_med = 0.0, 0.0
        if s_med > 100 and v_med > 80:
            if d < best_d:
                best_d, best_j = d, j
        if d < fallback_d:
            fallback_d, fallback_j = d, j
    if best_j > 0:
        return (labels == best_j)
    if fallback_j > 0:
        return (labels == fallback_j)

    # 兜底：最大连通域
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == best)


# ============================================================
# mask → 颜色聚类剔除（保留端点所在颜色簇，剔除异色岩点/杂质）
# ============================================================
def _kmeans_simple(X: np.ndarray, k: int, iters: int = 25, seed: int = 42
                   ) -> tuple[np.ndarray, np.ndarray]:
    """手写 k-means（无 sklearn 依赖）。返回 (labels, centers)。"""
    rng = np.random.RandomState(seed)
    centers = X[rng.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        d = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        lab = d.argmin(1)
        for i in range(k):
            if (lab == i).any():
                centers[i] = X[lab == i].mean(0)
    d = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
    return d.argmin(1), centers


def _keep_endpoint_cluster(image_bgr: np.ndarray, mask: np.ndarray,
                           px: float, py: float,
                           min_area: int = 60, n_clusters: int = 3) -> np.ndarray:
    """
    EdgeSAM 的 mask 可能包含"端点附近多个不同颜色的物体"（如 9.jpg R_hand：
    紫色岩点 + 白色 hold 共用一个 mask，紫色占 73%）。做 HSV 颜色聚类，
    只保留**端点像素所属的颜色簇** —— 端点代表攀岩者正在接触的那个 hold。

    返回聚类后的 mask（只含端点簇）。若 mask 像素太少无法聚类，或端点簇
    面积 < min_area，返回原 mask。
    """
    ys, xs = np.where(mask)
    n = len(xs)
    if n < 80:
        return mask

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    pix = hsv[ys, xs]  # (N, 3) H, S, V

    labels, centers = _kmeans_simple(pix, n_clusters)

    # 端点像素的颜色：端点本身可能在 person_mask 内被剥掉（保护圈外），
    # 取 mask 内距离端点最近的像素作为参考色
    iy, ix = int(round(py)), int(round(px))
    iy = max(0, min(iy, mask.shape[0] - 1))
    ix = max(0, min(ix, mask.shape[1] - 1))
    if mask[iy, ix]:
        ep_pix = hsv[iy, ix]
    else:
        d2 = (xs - ix) ** 2 + (ys - iy) ** 2
        ep_pix = pix[int(np.argmin(d2))]

    # 端点像素归属的簇
    dists = np.linalg.norm(centers - ep_pix, axis=1)
    ep_cluster = int(np.argmin(dists))
    ep_n = int((labels == ep_cluster).sum())
    if ep_n < min_area:
        # 端点簇太小 → 说明 mask 颜色本来就单一，聚类无意义
        return mask

    # 保护规则：只有"端点簇是白色系且占比小"才剔除异色块。
    #   - 白色 hold 低饱和（S 中位 < 75）且高亮（V 中位 > 140）
    #   - 彩色 hold（蓝/橙/紫…）S 高，不该聚类剔除（9 图回归的根因：
    #     k-means 把纯色 mask 拆成明暗簇，端点落在非主簇 → 面积骤减判错）
    #   - 端点簇占 mask 比例 > 25% → mask 颜色本就以端点色为主，无需剔除
    ep_pix_all = pix[labels == ep_cluster]
    ep_s_med = float(np.median(ep_pix_all[:, 1]))
    ep_v_med = float(np.median(ep_pix_all[:, 2]))
    ep_ratio = ep_n / n
    is_whiteish = (ep_s_med < 75) and (ep_v_med > 140)
    if ep_ratio > 0.25 or not is_whiteish:
        return mask

    # 只保留端点簇
    m2 = np.zeros_like(mask)
    m2[ys[labels == ep_cluster], xs[labels == ep_cluster]] = True
    return m2


# ============================================================
# mask → HSV 颜色统计（在 hold mask 内采多点 patch 圆形均值，借鉴 route_color）
# ============================================================
def _robust_hsv_from_mask(image_bgr: np.ndarray, mask: np.ndarray,
                          exclude_mask: np.ndarray | None = None) -> tuple[float, float, float]:
    """
    image_bgr: HxWx3 uint8
    mask: HxW bool（hold 区域，EdgeSAM 切出的"端点附近"区域）
    exclude_mask: HxW bool 可选 — exclude_mask 内像素不进颜色统计

    返回 (h, s, v)：hold 的代表色

    策略：
      - 在 mask 内只统计"不在 exclude_mask 内"的像素：
        当 provide exclude_mask=person_mask，hand+hold mask 内的人体皮肤/衣
        服像素被剔除 → 颜色统计只反映 hold 像素的真实颜色。
      - 如果 mask 像素太少 (<20) → 返回 (0,0,0)
      - 在 mask 的 bbox 中均匀采 5x5 网格点，每点 5x5 patch：
          - 不满足 S>=35/V>=40 的"灰白"像素参与"白岩点"判定
          - 满足的进入色相聚合
      - 如果"灰白像素"占 mask 大部分 (>50%) 且平均 V 高 → 返回 (0, 0, mean_v)（标志"白"）
      - 否则用色相圆形均值聚合
    """
    # 用 mask & ~exclude_mask 限定参与颜色统计的像素（剔出 person_mask 内的身体）
    if exclude_mask is not None:
        mask_for_color = mask & ~exclude_mask.astype(bool)
    else:
        mask_for_color = mask

    ys, xs = np.where(mask_for_color)
    if len(xs) < 20:
        # 退回到原始 mask（剔除后没足够像素）
        ys, xs = np.where(mask)
        if len(xs) < 20:
            return (0.0, 0.0, 0.0)
        mask_for_color = mask

    h_img, w_img = mask.shape

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask_pixels = hsv[ys, xs]  # (N, 3)
    mask_pixels_v = mask_pixels[:, 2]

    # ------ 白岩点判定 ------
    # 路径 A：绝对白（mask 内过半像素 S<35 且 V>140）——纯白 hold 直判
    white_mask = (mask_pixels[:, 1] < 35) & (mask_pixels[:, 2] > 140)
    white_ratio = float(white_mask.sum()) / len(mask_pixels)
    if white_ratio > 0.50:
        mean_v = float(np.mean(mask_pixels_v[white_mask])) if white_mask.sum() else 200.0
        return (0.0, 0.0, mean_v)

    # 路径 B：相对白 —— 白色 hold 是墙上"最不饱和"的物体。
    # 暖光/阴影下白色 hold 的 S 常在 35-75（不是绝对 <35），且 hold 只占
    # mask 一部分（其余是墙），所以"mask 内绝对占比"判不出来。改用相对判据：
    #   mask 饱和度中位数 << 环境环（mask 外扩 20px）→ 白色 hold
    #   且 mask 绝对 S 中位数 < 75（彩色 hold 即使相对低饱和也不低于此）
    #   且 mask 亮度不显著暗于环境（白 hold 是亮物体）
    # 实测 9.jpg R_foot (S_ratio=0.47, maskS=62, V_ratio=0.98) 判白；
    # 8.jpg 蓝 hold (S_ratio=0.50 但 maskS=114) 被绝对 S 挡掉，不会误判。
    if exclude_mask is not None:
        kernel = np.ones((20, 20), np.uint8)
        dil = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
        ring = dil & ~mask & ~exclude_mask.astype(bool)
        if ring.sum() > 50:
            ring_pix = hsv[ring]
            r_s = float(np.median(ring_pix[:, 1]))
            r_v = float(np.median(ring_pix[:, 2]))
            m_s = float(np.median(mask_pixels[:, 1]))
            m_v = float(np.median(mask_pixels[:, 2]))
            if (r_s > 20 and m_s / r_s < 0.60
                    and m_s < 75 and m_v >= r_v * 0.70):
                mean_v = float(np.mean(mask_pixels_v))
                return (0.0, 0.0, mean_v)

    # ------ bbox 5x5 网格采样的色相聚合 ------
    bx0, bx1 = xs.min(), xs.max()
    by0, by1 = ys.min(), ys.max()
    bw = max(1, bx1 - bx0)
    bh = max(1, by1 - by0)

    grid_n = 5
    pts: list[tuple[int, int]] = []
    for gy in range(grid_n):
        for gx in range(grid_n):
            cx_g = bx0 + (gx + 0.5) * bw / grid_n
            cy_g = by0 + (gy + 0.5) * bh / grid_n
            ix = int(round(cx_g))
            iy = int(round(cy_g))
            if 2 <= ix < w_img - 2 and 2 <= iy < h_img - 2:
                # patch 中心 5x5 检查：用 mask_for_color (排除身体) + mask
                # patch 中"hold 占比 >40%" → 这点可能采到 hold 颜色
                patch_hold = mask_for_color[max(0, iy - 2):iy + 3, max(0, ix - 2):ix + 3]
                patch_total = mask[max(0, iy - 2):iy + 3, max(0, ix - 2):ix + 3]
                if patch_total.size == 0:
                    continue
                if patch_hold.sum() / patch_total.size > 0.20:
                    pts.append((ix, iy))
    if not pts:
        return (0.0, 0.0, 0.0)

    hues, sats, vals = [], [], []
    for ix, iy in pts:
        # patch 是以 (ix, iy) 中心 5x5 — 直接采样 HSV，不考虑 mask_for_color 在 patch 内
        patch = hsv[iy - 2:iy + 3, ix - 2:ix + 3].reshape(-1, 3)
        keep = (patch[:, 1] >= 35) & (patch[:, 2] >= 40)
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
        # 没采到任何高饱和像素 — mask 内是浅色，用全部像素均值
        mean_v = float(np.mean(mask_pixels_v))
        mean_s = float(np.mean(mask_pixels[:, 1]))
        if mean_s < 35 and mean_v > 140:
            return (0.0, 0.0, mean_v)
        return (0.0, 0.0, mean_v)
    hr2 = np.array(hues) * (2 * np.pi / 180.0)
    h_final = float(np.degrees(np.arctan2(
        np.mean(np.sin(hr2)), np.mean(np.cos(hr2))
    )) / 2.0) % 180.0
    s_final = float(np.median(sats))
    v_final = float(np.median(vals))
    # 粉红/紫边界修正：hue 在粉红区 (>=150) 时，S 中位是判定粉红 vs 紫的关键。
    # 若 S 中位 < 120 判粉红。但 mask 内若混入高饱和杂质，中位 S 会被拉高。
    # 用"H>=150 像素的 S 中位"而非全 mask —— 更贴近 hold 本身的饱和度。
    if h_final >= route_color.PINK_HUE_MIN:
        hi_pix = mask_pixels[mask_pixels[:, 0] >= route_color.PINK_HUE_MIN]
        if len(hi_pix) >= 30:
            s_hi = float(np.median(hi_pix[:, 1]))
            # 粉红 hold 通常低饱和：S 中位 < 120 → 粉红；否则保持原值走紫 bin
            if s_hi < route_color.PINK_SAT_MAX:
                s_final = s_hi
    return (h_final, s_final, v_final)

