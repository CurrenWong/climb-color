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


# 默认 checkpoint / 设备
DEFAULT_CHECKPOINT = "weights/edge_sam.pth"
DEVICE_CPU = torch.device("cpu")


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
        n: int = 4, radius: float = 35.0,
    ) -> np.ndarray:
        """
        在端点周围 + 远离身体方向 / 紧贴身体等位置制造 negative points，
        把"端点附近的身体"从 hold mask 中剔除。
        返回 shape (n, 2) 的 array，labels 由调用方填 0。
        没身体信息时（极少见）：围一个半径 radius 的圆上 n 个点。
        """
        pts = []
        if body_cx is not None and body_cy is not None:
            # 从端点向身体方向的反方向 30px 拿 1 点（身体侧）：
            ux, uy = body_cx - px, body_cy - py
            norm = np.hypot(ux, uy) or 1.0
            ux, uy = ux / norm, uy / norm
            inward_x = px + 30 * ux
            inward_y = py + 30 * uy
            # 再四周加点（垂直端点方向上的"墙"）：
            for theta in (np.pi / 2, -np.pi / 2):
                rx = px + radius * (ux * np.cos(theta) - uy * np.sin(theta))
                ry = py + radius * (ux * np.sin(theta) + uy * np.cos(theta))
                pts.extend([(inward_x, inward_y), (rx, ry)])
        else:
            for theta in np.linspace(0, 2 * np.pi, n, endpoint=False):
                pts.append((px + radius * np.cos(theta),
                            py + radius * np.sin(theta)))
        return np.array(pts, dtype=np.float64)

    def segment_endpoint(
        self,
        px: float, py: float,
        body_cx: float | None = None, body_cy: float | None = None,
        num_multimask_outputs: int = 3,
        use_stability_score: bool = True,
    ) -> dict:
        """
        对 (px, py) 这个端点做 hold 分割。

        返回 dict:
          mask: HxW bool（端点所在 hold 的 mask，原图坐标系）
          score: float（mask 置信度）
          h, s, v: float（mask 内颜色统计）
          color_name: str（中文色名）
          area: int（mask 像素数）
        """
        if not self.predictor.is_image_set:
            raise RuntimeError("先调用 set_image() 再 segment_endpoint()")

        # 正点：1 个端点；负点：4 个（同方法）
        neg = self._make_negative_prompts(px, py, body_cx, body_cy, n=4)
        point_coords = np.vstack([
            [[px, py]],
            neg,
        ])
        point_labels = np.array([1] + [0] * len(neg), dtype=np.int64)

        masks, scores, _ = self.predictor.predict(
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

        # 过滤：必须包含端点、score ≥ 0.4、面积 < 30% 图（避免"整墙" mask）
        candidates = []
        for i, m in enumerate(masks):
            area = int(m.sum())
            if area < 100:                       # 太碎
                continue
            if area > img_area * 0.30:           # 太大 — 多半是"墙+hold"一起切了
                continue
            if not m[iy, ix]:
                continue
            if float(scores[i]) < 0.4:
                continue
            candidates.append((float(scores[i]), area, m))

        # 当所有候选都被"包含端点"过滤掉，兜底：
        if not candidates:
            # 在"包含端点 + score ≥ 0.3"的候选里选（放低 score 门槛）
            soft = []
            for i, m in enumerate(masks):
                area = int(m.sum())
                if area < 50 or area > img_area * 0.45:
                    continue
                if not m[iy, ix]:
                    continue
                if float(scores[i]) < 0.3:
                    continue
                soft.append((float(scores[i]), area, m))
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
        candidates.sort(key=lambda c: (c[1], -c[0]))
        chosen_score, _, chosen = candidates[0]

        # 用 mask 取 hold 颜色
        h, s, v = _robust_hsv_from_mask(self.last_image_bgr, chosen)
        # 白岩点：S<35 + V>140 → 直接判定为"白"（绕过 hue_to_color_name 对 H=0 的歧义）
        color_name = "白" if (s < 35 and v > 140) else route_color.hue_to_color_name(h, s)

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
# mask → HSV 颜色统计（在 hold mask 内采多点 patch 圆形均值，借鉴 route_color）
# ============================================================
def _robust_hsv_from_mask(image_bgr: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    """
    image_bgr: HxWx3 uint8
    mask: HxW bool（hold 区域）
    返回 (h, s, v)：hold 的代表色

    策略：
      - 如果 mask 像素太少 (<20) → 返回 (0,0,0)
      - 在 mask 的 bbox 中均匀采 5x5 网格点，每点 5x5 patch：
          - 不满足 S>=35/V>=40 的"灰白"像素参与"白岩点"判定
          - 满足的进入色相聚合
      - 如果"灰白像素"占 mask 大部分 (>50%) 且平均 V 高 → 返回 (0, 0, mean_v)（标志"白"）
      - 否则用色相圆形均值聚合
    """
    ys, xs = np.where(mask)
    if len(xs) < 20:
        return (0.0, 0.0, 0.0)
    h_img, w_img = mask.shape

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask_pixels = hsv[ys, xs]  # (N, 3)
    mask_pixels_v = mask_pixels[:, 2]

    # ------ 白岩点判定 ------
    white_mask = (mask_pixels[:, 1] < 35) & (mask_pixels[:, 2] > 140)
    white_ratio = float(white_mask.sum()) / len(mask_pixels)
    if white_ratio > 0.50:
        # mask 内大半像素是"低饱和高亮" → 白色 hold
        mean_v = float(np.mean(mask_pixels_v[white_mask])) if white_mask.sum() else 200.0
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
                patch_mask = mask[max(0, iy - 2):iy + 3, max(0, ix - 2):ix + 3]
                if patch_mask.mean() > 0.4:
                    pts.append((ix, iy))
    if not pts:
        return (0.0, 0.0, 0.0)

    hues, sats, vals = [], [], []
    for ix, iy in pts:
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
    return (h_final, float(np.median(sats)), float(np.median(vals)))
