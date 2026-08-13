"""
人体 mask 分割器：Ultralytics YOLO26 segmentation，COCO class 0 = person。

用途：在攀岩图片里识别攀岩者人体像素，把这个 mask 喂给 EdgeSAM 让它
知道"哪片是身体"，从而避免把手/脚/衣服/手表误当岩点。
"""
from __future__ import annotations

import numpy as np


# COCO class 0 (person)
PERSON_CLASS_ID = 0


class PersonMasker:
    """
    YOLO26 segmentation 包装。只保留 person class 的像素。

    用法:
        masker = PersonMasker(model_name="yolo26n-seg.pt")
        person_bool = masker.segment(image_bgr)   # HxW bool，True = 人体像素
    """

    def __init__(
        self,
        model_name: str = "yolo26n-seg.pt",
        device: str | None = None,
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.7,
        verbose: bool = False,
    ):
        """
        model_name: 默认 yolo26n-seg.pt（最小最快，COCO mAPmask 33.9）
                    其他选项: yolo26s-seg / m-seg / l-seg / x-seg
        device: None = 自动检测 cuda，否则 'cpu' / 'cuda'
        imgsz: 输入尺寸（默认 640）
        conf: 置信度阈值
        iou: NMS IoU 阈值
        """
        from ultralytics import YOLO  # 延迟 import，避免无 ultralytics 时 import 失败
        self.model = YOLO(model_name)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.verbose = verbose
        self._device = device

    def segment(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        返回 HxW bool mask：True = 人体像素。
        """
        h, w = image_bgr.shape[:2]
        # ultralytics 接受 BGR 或 RGB。我们传 BGR（与 OpenCV 一致）
        results = self.model.predict(
            image_bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self._device,
            verbose=self.verbose,
        )
        if not results or results[0].masks is None:
            return np.zeros((h, w), dtype=bool)
        result = results[0]

        # 取所有 class=0 (person) 的 mask，合并成一张人体 mask
        cls = result.boxes.cls.cpu().numpy().astype(int)  # (N,)
        if PERSON_CLASS_ID not in cls:
            return np.zeros((h, w), dtype=bool)

        # result.masks.data 是 (N, h', w') torch.uint8，0/1。resize 到原图。
        # 直接拿 numpy 转 HxW
        person_mask = np.zeros((h, w), dtype=bool)
        for i in range(len(cls)):
            if int(cls[i]) != PERSON_CLASS_ID:
                continue
            m = result.masks.data[i].cpu().numpy()
            if m.shape != (h, w):
                m = _resize_mask(m, (h, w))
            person_mask |= m.astype(bool)
        return person_mask


def _resize_mask(mask_2d: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """把 ultralytics 的 mask resize 到目标分辨率。0/1 重建阈值 0.5。"""
    import cv2
    out = cv2.resize(
        mask_2d.astype(np.uint8),
        (target_size[1], target_size[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return (out > 0.5)


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """mask 二值膨胀（边缘扩张 N px），让"贴近身体"也归到人体。"""
    if radius <= 0:
        return mask.astype(bool)
    import cv2
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    out = cv2.dilate(mask.astype(np.uint8), k, iterations=1)
    return out.astype(bool)


def sample_edge_points(person_mask: np.ndarray, n: int = 12,
                       max_radius: int = 80) -> np.ndarray:
    """
    在 person mask **边缘外** 均匀撒 n 个 negative prompt points。

    用来在 EdgeSAM prompt 时让 SAM 更清楚"端点附近哪些是身体、哪些是岩点"。
    返回 shape (n, 2) 的 np.ndarray；位置已是图像坐标。

    Strategy:
      - 找出 person mask 边缘的轮廓
      - 在轮廓外侧 30-80px 均匀采 n 个点（每隔 contour_len/n 个 contour 点取一个）
    """
    import cv2
    if person_mask.sum() < 50:
        return np.zeros((0, 2), dtype=np.float64)
    person_u8 = (person_mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(person_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros((0, 2), dtype=np.float64)
    all_pts = np.vstack(contours).reshape(-1, 2)
    if len(all_pts) < n:
        idx = np.arange(len(all_pts))
    else:
        idx = np.linspace(0, len(all_pts) - 1, n).astype(int)
    samples = all_pts[idx].astype(np.float64)  # (n, 2) 轮廓点

    # 用 2D 法向方向（粗略：从中心向外）膨胀
    ys, xs = np.where(person_mask)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    cx, cy = float(xs.mean()), float(ys.mean())
    ux = samples[:, 0] - cx
    uy = samples[:, 1] - cy
    norm = np.hypot(ux, uy)
    norm[norm < 1] = 1.0
    ux, uy = ux / norm, uy / norm
    out_pts = np.column_stack([
        samples[:, 0] + max_radius * 0.4 * ux,
        samples[:, 1] + max_radius * 0.4 * uy,
    ])
    return out_pts
