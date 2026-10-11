"""
rock_wall_retriever.py — 两阶段岩壁检索(生产级 MVP)

Stage 1 (召回):  DINOv2-S/14 全局特征 + FAISS IndexFlatIP → Top-K 候选
Stage 2 (精排):  ALIKED 局部特征 + Lowe ratio + RANSAC 单应性验证
硬门控:          inliers >= MIN_INLIERS AND ratio >= MIN_INLIER_RATIO 才判 "同一面墙"

相对 v1 文档的技术修正(已在我们 3 张照片的实测中验证):
1. 用 ALIKED 替代 SuperPoint:对重复纹理(岩点)鲁棒性更好,且 SuperPoint 官方
   torch.hub 路径已废弃(项目里用的是 kornia 0.8.3,内置 ALIKED 已验证可用)
2. RANSAC 硬门控,不与 DINOv2 做加权求和 —— 实测 DINOv2 cos 0.83 的"不同墙"
   仍然会得较高加权分,造成误召回;几何不通过就直接否定
3. Stage 2 候选数 Top-K=3(文档原案 Top-20 在纯 CPU 上跑不动,见评估报告)
4. RANSAC threshold 用 5.0(我们 super_point_test.py 已验证比默认 3.0 更稳)
5. DINOv2 召回用宽阈值 cos>=0(取 Top-K 即可),不卡阈值 —— 让 Stage 2 决定最终答案
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import faiss
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

import kornia.feature as KF


# ========== 默认配置(可根据场景调整) ==========
DEVICE = "cpu"
IMG_SIZE = (640, 480)                  # 局部特征输入尺寸(灰度)
DINO_IMG_SIZE = 518                    # DINOv2 原生训练分辨率(14 的倍数)
DEFAULT_TOP_K = 5                      # Stage 2 跑局部特征的候选数(性能预算关键;Top-K=3 在 34 张图实测中漏检了 1 对)
# 硬门控:三者同时满足才判"同一面墙"
# (阈值依据:在 34 张照片 + 视频帧实测中,
#  - 真同墙配对 inliers >= 22(good_matches >= 36,被视频中的人遮挡会降到 22)
#  - 误检配对 inliers <= 18(good_matches <= 28)
#  此阈值 MIN_INLIERS=20 留出余量,既能包容遮挡,又能挡掉误检)
MIN_INLIERS = 20                       # RANSAC 内点绝对数量下限
MIN_INLIER_RATIO = 0.40                # RANSAC 内点比例下限
MIN_GOOD_MATCHES = 20                  # Lowe ratio test 后留下的 good match 数下限
LOWE_RATIO = 0.75                      # Lowe ratio test 阈值
RANSAC_REPROJ_THRESHOLD = 5.0          # RANSAC 重投影误差阈值(像素)


@dataclass
class FrameRecord:
    """索引中一个视频帧的元数据"""
    frame_id: str
    image_path: str
    dino_embedding: np.ndarray | None = None   # (384,) L2-normalized
    video_id: str | None = None                # 如果来自视频,记录视频 ID
    timestamp_ms: int | None = None            # 视频内的毫秒时间戳


@dataclass
class MatchResult:
    """search() 返回的命中记录"""
    frame_id: str
    image_path: str
    dino_score: float            # Stage 1 cosine 相似度
    inliers: int                 # Stage 2 RANSAC 内点数
    inlier_ratio: float          # Stage 2 内点比例
    is_match: bool               # 是否通过硬门控
    video_id: str | None = None  # 命中的帧来自哪个视频(若来自视频)
    timestamp_ms: int | None = None  # 视频内毫秒时间戳(若来自视频)


class RockWallRetriever:
    """两阶段岩壁检索器:DINOv2 召回 + ALIKED/RANSAC 验证

    使用流程:
        retriever = RockWallRetriever(dino_model_path=".../dinov2-small")
        for vid_frame in video_keyframes:
            retriever.index_frame(frame_id, image_path)
        retriever.build_index()
        results = retriever.search(query_image_path)
    """

    def __init__(
        self,
        dino_model_path: str,
        device: str = DEVICE,
        min_inliers: int = MIN_INLIERS,
        min_inlier_ratio: float = MIN_INLIER_RATIO,
        min_good_matches: int = MIN_GOOD_MATCHES,
    ) -> None:
        self.device = device
        self.min_inliers = min_inliers
        self.min_inlier_ratio = min_inlier_ratio
        self.min_good_matches = min_good_matches

        # DINOv2 全局特征
        self.dino_processor = AutoImageProcessor.from_pretrained(dino_model_path)
        self.dino_model = AutoModel.from_pretrained(dino_model_path).to(self.device).eval()

        # ALIKED 局部特征(首次运行会下载 ~2.6MB aliked-n16.pth)
        self.aliked_model = KF.ALIKED.from_pretrained(device=torch.device(self.device)).eval()

        # FAISS 索引
        self.dim: int = 384
        self.index: faiss.IndexFlatIP | None = None

        # 帧元数据 + ALIKED 特征缓存(避免对同一张图重复跑推理)
        self.frames: list[FrameRecord] = []
        self._alikd_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # =========================================================
    # 离线建库
    # =========================================================

    def index_frame(
        self,
        frame_id: str,
        image_path: str,
        *,
        video_id: str | None = None,
        timestamp_ms: int | None = None,
    ) -> None:
        """把一张帧加入索引(只算 DINOv2 全局特征;ALIKED 检索时再算并缓存)

        若 frame 来自视频,传 video_id 和 timestamp_ms,检索返回时一并带回,方便 API
        返回"视频跳转时间戳"。
        """
        if not Path(image_path).is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        img = self._load_pil(image_path)
        emb = self._extract_dino(img)
        self.frames.append(FrameRecord(
            frame_id=frame_id,
            image_path=image_path,
            dino_embedding=emb,
            video_id=video_id,
            timestamp_ms=timestamp_ms,
        ))

    def build_index(self) -> None:
        """所有帧入完后,build FAISS 索引"""
        if not self.frames:
            raise ValueError("No frames indexed. Call index_frame() first.")
        # 过滤未初始化的 embedding(index_frame 已保证全部非空,这里仅为了满足类型推断)
        embeddings: list[np.ndarray] = [
            f.dino_embedding for f in self.frames if f.dino_embedding is not None
        ]
        vectors = np.stack(embeddings).astype(np.float32)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vectors)

    # =========================================================
    # 在线检索
    # =========================================================

    def search(
        self,
        query_image_path: str,
        top_k_recall: int = DEFAULT_TOP_K,
        exclude_frame_id: str | None = None,
        min_cosine: float = 0.0,
    ) -> list[MatchResult]:
        """两阶段检索,返回按 is_match → inliers → dino_score 降序的结果列表

        Args:
            query_image_path: 查询图片路径
            top_k_recall: Stage 2 候选数
            exclude_frame_id: 若指定,临时从索引中移除该 frame_id(leave-one-out 用)
            min_cosine: 若 >0,Stage 1 用余弦相似度阈值筛选候选(覆盖 top_k_recall)
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        if not Path(query_image_path).is_file():
            raise FileNotFoundError(f"Query image not found: {query_image_path}")

        # 选出本次实际参与召回的帧(exclude_frame_id 用于 leave-one-out)
        active_frames = [f for f in self.frames if f.frame_id != exclude_frame_id]
        if not active_frames:
            return []

        # ---- Stage 1: DINOv2 全局召回(在临时索引上跑) ----
        q_img = self._load_pil(query_image_path)
        q_emb = self._extract_dino(q_img).reshape(1, -1).astype(np.float32)
        # 过滤未初始化的 embedding(index_frame 已保证全部非空,这里仅为了满足类型推断)
        active_embeddings: list[np.ndarray] = [
            f.dino_embedding for f in active_frames if f.dino_embedding is not None
        ]
        active_vectors = np.stack(active_embeddings).astype(np.float32)
        temp_index = faiss.IndexFlatIP(self.dim)
        temp_index.add(active_vectors)
        # 设了 min_cosine 就拉满所有候选,然后在循环里按 cosine 过滤
        k = len(active_frames) if min_cosine > 0 else min(top_k_recall, len(active_frames))
        dino_scores, candidate_indices = temp_index.search(q_emb, k=k)

        # ---- Stage 2: 对每个候选跑几何验证 ----
        query_kp, query_desc = self._get_aliked(query_image_path)
        results: list[MatchResult] = []
        for score, idx in zip(dino_scores[0].tolist(), candidate_indices[0].tolist()):
            if idx < 0:
                continue
            if score < min_cosine:  # 余弦相似度阈值过滤
                continue
            cand = active_frames[idx]
            inliers, ratio = self._verify_geometry(query_kp, query_desc, cand.image_path)
            results.append(MatchResult(
                frame_id=cand.frame_id,
                image_path=cand.image_path,
                dino_score=float(score),
                inliers=inliers,
                inlier_ratio=ratio,
                is_match=(inliers >= self.min_inliers and ratio >= self.min_inlier_ratio),
                video_id=cand.video_id,
                timestamp_ms=cand.timestamp_ms,
            ))

        # 排序:match 优先,内点多优先,DINOv2 高优先
        results.sort(key=lambda r: (r.is_match, r.inliers, r.dino_score), reverse=True)
        return results

    # =========================================================
    # 内部:模型推理
    # =========================================================

    def _load_pil(self, image_path: str) -> Image.Image:
        return Image.open(image_path).convert("RGB")

    def _extract_dino(self, img: Image.Image) -> np.ndarray:
        """DINOv2 提取 384 维 L2 归一化全局向量"""
        inputs = self.dino_processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.dino_model(**inputs)
        # Dinov2Model 在 transformers >= 5.x 提供 pooler_output(shape (1, 384))
        # 在更老版本上需要退回到 last_hidden_state[:, 0, :]
        if outputs.pooler_output is not None:
            emb = outputs.pooler_output[0].cpu().numpy().astype(np.float32)
        else:
            emb = outputs.last_hidden_state[0, 0, :].cpu().numpy().astype(np.float32)
        norm = float(np.linalg.norm(emb))
        if norm > 0:
            emb = emb / norm
        return emb

    def _get_aliked(self, image_path: str) -> tuple[np.ndarray, np.ndarray]:
        """ALIKED 提取关键点和描述子,带内存缓存"""
        if image_path in self._alikd_cache:
            return self._alikd_cache[image_path]
        kps, desc = self._extract_aliked(image_path)
        self._alikd_cache[image_path] = (kps, desc)
        return kps, desc

    def _extract_aliked(self, image_path: str) -> tuple[np.ndarray, np.ndarray]:
        """ALIKED 实际推理;空图/失败时返回空数组"""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return (np.empty((0, 2), dtype=np.float32),
                    np.empty((0, 256), dtype=np.float32))
        # 收窄 cv2.imread 返回的 MatLike | ndarray 联合类型,以匹配 cv2.resize 的 ndarray 重载
        img = np.asarray(img)
        img = cv2.resize(img, IMG_SIZE)
        tensor = torch.from_numpy(img).float() / 255.0
        tensor = tensor.unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, H, W)
        with torch.no_grad():
            feats_list = self.aliked_model(tensor)
        feats = feats_list[0]
        kps = feats.keypoints.cpu().numpy().astype(np.float32)         # (N, 2) 像素坐标 [x, y]
        desc = feats.descriptors.cpu().numpy().astype(np.float32)       # (N, D),已 L2 归一化
        return kps, desc

    def _verify_geometry(
        self,
        query_kp: np.ndarray,
        query_desc: np.ndarray,
        candidate_image_path: str,
    ) -> tuple[int, float]:
        """对称 Lowe ratio + RANSAC 几何验证。返回 (inliers, inlier_ratio)

        关键设计:同一对图片,query 方向不同会得到不同的 knn good matches(因为
        knnMatch 是非对称的),进而 RANSAC inliers 不同。为保证结果方向无关,
        这里从 query↔candidate 两个方向各做一次完整 knn+Lowe+RANSAC,取 inliers 的
        较大者作为最终值。这样:
          - 真同墙(双向几何一致):两个方向都得到高 inliers
          - 部分遮挡或角度差异:取较好那个方向的结果
          - 不同墙:两个方向都不会超过阈值

        注意:不取 union 扩大 match 集合(否则 RANSAC 会把噪声当一致,引入误检)。
        """
        cand_kp, cand_desc = self._get_aliked(candidate_image_path)
        if len(query_kp) < 4 or len(cand_kp) < 4:
            return 0, 0.0

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        def _one_direction(q_kp: np.ndarray, q_desc: np.ndarray, c_kp: np.ndarray, c_desc: np.ndarray) -> tuple[int, float]:
            try:
                knn = bf.knnMatch(q_desc, c_desc, k=2)
            except cv2.error:
                return 0, 0.0
            good: list[cv2.DMatch] = []
            for m_pair in knn:
                if len(m_pair) == 2 and m_pair[0].distance < LOWE_RATIO * m_pair[1].distance:
                    good.append(m_pair[0])
            if len(good) < self.min_good_matches:
                return 0, 0.0
            src_pts = np.array([q_kp[m.queryIdx] for m in good], dtype=np.float32).reshape(-1, 1, 2)
            dst_pts = np.array([c_kp[m.trainIdx] for m in good], dtype=np.float32).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD)
            if mask is None or H is None:
                return 0, 0.0
            inliers = int(mask.sum())
            return inliers, inliers / len(good)

        # 两个方向各算一次,取 inliers 较大者
        inl1, ratio1 = _one_direction(query_kp, query_desc, cand_kp, cand_desc)
        inl2, ratio2 = _one_direction(cand_kp, cand_desc, query_kp, query_desc)
        if inl1 >= inl2:
            return inl1, ratio1
        return inl2, ratio2