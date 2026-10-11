"""
compare_pairs.py — 对指定两对(视频帧 ↔ 照片)做详细对比

输出每对的:
  - DINOv2 cosine 相似度
  - ALIKED good matches / Lowe ratio 阈值后
  - RANSAC inliers / ratio / reproj error
  - 系统硬门控判定(MATCH / REJECT)+ 各阈值实时值
"""
from __future__ import annotations

import cv2
from dataclasses import dataclass

from rock_wall_retriever import (
    LOWE_RATIO,
    MIN_GOOD_MATCHES,
    MIN_INLIER_RATIO,
    MIN_INLIERS,
    RANSAC_REPROJ_THRESHOLD,
    MatchResult,
    RockWallRetriever,
)
from _paths import DINO_MODEL_DIR, FRAME_DIR_3O, PHOTO_DIR


PAIRS = [
    ("frame_0090s.jpg",     "IMG_20260808_193612.jpg"),
    ("frame_0150s.jpg",     "IMG_20260808_193659.jpg"),
]


@dataclass
class CompareResult:
    results: list[MatchResult]
    good_matches: int
    q_kp_count: int
    c_kp_count: int


def detail_compare(query_path: str, candidate_path: str) -> CompareResult:
    """跑两阶段检索并返回每一步细节"""
    # 用最小 retriever 只索引 1 个候选
    r = RockWallRetriever(dino_model_path=DINO_MODEL_DIR)
    candidate_name = candidate_path
    r.index_frame(candidate_name, candidate_path)
    r.build_index()
    results = r.search(query_path)

    # 手工算 Lowe ratio 后的 good matches 数量(便于报告)
    q_kp, q_desc = r._get_aliked(query_path)
    c_kp, c_desc = r._get_aliked(candidate_path)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn = bf.knnMatch(q_desc, c_desc, k=2)
    good = [m for p in knn if len(p) == 2 and p[0].distance < LOWE_RATIO * p[1].distance
            for m in [p[0]]]

    return CompareResult(
        results=results,
        good_matches=len(good),
        q_kp_count=len(q_kp),
        c_kp_count=len(c_kp),
    )


def main() -> int:
    print("=" * 78)
    print(f"硬门控阈值实时值:")
    print(f"  MIN_GOOD_MATCHES={MIN_GOOD_MATCHES}  MIN_INLIERS={MIN_INLIERS}  "
          f"MIN_INLIER_RATIO={MIN_INLIER_RATIO}  RANSAC_THRESHOLD={RANSAC_REPROJ_THRESHOLD}")
    print("=" * 78)

    for query_name, candidate_name in PAIRS:
        # 拼接完整路径
        if query_name.startswith("frame_"):
            query_path = f"{FRAME_DIR_3O}\\{query_name}"
        else:
            query_path = f"{PHOTO_DIR}\\{query_name}"
        candidate_path = f"{PHOTO_DIR}\\{candidate_name}"

        print(f"\n┌─ 对比 ─────────────────────────────────────────────")
        print(f"│  查询: {query_path}")
        print(f"│  候选: {candidate_path}")
        print(f"└─────────────────────────────────────────────────────")

        info = detail_compare(query_path, candidate_path)
        r = info.results[0]

        print(f"\n  DINOv2 cosine:        {r.dino_score:.4f}")
        print(f"  ALIKED 关键点 (查询): {info.q_kp_count}")
        print(f"  ALIKED 关键点 (候选): {info.c_kp_count}")
        print(f"  Lowe ratio 后 good:   {info.good_matches}  "
              f"(阈值 >= {MIN_GOOD_MATCHES})")
        print(f"  RANSAC inliers:       {r.inliers}  "
              f"(阈值 >= {MIN_INLIERS})")
        print(f"  RANSAC ratio:         {r.inlier_ratio:.3f}  "
              f"(阈值 >= {MIN_INLIER_RATIO})")

        verdict = "✅ MATCH" if r.is_match else "❌ REJECT"
        print(f"\n  系统判定: {verdict}")

        # 解释为什么 REJECT(如果是)
        if not r.is_match:
            reasons = []
            if info.good_matches < MIN_GOOD_MATCHES:
                reasons.append(f"good_matches={info.good_matches} < {MIN_GOOD_MATCHES}")
            if r.inliers < MIN_INLIERS:
                reasons.append(f"inliers={r.inliers} < {MIN_INLIERS}")
            if r.inlier_ratio < MIN_INLIER_RATIO:
                reasons.append(f"ratio={r.inlier_ratio:.3f} < {MIN_INLIER_RATIO}")
            if reasons:
                print(f"  REJECT 原因: {'; '.join(reasons)}")

    print("\n" + "=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())