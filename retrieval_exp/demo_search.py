"""
demo_search.py — img/ 下所有 JPG 照片 leave-one-out 验证 RockWallRetriever

数据来源:自动扫描 img/ 下所有 *.jpg
  (HEIC 转换后共 34 张,用户已确认全部保留)

Ground truth(用户已确认):
  - 193724 ↔ 193731:同一面墙(连拍,肯定要 MATCH)
  - 193732 ↔ 193734:不是同一面墙(系统曾经误检,阈值收紧后应被消除)
  - 193637 ↔ 193659:有部分重合但不算同一面墙(系统曾经误检,应被消除)
  - 其他 6 对(193715↔193717 / 193648↔193650 / 193628↔193630 /
          193746↔193751 / 193617↔193621 / 193724↔193731):用户已确认是真同墙
"""
from pathlib import Path

from _paths import DINO_MODEL_DIR, IMG_DIR
from rock_wall_retriever import MatchResult, RockWallRetriever

# 用户反馈的 ground truth(frame_id 与 p.stem 一致,即 IMG_20260808_<时间戳>)
SAME_WALL_PAIRS: set[tuple[str, str]] = {
    ("IMG_20260808_193724", "IMG_20260808_193731"),
    ("IMG_20260808_193715", "IMG_20260808_193717"),
    ("IMG_20260808_193648", "IMG_20260808_193650"),
    ("IMG_20260808_193628", "IMG_20260808_193630"),
    ("IMG_20260808_193746", "IMG_20260808_193751"),
    ("IMG_20260808_193617", "IMG_20260808_193621"),
}
NOT_SAME_WALL_PAIRS: set[tuple[str, str]] = {
    ("IMG_20260808_193732", "IMG_20260808_193734"),   # 完全不同的墙
    ("IMG_20260808_193637", "IMG_20260808_193659"),   # 部分重合但不算同墙
}


def discover_photos(img_dir: Path) -> list[tuple[str, str]]:
    """自动扫描 img/ 下所有 .jpg,返回 [(frame_id, full_path), ...]"""
    photos: list[tuple[str, str]] = []
    for p in sorted(img_dir.glob("*.jpg")):
        photos.append((p.stem, str(p)))
    return photos


def evaluate(matches_by_query: dict[str, list[MatchResult]]) -> tuple[int, int, int, int]:
    """对照 ground truth 评估:返回 (tp, fp, tn, fn),按独立配对去重统计"""
    # 收集所有被判定为同墙的独立配对
    detected_pairs: set[tuple[str, str]] = set()
    for query_id, results in matches_by_query.items():
        for r in results:
            if not r.is_match:
                continue
            a, b = sorted([query_id, r.frame_id])
            detected_pairs.add((a, b))

    tp = len(detected_pairs & SAME_WALL_PAIRS)
    fp = len(detected_pairs & NOT_SAME_WALL_PAIRS)
    fn = len(SAME_WALL_PAIRS - detected_pairs)
    tn = len(NOT_SAME_WALL_PAIRS - detected_pairs)
    return tp, fp, tn, fn


def main() -> int:
    photos = discover_photos(IMG_DIR)
    print("=" * 78)
    print(f"RockWallRetriever — {len(photos)} 张照片 leave-one-out 验证")
    print("=" * 78)

    # 一次性建索引(模型只加载一次,后续查询用 exclude_frame_id 实现 leave-one-out)
    retriever = RockWallRetriever(dino_model_path=DINO_MODEL_DIR)
    print(f"\n索引 {len(photos)} 张照片到 retriever ...")
    for frame_id, path in photos:
        retriever.index_frame(frame_id, path)
    retriever.build_index()
    print("索引构建完成\n")

    # leave-one-out:每个 photo 作为查询,排除自身,看命中情况
    matches_by_query: dict[str, list[MatchResult]] = {}
    for query_id, path in photos:
        results = retriever.search(path, exclude_frame_id=query_id)
        matches_by_query[query_id] = results

    # ===== 详情:每个查询的 Top-3 候选 =====
    print(f"{'=' * 78}\n详情:每个查询的 Top-3 候选\n{'=' * 78}")
    for query_id, _ in photos:
        results = matches_by_query[query_id]
        print(f"\n[查 {query_id}]")
        for r in results[:3]:  # Top-3
            tag = "MATCH " if r.is_match else "reject"
            print(
                f"  → {r.frame_id:7s} [{tag}] "
                f"inliers={r.inliers:4d} ratio={r.inlier_ratio:.3f} "
                f"dino={r.dino_score:.4f}"
            )

    # ===== 同墙配对汇总 =====
    print(f"\n{'=' * 78}\n系统判定的同墙配对(只看 is_match=True)\n{'=' * 78}")
    seen_pairs: set[tuple[str, str]] = set()
    same_wall_pairs: list[tuple[str, str, int]] = []
    for query_id, results in matches_by_query.items():
        for r in results:
            if not r.is_match:
                continue
            a, b = sorted([query_id, r.frame_id])
            pair: tuple[str, str] = (a, b)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            same_wall_pairs.append((pair[0], pair[1], r.inliers))

    if same_wall_pairs:
        same_wall_pairs.sort(key=lambda x: -x[2])
        print(f"共 {len(same_wall_pairs)} 对:")
        for a, b, inl in same_wall_pairs:
            tag = ""
            if (a, b) in SAME_WALL_PAIRS:
                tag = " ✅ 真同墙(用户已确认)"
            elif (a, b) in NOT_SAME_WALL_PAIRS:
                tag = " ❌ 误检(用户已标记非同墙)"
            else:
                tag = "    (用户未标注)"
            print(f"  {a} ↔ {b}  (inliers={inl}){tag}")
    else:
        print("(无)")

    # ===== Ground truth 评估 =====
    print(f"\n{'=' * 78}\nGround truth 评估\n{'=' * 78}")
    tp, fp, tn, fn = evaluate(matches_by_query)
    print(f"  已知 ground truth: {len(SAME_WALL_PAIRS)} 对真同墙, {len(NOT_SAME_WALL_PAIRS)} 对非同墙")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"  Precision={precision:.2%}  Recall={recall:.2%}")
    ok = (tp == len(SAME_WALL_PAIRS) and fp == 0)
    print(f"  总体: {'✅ 全部判对' if ok else '❌ 存在误判'}")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())