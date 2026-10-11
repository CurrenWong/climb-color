"""
analyze_video.py — 从视频抽帧、分析"视频里有几面墙"、按墙分文件夹

步骤:
1. FFmpeg 从视频每 5 秒抽一张帧(纯 CPU,无需 GPU),保留到 video_frames_cache/<video_stem>/
2. 把 34 张照片 + 抽出的视频帧一起入索引
3. 对每张视频帧:
   - 用 DINOv2 cosine >= 0.7 召回候选(替代原 Top-K=5,避免漏检)
   - 对通过的候选跑 ALIKED 几何验证
   - 命中了哪些照片 → 该帧属于哪面墙
4. 视频帧按"互相匹配"聚类(union-find)
5. 对每个聚类,union 所有帧匹配到的照片 → 已知墙 vs 新墙
6. 把每面墙的帧 copy 到 walls/<wall_dir_name>/ 目录

用法:python analyze_video.py <video_path>
"""
from __future__ import annotations

import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from _paths import (
    CACHE_ROOT,
    DINO_MODEL_DIR,
    DINO_RECALL_THRESHOLD,
    FRAME_INTERVAL_SEC,
    IMG_DIR,
)
from rock_wall_retriever import MatchResult, RockWallRetriever
from video_utils import extract_frames, get_video_duration


def is_photo_id(frame_id: str) -> bool:
    """照片 ID 形如 IMG_20260808_NNNNN,视频帧 ID 形如 frame_NNNNs"""
    return frame_id.startswith("IMG_")


@dataclass
class WallGroup:
    """一个聚类(一面墙)的所有帧 + 匹配到的照片"""
    items: list[tuple[int, str]]       # [(timestamp_sec, frame_id), ...]
    matched_photos: set[str]           # 该聚类中所有帧匹配到的照片 ID 集合

    @property
    def timestamps(self) -> list[int]:
        return [ts for ts, _ in self.items]

    @property
    def ts_range(self) -> str:
        ts = self.timestamps
        return f"{ts[0]}s" if len(ts) == 1 else f"{ts[0]}s~{ts[-1]}s"


# ========== 主流程 ==========

def main() -> int:
    import sys
    if len(sys.argv) != 2:
        print("用法: python analyze_video.py <视频路径>")
        return 1
    video_path = Path(sys.argv[1])
    if not video_path.is_file():
        print(f"视频不存在: {video_path}")
        return 1

    cache_dir = CACHE_ROOT / video_path.stem
    walls_dir = cache_dir / "walls"

    print("=" * 78)
    print(f"分析视频: {video_path.name}")
    print(f"  总时长: {get_video_duration(video_path):.1f}s")
    print(f"  抽帧间隔: {FRAME_INTERVAL_SEC}s")
    print(f"  DINOv2 召回阈值: cosine >= {DINO_RECALL_THRESHOLD}")
    print(f"  帧缓存目录: {cache_dir}")
    print(f"  按墙分文件夹输出: {walls_dir}")
    print("=" * 78)

    print(f"\n[1/4] FFmpeg 抽帧 ({FRAME_INTERVAL_SEC}s 间隔) ...")
    video_frames = extract_frames(video_path, cache_dir, FRAME_INTERVAL_SEC)
    print(f"  抽出 {len(video_frames)} 帧 → {cache_dir}")

    print(f"\n[2/4] 索引 {len(list(IMG_DIR.glob('*.jpg')))} 张照片 + {len(video_frames)} 张视频帧 ...")
    retriever = RockWallRetriever(dino_model_path=DINO_MODEL_DIR)
    for p in sorted(IMG_DIR.glob("*.jpg")):
        retriever.index_frame(p.stem, str(p))
    for ts, fp in video_frames:
        retriever.index_frame(fp.stem, str(fp))
    retriever.build_index()

    # ===== 各视频帧匹配详情 =====
    print(f"\n[3/4] 用 DINOv2 cosine >= {DINO_RECALL_THRESHOLD} 召回,跑几何验证 ...")
    frame_to_matches: dict[str, list[MatchResult]] = {}
    for ts, fp in video_frames:
        frame_id = fp.stem
        results = retriever.search(
            str(fp), exclude_frame_id=frame_id, min_cosine=DINO_RECALL_THRESHOLD
        )
        frame_to_matches[frame_id] = results

    print(f"\n{'=' * 78}\n各视频帧匹配结果\n{'=' * 78}")
    for ts, fp in video_frames:
        frame_id = fp.stem
        results = frame_to_matches[frame_id]
        photo_matches = [r for r in results if r.is_match and is_photo_id(r.frame_id)]
        video_matches = [r for r in results if r.is_match and not is_photo_id(r.frame_id)]
        print(f"\n  t={ts:3d}s  ({frame_id})")
        if not photo_matches:
            print("    → 未命中任何已知照片墙(视频独有)")
        else:
            for r in photo_matches:
                print(f"    → 命中 {r.frame_id:25s}  inliers={r.inliers:4d}  dino={r.dino_score:.4f}")
        if video_matches:
            ids = [r.frame_id for r in video_matches]
            print(f"    + 互匹配视频帧({len(video_matches)} 个): {', '.join(ids[:5])}{'...' if len(ids) > 5 else ''}")

    # ===== 聚类:视频帧按"互相匹配"分组 =====
    print(f"\n[4/4] 视频帧互匹配聚类 ...")
    sub_retriever = RockWallRetriever(dino_model_path=DINO_MODEL_DIR)
    for _ts, fp in video_frames:
        sub_retriever.index_frame(fp.stem, str(fp))
    sub_retriever.build_index()

    sub_pairs: set[tuple[str, str]] = set()
    for _ts, fp in video_frames:
        fid = fp.stem
        for r in sub_retriever.search(str(fp), exclude_frame_id=fid):
            if r.is_match:
                a, b = sorted([fid, r.frame_id])
                sub_pairs.add((a, b))

    all_frame_ids = [fp.stem for _ts, fp in video_frames]
    parent: dict[str, str] = {fid: fid for fid in all_frame_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in sub_pairs:
        union(a, b)

    raw_groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for ts, fp in video_frames:
        raw_groups[find(fp.stem)].append((ts, fp.stem))

    walls: list[WallGroup] = []
    for root, items in raw_groups.items():
        matched_photos: set[str] = set()
        for _ts, fid in items:
            for r in frame_to_matches[fid]:
                if r.is_match and is_photo_id(r.frame_id):
                    matched_photos.add(r.frame_id)
        walls.append(WallGroup(items=sorted(items), matched_photos=matched_photos))
    walls.sort(key=lambda w: w.timestamps[0])

    # ===== 输出墙的总结 + 按墙分文件夹 =====
    print(f"\n{'=' * 78}\n视频里检测到的独立墙(共 {len(walls)} 面)\n{'=' * 78}")
    known_count = 0
    new_count = 0
    walls_dir.mkdir(parents=True, exist_ok=True)
    if walls_dir.exists():
        for child in walls_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)

    for i, wall in enumerate(walls, 1):
        if wall.matched_photos:
            known_count += 1
            wall_dir_name = f"wall_{i:02d}_known_{sorted(wall.matched_photos)[0]}"
            print(f"\n  墙 #{i} [已知墙]  时间: {wall.ts_range} ({len(wall.items)} 帧)")
            print(f"         匹配照片: {', '.join(sorted(wall.matched_photos))}")
        else:
            new_count += 1
            wall_dir_name = f"wall_{i:02d}_unknown"
            print(f"\n  墙 #{i} [新墙]    时间: {wall.ts_range} ({len(wall.items)} 帧)")
            print(f"         (视频独有,34 张照片中无对应)")

        wall_dir = walls_dir / wall_dir_name
        wall_dir.mkdir(parents=True, exist_ok=True)
        for _ts, fid in wall.items:
            src = cache_dir / f"{fid}.jpg"
            dst = wall_dir / f"{fid}.jpg"
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
        print(f"         → 已 copy {len(wall.items)} 帧到 walls/{wall_dir_name}/")

    print(f"\n{'=' * 78}")
    print(f"汇总: {len(walls)} 面独立墙,其中:")
    print(f"  - 已知墙(匹配到了 34 张照片中的某些): {known_count} 面")
    print(f"  - 新墙(视频独有,不在 34 张照片中):    {new_count} 面")
    print(f"  - 帧原始抽取:  {cache_dir}/frame_NNNNs.jpg")
    print(f"  - 按墙分文件夹: {walls_dir}/wall_XX_*/")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())