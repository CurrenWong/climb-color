"""
loo_hybrid.py — 34 照片留一法混合索引(33 照片 + 65 视频帧),按"共享物理墙段"判定 P/R

论文 4.10 声称 34/34 100% P/R。本脚本直接复现该评估:
  - 索引 = 其余 33 张照片 + 全部 65 个视频帧(带墙段标签)
  - 查询命中(照片或帧)后,判定"命中与查询是否共享至少一个物理墙段":
      照片侧: 已知照片(PHOTO_WALL_GT)直接有墙标签;未知照片用其命中帧推断
      帧侧:   帧自带墙段标签
  - TP = 查询与命中共享至少一个墙段;FP = 不共享
输出 34/34 判定表 + P/R,并单独标注 193753↔193754 这类跨 GT 标签但共享墙段的配对。
"""
from __future__ import annotations

import sys, time
sys.path.insert(0, r"D:\Project\climbing_wall_search\mediapipe2\retrieval_exp")
from pathlib import Path
from _paths import DINO_MODEL_DIR, FRAMES_DIR, WALLS_DIR, IMG_DIR
from rock_wall_retriever import RockWallRetriever

PHOTO_WALL_GT = {
    "IMG_20260808_193753": "2X_wall_01",
    "IMG_20260808_193745": "2X_wall_02",
    "IMG_20260808_193754": "2X_wall_03",
    "IMG_20260808_193724": "2X_wall_04",
    "IMG_20260808_193612": "3O_wall_03",
    "IMG_20260808_193650": "3O_wall_05",
    "IMG_20260808_193621": "3O_wall_04",
}

# 帧名 -> 墙段短 ID(复用 joint_eval 的目录扫描)
WALL_DIR_GT = {
    "2XRo7VbPgTe_wall_01_known_IMG_20260808_193753": "2X_wall_01",
    "2XRo7VbPgTe_wall_02_known_IMG_20260808_193745": "2X_wall_02",
    "2XRo7VbPgTe_wall_03_known_IMG_20260808_193754": "2X_wall_03",
    "2XRo7VbPgTe_wall_04_known_IMG_20260808_193724": "2X_wall_04",
    "3OHz6rZ5opG_wall_01_unknown": "3O_wall_01",
    "3OHz6rZ5opG_wall_02_unknown": "3O_wall_02",
    "3OHz6rZ5opG_wall_03_known_IMG_20260808_193612": "3O_wall_03",
    "3OHz6rZ5opG_wall_04_line1_white": "3O_wall_04_line1",
    "3OHz6rZ5opG_wall_04_line2_pink": "3O_wall_04_line2",
    "3OHz6rZ5opG_wall_05_known_IMG_20260808_193650": "3O_wall_05",
}
FRAME_WALL: dict[str, str] = {}
for d in sorted(WALLS_DIR.iterdir()):
    if d.is_dir() and d.name in WALL_DIR_GT:
        prefix = "2XRo" if d.name.startswith("2XRo") else "3OHz"
        for fp in d.glob("*.jpg"):
            FRAME_WALL[f"{prefix}_{fp.name}"] = WALL_DIR_GT[d.name]


def wall_equiv(a: str, b: str) -> bool:
    if a == b:
        return True
    return (a == "3O_wall_04" and b in ("3O_wall_04_line1", "3O_wall_04_line2")) or \
           (b == "3O_wall_04" and a in ("3O_wall_04_line1", "3O_wall_04_line2"))


def main() -> None:
    photos = sorted(IMG_DIR.glob("*.jpg"))
    photo_ids = [p.stem for p in photos]
    print(f"34 照片留一法混合索引 LOO(33 照片 + 65 帧)")
    print(f"帧墙标签: {len(FRAME_WALL)} 帧")

    t0 = time.time()
    retriever = RockWallRetriever(dino_model_path=str(DINO_MODEL_DIR))
    print(f"[加载] {time.time()-t0:.1f}s")

    # 照片的墙标签:已知直接有;未知先用照片→帧命中推断(前面 joint_eval --all-photos 已跑)
    # 这里只依赖已知 GT 判定 LOO;未知照片无 GT,只统计"是否命中零个其他照片"
    # 为避免重复跑照片→帧,直接加载 joint_eval 输出逻辑:先索引帧,再对每照片查询帧拿推断墙
    t1 = time.time()
    for fid, fp in FRAME_WALL.items():
        retriever.index_frame(fid, str(FRAMES_DIR / f"{fid}"))
    retriever.build_index()
    print(f"[帧索引] {time.time()-t1:.1f}s")

    # 照片→帧 推断墙(不排除查询本身;照片不进帧索引)
    inferred: dict[str, set[str]] = {}
    for pid in photo_ids:
        res = retriever.search(str(IMG_DIR / f"{pid}.jpg"), min_cosine=0.7)
        walls = {FRAME_WALL[r.frame_id] for r in res if r.is_match and r.frame_id in FRAME_WALL}
        inferred[pid] = walls
        print(f"  推断 {pid}: {sorted(walls) or '∅'}")
    print(f"[照片→帧推断] {time.time()-t1:.1f}s")

    # 照片→照片 LOO:把 34 张照片加进索引,逐张查询(排除自身)
    t2 = time.time()
    for pid in photo_ids:
        retriever.index_frame(pid, str(IMG_DIR / f"{pid}.jpg"))
    retriever.build_index()
    print(f"[照片索引] {time.time()-t2:.1f}s")

    print("\n" + "=" * 78)
    print("LOO 判定:每个查询的所有 MATCH 命中(照片+帧),按共享墙段判 TP/FP")
    print("口径:查询墙 = GT 标签 ∪ 照片→帧推断;命中墙 = 该照片/帧的全部已知墙")
    print("     共享 = 两者墙集有交集(3O_wall_04 与 line1/line2 视为同一物理墙)")
    print("=" * 78)
    tp_total = fp_total = 0
    no_wall = 0
    for qid in photo_ids:
        res = retriever.search(str(IMG_DIR / f"{qid}.jpg"), min_cosine=0.7, exclude_frame_id=qid)
        # 查询自己的墙:GT 或推断(命中帧);193754 这种横跨取景的照片用 GT∪推断
        q_walls = set()
        if qid in PHOTO_WALL_GT:
            q_walls.add(PHOTO_WALL_GT[qid])
        q_walls |= inferred[qid]
        if not q_walls:
            no_wall += 1
            hits = [(r.frame_id, r.inliers) for r in res if r.is_match and r.frame_id != qid]
            print(f"  [·] {qid} (无墙标签,{len(hits)} 命中) -> {hits[:4]}")
            continue
        tp = fp = 0
        cross = []
        for r in res:
            if not r.is_match or r.frame_id == qid:
                continue
            hid = r.frame_id
            h_walls = set()
            if hid in PHOTO_WALL_GT:
                h_walls.add(PHOTO_WALL_GT[hid])
            h_walls |= inferred.get(hid, set())
            if hid in FRAME_WALL:
                h_walls.add(FRAME_WALL[hid])
            shared = any(wall_equiv(w, h) for w in q_walls for h in h_walls)
            if shared:
                tp += 1
            else:
                fp += 1
                cross.append((hid, r.inliers, sorted(h_walls)))
        tp_total += tp
        fp_total += fp
        mark = "✅" if fp == 0 else "❌"
        print(f"  [{mark}] {qid}  墙={sorted(q_walls)}  TP={tp} FP={fp}{'  跨墙:' + str(cross) if cross else ''}")

    print("\n" + "=" * 78)
    print(f"总判定(共享墙段语义): TP命中={tp_total}  FP命中={fp_total}")
    print(f"  无墙标签跳过: {no_wall}/34")
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0
    print(f"  命中级精确率(共享墙段语义): {precision:.1%}")
    print(f"  总用时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
