"""诊断:2X_wall_01 帧与 2X_wall_03 帧是否同墙;193753/193754 照片与两墙段帧的匹配"""

import sys, time
sys.path.insert(0, r"D:\Project\climbing_wall_search\mediapipe2\retrieval_exp")
from pathlib import Path
from _paths import DINO_MODEL_DIR, FRAMES_DIR, WALLS_DIR, IMG_DIR
from rock_wall_retriever import RockWallRetriever

walls = sorted(WALLS_DIR.iterdir())
def wall_frames(name_part):
    for d in walls:
        if name_part in d.name:
            return sorted(d.glob("*.jpg"))
    return []

w01_frames = wall_frames("2XRo7VbPgTe_wall_01")
w03_frames = wall_frames("2XRo7VbPgTe_wall_03")
print(f"wall_01 frames: {[f.name for f in w01_frames]}")
print(f"wall_03 frames: {[f.name for f in w03_frames]}")

r = RockWallRetriever(dino_model_path=str(DINO_MODEL_DIR))
# 索引 wall_03 的帧
for f in w03_frames:
    r.index_frame(f.stem, str(f))
r.build_index()

# 用 wall_01 帧查询 -> 看能否命中 wall_03
print("\n== wall_01 帧查询 wall_03 索引 ==")
for f in w01_frames[:3]:
    res = r.search(str(f), min_cosine=0.5)
    hits = [(x.frame_id, x.inliers, round(x.inlier_ratio,3), round(x.dino_score,4)) for x in res if x.is_match]
    print(f"  {f.name} -> {hits[:5]}")

# 照片查询 wall_03 索引
print("\n== 照片查询 wall_03 索引 ==")
for pname in ["IMG_20260808_193753", "IMG_20260808_193754"]:
    p = IMG_DIR / f"{pname}.jpg"
    res = r.search(str(p), min_cosine=0.5)
    hits = [(x.frame_id, x.inliers, round(x.inlier_ratio,3), round(x.dino_score,4)) for x in res if x.is_match]
    print(f"  {pname} -> {hits[:6]}")
