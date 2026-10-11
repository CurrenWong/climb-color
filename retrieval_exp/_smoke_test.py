"""_smoke_test.py — retrieval_exp 自检：路径 + 核心导入 + 最小检索

用法:
    cd retrieval_exp
    <python-with-faiss> -u _smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    ok = True

    # 1) 路径模块
    try:
        import _paths  # noqa: F401
        from _paths import IMG_DIR, FRAMES_DIR, WALLS_DIR, DINO_MODEL_DIR
        print(f"[ok] _paths: IMG_DIR={IMG_DIR}  DINO_MODEL_DIR={DINO_MODEL_DIR}")
        for name, d in [("IMG_DIR", IMG_DIR), ("FRAMES_DIR", FRAMES_DIR),
                        ("WALLS_DIR", WALLS_DIR), ("DINO_MODEL_DIR", DINO_MODEL_DIR)]:
            if not Path(d).exists():
                print(f"[!!] {name} 不存在: {d}")
                ok = False
    except Exception as e:  # noqa: BLE001
        print(f"[!!] _paths 导入失败: {e}")
        return 1

    # 2) 核心依赖
    for mod in ("faiss", "cv2", "numpy", "torch"):
        try:
            __import__(mod)
            print(f"[ok] 依赖 {mod}")
        except Exception as e:  # noqa: BLE001
            print(f"[!!] 缺少依赖 {mod}: {e}")
            ok = False

    # 3) 检索器导入 + DINOv2 加载
    try:
        from rock_wall_retriever import RockWallRetriever
        r = RockWallRetriever(dino_model_path=str(DINO_MODEL_DIR))
        print(f"[ok] RockWallRetriever 加载 DINOv2: {r}")
    except Exception as e:  # noqa: BLE001
        print(f"[!!] 检索器初始化失败: {e}")
        ok = False

    # 4) 索引一个真实帧做最小检索
    if ok:
        try:
            import _paths
            frames = sorted(FRAMES_DIR.glob("*.jpg"))
            if frames:
                fid = frames[0].stem
                r.index_frame(fid, str(frames[0]))
                r.build_index()
                res = r.search(str(frames[0]), min_cosine=0.5)
                print(f"[ok] 最小检索: {fid} -> {len([x for x in res if x.is_match])} 命中")
            else:
                print("[!!] FRAMES_DIR 无帧文件")
                ok = False
        except Exception as e:  # noqa: BLE001
            print(f"[!!] 最小检索失败: {e}")
            ok = False

    print("\nSMOKE " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
