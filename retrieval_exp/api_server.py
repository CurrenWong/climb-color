"""
api_server.py — 岩壁照片/视频检索 HTTP API

数据来源:
  - img/     目录:用户拍摄的照片(.jpg/.png/...)
  - videos/  目录:攀岩视频(.mp4),启动时按 VIDEO_FRAME_INTERVAL_SEC 抽帧入索引

端点:
  GET  /healthz    — 健康检查,返回索引规模(照片 + 视频帧)
  GET  /frames     — 列出所有已索引的帧(可按 ?kind=photo|video 过滤)
  POST /search     — 上传查询图片,返回 Top-K 候选(照片或视频帧,带时间戳)

启动:uvicorn api_server:app --host 0.0.0.0 --port 8000
     或 python api_server.py
"""
from __future__ import annotations

import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from rock_wall_retriever import MatchResult, RockWallRetriever
from video_utils import extract_frames
from _paths import (
    DINO_MODEL_DIR,
    DINO_RECALL_THRESHOLD,
    IMG_DIR,
    VIDEO_FRAME_INTERVAL_SEC,
    VIDEOS_DIR,
)


# ========== 配置 ==========
# IMG_DIR / VIDEOS_DIR / DINO_MODEL_DIR 来自 _paths.py(env 可覆盖)
MAX_RECALL_CANDIDATES = 200            # cosine 过滤前的最大候选数
PHOTO_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi"}
ALLOWED_UPLOAD_EXT = PHOTO_EXT          # /search 接受的上传扩展名
MAX_UPLOAD_BYTES = 20 * 1024 * 1024     # 20MB 上限


# ========== Pydantic 响应模型 ==========
class MatchItem(BaseModel):
    """单条匹配(可以是照片或视频帧)"""
    frame_id: str
    image_path: str
    kind: Literal["photo", "video"]
    dino_score: float
    inliers: int
    inlier_ratio: float
    is_match: bool
    video_id: str | None = None
    timestamp_ms: int | None = None
    jump_to_seconds: int | None = None


class SearchResponse(BaseModel):
    query_filename: str
    top_k: int
    matches: list[MatchItem]


class FrameInfo(BaseModel):
    frame_id: str
    image_path: str
    kind: Literal["photo", "video"]
    video_id: str | None = None
    timestamp_ms: int | None = None


class HealthResponse(BaseModel):
    status: str
    photo_count: int
    video_frame_count: int
    video_count: int
    device: str


# ========== 全局 retriever(由 lifespan 初始化) ==========
retriever: RockWallRetriever | None = None


def _is_photo_frame(f) -> bool:
    """判断索引里的帧是来自照片还是视频"""
    return f.video_id is None


def _build_index(r: RockWallRetriever, img_dir: Path, videos_dir: Path) -> tuple[int, int, int]:
    """索引照片和视频,返回 (photo_count, video_frame_count, video_count)"""
    photo_count = 0
    video_frame_count = 0
    video_count = 0

    # 索引照片
    if img_dir.is_dir():
        for p in sorted(img_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in PHOTO_EXT:
                try:
                    r.index_frame(p.stem, str(p))
                    photo_count += 1
                except Exception as exc:
                    print(f"  [跳过照片] {p.name}: {exc}")

    # 索引视频(抽帧)
    if videos_dir.is_dir():
        for video_path in sorted(videos_dir.iterdir()):
            if not video_path.is_file():
                continue
            if video_path.suffix.lower() not in VIDEO_EXT:
                continue
            video_id = video_path.stem
            try:
                frames = extract_frames(video_path, interval_sec=VIDEO_FRAME_INTERVAL_SEC)
            except Exception as exc:
                print(f"  [跳过视频] {video_path.name}: {exc}")
                continue
            video_count += 1
            for ts_sec, frame_path in frames:
                try:
                    r.index_frame(
                        frame_id=f"{video_id}_t{ts_sec:04d}s",
                        image_path=str(frame_path),
                        video_id=video_id,
                        timestamp_ms=ts_sec * 1000,
                    )
                    video_frame_count += 1
                except Exception as exc:
                    print(f"  [跳过视频帧] {frame_path.name}: {exc}")

    r.build_index()
    return photo_count, video_frame_count, video_count


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载模型 + 建索引;关闭时清理"""
    global retriever
    print("启动中:加载 DINOv2 + ALIKED 模型...")
    retriever = RockWallRetriever(dino_model_path=DINO_MODEL_DIR)
    print(f"扫描 {IMG_DIR} 下的照片 ...")
    print(f"扫描 {VIDEOS_DIR} 下的视频 ...")
    photo_count, video_frame_count, video_count = _build_index(retriever, IMG_DIR, VIDEOS_DIR)
    print(f"索引完成: {photo_count} 张照片 + {video_frame_count} 个视频帧 ({video_count} 个视频)")

    yield
    print("服务关闭")


app = FastAPI(
    title="岩壁照片/视频检索 API",
    description="上传一张岩壁照片,返回 Top-K 候选(照片或带时间戳的视频帧)",
    version="0.2.0",
    lifespan=lifespan,
)


# ========== 端点 ==========
@app.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    if retriever is None or retriever.index is None:
        raise HTTPException(status_code=503, detail="Retriever not ready")
    photo_count = sum(1 for f in retriever.frames if _is_photo_frame(f))
    video_frame_count = sum(1 for f in retriever.frames if not _is_photo_frame(f))
    video_count = len({f.video_id for f in retriever.frames if f.video_id})
    return HealthResponse(
        status="ok",
        photo_count=photo_count,
        video_frame_count=video_frame_count,
        video_count=video_count,
        device=retriever.device,
    )


@app.get("/frames", response_model=list[FrameInfo])
async def list_frames(
    kind: Literal["all", "photo", "video"] = Query("all", description="过滤类型"),
) -> list[FrameInfo]:
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not ready")
    items: list[FrameInfo] = []
    for f in retriever.frames:
        is_photo = _is_photo_frame(f)
        if kind == "photo" and not is_photo:
            continue
        if kind == "video" and is_photo:
            continue
        items.append(FrameInfo(
            frame_id=f.frame_id,
            image_path=f.image_path,
            kind="photo" if is_photo else "video",
            video_id=f.video_id,
            timestamp_ms=f.timestamp_ms,
        ))
    return items


@app.post("/search", response_model=SearchResponse)
async def search(
    file: Annotated[UploadFile, File(description="查询图片(jpg/png/...)")],
) -> SearchResponse:
    if retriever is None or retriever.index is None:
        raise HTTPException(status_code=503, detail="Retriever not ready")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_UPLOAD_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的扩展名 {suffix},允许: {sorted(ALLOWED_UPLOAD_EXT)}",
        )

    tmp_path = Path(tempfile.mkstemp(suffix=suffix)[1])
    try:
        with tmp_path.open("wb") as out:
            shutil.copyfileobj(file.file, out)
        if tmp_path.stat().st_size > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="文件超过 20MB")

        # 用 DINOv2 cosine >= DINO_RECALL_THRESHOLD 做召回,再在召回集上跑几何验证
        # (B 方案:避免 Top-K 不够大时漏检真同墙)
        results = retriever.search(
            str(tmp_path),
            top_k_recall=MAX_RECALL_CANDIDATES,
            min_cosine=DINO_RECALL_THRESHOLD,
        )
        # top_k 字段保留以兼容旧调用(不影响结果,只是标记给前端用的"想要几个")
        return SearchResponse(
            query_filename=file.filename or "uploaded",
            top_k=len(results),
            matches=[_to_item(r) for r in results],
        )
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def _to_item(r: MatchResult) -> MatchItem:
    is_photo = r.video_id is None
    return MatchItem(
        frame_id=r.frame_id,
        image_path=r.image_path,
        kind="photo" if is_photo else "video",
        dino_score=r.dino_score,
        inliers=r.inliers,
        inlier_ratio=r.inlier_ratio,
        is_match=r.is_match,
        video_id=r.video_id,
        timestamp_ms=r.timestamp_ms,
        jump_to_seconds=(r.timestamp_ms // 1000) if r.timestamp_ms is not None else None,
    )


# ========== 启动入口 ==========
if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, log_level="info")