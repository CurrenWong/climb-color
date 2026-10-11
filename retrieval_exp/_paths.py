"""
_paths.py — retrieval_exp 的路径配置(自包含)

仓库内数据(相对 retrieval_exp/ 上一级 = mediapipe2 仓库根):
  img_xhs/       — 两段视频的 5s 抽帧(65 帧,论文数据集)
  img_xhs_walls/ — 按墙分帧的目录(10 路线实例)
  weights/       — 模型权重(gitignored,用 download_weights.py 获取)

本地私有数据(默认指向项目根,可用环境变量覆盖;不入仓库):
  img/           — 用户拍摄的照片库(34 张,检索查询集)
  videos/        — 原始视频(2 段小红书视频)
  video_frames_cache/ — 视频抽帧缓存(与仓库 img_xhs 同源)

用法:
  # 本地开发(默认即项目根)
  python demo_search.py
  # 其他机器:设置环境变量指向自己的数据
  set CLIMB_PROJECT_ROOT=D:/somewhere
  python demo_search.py
"""
from __future__ import annotations

import os
from pathlib import Path

RETRIEVAL_EXP = Path(__file__).resolve().parent
REPO_ROOT = RETRIEVAL_EXP.parent  # mediapipe2/

# ---- 仓库内数据(始终存在,git 已跟踪 img_xhs/ img_xhs_walls/) ----
FRAMES_DIR = REPO_ROOT / "img_xhs"                       # 视频帧(65)
WALLS_DIR = REPO_ROOT / "img_xhs_walls"                  # 按墙分帧
DINO_MODEL_DIR = REPO_ROOT / "weights" / "dinov2-small"  # gitignored

# ---- 本地私有数据(默认项目根;env 可覆盖) ----
PROJECT_ROOT = Path(os.environ.get(
    "CLIMB_PROJECT_ROOT", r"D:\Project\climbing_wall_search"))
IMG_DIR = Path(os.environ.get("CLIMB_IMG_DIR", str(PROJECT_ROOT / "img")))
VIDEOS_DIR = Path(os.environ.get("CLIMB_VIDEOS_DIR", str(PROJECT_ROOT / "videos")))
CACHE_ROOT = Path(os.environ.get(
    "CLIMB_CACHE_ROOT", str(PROJECT_ROOT / "video_frames_cache")))

FFMPEG_BIN = os.environ.get(
    "FFMPEG_BIN", r"D:\Program\ffmpeg-8.0-full_build\bin\ffmpeg.exe")

# 视频帧缓存目录(本地私有,env 可覆盖)
FRAME_DIR_3O = CACHE_ROOT / "小红书_3OHz6rZ5opG"
FRAME_DIR_2X = CACHE_ROOT / "小红书_2XRo7VbPgTe"

# 抽帧/检索参数(与根目录原脚本保持一致)
FRAME_INTERVAL_SEC = 5
DINO_RECALL_THRESHOLD = 0.7
