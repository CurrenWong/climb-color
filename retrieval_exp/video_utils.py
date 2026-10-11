"""
video_utils.py — 共享的视频抽帧工具(供 analyze_video.py 和 api_server.py 共用)

FFmpeg 抽帧到 cache_dir/<video_stem>/frame_NNNNs.jpg,已存在则跳过。
路径来自 _paths.py(仓库内数据用相对路径,本地私有数据用环境变量覆盖)。
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from _paths import CACHE_ROOT, FFMPEG_BIN


def get_video_duration(video_path: Path) -> float:
    """从 ffprobe 输出解析 Duration(秒)"""
    ffprobe = str(Path(FFMPEG_BIN).parent / "ffprobe.exe")
    out = subprocess.check_output(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        text=True,
    )
    return float(out.strip())


def extract_frames(
    video_path: Path,
    output_dir: Path | None = None,
    interval_sec: int = 5,
) -> list[tuple[int, Path]]:
    """FFmpeg 抽帧,默认缓存到 CACHE_ROOT/<video_stem>/frame_NNNNs.jpg

    返回 [(timestamp_sec, frame_path), ...],已存在的帧跳过(增量)
    """
    if output_dir is None:
        output_dir = CACHE_ROOT / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    duration = get_video_duration(video_path)
    timestamps = list(range(0, int(duration), interval_sec))
    if timestamps and timestamps[-1] >= duration:
        timestamps.pop()

    frames: list[tuple[int, Path]] = []
    for ts in timestamps:
        out_path = output_dir / f"frame_{ts:04d}s.jpg"
        if out_path.exists():
            frames.append((ts, out_path))
            continue
        cmd = [
            FFMPEG_BIN, "-y",
            "-ss", str(ts),
            "-i", str(video_path),
            "-frames:v", "1",
            "-q:v", "2",
            str(out_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        frames.append((ts, out_path))
    return frames