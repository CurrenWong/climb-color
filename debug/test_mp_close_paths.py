"""
严格验证 MediaPipe 清理路径：close() / del / with 上下文 是否还卡死。
当年 test_mp.py 记载：MediaPipe 1.0.0 + TFLite XNNPACK 在 Windows 上清理阶段无限阻塞。
"""
import os
import sys
import time

import cv2
import mediapipe as mp

IMG = r"D:\Project\climbing_wall_search\video_frames_cache\小红书_2XRo7VbPgTe\walls\wall_01_known_IMG_20260808_193753\frame_0005s.jpg"

print(f"mediapipe version: {mp.__version__}")


def detect_once(landmarker):
    image_bgr = cv2.imread(IMG)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = landmarker.detect(mp_img)
    return bool(result.pose_landmarks)


def test_close():
    """测试 1: 显式 close() 是否卡死"""
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    landmarker = PoseLandmarker.create_from_options(
        PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='pose_landmarker_full.task'),
            running_mode=VisionRunningMode.IMAGE,
            num_poses=1,
        )
    )
    print(f"  detect: {detect_once(landmarker)}")
    t0 = time.time()
    landmarker.close()
    print(f"  close() 耗时: {time.time()-t0:.2f}s ✅ 正常")
    return 0


def test_del():
    """测试 2: del + gc 是否卡死"""
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    landmarker = PoseLandmarker.create_from_options(
        PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='pose_landmarker_full.task'),
            running_mode=VisionRunningMode.IMAGE,
            num_poses=1,
        )
    )
    print(f"  detect: {detect_once(landmarker)}")
    t0 = time.time()
    del landmarker
    import gc
    gc.collect()
    print(f"  del + gc.collect() 耗时: {time.time()-t0:.2f}s ✅ 正常")
    return 0


def test_with():
    """测试 3: with 上下文管理器是否卡死"""
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    t0 = time.time()
    with PoseLandmarker.create_from_options(
        PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='pose_landmarker_full.task'),
            running_mode=VisionRunningMode.IMAGE,
            num_poses=1,
        )
    ) as landmarker:
        print(f"  detect: {detect_once(landmarker)}")
    print(f"  with 块退出耗时: {time.time()-t0:.2f}s ✅ 正常")
    return 0


if __name__ == '__main__':
    print("=== 测试1: close() ===")
    test_close()
    print("=== 测试2: del ===")
    test_del()
    print("=== 测试3: with ===")
    test_with()
    print("\n全部通过，正常退出（不调 os._exit）")
    sys.exit(0)
