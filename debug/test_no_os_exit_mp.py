"""
验证：MediaPipe 架构是否仍需要 os._exit(0)。

测试方法：创建 PoseLandmarker → 跑一帧 → 正常 return（不调 os._exit、不 close）。
如果进程卡死在解释器清理（TFLite XNNPACK 析构），说明 os._exit 仍必要；
如果正常退出，说明可以移除。
"""
import os
import sys
import time

import cv2
import mediapipe as mp

IMG = r"D:\Project\climbing_wall_search\video_frames_cache\小红书_2XRo7VbPgTe\walls\wall_01_known_IMG_20260808_193753\frame_0005s.jpg"


def main():
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

    image_bgr = cv2.imread(IMG)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = landmarker.detect(mp_img)
    print(f"检测到人体: {bool(result.pose_landmarks)}")

    print("✅ 主流程完成，正常 return（不调 os._exit、不 close landmarker）")
    return 0


if __name__ == '__main__':
    rc = main()
    print(f"exit code = {rc}")
    sys.exit(rc)
