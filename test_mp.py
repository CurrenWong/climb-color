"""
MediaPipe 1.0.0 Tasks API 姿态检测 + 攀岩线路色识别

【为什么不用 mp.solutions】
MediaPipe 1.0.0 已删除 legacy 的 `mp.solutions` 命名空间，新 API 全部走
`mp.tasks.vision.PoseLandmarker`，需要配套一个 .task 模型文件。

【为什么不要 landmarker.close() / with 上下文 / del】
MediaPipe 1.0.0 + TFLite XNNPACK 在 Windows 上清理阶段会无限阻塞（已复现）。
变通：跳过 close() 与 del，最后用 os._exit(0) 跳过解释器清理。代价是单个
Python 进程只能创建有限个 landmarker（每次新建不释放会涨内存），但单次
或少量检测场景够用。
"""
import os
import cv2
import mediapipe as mp

from route_color import detect_route_color, draw_route_info, merge_flip_landmarks


def analyze_image(image_path: str, output_path: str, landmarker,
                  DrawingUtils, PoseLandmarksConnections, DrawingStyles) -> dict | None:
    """
    处理一张图片：姿态检测（原图 + 翻转集成）+ 接触点颜色采样 + 线路色投票
    返回 route_info 或 None
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"  ❌ 无法读取: {image_path}")
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    # 姿态检测：原图 + 水平翻转 双跑，逐关节取可见度高的（翻转集成）
    # 解决 MediaPipe 对高抬腿/遮挡/背对姿态单方向定位失败的问题（实测 4.jpg 左腿）
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = landmarker.detect(mp_image)

    if not result.pose_landmarks:
        # 原图没检测到 → 试翻转图
        flip_rgb = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        mp_flip = mp.Image(image_format=mp.ImageFormat.SRGB, data=flip_rgb)
        result = landmarker.detect(mp_flip)
        if result.pose_landmarks:
            print(f"  ⚠️ 原图未检测到，翻转图检测到人体")
            # 翻转检测：坐标翻回 + 左右标签对调
            landmarks = merge_flip_landmarks(
                result.pose_landmarks[0], result.pose_landmarks[0], w)
        else:
            print(f"  ⚠️ 未检测到人体")
            cv2.imwrite(output_path, image)  # 原图也存一下
            return None
    else:
        flip_rgb = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        mp_flip = mp.Image(image_format=mp.ImageFormat.SRGB, data=flip_rgb)
        result_flip = landmarker.detect(mp_flip)
        if result_flip.pose_landmarks:
            landmarks = merge_flip_landmarks(
                result.pose_landmarks[0], result_flip.pose_landmarks[0], w)
        else:
            landmarks = result.pose_landmarks[0]

    # 关键顺序：先在原图上做颜色分析，再画骨架覆盖
    # 否则骨架的彩色线/圆点会被 detect_holds 当成"大岩点"干扰匹配
    route_info = detect_route_color(image, landmarks, w, h)

    # 再画姿态骨架
    DrawingUtils.draw_landmarks(
        image=image,
        landmark_list=landmarks,
        connections=PoseLandmarksConnections.POSE_LANDMARKS,
        landmark_drawing_spec=DrawingStyles.get_default_pose_landmarks_style(),
    )
    draw_route_info(image, route_info)

    # 打印
    print(f"  ✅ 关键点={len(landmarks)}  "
          f"鼻子=({landmarks[0].x*w:.0f},{landmarks[0].y*h:.0f})")
    print(f"  接触点采样:")
    for s in route_info['samples']:
        if s['samples'] == 0:
            print(f"    {s['point']:8s}: {s['name']}  vis={s.get('visibility', 0):.2f}")
        else:
            print(f"    {s['point']:8s}: {s['name']:6s}  "
                  f"HSV=({s['h']:5.1f},{s['s']:5.1f},{s['v']:5.1f})  "
                  f"samples={s['samples']}")
    print(f"  >>> 线路色 = {route_info['route_color']}  "
          f"置信度 = {route_info['confidence']:.0%}  "
          f"票数 = {route_info['votes']}")

    cv2.imwrite(output_path, image)
    print(f"  💾 {output_path}")
    return route_info


def main():
    # ========== 初始化（一个 landmarker 处理多张图） ==========
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    PoseLandmarksConnections = mp.tasks.vision.PoseLandmarksConnections
    DrawingUtils = mp.tasks.vision.drawing_utils
    DrawingStyles = mp.tasks.vision.drawing_styles

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='pose_landmarker_full.task'),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.3,   # 0.5 会漏检小尺寸/被挡的人（8.jpg 实测需 <=0.4）
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    landmarker = PoseLandmarker.create_from_options(options)

    # ========== 处理所有图（按文件名数字排序） ==========
    import re
    import os as _os

    img_dir = 'img'
    files = [f for f in _os.listdir(img_dir)
             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    files.sort(key=lambda f: (int(re.sub(r'\D', '', f)) if re.sub(r'\D', '', f) else 99999, f))

    print("=" * 60)
    print(f"  攀岩线路色识别  ({len(files)} 张图)")
    print("=" * 60)
    for i, name in enumerate(files, 1):
        print(f"\n[{i}] {name}")
        analyze_image(
            image_path=os.path.join(img_dir, name),
            output_path=f'result_route_{i}.jpg',
            landmarker=landmarker,
            DrawingUtils=DrawingUtils,
            PoseLandmarksConnections=PoseLandmarksConnections,
            DrawingStyles=DrawingStyles,
        )

    # 立即终止，绕开 mediapipe 1.0.0 的 TFLite 析构卡死
    os._exit(0)


if __name__ == '__main__':
    main()
