"""
实时校准工具（精简版）：4 个关键 trackbar，调 绿->紫 是核心
操作：N 切图  S 保存  ESC 退出
"""
import os
import json
import cv2
import mediapipe as mp
import route_color


# ============== 加载 3 张图 + 跑一次姿态检测 ==============
NAMES = ['1.jpg', '2.jpg', '3.jpg']
IMAGES: list = []
LANDMARKS: list = [None] * 3

for n in NAMES:
    img = cv2.imread(f'img/{n}')
    assert img is not None, f'failed to load img/{n}'
    IMAGES.append(img)

B = mp.tasks.BaseOptions
P = mp.tasks.vision.PoseLandmarker
O = mp.tasks.vision.PoseLandmarkerOptions
R = mp.tasks.vision.RunningMode
_OPTS = O(
    base_options=B(model_asset_path='weights/pose_landmarker_full.task'),
    running_mode=R.IMAGE, num_poses=1,
)
_LANDMARKER = P.create_from_options(_OPTS)
for i, img in enumerate(IMAGES):
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = _LANDMARKER.detect(mp_img)
    if result.pose_landmarks:
        LANDMARKS[i] = result.pose_landmarks[0]
print(f'loaded {sum(l is not None for l in LANDMARKS)}/3 images with pose', flush=True)


# ============== 固定参数（不暴露成 trackbar）==============
FIXED = {
    'min_sat': 35, 'min_val': 40,
    'min_area': 80, 'max_area': 10000,
    'max_match_distance': 150,
    '蓝->紫 (H)': 128,  # 蓝→紫回卷点，几乎不动
}


# ============== Trackbar 窗口 ==============
WINDOW = 'Calibration'
cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW, 1100, 1300)


def _nothing(x):
    pass


# 只暴露 4 个最关键参数（用纯英文名 + 下划线，避免 OpenCV 解析特殊字符崩）
cv2.createTrackbar('green_to_purple', WINDOW, 70,  130, _nothing)
cv2.createTrackbar('yellow_to_green', WINDOW, 45,  70,  _nothing)
cv2.createTrackbar('purple_to_blue',  WINDOW, 100, 150, _nothing)
cv2.createTrackbar('torso_pen',      WINDOW, 5,   50,  _nothing)

# 强制窗口初始化（否则有些 Windows OpenCV 在 getTrackbarPos 之前窗口是 NULL）
cv2.waitKey(100)


def _clamp_bins(yg, gp, pb):
    yg = max(20, min(yg, gp - 1))
    gp = max(yg + 1, min(gp, pb - 1))
    pb = max(gp + 1, min(pb, 170))
    return yg, gp, pb


def _update_hue_bins(yg, gp, pb, bp):
    route_color.HUE_BINS[:] = [
        ((0, 8),                '红'),
        ((8, 20),               '橙'),
        ((20, yg),              '黄'),
        ((yg, gp),              '绿'),
        ((gp, pb),              '紫'),
        ((pb, bp),              '蓝'),
        ((bp, 170),             '紫'),
        ((170, 180),            '红'),
    ]


current_idx = 0
print('打开校准窗口... (4 个 trackbar)', flush=True)

while True:
    # 读 trackbar（带 try/except 兜底，防止某些 Windows 环境下窗口 NULL）
    try:
        gp = cv2.getTrackbarPos('green_to_purple', WINDOW)
        yg = cv2.getTrackbarPos('yellow_to_green', WINDOW)
        pb = cv2.getTrackbarPos('purple_to_blue', WINDOW)
        torso_pen = cv2.getTrackbarPos('torso_pen', WINDOW) / 100.0
    except cv2.error:
        # 窗口还没好，等一下
        cv2.waitKey(100)
        continue
    bp = FIXED['蓝->紫 (H)']
    yg, gp, pb = _clamp_bins(yg, gp, pb)
    _update_hue_bins(yg, gp, pb, bp)

    img = IMAGES[current_idx]
    h, w = img.shape[:2]
    landmarks = LANDMARKS[current_idx]

    if landmarks is not None:
        route_info = route_color.detect_route_color(
            img, landmarks, w, h,
            max_match_distance=FIXED['max_match_distance'],
            min_area=FIXED['min_area'], max_area=FIXED['max_area'],
            min_sat=FIXED['min_sat'], min_val=FIXED['min_val'],
            torso_penalty=torso_pen,
        )
    else:
        route_info = {'route_color': '无', 'samples': [], 'votes': {},
                     'confidence': 0, 'holds': [], 'matched_hold_ids': []}

    # 画结果
    display = img.copy()
    if landmarks is not None:
        route_color.draw_route_info(display, route_info)

    # 顶部小信息条（替代原来的底部多行）
    info = f"[{current_idx+1}/3] {NAMES[current_idx]}  |  yg={yg}  gp={gp}  pb={pb}  bp={bp}  pen={torso_pen:.2f}  |  N=next  S=save  ESC=quit"
    info = info.encode('ascii', 'replace').decode('ascii')  # 防止中文路径在某些系统上报错
    cv2.rectangle(display, (0, 0), (w, 22), (0, 0, 0), -1)
    cv2.putText(display, info, (8, 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow(WINDOW, display)

    key = cv2.waitKey(60) & 0xFF
    if key == 27:  # ESC
        break
    elif key in (ord('n'), ord('N')):
        current_idx = (current_idx + 1) % 3
    elif key in (ord('s'), ord('S')):
        params = {
            'hue_yellow_green': yg,
            'hue_green_purple': gp,
            'hue_purple_blue':  pb,
            'hue_blue_purple':  bp,
            **{k: v for k, v in FIXED.items() if 'H' not in k},
            'max_match_distance': FIXED['max_match_distance'],
            'torso_penalty': torso_pen,
        }
        with open('calibrate.json', 'w', encoding='utf-8') as f:
            json.dump(params, f, indent=2, ensure_ascii=False)
        print(f"saved: {params}", flush=True)

os._exit(0)
