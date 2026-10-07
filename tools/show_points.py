"""
只标 6 个接触点（无姿态骨架、无匹配 hold、无 header），方便肉眼核对位置
输出：points_1.jpg / points_2.jpg / points_3.jpg
"""
import os
import cv2
import mediapipe as mp

NAMES = ['1.jpg', '2.jpg', '3.jpg']
CONTACT_POINTS = [
    ('L_wrist',  15),
    ('R_wrist',  16),
    ('L_index',  19),
    ('R_index',  20),
    ('L_foot',   31),
    ('R_foot',   32),
]

# 配色：手 = 黄色，脚 = 洋红
POINT_COLOR = {
    'wrist': (0, 255, 255),
    'index': (0, 200, 255),
    'foot':  (255, 0, 255),
}

B = mp.tasks.BaseOptions
P = mp.tasks.vision.PoseLandmarker
O = mp.tasks.vision.PoseLandmarkerOptions
R = mp.tasks.vision.RunningMode

opts = O(
    base_options=B(model_asset_path='pose_landmarker_full.task'),
    running_mode=R.IMAGE, num_poses=1,
)
lk = P.create_from_options(opts)

for i, name in enumerate(NAMES, 1):
    img = cv2.imread(f'img/{name}')
    assert img is not None
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = lk.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.pose_landmarks:
        print(f'[{i}] {name}: no pose')
        continue
    landmarks = res.pose_landmarks[0]

    out = img.copy()

    print(f'==== {name} ({w}x{h}) ====')
    # 第一次扫描：定每个标签的"目标"位置（按部位分象限，避免挤在一起）
    targets = {
        'L_wrist':  (-1, -1, 'L'),  # (dx_sign, dy_sign, tag)
        'L_index':  (-1, -1, 'L'),
        'R_wrist':  (+1, -1, 'R'),
        'R_index':  (+1, -1, 'R'),
        'L_foot':   (-1, +1, 'L'),
        'R_foot':   (+1, +1, 'R'),
    }
    OFFSET = 60  # 标签距离点的像素

    for label, idx in CONTACT_POINTS:
        lm = landmarks[idx]
        px, py = int(lm.x * w), int(lm.y * h)
        kind = 'foot' if 'foot' in label else ('index' if 'index' in label else 'wrist')
        color = POINT_COLOR[kind]

        # 小点（3px，不再覆盖采样点本身）
        cv2.circle(out, (px, py), 3, color, -1)
        cv2.circle(out, (px, py), 4, (255, 255, 255), 1)  # 细白边

        # 标签放在远端（按 dx_sign/dy_sign 决定象限）
        dx_sign, dy_sign, _ = targets[label]
        text = f"{label}  ({px},{py})"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # 标签的左上角位置（远离点）
        if dx_sign > 0:
            bx = px + OFFSET
        else:
            bx = px - OFFSET - tw
        if dy_sign > 0:
            by = py + OFFSET + th
        else:
            by = py - OFFSET

        # 边界保护：贴图边了就往里推
        bx = max(2, min(bx, w - tw - 6))
        by = max(th + 4, min(by, h - 4))

        # 引线（点 -> 标签中心）
        cv2.line(out, (px, py), (bx + tw // 2, by - th // 2), color, 1, cv2.LINE_AA)

        # 标签背景 + 文字
        cv2.rectangle(out, (bx - 3, by - th - 3), (bx + tw + 3, by + 3), (0, 0, 0), -1)
        cv2.putText(out, text, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        print(f'  {label:8s}  px=({px:4d},{py:4d})  vis={lm.visibility:.2f}')

    out_path = f'points_{i}.jpg'
    cv2.imwrite(out_path, out)
    print(f'  saved: {out_path}\n')

os._exit(0)
