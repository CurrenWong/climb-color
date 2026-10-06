"""
v1 baseline: MediaPipe Pose + detect_holds (HSV connected-components) + endpoint voting
在同一个 10-wall 视频集合上做 wall-level 回归，用于论文 Table 1 的 baseline 对比。

与 EdgeSAM 管线的差别：
  - hold 提取: HSV 阈值 + 形态学 + 连通域（detect_holds），无语义分割
  - 端点采样: 岩点 bbox 匹配 / 环形采样（sample_local_color）
  - 无 ring 黑票修正、无 person mask 引导、无自适应 box

用法:
  python baseline_detect_holds_walls.py            # 全部 wall
  python baseline_detect_holds_walls.py 2          # 前 2 个 wall
输出:
  baseline_v1_summary.txt
"""
import os
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

from route_color import detect_route_color, merge_flip_landmarks, hue_to_color_name

WALLS_DIR = Path("img_xhs_walls")

TRUTH = {
    '2XRo7VbPgTe_wall_01_known_IMG_20260808_193753': '黑',
    '2XRo7VbPgTe_wall_02_known_IMG_20260808_193745': '蓝',
    '2XRo7VbPgTe_wall_03_known_IMG_20260808_193754': '粉红',
    '2XRo7VbPgTe_wall_04_known_IMG_20260808_193724': '青',
    '3OHz6rZ5opG_wall_01_unknown': '蓝',
    '3OHz6rZ5opG_wall_02_unknown': '红',
    '3OHz6rZ5opG_wall_03_known_IMG_20260808_193612': '黄',
    '3OHz6rZ5opG_wall_04_line1_white': '白',
    '3OHz6rZ5opG_wall_04_line2_pink': '粉红',
    '3OHz6rZ5opG_wall_05_known_IMG_20260808_193650': '绿',
}

SKIP = {'非岩点', '低置信度', '未知', '无法判断', '未知(暗)'}


def detect_pose(image_bgr, landmarker, w, h):
    """原图 + 水平翻转双跑，逐关节取可见度高者（与 v1 一致）。"""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    res = landmarker.detect(mp_img)

    flip_rgb = cv2.cvtColor(cv2.flip(image_bgr, 1), cv2.COLOR_BGR2RGB)
    mp_flip = mp.Image(image_format=mp.ImageFormat.SRGB, data=flip_rgb)
    res_flip = landmarker.detect(mp_flip)

    if res.pose_landmarks and res_flip.pose_landmarks:
        return merge_flip_landmarks(res.pose_landmarks[0], res_flip.pose_landmarks[0], w)
    if res.pose_landmarks:
        return res.pose_landmarks[0]
    if res_flip.pose_landmarks:
        # 只有翻转检测到：坐标翻回 + 左右对调（merge 自己对自己，w 做镜像）
        return merge_flip_landmarks(res_flip.pose_landmarks[0], res_flip.pose_landmarks[0], w)
    return None


def hsv_only_vote(image_bgr, landmarks, w, h):
    """
    最原始的 baseline: 不做任何 hold 分割，直接在端点周边 20-45px 环形带
    取像素 HSV，过掉低饱和/暗像素后用圆形均值算色相 → 颜色名。
    代表"纯颜色空间 + 姿态端点"方案的下限。
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    yy, xx = np.ogrid[:h, :w]
    votes = Counter()
    for _, idx in (('L_hand', 15), ('R_hand', 16), ('L_foot', 31), ('R_foot', 32)):
        p = landmarks[idx]
        vis = float(getattr(p, 'visibility', 1.0))
        if vis < 0.3:
            continue
        px, py = int(p.x * w), int(p.y * h)
        px = max(0, min(px, w - 1))
        py = max(0, min(py, h - 1))
        d = np.hypot(xx - px, yy - py)
        ring = (d >= 20) & (d <= 45)
        if ring.sum() == 0:
            continue
        pix = hsv[ring]
        keep = (pix[:, 1] >= 35) & (pix[:, 2] >= 40)
        if keep.sum() < 10:
            continue
        hues = pix[keep, 0].astype(np.float64) * (2 * np.pi / 180.0)
        hmean = (np.degrees(np.arctan2(np.sin(hues).mean(),
                                       np.cos(hues).mean())) / 2.0) % 180.0
        s_med = float(np.median(pix[keep, 1]))
        votes[hue_to_color_name(float(hmean), s_med)] += 1
    return votes


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    hsv_only = '--hsv-only' in sys.argv
    limit = int(args[0]) if args and args[0].isdigit() else None

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='pose_landmarker_full.task'),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    landmarker = PoseLandmarker.create_from_options(options)

    walls = sorted([d for d in WALLS_DIR.iterdir() if d.is_dir()])
    if limit:
        walls = walls[:limit]
    method = 'HSV-only (endpoint ring sampling)' if hsv_only else 'v1 detect_holds (HSV connected components)'
    out_file = 'baseline_hsv_summary.txt' if hsv_only else 'baseline_v1_summary.txt'
    print(f"baseline = {method} — {len(walls)} walls\n")

    lines, hits, known = [], 0, 0
    for wall_dir in walls:
        frames = sorted(wall_dir.glob('*.jpg'))
        votes = Counter()
        n_ok = 0
        print(f"=== {wall_dir.name} ({len(frames)} 帧) ===")
        for fp in frames:
            img = cv2.imread(str(fp))
            if img is None:
                continue
            h, w = img.shape[:2]
            lm = detect_pose(img, landmarker, w, h)
            if lm is None:
                continue
            if hsv_only:
                v = hsv_only_vote(img, lm, w, h)
                for k, n in v.items():
                    votes[k] += n
                n_ok += 1
                continue
            try:
                info = detect_route_color(img, lm, w, h)
            except Exception as e:  # noqa: BLE001
                print(f"  ⚠️ {fp.name}: {e}")
                continue
            n_ok += 1
            for s in info['samples']:
                if s.get('name') not in SKIP:
                    votes[s['name']] += 1

        total = sum(votes.values())
        top = votes.most_common(1)[0][0] if votes else '无法判断'
        top_n = votes.most_common(1)[0][1] if votes else 0
        truth = TRUTH.get(wall_dir.name)
        if truth is None:
            hit = '?'
        else:
            known += 1
            if top == truth:
                hit = '✅'
                hits += 1
            else:
                hit = '❌'
        print(f"  >>> 判={top} ({top_n}/{total})  真值={truth or '?'}  {hit}\n")
        lines.append(f"{wall_dir.name[:55]:55s}  判={top:4s} ({top_n}/{total})  真值={truth or '?':6s}  {hit}")

    print(f"\n=== 汇总 (baseline: {method}) ===")
    print(f"命中: {hits}/{known}")
    for l in lines:
        print(f"  {l}")

    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(f"baseline ({method}) 命中: {hits}/{known}\n")
        for l in lines:
            f.write(l + "\n")
    print(f"\n已保存 {out_file}")

    # MediaPipe 1.0.0 TFLite 析构在 Windows 上卡顿（~42s），直接退出
    os._exit(0)


if __name__ == '__main__':
    main()
