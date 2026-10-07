"""
诊断 3O wall_04 line2_pink frame2/frame3 左手为什么判红。

复现 exp_dwpose_full.py 的流程，打印 L_hand 的：
  - 关键点坐标/置信度
  - mask 采样 HSV（_sample_color 返回值）
  - hue_to_color_name 判断过程（手动走各分支）
"""
import os
import sys
from pathlib import Path

_BASE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_BASE)
# sys.path 顺序: _BASE (dwpose_exp 本地副本) 优先, 然后根目录
# (edgesam_stub 等共享模块), 最后 EdgeSAM 源码。注意: 必须用
# 一次 slice insert 保留原 sys.path, 多次 insert(0, ...) 会反向
# 把 _BASE 推到后面, 导致 dwpose_exp 本地副本永远不被加载。
sys.path[0:0] = [_BASE, _ROOT, os.path.join(_ROOT, "EdgeSAM")]

import cv2
import numpy as np

import edgesam_stub  # noqa: F401

from dwpose_onnx import DWPoseONNX
from edgesam_holds import EdgeSamHoldSegmenter, _sample_color
from yolo_person_mask import PersonMasker
import route_color

WALL = r"..\img_xhs_walls\3OHz6rZ5opG_wall_04_line2_pink"

# exp_dwpose_full 里 frame_0X.jpg 按文件数字排序
frames = sorted(
    (p for p in Path(WALL).glob('*.jpg')),
    key=lambda p: (int(''.join(c for c in p.stem if c.isdigit()) or 0), p.name)
)
print(f"墙目录 {len(frames)} 帧, 前5帧:")
for f in frames[:8]:
    print(f"  {f.name}")

dw = DWPoseONNX('../weights/yolox_l.onnx', '../weights/dw-ll_ucoco_384.onnx')
segmenter = EdgeSamHoldSegmenter('../weights/edge_sam.pth', device='cpu')
person_masker = PersonMasker(model_name="../weights/yolo26m-seg.pt", device='cpu')

for idx in [1, 2]:  # frame2, frame3 (0-based)
    img_path = frames[idx]
    print(f"\n{'='*60}\n{img_path.name}")
    image_bgr = cv2.imread(str(img_path))

    boxes, kps, scores = dw.run(image_bgr)
    if kps is None:
        print("  无人体")
        continue
    k, s = kps[0], scores[0]
    body_pts = k[[5, 6, 11, 12]]
    body_cx = float(body_pts[:, 0].mean())
    body_cy = float(body_pts[:, 1].mean())

    person_mask = person_masker.segment(image_bgr)
    segmenter.set_image(image_bgr)

    for ep_name, kidx in [('L_hand', 9), ('R_hand', 10)]:
        px, py = float(k[kidx, 0]), float(k[kidx, 1])
        conf = float(s[kidx])
        print(f"\n  [{ep_name}] keypoint=({px:.1f},{py:.1f}) conf={conf:.2f}")
        if conf < 0.3:
            print("    低置信度，跳过")
            continue

        res = segmenter.segment_endpoint(px, py, body_cx, body_cy,
                                         person_mask=person_mask)
        if res is None or res['area'] < 100:
            print("    非岩点/无 mask")
            continue
        h, ss, v = res['h'], res['s'], res['v']
        print(f"    mask area={res['area']} score={res['score']:.2f}")
        print(f"    采样 HSV=({h:.1f},{ss:.1f},{v:.1f})  "
              f"-> _sample_color 判定: {res['color_name']}")

        # 手动走 hue_to_color_name
        name = route_color.hue_to_color_name(h, ss)
        print(f"    hue_to_color_name({h:.1f},{ss:.1f}) = {name}")
        # 分支解释
        if h >= 150:
            if ss < 200:
                print(f"      → h>=150 且 s<200 → 粉红（PINK_SAT_MAX=200 内）")
            else:
                print(f"      → h>=150 但 s={ss:.0f}>=200 → 进 HUE_BINS: "
                      f"{'粉红' if h<170 else '红'} ({(150,170) if h<170 else (170,180)})")
        # mask 内全部像素统计（验证 _sample_color 是否被网格采样偏置）
        mask_bool = res['mask_bool']
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        pix = hsv[mask_bool]
        print(f"    mask 全像素统计: H中位={np.median(pix[:,0]):.1f} "
              f"S中位={np.median(pix[:,1]):.1f} V中位={np.median(pix[:,2]):.1f} "
              f"(n={len(pix)})")
        # 各颜色桶占比（用 _classify_pixel 逻辑）
        from edgesam_holds import _classify_pixel
        cnt = {}
        for p in pix[:: max(1, len(pix)//2000)]:
            c = _classify_pixel(float(p[0]), float(p[1]), float(p[2]))
            cnt[c] = cnt.get(c, 0) + 1
        print(f"    mask 颜色桶(抽样): {dict(sorted(cnt.items(), key=lambda x:-x[1]))}")
