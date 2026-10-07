"""实验3 通用版：环形 8 方向偏移锚点做多正点，融合投票。

不预设偏移方向（避免 L_hand 式破坏），让 EdgeSAM mask 覆盖端点周围
环形区域，黑岩点所在方向的 mask 自然判黑。
"""
import os, sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "EdgeSAM"))
import edgesam_holds

IMG = r"D:\Project\climbing_wall_search\mediapipe2\img_xhs_walls\2XRo7VbPgTe_wall_01_known_IMG_20260808_193753\frame_0005s.jpg"
image_bgr = cv2.imread(IMG)
h, w = image_bgr.shape[:2]

import mediapipe as mp
from test_edge_sam import detect_pose
from yolo_person_mask import PersonMasker

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
landmarker = PoseLandmarker.create_from_options(
    PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=r"D:\Project\climbing_wall_search\mediapipe2\pose_landmarker_full.task"),
        running_mode=VisionRunningMode.IMAGE, num_poses=1,
        min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3))

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
landmarks = detect_pose(image_rgb, landmarker, w, h, flip_ensemble=True)
person_masker = PersonMasker(model_name="yolo26m-seg.pt", device="cpu")
person_mask = person_masker.segment(image_bgr)

print("加载 EdgeSAM…")
segmenter = edgesam_holds.EdgeSamHoldSegmenter(
    r"D:\Project\climbing_wall_search\mediapipe2\weights\edge_sam.pth", device="cpu")
segmenter.set_image(image_bgr)

BLACK_CENTER = (250, 646)


def mask_covers(mask_bool, pt, radius=20):
    yy, xx = np.ogrid[:h, :w]
    disk = (xx - pt[0])**2 + (yy - pt[1])**2 <= radius*radius
    return int((mask_bool & disk).sum())


def run_ring(px, py, radius, label):
    """端点 + 环形 radius 上 8 个锚点做正点"""
    body_cx = (landmarks[11].x + landmarks[12].x + landmarks[23].x + landmarks[24].x) * w / 4
    body_cy = (landmarks[11].y + landmarks[12].y + landmarks[23].y + landmarks[24].y) * h / 4
    neg = segmenter._make_negative_prompts(px, py, body_cx, body_cy)

    anchors = [np.array([px, py])]
    for t in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        anchors.append(np.array([px + radius * np.cos(t), py + radius * np.sin(t)]))
    point_coords = np.vstack(anchors + [neg])
    point_labels = np.array([1] * len(anchors) + [0] * len(neg), dtype=np.int64)

    eff_box_half = max(edgesam_holds.BOX_HALF_MIN, h // edgesam_holds.BOX_HALF_HEIGHT_DIV)
    eff_box_half = min(eff_box_half, edgesam_holds.BOX_HALF_MAX)
    ix, iy = int(round(px)), int(round(py))
    box = np.array([max(0, ix-eff_box_half), max(0, iy-eff_box_half),
                    min(w-1, ix+eff_box_half), min(h-1, iy+eff_box_half)], dtype=np.float64)

    masks, scores, _ = segmenter.predictor.predict(
        box=box, point_coords=point_coords, point_labels=point_labels,
        num_multimask_outputs=4, use_stability_score=True)
    if masks.ndim == 2:
        masks = masks[None, :, :]
        scores = np.array([float(scores)])

    print(f"\n{label}: 环形 r={radius} ({len(anchors)} 正点)")
    per = []
    for i, m in enumerate(masks):
        m = m & ~person_mask.astype(bool)
        cc = edgesam_holds._keep_main_component(m.copy(), ix, iy)
        if cc.sum() < 20:
            continue
        vote = edgesam_holds._pixel_color_vote(image_bgr, cc)
        cov = mask_covers(cc, BLACK_CENTER)
        per.append({'i': i, 'score': float(scores[i]), 'area': int(cc.sum()),
                    'cov': cov, 'top': vote['top_color'],
                    'votes': vote['votes'], 'mask': cc})
        print(f"  mask{i} score={scores[i]:.3f} area={int(cc.sum())} 黑岩点覆盖={cov}px "
              f"top={vote['top_color']} 黑={vote['votes'].get('黑',0)/sum(vote['votes'].values())*100:.1f}%")

    if not per:
        print("  → 无有效 mask")
        return

    # 融合1: 各 mask 像素合并投票（并集）
    union = np.zeros((h, w), dtype=bool)
    for p in per:
        union |= p['mask']
    uv = edgesam_holds._pixel_color_vote(image_bgr, union)
    print(f"  融合[像素并集]: top={uv['top_color']} 黑={uv['votes'].get('黑',0)/sum(uv['votes'].values())*100:.1f}%")

    # 融合2: 各 mask top 色计数
    from collections import Counter
    cnt = Counter(p['top'] for p in per)
    print(f"  融合[top色计数]: {dict(cnt)}")

    # 融合3: score 加权 top 色
    wsum = {}
    for p in per:
        wsum[p['top']] = wsum.get(p['top'], 0) + p['score']
    print(f"  融合[score加权]: { {k: round(v, 2) for k, v in sorted(wsum.items(), key=lambda x: -x[1])} }")

    # 融合4: 黑票占比（并集 mask 的黑票 vs 橙票）
    uv2 = uv['votes']
    total = sum(uv2.values())
    print(f"  黑票占比={uv2.get('黑',0)/total*100:.1f}% 橙票={uv2.get('橙',0)/total*100:.1f}%")


R = landmarks[16]
rx, ry = R.x * w, R.y * h
run_ring(rx, ry, 30, "R_hand")
run_ring(rx, ry, 40, "R_hand")

L = landmarks[15]
lx, ly = L.x * w, L.y * h
run_ring(lx, ly, 30, "L_hand")
