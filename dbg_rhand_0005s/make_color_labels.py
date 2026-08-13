"""颜色标签图：cand3 mask 内每个像素按颜色分类着色 + 端点位置，直观展示黑岩点与端点空间关系。"""
import os, sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "EdgeSAM"))
import edgesam_holds

IMG = r"D:\Project\climbing_wall_search\mediapipe2\img_xhs_walls\2XRo7VbPgTe_wall_01_known_IMG_20260808_193753\frame_0005s.jpg"
OUT = r"D:\Project\climbing_wall_search\mediapipe2\dbg_rhand_0005s\7_color_labels.jpg"
image_bgr = cv2.imread(IMG)
h, w = image_bgr.shape[:2]

import mediapipe as mp
from test_edge_sam import detect_pose
from yolo_person_mask import PersonMasker

segmenter = edgesam_holds.EdgeSamHoldSegmenter(
    r"D:\Project\climbing_wall_search\mediapipe2\weights\edge_sam.pth", device="cpu")
person_masker = PersonMasker(model_name="yolo26m-seg.pt", device="cpu")

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
sh_l, sh_r, hp_l, hp_r = landmarks[11], landmarks[12], landmarks[23], landmarks[24]
body_cx = (sh_l.x + sh_r.x + hp_l.x + hp_r.x) * w / 4
body_cy = (sh_l.y + sh_r.y + hp_l.y + hp_r.y) * h / 4
R_hand = landmarks[16]
px, py = R_hand.x * w, R_hand.y * h
person_mask = person_masker.segment(image_bgr)

segmenter.set_image(image_bgr)
eff_box_half = max(edgesam_holds.BOX_HALF_MIN,
                   min(h // edgesam_holds.BOX_HALF_HEIGHT_DIV, edgesam_holds.BOX_HALF_MAX))
ix = max(0, min(int(round(px)), w - 1))
iy = max(0, min(int(round(py)), h - 1))
bx1, by1 = max(0, ix - eff_box_half), max(0, iy - eff_box_half)
bx2, by2 = min(w - 1, ix + eff_box_half), min(h - 1, iy + eff_box_half)
box = np.array([bx1, by1, bx2, by2], dtype=np.float64)

neg = edgesam_holds.EdgeSamHoldSegmenter._make_negative_prompts(px, py, body_cx, body_cy)
edge_pts = edgesam_holds._sample_person_edge_points(
    person_mask, n=edgesam_holds.PERSON_EDGE_NEG_N,
    max_radius=edgesam_holds.PERSON_EDGE_MAX_RADIUS)
if len(edge_pts) > 0:
    neg = np.vstack([neg, edge_pts])
point_coords = np.vstack([[[px, py]], neg])
point_labels = np.array([1] + [0] * len(neg), dtype=np.int64)
masks, scores, _ = segmenter.predictor.predict(
    box=box, point_coords=point_coords, point_labels=point_labels,
    num_multimask_outputs=4, use_stability_score=True)
if masks.ndim == 2:
    masks = masks[None, :, :]
    scores = np.array([float(scores)])

hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
m3 = masks[3].astype(bool)

# 颜色标签图：放大端点周围区域 3x，逐像素着色
zoom = 3
x0, x1 = max(0, ix - 130), min(w, ix + 130)
y0, y1 = max(0, iy - 150), min(h, iy + 150)
crop_w, crop_h = x1 - x0, y1 - y0
canvas = cv2.resize(image_bgr[y0:y1, x0:x1], (crop_w * zoom, crop_h * zoom),
                    interpolation=cv2.INTER_NEAREST)

# 颜色分类着色（在 canvas 坐标系）
def classify(h, s, v):
    if s < 35:
        return "黑" if v < 140 else "白"
    hh = h
    if 8 <= hh <= 30: return "橙"
    if hh < 8 or hh >= 170: return "红"
    if 30 < hh <= 60: return "黄"
    if 60 < hh <= 95: return "绿"
    if 95 < hh <= 130: return "青"
    if 130 < hh <= 155: return "蓝"
    return "紫" if hh < 170 else "红"

COLORS = {"黑": (30, 30, 30), "白": (255, 255, 255), "橙": (0, 128, 255),
          "红": (0, 0, 255), "黄": (0, 255, 255), "绿": (0, 255, 0),
          "青": (255, 255, 0), "蓝": (255, 0, 0), "紫": (255, 0, 255)}

# 只在 mask 内着色，保持背景原样
for yy in range(y0, y1):
    for xx in range(x0, x1):
        if m3[yy, xx]:
            hh, ss, vv = hsv[yy, xx]
            name = classify(hh, ss, vv)
            c = COLORS[name]
            cy0, cx0 = (yy - y0) * zoom, (xx - x0) * zoom
            canvas[cy0:cy0+zoom, cx0:cx0+zoom] = c

# 端点 + 圆盘标注
ex, ey = (ix - x0) * zoom, (iy - y0) * zoom
cv2.circle(canvas, (ex, ey), 6 * zoom, (0, 255, 0), 3)
for r in [30, 60]:
    cv2.circle(canvas, (ex, ey), r * zoom, (0, 255, 0), 1)
cv2.putText(canvas, "endpoint", (ex + 8, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# 黑像素区域中心标注
black = (hsv[:,:,1] < 35) & (hsv[:,:,2] < 140) & m3
by_, bx_ = np.where(black)
if len(bx_):
    b_cx, b_cy = int(np.mean(bx_)), int(np.mean(by_))
    b_ex, b_ey = (b_cx - x0) * zoom, (b_cy - y0) * zoom
    cv2.circle(canvas, (b_ex, b_ey), 8, (0, 0, 255), -1)
    cv2.putText(canvas, f"black-hold center ({b_cx},{b_cy})", (b_ex - 260, b_ey - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    print(f"黑岩点中心=({b_cx},{b_cy})  端点=({ix},{iy})  距离={np.hypot(b_cx-ix, b_cy-iy):.0f}px")

cv2.putText(canvas, "CAND#3 mask color labels (black/gray=黑, white=白, orange=橙)", (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv2.imwrite(OUT, canvas)
print(f"💾 {OUT}  ({canvas.shape[1]}x{canvas.shape[0]})")
