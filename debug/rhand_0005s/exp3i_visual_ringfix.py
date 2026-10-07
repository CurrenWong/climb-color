"""实验3i：可视化环形带黑修正判断逻辑（R_hand + L_hand 并排）。"""
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

sh_l, sh_r, hp_l, hp_r = landmarks[11], landmarks[12], landmarks[23], landmarks[24]
body_cx = (sh_l.x + sh_r.x + hp_l.x + hp_r.x) * w / 4
body_cy = (sh_l.y + sh_r.y + hp_l.y + hp_r.y) * h / 4

hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
yy, xx = np.ogrid[:h, :w]

D1, D2 = 20, 70
RATIO_MIN = 0.18

# 色名→BGR 图例色
NAME_BGR = {
    '红': (60, 60, 255), '橙': (0, 140, 255), '黄': (0, 215, 255),
    '绿': (80, 200, 60), '青': (220, 220, 0), '蓝': (220, 120, 30),
    '紫': (200, 80, 150), '粉红': (200, 110, 250), '白': (240, 240, 240),
    '黑': (30, 30, 30), '未知(暗)': (90, 90, 90),
}

CROP_HALF = 90  # 放大图半宽


def classify_pixel_name(hh, ss, vv):
    return edgesam_holds._classify_pixel(float(hh), float(ss), float(vv))


def build_panel(ep_name, idx, crop_center=None):
    lm = landmarks[idx]
    px, py = lm.x * w, lm.y * h
    ix, iy = int(round(px)), int(round(py))
    res = segmenter.segment_endpoint(px, py, body_cx, body_cy, person_mask=person_mask)
    mask = res['mask_bool']
    base_color = res['color_name']

    dist = np.hypot(xx - ix, yy - iy)
    ring = (dist >= D1) & (dist <= D2) & ~mask
    if person_mask is not None and person_mask.any():
        ring = ring & ~person_mask.astype(bool)
    ring_pix = hsv[ring]
    ring_names = np.array([classify_pixel_name(hh, ss, vv) for hh, ss, vv in ring_pix])
    ring_black = int((ring_names == '黑').sum())
    ring_total = len(ring_names)
    ratio = ring_black / ring_total if ring_total else 0.0
    fixed = (base_color != '黑' and ratio >= RATIO_MIN)
    final = '黑' if fixed else base_color

    # ---- 裁剪区域（含环带外圈 70px + 余量）----
    if crop_center is None:
        cx0, cy0 = ix, iy
    else:
        cx0, cy0 = crop_center
    x0 = max(0, cx0 - CROP_HALF); x1 = min(w, cx0 + CROP_HALF)
    y0 = max(0, cy0 - CROP_HALF); y1 = min(h, cy0 + CROP_HALF)
    crop = image_bgr[y0:y1, x0:x1].copy()
    scale = 3
    crop_big = cv2.resize(crop, ((x1-x0)*scale, (y1-y0)*scale), interpolation=cv2.INTER_NEAREST)

    # mask 轮廓（绿）
    mcrop = mask[y0:y1, x0:x1].astype(np.uint8)
    conts, _ = cv2.findContours(mcrop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in conts:
        cv2.drawContours(crop_big, [c * scale], -1, (80, 220, 80), 2)

    # 环带内黑票像素高亮（青）+ 非黑像素暗化
    rcrop = ring[y0:y1, x0:x1]
    names_crop = np.array([classify_pixel_name(hh, ss, vv) for hh, ss, vv in hsv[y0:y1, x0:x1][rcrop]])
    idxs = list(zip(*np.where(rcrop)))
    for k, (rr, cc) in enumerate(idxs):
        bl = 3
        if names_crop[k] == '黑':
            cv2.circle(crop_big, (cc*scale + bl, rr*scale + bl), bl, (255, 255, 0), -1)  # 青亮
        else:
            for dy in range(bl*2+1):
                for dx in range(bl*2+1):
                    yy2, xx2 = rr*scale+dy, cc*scale+dx
                    if 0 <= yy2 < crop_big.shape[0] and 0 <= xx2 < crop_big.shape[1]:
                        crop_big[yy2, xx2] = (crop_big[yy2, xx2] * 0.45).astype(np.uint8)

    # 环带边界圆（黄）
    cv2.circle(crop_big, ((ix - x0) * scale, (iy - y0) * scale), D1 * scale, (0, 220, 255), 2)
    cv2.circle(crop_big, ((ix - x0) * scale, (iy - y0) * scale), D2 * scale, (0, 220, 255), 2)
    # 端点（青十字）
    cv2.drawMarker(crop_big, ((ix - x0) * scale, (iy - y0) * scale), (255, 255, 0),
                   cv2.MARKER_CROSS, 24, 2)

    # 黑岩点真值位置（红十字，仅 R_hand 有参考意义；画在图上如可见）
    bpx, bpy = 250, 646
    if x0 <= bpx < x1 and y0 <= bpy < y1:
        cv2.drawMarker(crop_big, ((bpx - x0) * scale, (bpy - y0) * scale), (0, 0, 255),
                       cv2.MARKER_TILTED_CROSS, 30, 2)

    # 统计条（顶部）
    bar = np.full((56, crop_big.shape[1], 3), 245, dtype=np.uint8)
    cv2.putText(bar, f"{ep_name}  base={base_color}  ring_black={ratio*100:.1f}%",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2)
    cv2.putText(bar, f"ring={D1}-{D2}px black={ring_black}/{ring_total}  threshold={RATIO_MIN:.2f}  => {final}",
                (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2)
    if fixed:
        cv2.putText(bar, "FIXED->BLACK", (bar.shape[1]-170, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)

    panel = np.vstack([bar, crop_big])
    return panel, (ix, iy), base_color, ratio, fixed, final


# R_hand 面板
panel_r, pos_r, base_r, ratio_r, fixed_r, final_r = build_panel('R_hand', 16)
# L_hand 面板
panel_l, pos_l, base_l, ratio_l, fixed_l, final_l = build_panel('L_hand', 15)

# 上下文图：全帧 + 两个端点圆环
ctx = image_bgr.copy()
ctx = cv2.resize(ctx, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
sc = 0.5
for (ix, iy), color, tag in [(pos_r, (0, 220, 255), 'R'), (pos_l, (0, 220, 255), 'L')]:
    cv2.circle(ctx, (int(ix*sc), int(iy*sc)), int(D1*sc), color, 1)
    cv2.circle(ctx, (int(ix*sc), int(iy*sc)), int(D2*sc), color, 1)
    cv2.drawMarker(ctx, (int(ix*sc), int(iy*sc)), color, cv2.MARKER_CROSS, 20, 1)
cv2.putText(ctx, "Ring 20-70px around endpoints", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# 图例条
legend = np.full((36, ctx.shape[1], 3), 235, dtype=np.uint8)
cv2.putText(legend, "green=mask  yellow=ring  cyan dots=black-voted pixels  red X=black hold center  dark=non-black",
            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

top = np.vstack([ctx, legend])

# 左右并排（等高）
H = max(panel_r.shape[0], panel_l.shape[0])
def pad(p):
    if p.shape[0] < H:
        p = np.vstack([p, np.full((H - p.shape[0], p.shape[1], 3), 255, dtype=np.uint8)])
    return p
panel_r = pad(panel_r); panel_l = pad(panel_l)
# 画面坐标：L_hand(x=258) 在左，R_hand(x=291) 在右 → 面板顺序 L | R
bottom = np.hstack([panel_l, np.full((H, 8, 3), 255, dtype=np.uint8), panel_r])

# top 横向补白对齐 bottom
W = bottom.shape[1]
if top.shape[1] < W:
    extra = np.full((top.shape[0], W - top.shape[1], 3), 255, dtype=np.uint8)
    top = np.hstack([top, extra])
else:
    top = cv2.resize(top, (W, top.shape[0]), interpolation=cv2.INTER_AREA)

out = np.vstack([top, np.full((8, top.shape[1], 3), 255, dtype=np.uint8), bottom])

out_path = r"D:\Project\climbing_wall_search\mediapipe2\dbg_rhand_0005s\exp3i_visual_ringfix.png"
cv2.imwrite(out_path, out)
print(f"已保存: {out_path}")
print(f"R_hand: base={base_r} ring_black={ratio_r*100:.1f}% fixed={fixed_r} -> {final_r}")
print(f"L_hand: base={base_l} ring_black={ratio_l*100:.1f}% fixed={fixed_l} -> {final_l}")
