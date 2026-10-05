"""
诊断：2XRo 黑色线路 0005s 帧 R_hand mask 判断全过程可视化。

用户反馈：0005s 的 R_hand mask 跑到墙上去了（mask 覆盖大面积墙面）。
本脚本逐步复现 segment_endpoint 流程，把每一步画出来：
  1. 原帧 + 姿态端点 + body 中心
  2. box prompt（自适应 box）
  3. 负点分布（身体方向 1 + 环绕 5 + person mask 边缘 8）
  4. EdgeSAM 4 个候选 mask（multimask）各自 score/area
  5. 候选筛选（score/area/dist 过滤）+ 选中的 mask
  6. person mask 减除 → 主连通域 → 颜色统计
"""
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "EdgeSAM"))
import mediapipe as mp
import edgesam_holds
from test_edge_sam import detect_pose
from yolo_person_mask import PersonMasker

IMG = r"D:\Project\climbing_wall_search\mediapipe2\img_xhs_walls\2XRo7VbPgTe_wall_01_known_IMG_20260808_193753\frame_0005s.jpg"
OUT_DIR = r"D:\Project\climbing_wall_search\mediapipe2\dbg_rhand_0005s"
CKPT = r"D:\Project\climbing_wall_search\mediapipe2\weights\edge_sam.pth"
POSE_MODEL = r"D:\Project\climbing_wall_search\mediapipe2\pose_landmarker_full.task"

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- 加载 ----------------
image_bgr = cv2.imread(IMG)
h, w = image_bgr.shape[:2]
print(f"帧: {IMG}  ({w}x{h})")

# EdgeSAM
segmenter = edgesam_holds.EdgeSamHoldSegmenter(CKPT, device="cpu")
print("✅ EdgeSAM 加载完成")

# YOLO26 person masker
person_masker = PersonMasker(model_name="yolo26m-seg.pt", device="cpu")
print("✅ YOLO26 person masker 加载完成")

# MediaPipe pose
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
PoseLandmarksConnections = mp.tasks.vision.PoseLandmarksConnections
DrawingUtils = mp.tasks.vision.drawing_utils
DrawingStyles = mp.tasks.vision.drawing_styles
landmarker = PoseLandmarker.create_from_options(
    PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_MODEL),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
)

# ---------------- 1. 姿态 ----------------
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
landmarks = detect_pose(image_rgb, landmarker, w, h, flip_ensemble=True)
assert landmarks is not None, "未检测到人体"

shoulder_l, shoulder_r = landmarks[11], landmarks[12]
hip_l, hip_r = landmarks[23], landmarks[24]
body_cx = (shoulder_l.x + shoulder_r.x + hip_l.x + hip_r.x) * w / 4
body_cy = (shoulder_l.y + shoulder_r.y + hip_l.y + hip_r.y) * h / 4

R_hand = landmarks[16]
px, py = R_hand.x * w, R_hand.y * h
print(f"R_hand 端点: ({px:.1f}, {py:.1f})  vis={R_hand.visibility:.2f}")
print(f"body 中心: ({body_cx:.1f}, {body_cy:.1f})")

# 1 图：原帧 + 端点 + body
fig1 = image_bgr.copy()
cv2.circle(fig1, (int(px), int(py)), 8, (0, 0, 255), -1)
cv2.circle(fig1, (int(px), int(py)), 12, (255, 255, 255), 2)
cv2.circle(fig1, (int(body_cx), int(body_cy)), 8, (0, 255, 0), -1)
cv2.line(fig1, (int(px), int(py)), (int(body_cx), int(body_cy)), (0, 255, 0), 2)
cv2.putText(fig1, f"R_hand ({px:.0f},{py:.0f})", (int(px) + 20, int(py) - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
cv2.putText(fig1, f"body ({body_cx:.0f},{body_cy:.0f})", (int(body_cx) + 20, int(body_cy) - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
cv2.putText(fig1, "STEP1: pose endpoint + body center", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
cv2.imwrite(os.path.join(OUT_DIR, "1_pose.jpg"), fig1)
print("💾 1_pose.jpg")

# ---------------- 2. box prompt ----------------
eff_box_half = max(edgesam_holds.BOX_HALF_MIN,
                   min(h // edgesam_holds.BOX_HALF_HEIGHT_DIV,
                       edgesam_holds.BOX_HALF_MAX))
ix = max(0, min(int(round(px)), w - 1))
iy = max(0, min(int(round(py)), h - 1))
bx1 = max(0, ix - eff_box_half)
by1 = max(0, iy - eff_box_half)
bx2 = min(w - 1, ix + eff_box_half)
by2 = min(h - 1, iy + eff_box_half)
box = np.array([bx1, by1, bx2, by2], dtype=np.float64)
print(f"box_half={eff_box_half}  box=[{bx1},{by1},{bx2},{by2}]  (尺寸 {bx2-bx1}x{by2-by1})")

fig2 = image_bgr.copy()
cv2.rectangle(fig2, (bx1, by1), (bx2, by2), (0, 165, 255), 3)
cv2.circle(fig2, (ix, iy), 6, (0, 0, 255), -1)
cv2.putText(fig2, f"box_half={eff_box_half}", (bx1 + 8, by1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
cv2.putText(fig2, "STEP2: box prompt (adaptive)", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
cv2.imwrite(os.path.join(OUT_DIR, "2_box.jpg"), fig2)
print("💾 2_box.jpg")

# ---------------- 3. person mask + 负点 ----------------
person_mask = person_masker.segment(image_bgr)
print(f"person mask: {person_mask.sum()} px ({person_mask.mean()*100:.1f}%)")

neg_manual = edgesam_holds.EdgeSamHoldSegmenter._make_negative_prompts(
    px, py, body_cx, body_cy)
edge_pts = edgesam_holds._sample_person_edge_points(
    person_mask, n=edgesam_holds.PERSON_EDGE_NEG_N,
    max_radius=edgesam_holds.PERSON_EDGE_MAX_RADIUS)
neg = neg_manual.copy()
if len(edge_pts) > 0:
    neg = np.vstack([neg, edge_pts])

fig3 = image_bgr.copy()
# person mask 半透明紫
pm_u8 = person_mask.astype(np.uint8) * 255
overlay = fig3.copy()
overlay[pm_u8 > 0] = (255, 180, 180)
cv2.addWeighted(overlay, 0.25, fig3, 0.75, 0, fig3)
# box
cv2.rectangle(fig3, (bx1, by1), (bx2, by2), (0, 165, 255), 2)
# 正点
cv2.circle(fig3, (ix, iy), 8, (0, 255, 0), -1)
cv2.putText(fig3, "pos", (ix + 10, iy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
# 负点：手动（蓝）、person edge（红）
for j, (nx, ny) in enumerate(neg_manual):
    cv2.circle(fig3, (int(nx), int(ny)), 6, (255, 0, 0), -1)
    cv2.putText(fig3, f"n{j}", (int(nx) + 8, int(ny) + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
for j, (nx, ny) in enumerate(edge_pts):
    cv2.circle(fig3, (int(nx), int(ny)), 6, (0, 0, 255), -1)
cv2.putText(fig3, f"person={person_mask.mean()*100:.1f}%", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 130, 255), 2)
cv2.putText(fig3, "STEP3: person mask + negative prompts (blue=manual, red=edge)", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv2.imwrite(os.path.join(OUT_DIR, "3_negprompts.jpg"), fig3)
print(f"💾 3_negprompts.jpg  (neg total {len(neg)})")

# ---------------- 4. EdgeSAM 4 候选 ----------------
segmenter.set_image(image_bgr)
point_coords = np.vstack([[[px, py]], neg])
point_labels = np.array([1] + [0] * len(neg), dtype=np.int64)
masks, scores, _ = segmenter.predictor.predict(
    box=box,
    point_coords=point_coords,
    point_labels=point_labels,
    num_multimask_outputs=4,
    use_stability_score=True,
)
if masks.ndim == 2:
    masks = masks[None, :, :]
    scores = np.array([float(scores)])

img_area = w * h
area_cap = int(img_area * edgesam_holds.AREA_MAX_RATIO)
print(f"\n--- EdgeSAM {len(masks)} 个候选 ---")
candidates = []
for i, m in enumerate(masks):
    area = int(m.sum())
    ys_m, xs_m = np.where(m)
    dist = float(np.min(np.hypot(xs_m - ix, ys_m - iy))) if len(xs_m) else 9999
    ok_area = edgesam_holds.AREA_MIN <= area <= area_cap
    ok_score = float(scores[i]) >= edgesam_holds.SCORE_MIN
    ok_dist = dist <= edgesam_holds.DIST_MAX
    print(f"  mask#{i}: score={scores[i]:.3f}  area={area:6d} ({area/img_area*100:.1f}%)  "
          f"dist={dist:.1f}  [{'PASS' if ok_area and ok_score and ok_dist else 'FAIL'}]  "
          f"(area{'✓' if ok_area else '✗'} score{'✓' if ok_score else '✗'} dist{'✓' if ok_dist else '✗'})")
    if ok_area and ok_score and ok_dist:
        candidates.append((float(scores[i]), dist, area, m))

# 4 候选图：拼成 2x2
cand_imgs = []
for i, m in enumerate(masks):
    disp = image_bgr.copy()
    idx = m > 0
    blend = disp[idx].astype(np.float32) * 0.45 + np.array([0, 200, 255], np.float32) * 0.55
    disp[idx] = np.clip(blend, 0, 255).astype(np.uint8)
    cv2.circle(disp, (ix, iy), 6, (0, 0, 255), -1)
    area = int(m.sum())
    ys_m, xs_m = np.where(m)
    dist = float(np.min(np.hypot(xs_m - ix, ys_m - iy))) if len(xs_m) else 9999
    status = "PASS" if (edgesam_holds.AREA_MIN <= area <= area_cap
                        and float(scores[i]) >= edgesam_holds.SCORE_MIN
                        and dist <= edgesam_holds.DIST_MAX) else "FAIL"
    cv2.putText(disp, f"cand#{i} score={scores[i]:.3f} area={area} ({area/img_area*100:.1f}%) [{status}]",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
    cv2.putText(disp, f"cand#{i} score={scores[i]:.3f} area={area} ({area/img_area*100:.1f}%) [{status}]",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
    cv2.putText(disp, f"dist={dist:.1f}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(disp, f"dist={dist:.1f}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cand_imgs.append(disp)

# 拼接 2x2
h_c, w_c = cand_imgs[0].shape[:2]
grid = np.zeros((h_c * 2, w_c * 2, 3), dtype=np.uint8)
for i, img in enumerate(cand_imgs):
    r, c = divmod(i, 2)
    grid[r * h_c:(r + 1) * h_c, c * w_c:(c + 1) * w_c] = img
cv2.imwrite(os.path.join(OUT_DIR, "4_candidates.jpg"), grid)
print("💾 4_candidates.jpg")

# ---------------- 5. 候选筛选 + 选定 ----------------
if not candidates:
    print("\n❌ 无候选通过筛选 → 判为非岩点")
    chosen = None
else:
    candidates.sort(key=lambda c: (-c[0], c[1], -c[2]))
    chosen_score, chosen_dist, chosen_area, chosen = candidates[0]
    print(f"\n✅ 选定: score={chosen_score:.3f} dist={chosen_dist:.1f} area={chosen_area}")

    # 减 person
    chosen_raw = chosen.copy()
    chosen_person = chosen & ~person_mask.astype(bool)
    print(f"减 person: {chosen_raw.sum()} → {chosen_person.sum()}  (扣 {chosen_raw.sum() - chosen_person.sum()})")

    if chosen_person.sum() < edgesam_holds.COLOR_MIN_PIXELS:
        print(f"❌ 减 person 后 < {edgesam_holds.COLOR_MIN_PIXELS} px → 判为非岩点")

    # 主连通域
    chosen_cc = edgesam_holds._keep_main_component(chosen_person.copy(), ix, iy)
    print(f"主连通域: {chosen_person.sum()} → {chosen_cc.sum()}")

    # 颜色
    h_val, s_val, v_val = edgesam_holds._sample_color(image_bgr, chosen_cc)
    if v_val < edgesam_holds.DARK_V_MAX and s_val < edgesam_holds.DARK_S_MAX:
        color_name = "未知(暗)"
    elif s_val < edgesam_holds.WHITE_SAT_MAX:
        color_name = "白" if v_val >= edgesam_holds.WHITE_VAL_MIN else "黑"
    else:
        color_name = edgesam_holds.route_color.hue_to_color_name(h_val, s_val)
    print(f"颜色: HSV=({h_val:.1f},{s_val:.1f},{v_val:.1f}) → {color_name}")

    # 5 图：原始选定 mask vs 减 person vs 主连通域
    def mask_overlay(base, m, color_bgr, label, color_label=(0,0,0)):
        disp = base.copy()
        idx = m > 0
        disp[idx] = np.clip(disp[idx].astype(np.float32) * 0.4
                            + np.array(color_bgr, np.float32) * 0.6, 0, 255).astype(np.uint8)
        cv2.circle(disp, (ix, iy), 6, (0, 0, 255), -1)
        cv2.putText(disp, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_label, 2)
        return disp

    base = image_bgr.copy()
    pm_ov = base.copy(); pm_ov[pm_u8 > 0] = (255, 180, 180)
    base_pm = cv2.addWeighted(pm_ov, 0.2, base, 0.8, 0, base.copy())

    fig5a = mask_overlay(base_pm, chosen_raw, (0, 200, 255), "STEP5a: chosen mask (raw, before person subtract)")
    fig5b = mask_overlay(base_pm, chosen_person, (0, 200, 255), "STEP5b: after person subtract")
    fig5c = mask_overlay(base_pm, chosen_cc, (0, 200, 255), f"STEP5c: main component  HSV=({h_val:.0f},{s_val:.0f},{v_val:.0f}) -> {color_name}")
    hh = max(fig5a.shape[0], fig5b.shape[0], fig5c.shape[0])
    ww = fig5a.shape[1]
    strip = np.zeros((hh, ww * 3, 3), dtype=np.uint8)
    strip[:fig5a.shape[0], :ww] = fig5a
    strip[:fig5b.shape[0], ww:ww*2] = fig5b
    strip[:fig5c.shape[0], ww*2:ww*3] = fig5c
    cv2.imwrite(os.path.join(OUT_DIR, "5_after_person.jpg"), strip)
    print("💾 5_after_person.jpg")

    # 6 图：最终判定
    fig6 = image_bgr.copy()
    idx = chosen_cc > 0
    fig6[idx] = np.clip(fig6[idx].astype(np.float32) * 0.4
                        + np.array([0, 200, 255], np.float32) * 0.6, 0, 255).astype(np.uint8)
    cv2.circle(fig6, (ix, iy), 6, (0, 0, 255), -1)
    cv2.putText(fig6, f"FINAL: R_hand -> {color_name}  HSV=({h_val:.0f},{s_val:.0f},{v_val:.0f})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(fig6, f"FINAL: R_hand -> {color_name}  HSV=({h_val:.0f},{s_val:.0f},{v_val:.0f})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(OUT_DIR, "6_final.jpg"), fig6)
    print("💾 6_final.jpg")

print("\n完成。输出目录:", OUT_DIR)
