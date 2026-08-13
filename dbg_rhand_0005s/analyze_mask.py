"""定量分析：R_hand 选定 mask 的几何 + 颜色分布，判断 mask 是否跑到墙上。"""
import os, sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "EdgeSAM"))
import edgesam_holds

IMG = r"D:\Project\climbing_wall_search\mediapipe2\img_xhs_walls\2XRo7VbPgTe_wall_01_known_IMG_20260808_193753\frame_0005s.jpg"
image_bgr = cv2.imread(IMG)
h, w = image_bgr.shape[:2]

# 复现选定的 mask#1（score=0.828 area=3550）
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
res = segmenter.segment_endpoint(px, py, body_cx, body_cy, person_mask=person_mask)
mask = res["mask_bool"]
print(f"端点=({px:.1f},{py:.1f})  mask area={mask.sum()}  color={res['color_name']} HSV=({res['h']:.1f},{res['s']:.1f},{res['v']:.1f})")

ys, xs = np.where(mask)
print(f"\n--- mask 几何 ---")
print(f"bbox: x=[{xs.min()},{xs.max()}] y=[{ys.min()},{ys.max()}]  (宽 {xs.max()-xs.min()}, 高 {ys.max()-ys.min()})")
print(f"mask 中心=({xs.mean():.0f},{ys.mean():.0f})  端点=({px:.0f},{py:.0f})")

# mask 内的颜色直方图（BGR→HSV）
hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
pix = hsv[ys, xs]
print(f"\n--- mask 内像素颜色分布 ---")
for name, cond in [
    ("白  (S<35,V>140)", (pix[:,1] < 35) & (pix[:,2] > 140)),
    ("黑  (S<35,V<140)", (pix[:,1] < 35) & (pix[:,2] <= 140)),
    ("低饱 (S<35)", pix[:,1] < 35),
    ("橙   (H 8-30,S>60,V>100)", (pix[:,0] >= 8) & (pix[:,0] <= 30) & (pix[:,1] > 60) & (pix[:,2] > 100)),
    ("高饱 (S>100)", pix[:,1] > 100),
    ("亮 (V>150)", pix[:,2] > 150),
]:
    n = cond.sum()
    print(f"  {name:28s}: {n:5d}  ({n/len(pix)*100:5.1f}%)")

# S/V 分位数
print(f"\n  S 中位数={np.median(pix[:,1]):.0f}  V 中位数={np.median(pix[:,2]):.0f}")
print(f"  H 分布: {np.percentile(pix[:,0], [10,25,50,75,90]).round(0)}")

# 端点局部 vs mask 整体颜色对比
patch = hsv[max(0,int(py)-20):int(py)+20, max(0,int(px)-20):int(px)+20].reshape(-1,3)
print(f"\n--- 端点 40x40 patch 颜色 (手部区域) ---")
print(f"  S 中位数={np.median(patch[:,1]):.0f}  V 中位数={np.median(patch[:,2]):.0f}")
white_patch = (patch[:,1] < 35) & (patch[:,2] > 140)
print(f"  白像素占比={white_patch.sum()/len(patch)*100:.1f}%")
dark_patch = (patch[:,1] < 35) & (patch[:,2] <= 140)
print(f"  黑像素占比={dark_patch.sum()/len(patch)*100:.1f}%")

# 墙面色（远离人体、远离 hold 的背景）
bg_ys, bg_xs = np.where(np.ones_like(mask))
bg_patch = hsv[300:500, 100:500].reshape(-1,3)  # 端点左侧大片区域
print(f"\n--- 墙面背景参考 (x 100-500, y 300-500) ---")
print(f"  S 中位数={np.median(bg_patch[:,1]):.0f}  V 中位数={np.median(bg_patch[:,2]):.0f}")

# mask 主连通域数量
n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
print(f"\n--- 连通域 ---")
print(f"  连通域数(含背景)={n_lbl}")
for i in range(1, n_lbl):
    print(f"  cc#{i}: area={stats[i, cv2.CC_STAT_AREA]}, bbox=({stats[i,cv2.CC_STAT_LEFT]},{stats[i,cv2.CC_STAT_TOP]} {stats[i,cv2.CC_STAT_WIDTH]}x{stats[i,cv2.CC_STAT_HEIGHT]})")
