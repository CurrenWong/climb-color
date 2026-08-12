"""
攀岩线路颜色检测 v2
策略：先全图检测岩点（连通域），再把接触点匹配到最近的岩点，按岩点颜色投票。
这能解决 v1 的核心缺陷：手/鞋遮住岩点时，半径采样拿到的是手+墙+邻接岩点的混合色。
"""
import cv2
import numpy as np
from collections import Counter


# Hue (0-179) → 中文色名
# 色相圆形均值已修正（除2），现在是标准 OpenCV 色相：
#   红=0/180  橙≈15  黄≈30  绿≈60  青≈90  蓝≈120  紫≈150  粉红≈165-180
# 粉红 vs 紫：H>=150 是粉红/品红区（4.jpg 深粉红 hold H=163, S=144 之前被
# (118,170) 归紫）。粉红 vs 红 靠饱和度区分（粉红 S 低/中，真红 S 高）。
HUE_BINS = [
    ((0, 10),    '红'),
    ((10, 20),   '橙'),
    ((20, 30),   '黄'),
    ((30, 80),   '绿'),
    ((80, 95),   '青'),
    ((95, 118),  '蓝'),
    ((118, 150), '紫'),     # 蓝紫/紫 hold（1.jpg 肾形紫 hold H=127）
    ((150, 170), '粉红'),   # 粉红/品红 hold（4.jpg 深粉红 H=163）
    ((170, 180), '红'),
]

# 粉红判别：色相 ≥150 已归粉红 bin；此处的 S 条件用于 (150,170) 内
# 极高饱和（S>=200 接近真红/洋红）的兜底 —— 保持向后兼容 v1 行为。
PINK_HUE_MIN = 150
PINK_SAT_MAX = 200


# 肤色 HSV（仅用来排除攀岩者的身体，避免把胳膊/腿当成大岩点）
# 收窄到更典型的肤色范围，减少误伤暖色系岩点（黄/橙/红）
SKIN_HUE = (0, 22)
SKIN_SAT = (40, 170)
SKIN_VAL = (90, 220)


def hue_to_color_name(h: float, s: float = 255) -> str:
    if h >= PINK_HUE_MIN and s < PINK_SAT_MAX:
        return '粉红'
    for (lo, hi), name in HUE_BINS:
        if lo <= h < hi:
            return name
    return '未知'


# ============================================================
# 接触点定义
# ============================================================
CONTACT_POINTS = [
    ('L_wrist', 15, True),
    ('R_wrist', 16, True),
    ('L_index', 19, True),
    ('R_index', 20, True),
    ('L_foot',  31, False),
    ('R_foot',  32, False),
]


# ============================================================
# v2 核心：岩点检测
# ============================================================
def _robust_hold_hsv(hsv: np.ndarray, bbox: tuple) -> tuple | None:
    """
    在岩点 bbox 内部采多个点（中心 + 4 个象限），每点取 5x5 patch 的中位色相，
    再聚合。避免整个连通域统计时被边缘/墙/邻点像素拉偏色相。
    返回 (h, s, v) 或 None
    """
    bx, by, bw, bh = bbox
    h_img, w_img = hsv.shape[:2]
    pts = [
        (bx + bw // 2,       by + bh // 2),
        (bx + bw // 4,       by + bh // 4),
        (bx + 3 * bw // 4,   by + bh // 4),
        (bx + bw // 4,       by + 3 * bh // 4),
        (bx + 3 * bw // 4,   by + 3 * bh // 4),
    ]
    hues, sats, vals = [], [], []
    for px, py in pts:
        if not (2 <= px < w_img - 2 and 2 <= py < h_img - 2):
            continue
        patch = hsv[py - 2:py + 3, px - 2:px + 3].reshape(-1, 3)
        keep = (patch[:, 1] >= 35) & (patch[:, 2] >= 40)
        if keep.sum() < 3:
            continue
        pix = patch[keep]
        hr = pix[:, 0].astype(np.float64) * (2 * np.pi / 180.0)
        h_patch = float(np.degrees(np.arctan2(
            np.mean(np.sin(hr)), np.mean(np.cos(hr))
        )) / 2.0) % 180.0
        hues.append(h_patch)
        sats.append(float(np.median(pix[:, 1])))
        vals.append(float(np.median(pix[:, 2])))
    if not hues:
        return None
    # 聚合多个采样点（圆形均值）
    hr2 = np.array(hues) * (2 * np.pi / 180.0)
    h_final = float(np.degrees(np.arctan2(
        np.mean(np.sin(hr2)), np.mean(np.cos(hr2))
    )) / 2.0) % 180.0
    return h_final, float(np.median(sats)), float(np.median(vals))


def detect_holds(image: np.ndarray,
                 min_area: int = 80,
                 max_area: int = 10000,
                 min_sat: int = 35,
                 min_val: int = 40,
                 shadow_band: int = 15,
                 shadow_rel_thr: float = 0.85) -> tuple[list[dict], np.ndarray]:
    """
    把图片里所有"非墙、非皮肤"的高饱和连通块当成候选岩点。
    返回 (holds, labels)：
      holds: [{id, center, bbox, area, h, s, v, name}, ...]
      labels: 整图的 connected-component label 图（用于调试可视化）
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 肤色 mask（攀岩者的胳膊、腿、脸）
    skin_mask = (
        (hsv[:, :, 0] >= SKIN_HUE[0]) & (hsv[:, :, 0] < SKIN_HUE[1]) &
        (hsv[:, :, 1] >= SKIN_SAT[0]) & (hsv[:, :, 1] < SKIN_SAT[1]) &
        (hsv[:, :, 2] >= SKIN_VAL[0]) & (hsv[:, :, 2] < SKIN_VAL[1])
    )

    # 岩点 mask：饱和 + 亮 + 不是皮肤
    hold_mask = (
        (hsv[:, :, 1] >= min_sat) &
        (hsv[:, :, 2] >= min_val) &
        ~skin_mask
    ).astype(np.uint8)

    # 形态学清理：只去噪（OPEN），不做 CLOSE——CLOSE 会把相邻的不同色岩点桥接成一个大连通域，
    # 中位色相被平均出假颜色（比如绿+红+墙平均成"蓝"）
    kernel = np.ones((3, 3), np.uint8)
    hold_mask = cv2.morphologyEx(hold_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # 排除背景墙：找最大连通域的**平均颜色**（= 墙色），
    # 掩膜所有接近墙色的像素（包括被岩点切碎的墙碎片）。
    # 比"找最大连通域直接 mask"更鲁棒：即使墙不连通，也能识别墙色。
    img_area = image.shape[0] * image.shape[1]
    num_labels_pre, labels_pre, stats_pre, _ = cv2.connectedComponentsWithStats(
        hold_mask, connectivity=8
    )
    if num_labels_pre > 1:
        areas_pre = stats_pre[1:, cv2.CC_STAT_AREA]
        largest_idx = 1 + int(np.argmax(areas_pre))
        largest_area_pre = int(stats_pre[largest_idx, cv2.CC_STAT_AREA])
        if largest_area_pre > img_area * 0.1:  # 最大片 > 10% 图像（够大能算均值）
            # 用最大连通域的像素算平均颜色 = 墙色
            wall_pixels = hsv[labels_pre == largest_idx]
            wall_h = float(np.mean(wall_pixels[:, 0]))
            wall_s = float(np.mean(wall_pixels[:, 1]))
            wall_v = float(np.mean(wall_pixels[:, 2]))
            # 圆形距离（hue）
            d_h = np.abs(hsv[:, :, 0].astype(float) - wall_h)
            d_h = np.minimum(d_h, 180 - d_h)
            d_s = np.abs(hsv[:, :, 1].astype(float) - wall_s)
            d_v = np.abs(hsv[:, :, 2].astype(float) - wall_v)
            wall_dist = np.sqrt(d_h ** 2 + d_s ** 2 + d_v ** 2)
            wall_mask = wall_dist < 40  # 距离墙色 < 40 的像素 = 墙
            hold_mask[wall_mask] = False

    # 重新连通域（排墙后）
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        hold_mask.astype(np.uint8), connectivity=8
    )

    holds = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue

        w_box = int(stats[i, cv2.CC_STAT_WIDTH])
        h_box = int(stats[i, cv2.CC_STAT_HEIGHT])
        if w_box == 0 or h_box == 0:
            continue
        # 太瘦长的可能是阴影或胶带，跳过
        if max(w_box, h_box) / min(w_box, h_box) > 5:
            continue

        hold_pixels = hsv[labels == i]
        if len(hold_pixels) < 20:
            continue

        # 阴影剔除：hold 明显比周围环境暗（相对亮度 < shadow_rel_thr）→ 墙上的投影阴影
        # 实测 9.jpg：阴影块 V 均值 50-90、相对亮度 0.58-0.82；真岩点 ≥ 0.86。
        # 环境 = hold bbox 外扩 shadow_band px 且不属于本 hold 的像素
        bx0 = int(stats[i, cv2.CC_STAT_LEFT])
        by0 = int(stats[i, cv2.CC_STAT_TOP])
        bx1 = min(image.shape[1], bx0 + w_box + shadow_band)
        by1 = min(image.shape[0], by0 + h_box + shadow_band)
        bx0e = max(0, bx0 - shadow_band)
        by0e = max(0, by0 - shadow_band)
        env_region = hsv[by0e:by1, bx0e:bx1, 2].astype(np.float64)
        env_not_hold = labels[by0e:by1, bx0e:bx1] != i
        if env_not_hold.sum() >= 30:
            env_v_mean = float(np.mean(env_region[env_not_hold]))
            hold_v_mean = float(np.mean(hold_pixels[:, 2]))
            if hold_v_mean < env_v_mean * shadow_rel_thr:
                continue  # 阴影，跳过

        # 多点多 patch 采样岩点内部颜色（避开边缘/墙/邻点干扰）
        robust = _robust_hold_hsv(hsv, (int(stats[i, cv2.CC_STAT_LEFT]),
                                        int(stats[i, cv2.CC_STAT_TOP]),
                                        int(stats[i, cv2.CC_STAT_WIDTH]),
                                        int(stats[i, cv2.CC_STAT_HEIGHT])))
        if robust is None:
            continue
        h_circular, s_med, v_med = robust

        holds.append({
            'id': int(i),
            'center': (float(centroids[i][0]), float(centroids[i][1])),
            'bbox': (int(stats[i, cv2.CC_STAT_LEFT]),
                     int(stats[i, cv2.CC_STAT_TOP]), w_box, h_box),
            'area': area,
            'h': h_circular,
            's': float(np.median(hold_pixels[:, 1])),
            'v': float(np.median(hold_pixels[:, 2])),
            'name': hue_to_color_name(h_circular, float(np.median(hold_pixels[:, 1]))),
        })

    return holds, labels


# ============================================================
# v3 核心：四肢端点匹配 + 投票
# 思路：直接用四肢端点（双手腕 / 双脚尖），找距离最近的岩点中心，
# 评分 = area / (1 + distance)，四肢各投一票。
# ============================================================
LIMB_ENDPOINTS = [
    ('L_hand', 15),   # L_wrist
    ('R_hand', 16),   # R_wrist
    ('L_foot', 31),   # L_foot_index（脚尖，踩点处）
    ('R_foot', 32),   # R_foot_index
]


def sample_local_color(image: np.ndarray, x: float, y: float,
                       inner: int = 20, outer: int = 45,
                       body_x: float | None = None,
                       body_y: float | None = None) -> dict | None:
    """
    方向加权环采样端点周围区域的主色。
    - 内半径 inner 跳过手/脚主体
    - 外半径 outer 采岩点环绕
    - 若提供 body_x/body_y（身体中心），按"从身体到端点方向"加权：
      在该方向上的像素权重大，反方向（身体侧）权重=0。
      这样手/脚指向的岩点（远端）会盖过身体侧的其他东西。
    - 皮肤 + 低饱和过滤
    - 饱和度 × 方向权重 加权圆形均值
    """
    h_img, w_img = image.shape[:2]
    x, y = int(x), int(y)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    Y, X = np.ogrid[:h_img, :w_img]
    dx_pix, dy_pix = X - x, Y - y
    dist = np.sqrt(dx_pix ** 2 + dy_pix ** 2)
    ring = (dist >= inner) & (dist <= outer)
    if ring.sum() < 30:
        return None

    # 方向权重：从身体到端点的方向上权重=1，反方向=0
    if body_x is not None and body_y is not None:
        bx, by = int(body_x), int(body_y)
        ux, uy = x - bx, y - by
        norm = np.sqrt(ux ** 2 + uy ** 2)
        if norm >= 1:
            ux, uy = ux / norm, uy / norm
        else:
            ux, uy = 0.0, 0.0
        # 点积：环像素在 (ux,uy) 方向上为正（远端），反方向为负
        outward = dx_pix * ux + dy_pix * uy
        dir_w = np.maximum(outward, 0) / outer  # 归一化到 [0, 1]
    else:
        dir_w = np.ones_like(dist)

    # 过滤皮肤 + 低饱和
    keep = (hsv[:, :, 1] >= 35) & (hsv[:, :, 2] >= 40)
    skin = (
        (hsv[:, :, 0] < 25) & (hsv[:, :, 1] >= 30) &
        (hsv[:, :, 1] < 180) & (hsv[:, :, 2] >= 80) & (hsv[:, :, 2] < 230)
    )
    keep = keep & ~skin

    mask = ring & keep
    if mask.sum() < 30:
        return None

    pix = hsv[mask]                # (N, 3)
    wts = dir_w[mask] * (pix[:, 1].astype(np.float64) / 255.0)  # 方向 × 饱和度
    if wts.sum() < 1e-6:
        return None

    hr = pix[:, 0].astype(np.float64) * (2 * np.pi / 180.0)
    h_mean = float(np.degrees(np.arctan2(
        np.sum(np.sin(hr) * wts),
        np.sum(np.cos(hr) * wts)
    )) / 2.0) % 180.0
    return {
        'h': h_mean,
        's': float(np.median(pix[:, 1])),
        'v': float(np.median(pix[:, 2])),
        'name': hue_to_color_name(h_mean, float(np.median(pix[:, 1]))),
        'samples': int(mask.sum()),
    }


def _is_body_attachment(hsv: np.ndarray, bbox: tuple,
                        body_cx: float, body_cy: float,
                        band: int = 20, thr: float = 0.40) -> bool:
    """
    判断一个 hold 是否"嵌在身体里"（手表/护腕/手部碎片/衣服等）。
    规律：真岩点在墙上，皮肤只出现在身体那一侧（攀爬者伸手去抓时，
    皮肤从身体方向盖住岩点一侧）；而手表/护腕嵌在手臂里，皮肤从
    四面八方包围它——包括远离身体的那一侧。
    规则：把 hold bbox 外扩 band px 的环形带，按"身体→hold中心"方向投影，
    分身体侧(<0)与远离侧(>0)。若**远离侧**皮肤占比 > thr → 身体附件。
    """
    h_img, w_img = hsv.shape[:2]
    bx, by, bw, bh = bbox
    x0, y0 = max(0, bx - band), max(0, by - band)
    x1, y1 = min(w_img, bx + bw + band), min(h_img, by + bh + band)
    if x1 - x0 < 5 or y1 - y0 < 5:
        return False

    hx, hy = bx + bw / 2.0, by + bh / 2.0
    ux, uy = hx - body_cx, hy - body_cy
    n = np.hypot(ux, uy)
    if n < 1:
        return False
    ux, uy = ux / n, uy / n

    # 皮肤 mask（与 detect_holds 一致）
    skin = (
        (hsv[:, :, 0] >= SKIN_HUE[0]) & (hsv[:, :, 0] < SKIN_HUE[1]) &
        (hsv[:, :, 1] >= SKIN_SAT[0]) & (hsv[:, :, 1] < SKIN_SAT[1]) &
        (hsv[:, :, 2] >= SKIN_VAL[0]) & (hsv[:, :, 2] < SKIN_VAL[1])
    )

    Y, X = np.ogrid[y0:y1, x0:x1]
    proj = (X - hx) * ux + (Y - hy) * uy
    far_region = proj > 0
    if far_region.sum() < 10:
        return False
    far_skin = skin[y0:y1, x0:x1] & far_region
    far_ratio = float(far_skin.sum()) / float(far_region.sum())
    return far_ratio > thr


# ============================================================
# 翻转集成（flip ensemble）
# ============================================================
# 水平翻转后的左右标签对调映射（MediaPipe 左/右是解剖学视角，
# 翻转后模型会把左右标反——实测 4.jpg 翻转检测的 L_shoulder 实际是
# 解剖学 R_shoulder）。因此使用翻转检测结果时，需按此映射交换标签。
_FLIP_LR_PAIR = {
    1: 2, 2: 1,      # 眼内
    3: 4, 4: 3,      # 眼
    5: 6, 6: 5,      # 眼外
    7: 8, 8: 7,      # 耳
    9: 10, 10: 9,    # 嘴角
    11: 12, 12: 11,  # 肩
    13: 14, 14: 13,  # 肘
    15: 16, 16: 15,  # 腕
    23: 24, 24: 23,  # 髋
    25: 26, 26: 25,  # 膝
    27: 28, 28: 27,  # 踝
    29: 30, 30: 29,  # 脚跟
    31: 32, 32: 31,  # 脚
}


class _LM:
    """轻量 landmark 兼容对象（与 mediapipe NormalizedLandmark 的字段一致）"""
    __slots__ = ('x', 'y', 'z', 'visibility', 'presence')

    def __init__(self, x, y, z, visibility, presence):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.visibility = float(visibility)
        self.presence = float(presence)


def merge_flip_landmarks(orig_landmarks, flip_landmarks,
                         w: int, swap_ratio: float = 1.2) -> list:
    """
    翻转集成：对每个关节，取"原图检测"与"翻转检测"中可见度更高的那个。

    为什么需要：MediaPipe 对某些极端姿态（高抬腿/遮挡/背对）在单一方向
    上定位失败，可见度掉到 0.03；但同一姿态水平翻转后再检测，可见度能
    恢复到 0.9+（实测 4.jpg 左腿）。原因可能是模型的训练数据对某些
    朝向/遮挡模式有偏好。

    翻转检测结果的还原：
      1. 坐标翻回：翻转图 x' = 1 - x（归一化）
      2. 标签对调：flip_landmarks 的左/右标反了（_FLIP_LR_PAIR）
    选择规则：flip 可见度 > 原图可见度 × swap_ratio 时采用 flip。

    返回：合并后的 landmark 列表（原图坐标系，每个元素有 x/y/z/
    visibility/presence 属性，兼容 detect_route_color 与 draw_landmarks）。
    """
    merged = []
    for i in range(33):
        po = orig_landmarks[i]
        pf = flip_landmarks[_FLIP_LR_PAIR.get(i, i)]
        if pf.visibility > po.visibility * swap_ratio:
            # 用翻转检测：x 翻回原图系，标签已由 _FLIP_LR_PAIR 对调
            merged.append(_LM(1.0 - pf.x, pf.y, pf.z, pf.visibility, pf.presence))
        else:
            merged.append(_LM(po.x, po.y, po.z, po.visibility, po.presence))
    return merged


# ============================================================
# 白色岩点检测（白路线兜底）
# ============================================================
# 背景：白色/浅色岩点（S<60）过不了彩色岩点的 min_sat=35 过滤，
# 连通域检测不到，环采样被饱和的高饱和墙色淹没。
# 但白岩点是暗墙上**最亮的物体**：高分辨率下可检测为
# "低饱和(S<60) + 高亮(V>140)" 的连通块，且紧贴端点（40px 内）。
# 该检测只在"彩色投票不明确（置信度 < white_gate_conf）"时启用，
# 避免误伤彩色路线（彩色岩点上的镁粉/高光也会形成白块）。
WHITE_SAT_MAX = 60
WHITE_VAL_MIN = 140
WHITE_MIN_AREA = 100
WHITE_RADIUS = 40


def detect_white_endpoints(image: np.ndarray, landmarks, w: int, h: int,
                           radius: int = WHITE_RADIUS,
                           min_area: int = WHITE_MIN_AREA,
                           s_max: int = WHITE_SAT_MAX,
                           v_min: int = WHITE_VAL_MIN) -> dict:
    """
    检测每个端点附近（radius px 内）的白色岩点连通块。
    返回 {端点名: 白块面积}，只包含命中（面积 >= min_area）的端点。
    只考虑可见度 >= 0.3 的端点。
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1].astype(np.float32)
    V = hsv[:, :, 2].astype(np.float32)
    skin = (
        (hsv[:, :, 0] >= SKIN_HUE[0]) & (hsv[:, :, 0] < SKIN_HUE[1]) &
        (hsv[:, :, 1] >= SKIN_SAT[0]) & (hsv[:, :, 1] < SKIN_SAT[1]) &
        (hsv[:, :, 2] >= SKIN_VAL[0]) & (hsv[:, :, 2] < SKIN_VAL[1])
    )
    white_mask = ((S < s_max) & (V > v_min) & ~skin).astype(np.uint8)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(
        white_mask, connectivity=8)
    comps = [(int(stats[i, cv2.CC_STAT_AREA]),
              float(centroids[i][0]), float(centroids[i][1]))
             for i in range(1, num) if int(stats[i, cv2.CC_STAT_AREA]) >= min_area]

    hits = {}
    for ep_name, idx in LIMB_ENDPOINTS:
        lm = landmarks[idx]
        if lm.visibility < 0.3:
            continue
        px, py = lm.x * w, lm.y * h
        best = None
        for a, cx, cy in comps:
            d = np.hypot(cx - px, cy - py)
            if d <= radius and (best is None or a > best[0]):
                best = (a, d)
        if best is not None:
            hits[ep_name] = best[0]
    return hits


def detect_route_color(image: np.ndarray, landmarks, w: int, h: int,
                       max_match_distance: float = 150,
                       min_area: int = 80, max_area: int = 10000,
                       min_sat: int = 35, min_val: int = 40,
                       torso_penalty: float = 0.05,
                       bbox_margin: float = 5.0,
                       body_band: int = 20,
                       body_thr: float = 0.40,
                       shadow_band: int = 15,
                       shadow_rel_thr: float = 0.85,
                       white_gate_conf: float = 0.6) -> dict:
    """
    1. 全图检测岩点
    2. 每个四肢端点：若在某个岩点 bbox 内（+margin）→ 用该岩点中心+颜色
    3. 不在任何岩点 bbox 内 → 环采样局部主色兜底（跳过手/脚主体）
       （hold 检测可能把岩点碎片化漏掉，但环上像素主色仍有效）
    4. 局部也采不到 → 判定为"非岩点"（腾空/踩墙），不参与投票
    """
    holds, _ = detect_holds(image, min_area=min_area, max_area=max_area,
                             min_sat=min_sat, min_val=min_val,
                             shadow_band=shadow_band,
                             shadow_rel_thr=shadow_rel_thr)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 身体中心（肩 + 髋 平均），用于环采样的方向加权
    shoulder_l = landmarks[11]
    shoulder_r = landmarks[12]
    hip_l = landmarks[23]
    hip_r = landmarks[24]
    body_cx = (shoulder_l.x + shoulder_r.x + hip_l.x + hip_r.x) * w / 4
    body_cy = (shoulder_l.y + shoulder_r.y + hip_l.y + hip_r.y) * h / 4

    samples = []
    matched_hold_ids: set[int] = set()

    for ep_name, idx in LIMB_ENDPOINTS:
        lm = landmarks[idx]
        px, py = lm.x * w, lm.y * h

        if lm.visibility < 0.3:
            samples.append({
                'point': ep_name, 'name': '低置信度', 'samples': 0,
                'px': (px, py), 'visibility': float(lm.visibility),
            })
            continue

        # 1) 找"包含端点"的岩点（bbox 内 + margin），取面积最大的
        on_hold = None
        for ho in holds:
            bx, by, bw, bh = ho['bbox']
            if (bx - bbox_margin <= px <= bx + bw + bbox_margin and
                    by - bbox_margin <= py <= by + bh + bbox_margin):
                if on_hold is None or ho['area'] > on_hold['area']:
                    on_hold = ho

        if on_hold is not None:
            # 身体附件过滤：手表/护腕/手部碎片嵌在身体里，远离侧也被皮肤包围。
            # 真岩点（攀爬者正在抓的）只有身体侧有皮肤。被判定为身体附件 →
            # 不用它投票，降级走局部环采样（src='local'）。
            if _is_body_attachment(hsv_image, on_hold['bbox'],
                                   body_cx, body_cy,
                                   band=body_band, thr=body_thr):
                on_hold = None

        if on_hold is not None:
            # 端点在岩点 bbox 内 → 用岩点中心 + 颜色
            matched_hold_ids.add(on_hold['id'])
            samples.append({
                'point': ep_name,
                'name': on_hold['name'],
                'h': on_hold['h'],
                's': on_hold['s'],
                'v': on_hold['v'],
                'samples': on_hold['area'],
                'px': (px, py),
                'hold_center': on_hold['center'],
                'hold_bbox': on_hold['bbox'],
                'distance': float(np.hypot(px - on_hold['center'][0],
                                           py - on_hold['center'][1])),
                'src': 'hold',
                'visibility': float(lm.visibility),
            })
            continue

        # 2) 不在任何岩点 bbox 内 → 方向加权环采样（body→endpoint 方向上权重高）
        local = sample_local_color(image, px, py, body_x=body_cx, body_y=body_cy)
        if local is None:
            # 局部也采不到有效颜色 → 腾空/踩墙，非岩点
            samples.append({
                'point': ep_name,
                'name': '非岩点',
                'samples': 0,
                'px': (px, py),
                'visibility': float(lm.visibility),
            })
            continue

        samples.append({
            'point': ep_name,
            'name': local['name'],
            'h': local['h'],
            's': local['s'],
            'v': local['v'],
            'samples': local['samples'],
            'px': (px, py),
            'hold_center': (px, py),
            'hold_bbox': None,
            'distance': 0.0,
            'src': 'local',
            'visibility': float(lm.visibility),
        })

    # 投票（'非岩点' 和 '低置信度' 不参与）
    color_names = {name for (_, _), name in HUE_BINS} | {'粉红', '未知'}
    valid = [s for s in samples if s.get('name') in color_names]

    if not valid:
        return {
            'route_color': '无法判断',
            'samples': samples,
            'votes': {},
            'confidence': 0.0,
            'holds': holds,
            'matched_hold_ids': [],
        }

    votes = Counter(s['name'] for s in valid)
    route_color, top_count = votes.most_common(1)[0]
    confidence = top_count / len(valid)

    # ---- 白路线兜底 ----
    # 彩色投票不明确（置信度 < white_gate_conf）时，启用白色岩点检测：
    # 白岩点（S<60）过不了彩色岩点的 min_sat 过滤，环采样被高饱和墙色淹没，
    # 但高分辨率下可检测为端点附近（40px）的低饱和高亮连通块。
    # 命中的端点改投"白"。彩色路线置信度足够时不触发，避免镁粉/高光误伤。
    if confidence < white_gate_conf:
        white_hits = detect_white_endpoints(image, landmarks, w, h)
        if white_hits:
            new_votes = Counter()
            for s in valid:
                if s['point'] in white_hits:
                    new_votes['白'] += 1
                else:
                    new_votes[s['name']] += 1
            new_route, new_top = new_votes.most_common(1)[0]
            new_conf = new_top / len(valid)
            if new_top >= top_count:  # 白检测让票数更集中才采用
                route_color, votes, confidence = new_route, new_votes, new_conf
                # 同步更新样本名，可视化一致
                for s in samples:
                    if s['point'] in white_hits and s.get('name') in color_names:
                        s['name'] = '白'

    return {
        'route_color': route_color,
        'samples': samples,
        'votes': dict(votes),
        'confidence': confidence,
        'holds': holds,
        'matched_hold_ids': list(matched_hold_ids),
    }


# ============================================================
# 可视化
# ============================================================
def _hsv_to_bgr(h: float, s: float, v: float) -> tuple:
    color = np.array([[[int(h), int(s), int(v)]]], dtype=np.uint8)
    return tuple(int(c) for c in cv2.cvtColor(color, cv2.COLOR_HSV2BGR)[0, 0])


def draw_route_info(image: np.ndarray, route_info: dict) -> None:
    """
    绘制（v3 精简版）：
      - 顶部 header：线路色 + 置信度 + 票数
      - 采样点（四肢端点）：小黄点 + 黑圈（不覆盖画面）
      - 匹配岩点：只画彩色矩形边框 + 中心小十字（不填充，露出岩点本色）
      - 采样点 -> 岩点中心：细白连线
      - 标签放在采样点旁边，明确标"采样点"
    """
    h_img, w_img = image.shape[:2]

    # ---------- Header ----------
    header_h = 80
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w_img, header_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)

    route_color = route_info['route_color']
    confidence = route_info['confidence']
    n_holds = len(route_info.get('holds', []))
    n_matched = len(route_info.get('matched_hold_ids', []))

    cv2.putText(image, f"ROUTE COLOR  holds={n_holds}  matched={n_matched}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(image, f"Color: {route_color}    Confidence: {confidence:.0%}",
                (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    votes = route_info.get('votes', {})
    vote_str = "  ".join(f"{c}={n}" for c, n in sorted(votes.items(), key=lambda x: -x[1]))
    if vote_str:
        cv2.putText(image, vote_str, (10, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 200, 255), 1, cv2.LINE_AA)

    # ---------- 采样点 + 匹配岩点 ----------
    # 采样点中心统一红色（容易认），标签偏移方向（L 左 R 右）
    LABEL_DX = {'L_hand': -1, 'R_hand': +1, 'L_foot': -1, 'R_foot': +1}
    LABEL_DIST = 40  # 标签离采样点的像素距离

    for s in route_info['samples']:
        x, y = int(s['px'][0]), int(s['px'][1])

        if s.get('samples', 0) == 0:
            # 没匹配到岩点：小灰叉
            cv2.drawMarker(image, (x, y), (160, 160, 160),
                           markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)
            cv2.putText(image, f"{s['point']}采样点({s['name']})", (x + 12, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
            continue

        # 1) 采样点中心：红色实心点（白底圈，画面里最醒目）
        cv2.circle(image, (x, y), 5, (255, 255, 255), -1)
        cv2.circle(image, (x, y), 3, (0, 0, 255), -1)

        # 2) 匹配岩点：彩色矩形边框（1px，不填充）+ 中心粉色标记
        if s.get('hold_bbox'):
            bx, by, bw, bh = s['hold_bbox']
            bgr = _hsv_to_bgr(s['h'], s['s'], s['v'])
            cv2.rectangle(image, (bx, by), (bx + bw, by + bh), bgr, 1)
        if s.get('hold_center'):
            hx, hy = int(s['hold_center'][0]), int(s['hold_center'][1])
            # 岩点中心：粉色圆点 + 十字（与红色采样点区分）
            cv2.circle(image, (hx, hy), 6, (255, 128, 255), 1)
            cv2.circle(image, (hx, hy), 3, (255, 128, 255), -1)
            cv2.drawMarker(image, (hx, hy), (255, 128, 255), cv2.MARKER_CROSS, 10, 1)
            # 3) 采样点 -> 岩点中心细连线
            cv2.line(image, (x, y), (hx, hy), (255, 255, 255), 1, cv2.LINE_AA)

        # 4) 标签：离采样点 LABEL_DIST 远，带引线
        dx = LABEL_DX.get(s['point'], +1)
        label = f"{s['point']}采样点->{s['name']}({s['distance']:.0f}px)"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        if dx > 0:
            bx, by = x + LABEL_DIST, y - 14
        else:
            bx, by = x - LABEL_DIST - tw, y - 14
        bx = max(2, min(bx, w_img - tw - 4))
        by = max(th + 4, min(by, h_img - 4))
        # 引线：采样点 -> 标签
        cv2.line(image, (x, y), (bx + tw // 2, by - th // 2), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(image, (bx - 2, by - th - 2), (bx + tw + 2, by + 2), (0, 0, 0), -1)
        cv2.putText(image, label, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)
