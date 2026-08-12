# 攀岩线路颜色识别 — 完整流程说明

> 项目根目录：`D:\Project\mediapipe`
> 当前时间：2026-08-12
> 维护版本：v1 (route_color.py / test_mp.py) + v2 (edgesam_holds.py / test_edge_sam.py)

---

## 文件地图

```
D:\Project\mediapipe\
├── 输入：                  img/1.jpg ~ 9.jpg
├── 输出 v1：              result_route_*.jpg     ← test_mp.py
├── 输出 v2：              result_edgesam_*.jpg   ← test_edge_sam.py
├── 模型权重：             weights/
│   ├── edge_sam.pth                              (~38 MB)  ← EdgeSAM 主权重
│   ├── edge_sam_3x.pth                           (~38 MB)
│   └── edge_sam_3x_{encoder,decoder}.onnx
├── 一级代码：
│   ├── test_mp.py                                v1 主程序（不动）
│   ├── test_edge_sam.py                          v2 主程序
│   ├── route_color.py                            v1 核心算法（detect_holds / 等）
│   ├── edgesam_holds.py                          v2 核心算法（EdgeSamHoldSegmenter）
│   ├── yolo_person_mask.py                       v2 子模块（PersonMasker / dilate / sample_edge_points）
│   ├── edgesam_stub.py                           v2 子模块（绕过训练依赖）
│   ├── calibrate.py / show_points.py            调试工具
│   ├── test_box_prompt_9.py                      实验脚本（box + detect_holds，已废弃）
│   └── test_box_v2_9.py                          实验脚本（box 不依赖 detect_holds，已并入生产）
└── MediaPipe 模型：
    ├── pose_landmarker_full.task
    ├── pose_landmarker_{lite,heavy}.task
    ├── hand_landmarker.task                       （README 说不能用）
    └── efficientdet_lite0.tflite                 （README 说不能用）
```

---

## v1 pipeline（`test_mp.py`，长期存在）

```
img/1.jpg
  ↓
[1] MediaPipe Pose 单次检测
  ↓
[2] MediaPipe Pose 水平翻转检测
  ↓
[3] merge_flip_landmarks()             ← 谁可见度高取谁 + 左右标签对调
  ↓
[4] route_color.detect_route_color()
    ├─ detect_holds()                  ← HSV 阈值 + 连通域
    │  └─ 过滤 BGR→HSV: S>=35, V>=40, 不在肤色
    │  └─ 形态学 OPEN ×2
    │  └─ 最大连通域当墙色 → 距离<40 也归墙
    │  └─ 阴影剔除（相对亮度<0.85）
    │  └─ 连通域 面积 80~10000, 长宽比≤5
    │
    ├─ 对 4 个端点 (L_hand 15, R_hand 16, L_foot 31, R_foot 32)
    │  ├─ 是否在某个 hold bbox 内？
    │  │  ├─ 是 → _is_body_attachment() 过滤（手表/护腕）
    │  │  └─ 否 → 环采样 (内 20px, 外 45px)
    │  ├─ 4 个端点投票，颜色名 (红/橙/黄/绿/青/蓝/紫/粉红)
    │  └─ 白路线兜底（置信度<60% 时查端点 40px 内低饱/高亮块）
    │
    └─ 多数票 = 线路色 + 置信度
  ↓
[5] draw_route_info()                 ← 绘制岩点 bbox / 骨架 / header
  ↓
result_route_*.jpg
```

**v1 的根本瓶颈（README 已知）**：
- 同色伪装墙（4.jpg 粉墙粉点）：颜色空间硬限制
- 白色岩点（9.jpg 白路线）：`S<60` 全过滤
- `hand_landmarker.task` 不可用：攀岩者背对相机

---

## v2 pipeline（`test_edge_sam.py`，新增）

```
img/1.jpg
  ↓
[一次性] load EdgeSAM (edge_sam.pth, CPU) + YOLO26 (yolo26n-seg.pt) + MediaPipe Pose
  ↓
[每张图]
  image_bgr ← cv2.imread(...)
  ↓
  [A] YOLO26.person_masker.segment(image_bgr)
      yolo26n-seg 在 COCO class=0 (person)
      → 合并所有 person 检测 → HxW bool (person_mask)
      ↓
      person_mask_d6   = dilate_mask(person_mask, 6)         ← 给 negative prompt
      person_mask_excl  = dilate_mask(person_mask, 15)        ← 给颜色统计排除
  ↓
  [B] detect_pose(image_bgr)             ← MediaPipe Pose 原图检测
      detect_pose(flip)                   ← 翻转图检测
      merge_flip_landmarks()               ← 谁可见度高取谁 + 左右标签对调
      → 4 个端点 (15, 16, 31, 32) landmark 坐标
  ↓
  EdgeSAM encoder (image_bgr)             ← 一次性 image embedding
  ↓
  对每个端点 (4 次, 顺序 loop)：
    segment_endpoint(px, py,
                     person_mask=person_d6,
                     exclude_mask=person_excl,
                     use_box_prompt=True,      # box prompt 启用
                     mask_subtract_person=True)  # mask 剥离人体
      │
      ├─ box prompt: 端点周围 box_half*2 px 方框 (默认 80 → 160x160)
      │              box 中心固定 = 端点，完全不依赖 detect_holds
      ├─ positive point: 端点 (px, py)
      ├─ negative points: 4-6 手工远处 + 12 body 边缘外
      └─ EdgeSAM.predict(
            box=box,
            point_coords=[1+端点, 16 neg],
            point_labels=[1, 0, 0, …],
            num_multimask_outputs=4,          # 4 个候选 (was 3)
            use_stability_score=True,
          )
         → 4 个 mask 候选
      ↓
      mask_subtract_person=True:
        mask -= person_mask（把人体/鞋/腿/手像素剥掉）
        但保留端点周围 18px 保护圈（否则端点本身在 person 内，
        扣光后 m[iy, ix] 恒 False → 全被过滤）
      ↓
      筛选候选：
        area: 100 < area < img_area * 0.3  (白路线 0.2)
        score >= 0.4
        包含端点 (px, py)
        in_body_ratio (与 person_mask 重叠 / mask_total) <= 0.65
      ↓
      排序：stability_score 降序优先，同分 area 降序（大 mask 优先）
        candidates.sort(key=lambda c: (-c[0], -c[1]))
      → 选中 chosen_mask
      ↓
      _robust_hsv_from_mask(chosen_mask,
                            exclude_mask=person_excl)
        ├─ 取出 mask & ~exclude_mask 内像素
        ├─ bbox 5×5 网格采 25 个点，每点 5×5 patch
        ├─ 圆形 hue 均值聚合
        └─ 返回 (h, s, v)
      ↓
      颜色判定顺序：
        V<60 AND S<40  → '未知(暗)'
        S<35 AND V>140  → '白'
        else            → route_color.hue_to_color_name(h, s)
  ↓
  4 个端点投票 → 多数票 = 线路色 + 置信度
  ↓
  draw_results()  ← mask 半透明覆盖 + header + 端点 + 标签
  ↓
result_edgesam_*.jpg
```

---

## 关键依赖 & 当前状态

| 依赖 | 来源 | 当前可用 |
|---|---|---|
| EdgeSAM `.pth` | 自带 / 手放 | ✅ `weights/edge_sam.pth` (38 MB) + 3x |
| EdgeSAM Python 包 | `pip install -e ./EdgeSAM`（绕过训练依赖） | ✅ |
| YOLO26 `.pt` | ultralytics 自动下载 | ✅ `yolo26n-seg.pt` (6.4 MB) |
| MediaPipe Pose `.task` | 自带 | ✅ `pose_landmarker_full.task` 等 |

`edgesam_stub.py` 在 `import edge_sam` 之前注入 `mmdet` / `mmengine` / `loralib` / `projects.EfficientDet` 假模块（仓库顶层训练-only 依赖），并 monkeypatch `torch.load(map_location='cpu')`（checkpoint 是 CUDA 训练的）。

---

## 已知未解决：9.jpg R_foot mask 与人体重合 / 颜色不准

### 已做改动（commit 历史）

| Commit | 改动 | 效果 |
|---|---|---|
| `dba9f22` | 颜色统计排除 person_mask 内像素 | hand 皮肤不拉偏 hue |
| `b7729b1` | `candidates.sort(key=lambda c: (-c[0], -c[1]))` — stability_score ↓, area ↓ tie-break | 完整 mask 优先 |
| `ec30e80` | **box prompt 启用**：box=端点周围 160px, `num_multimask_outputs=3→4`, 不依赖 detect_holds | 9.jpg R_foot mask 1688→13527 (完整 hold shape) |
| (前 commit) | `mask_subtract_person=True` 默认 + 端点 18px 保护圈 | **mask 与人体不重合** |
| `2a127c4` | **相对白判定** + **颜色聚类剔除** | 见下 |

### `2a127c4` 双修复

**1. 相对白判定（`_robust_hsv_from_mask` 路径 B）**
- 问题：暖光下白色 hold 的 S=35-75（不是绝对 <35），且 hold 只占 mask 一部分 → 绝对白判定 `S<35 占比>50%` 失效，9.jpg R_foot 被判"黄"
- 修复：对比 mask 与 mask 外扩 20px 环境环的中位 S/V — `S_ratio<0.60 AND maskS<75 AND V_ratio>=0.70` → 白
- 效果：9.jpg R_foot 黄→白；8.jpg 蓝 hold (S_ratio=0.50 但 maskS=114) 被绝对 S 挡掉不误判

**2. 颜色聚类剔除（`_keep_endpoint_cluster`）**
- 问题：9.jpg R_hand 的 EdgeSAM mask 同时包含上方紫色岩点 (73.6%) + 被抓的白色 hold (14.3%) → 颜色统计混成"粉红"
- 修复：mask 内手写 k-means (k=3, 无 sklearn 依赖)，只保留**端点像素所属的颜色簇**，剔除异色大块
- 保护规则（防回归）：仅当端点簇"白色系"（S_med<75, V_med>140）且占比 <25% 时剔除；纯色 hold（蓝/橙/绿）原样保留
- 效果：9.jpg R_hand 粉红→白；9.jpg 线路色 黄33%→**白67%**；图 1-8 零回归

### mask_subtract_person 效果（9.jpg）

| 端点 | 之前 (含人体) | 现在 (剥离后) |
|---|---|---|
| L_hand | 22273 | 2951 |
| R_hand | 15206 | 7476 |
| R_foot | 13527 | 7228 |

### 当前观察（9.jpg，commit `2a127c4` 后）

- ✅ **R_foot 白**（相对白判定）— mask 正确贴合白色 hold
- ✅ **R_hand 白**（颜色聚类剔除紫色岩点）— mask 不再含上方紫色岩点
- ✅ **线路色 = 白 67%**（L_hand=黄 是残余误判，hold 也是白色但 mask 含杂质）
- ⚠️ L_hand 仍判"黄"（area=2951，S=53.5 V=99.5 — 相对白判定 V_ratio 不足未触发）

### 剩余修复方向

| 修复 | 改动量 | 风险 |
|---|---|---|
| L_hand 同款处理（诊断 mask 内白色占比为何不足） | 中等 | 低 |
| 缩 `box_half` (80→50) 减小 box 贪心 | 一行 | 可能切不到完整 hold |
| `mask_input` 迭代扩展（EdgeSAM 官方推荐） | 中等 | 慢 1.5-2x |
| 换 EdgeSAM-3x 模型（语义更强） | 中等 | 慢一点，无逻辑风险 |

---

## 文件角色摘要

| 文件 | 角色 | 备注 |
|---|---|---|
| `test_mp.py` | v1 main entry | 长期存在，不动 |
| `route_color.py` | v1 核心逻辑 | `detect_holds` / `detect_route_color` 等 |
| `test_edge_sam.py` | v2 main entry | 调 3 个模型：YOLO26 / EdgeSAM / MediaPipe |
| `edgesam_holds.py` | v2 核心逻辑 | `EdgeSamHoldSegmenter` 类 |
| `yolo_person_mask.py` | v2 子 | `PersonMasker` / `dilate_mask` / `sample_edge_points` |
| `edgesam_stub.py` | v2 子 | stub mmdet / mmcv / loralib / projects.EfficientDet |
| `edgesam_stub.py` 修复 | 注入 stub 必须 | `import edgesam_stub` 必须在 `from edge_sam` 之前 |
| `test_box_prompt_9.py` | 实验：box + detect_holds | 失败案例（已废弃） |
| `test_box_v2_9.py` | 实验：box 不依赖 detect_holds | 验证后并入 production（`use_box_prompt=True`） |
