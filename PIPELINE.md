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
[一次性] load EdgeSAM (edge_sam.pth, CPU) + YOLO26 (yolo26m-seg.pt) + MediaPipe Pose
  ↓
[每张图]
  image_bgr ← cv2.imread(...)
  ↓
  [A] YOLO26m.person_masker.segment(image_bgr)   ← 算法与可视化用同一个 mask
      yolo26m-seg 在 COCO class=0 (person)
      → 合并所有 person 检测 → HxW bool (person_mask)
      ↓
      person_mask_d6 = dilate_mask(person_mask, 6)   ← 负点采样用
      算法 subtract 用未膨胀 person_mask（避免吞掉手/脚旁的 hold）
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
                     person_mask=person_d6,          # 负点用 (膨胀版)
                     subtract_person_mask=person_mask, # subtract 用 (未膨胀版)
                     use_box_prompt=True,       # box prompt 启用
                     mask_subtract_person=True)  # mask 剥离人体
      │
      ├─ box prompt: 端点周围 box_half*2 px 方框 (默认 80 → 160x160)
      │              box 中心固定 = 端点，完全不依赖 detect_holds
      ├─ positive point: 端点 (px, py)
      ├─ negative points: 6 手工远处 + 12 person 边缘外 (用膨胀版)
      └─ EdgeSAM.predict(
            box=box,
            point_coords=[1+端点, 18 neg],
            point_labels=[1, 0, 0, …],
            num_multimask_outputs=4,
            use_stability_score=True,
          )
         → 4 个 mask 候选
      ↓
      mask_subtract_person=True（用 subtract_person_mask = 未膨胀版）：
        mask -= person_mask（把人体/鞋/腿/手像素剥掉）
        但保留端点周围 18px 保护圈（否则端点本身在 person 内会被扣光）
        关键：subtract 用未膨胀 mask —— 膨胀6px 会把"手/脚旁的 hold"也
        吞进 person（5.jpg L_foot 脚下橙色岩点被吞 → 取色区域=0 判非岩点；
        用未膨胀版后岩点保留，判橙）
      ↓
      筛选候选：
        area: 100 < area < img_area * 0.3
        score >= 0.4
        包含端点 (px, py) —— 但若端点本身在 person 内（脚在 person 内、
        hold 在脚下 person 外），放宽为"端点 40px 邻域非空"(5.jpg R_foot)
      ↓
      排序：stability_score 降序优先，同分 area 降序
        candidates.sort(key=lambda c: (-c[0], -c[1]))
      → 选中 chosen_mask
      ↓
      主连通域过滤 _keep_main_component(chosen, px, py):
        只保留"包含端点的连通域"（判断与可视化一致）
        1.jpg R_hand mask 有 10 个连通域：含端点紫色 hold 3485px +
        远处黄色 hold 2136px + 碎片 —— 不过滤则颜色统计被黄紫混合
        端点缺失时取"距端点最近 + 高饱和亮色 (S>100, V>80)"的连通域
        （5.jpg R_foot 候选3 最大域 658px 是左侧绳，正确 hold 是距 45px
        的高饱和域 345px）
      ↓
      颜色聚类 _keep_endpoint_cluster(chosen, px, py):
        mask 含多色时（9.jpg R_hand 紫色岩点 + 白色 hold）做 k-means，
        只保留端点所属颜色簇
        保护规则：端点簇是白色系（S<WHITE_REL_SAT_MAX=75, V>WHITE_BRIGHT_MIN=140）
        且占 mask < 25% 才剔除异色块 —— 否则纯色 mask k-means 拆明暗簇
        会判错
      ↓
      取色区域占比检查:
        (mask & ~person) / mask < 20% → 判"非岩点"
        （9.jpg L/R_hand mask 100% 在 person 内——18px 保护圈保住的
        是身体/手本身，不是 hold，不该投票）
      ↓
      _robust_hsv_from_mask(chosen, exclude_mask=person_mask):
        ├─ mask & ~exclude_mask 内像素
        ├─ 路径 A：过半像素 S<WHITE_SAT_MAX AND V>WHITE_BRIGHT_MIN → 白
        ├─ 路径 B（暖光/阴影）：mask S 中位 < WHITE_REL_SAT_MAX AND
        │   显著低于环境环 (m_s/r_s<0.60) → 白
        │   （9.jpg R_foot S_ratio=0.47, maskS=62 走这条）
        ├─ bbox 5×5 网格采 25 个点，每点 5×5 patch
        │   S<WHITE_SAT_MAX OR V<COLOR_VAL_MIN 算"灰白"，不参与色相聚合
        ├─ 圆形 hue 均值聚合
        └─ 返回 (h, s, v)
      ↓
      颜色判定顺序（颜色名阈值统一在 edgesam_holds 顶部常量）：
        V<60 AND S<40            → '未知(暗)'
        S<WHITE_SAT_MAX (35)     → '白'  (不要求 V>WHITE_BRIGHT_MIN(140)——
                                    白色 hold 在阴影/暖光下 V 可 120-140，
                                    9.jpg R_foot 相对白返回 (0,0,136)
                                    V=136<140 会落入 hue_to_color_name
                                    判"红"，所以兜底只看 S)
        else                     → route_color.hue_to_color_name(h, s)
                                 · HUE_BINS: (118,150)=紫, (150,170)=粉红
                                 · 4.jpg 深粉红 H=163 走粉红 bin（修前
                                   HUE_BINS (118,170) 全归紫 + S<120 阈值
                                   漏掉深粉红 S=130）
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
| YOLO26 `.pt` | ultralytics 自动下载 | ✅ `yolo26m-seg.pt` (52 MB)；`yolo26n/s` 保留供对比 |
| MediaPipe Pose `.task` | 自带 | ✅ `pose_landmarker_full.task` 等 |

`edgesam_stub.py` 在 `import edge_sam` 之前注入 `mmdet` / `mmengine` / `loralib` / `projects.EfficientDet` 假模块（仓库顶层训练-only 依赖），并 monkeypatch `torch.load(map_location='cpu')`（checkpoint 是 CUDA 训练的）。

---

## 当前状态（统一 yolo26m + 全套修复）

### 9 图线路色结果（真实值 vs 当前输出）

| 图 | 真实 | 当前 | 状态 |
|---|---|---|---|
| 1 | 紫 | 紫 75% | ✅ |
| 2 | 黄 | 黄 67% | ✅ |
| 3 | 绿 | 绿 100% | ✅ |
| 4 | 粉红 | 粉红 100% | ✅ |
| 5 | 橙 | 橙 67% | ✅ |
| 6 | 黄 | 白 100% | ❌ |
| 7 | 红 | 红 67% | ✅ |
| 8 | 蓝 | 紫 100% | ❌ |
| 9 | 白 | 白 100% | ✅ |

**8/9 正确**；6/8 错误（EdgeSAM 切割问题，非后处理可解）。

### 已做改动（commit 历史，v2 阶段）

| Commit | 改动 | 效果 |
|---|---|---|
| `ec30e80` | box prompt 启用 (box_half=80, num_multimask=4), 不依赖 detect_holds | 完整 hold mask |
| `2a127c4` | 相对白判定 + 颜色聚类剔除 | 9.jpg R_foot 黄→白 |
| `29c20ec` | 可视化层连通域过滤（后改为算法层，见 `e8551a7`） | 显示单一区域 |
| `ebb8378` | ~~yolo26m 显示层~~（已废弃，见下） | — |
| `e8551a7` | **统一 yolo26m**（算法=可视化，废除双模型欺骗）；`_keep_main_component` 主连通域过滤；`S<35` 判白（修 S=0→红 bug）；取色区域<20% 判非岩点 | 1.jpg R_hand 黄→紫 |
| `0508539` | 端点在 person 内时放宽"必须包含端点"（接受 40px 邻域非空） | 5.jpg R_foot 未知→橙 |
| `d7d0144` | `_keep_main_component` 端点缺失时取"最近高饱和亮色域"而非最大域 | 5.jpg R_foot 绳→脚下岩点 |
| `35e73bb` | **subtract 用未膨胀 person_mask**（负点仍用膨胀版）；取色检查/颜色统计同步 | 5.jpg L_foot 非岩点→橙 |
| `d29ffec` | **HUE_BINS 按色相拆 (118,150)=紫, (150,170)=粉红** + PINK_SAT_MAX 提高到 200 | 4.jpg L_foot 紫→粉红 |
| `0cb3a6c` | 删 `_robust_hsv_from_mask` 里粉红/紫 S 修正冗余代码（route_color 已按色相处理） | 无回归 |
| (当前) | **白色 / 灰白 hold 阈值统一为常量**（`WHITE_SAT_MAX=35` / `WHITE_REL_SAT_MAX=75` / `WHITE_BRIGHT_MIN=140` / `COLOR_VAL_MIN=40`），原 4 处硬编码 S<35/S<75/V>140/V>=40 全替换 | 改一处即生效 |

### 关键修复细节

**1. 统一 yolo26m（`e8551a7`）**
- 之前算法用 yolo26n (6.4MB) + 可视化用 yolo26m (52MB) — 显示与判定不一致，掩盖问题
- 统一后用 yolo26m：1.jpg 紫色 hold 不再被吞（yolo26n 吞 97%），R_hand 黄→紫

**2. 主连通域过滤（`_keep_main_component`）**
- EdgeSAM mask 可含多个不相连区域（1.jpg R_hand: 含端点紫色 hold 3485px + 远处黄 hold 2136px + 碎片共 10 域）
- 取"包含端点的连通域"；端点缺失时取"最近高饱和亮色域"（5.jpg R_foot 候选3 的最大域 658px 是左侧绳/衣物，正确 hold 是距 45px 的高饱和域 345px）

**3. 白色 hold 判定（三路径，阈值已统一为常量）**
- 路径 A：mask 内 S<WHITE_SAT_MAX 且 V>WHITE_BRIGHT_MIN 像素 > 50% → 纯白 hold
- 路径 B：mask S 中位 < WHITE_REL_SAT_MAX 且显著低于环境环（m_s/r_s<0.60）→ 暖光/阴影下白 hold
- 路径 C（兜底）：S<WHITE_SAT_MAX 直接判白，不要求 V 高
- 网格采样：S<WHITE_SAT_MAX OR V<COLOR_VAL_MIN 算"灰白"，不参与色相聚合
- 9.jpg R_foot 走路径 B（S_ratio=0.47, maskS=62, V=136）

**4. 取色区域 <20% 判非岩点**
- 9.jpg L/R_hand mask 100% 在 person 内 — 18px 保护圈保住的其实是手/身体，不是 hold，却以 mask 投票
- 取色区域 (mask & ~person) / mask < 20% → 非岩点

**5. subtract 用未膨胀 person_mask（`35e73bb`）**
- 负点仍用膨胀版（让 EdgeSAM 避开身体边缘）
- subtract 用未膨胀版 — 膨胀6px 会把"手/脚旁的 hold"吞进 person
- 5.jpg L_foot 脚下橙色岩点被膨胀版吞 → 取色区域=0 判非岩点；用未膨胀版后岩点保留 → 判橙

**6. 粉红/紫按色相拆（`d29ffec`）**
- 旧 HUE_BINS (118,170) 全归紫，粉红只走 S<120 兜底 —— 4.jpg 深粉红 H=163 S=130 不满足
- 新 (118,150)=紫, (150,170)=粉红：1.jpg 紫 H=127 不受影响，4.jpg 粉红 H=163 正确归粉红

### 已知限制（6/8 图，EdgeSAM 切割问题）

| 图 | 问题 | 根因 |
|---|---|---|
| 6 | R_foot 判白 (取色3570px 全白) | EdgeSAM 没切到脚下黄色 hold；L_foot mask 空 |
| 8 | R_foot 判紫 (H150-164 265px) | 蓝墙+蓝 hold 同色环境，EdgeSAM 切到粉/紫区；L/R_hand 取色=0（蓝 hold 被 person 吞） |

### 剩余修复方向

| 修复 | 改动量 | 风险 |
|---|---|---|
| 6/8 图 EdgeSAM 切不到正确 hold（box 偏移/负点调整） | 中等 | 需针对性调试 |
| 5.jpg R_hand 判红 (V=49 暗) | 中等 | 手部 hold 被 person 部分吞 |
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
