# 攀岩线路颜色识别 — 完整流程说明

> 项目根：`climb-color`（本地目录名可以是 `mediapipe2`，GitHub 仓库名为 `climb-color`）
> 最后更新：2026-10-06
> 生产路径：**v2**（`edgesam_holds.py` / `test_edge_sam.py`）；v1（`tools/test_mp.py`）保留作对照

---

## 文件地图

```
climb-color/
├── 主流程（生产）
│   ├── test_edge_sam.py          主入口：单帧识别 / 墙段聚合
│   ├── edgesam_holds.py          EdgeSamHoldSegmenter（box prompt / 候选筛选 / ring 黑修正）
│   ├── route_color.py            HSV 颜色分类 + 端点邻域像素投票（v1 v2 共用）
│   ├── yolo_person_mask.py       PersonMasker（YOLO26m 人体掩码）
│   └── edgesam_stub.py           绕过 mmdet/mmcv 训练依赖
├── 输入 / 输出
│   ├── img/                      单图测试集（1-9.jpg + 扩展 10-15）
│   ├── img_xhs_walls/            视频墙段帧（10 墙段 / 65 帧，5s 间隔）
│   ├── img_xhs_walls_1s/         加密采样实验帧（1s 间隔，gitignored）
│   └── result_edgesam_*.jpg      单帧识别结果（gitignored）
├── 实验与工具
│   ├── baselines/                论文对比基线（仅 HSV 4/10、detect_holds 6/10）
│   ├── tools/                    v1 主程序 test_mp.py、calibrate.py、show_points.py
│   ├── dwpose_exp/               DWPose 实验子项目（含自己的 README）
│   ├── vlm_exp/                  Qwen-VL 基线（5/10）
│   └── debug/                    探索性脚本（rhand_0005s/ = ring 黑票修正的发现过程）
├── weights/                      全部模型权重（gitignored）
│   ├── edge_sam.pth              EdgeSAM 主权重
│   ├── yolo26m-seg.pt            人体分割
│   ├── yolox_l.onnx / dw-ll_ucoco_384.onnx        DWPose
│   └── pose_landmarker_full.task MediaPipe Pose
├── 文档
│   ├── README.md / PIPELINE.md / LESSONS.md / CHANGELOG.md
│   ├── paper/                    论文（中英 Markdown + 图 + 数据 + DOCX 导出脚本）
│   └── docs/demo/                README 引用的 demo 图
└── EdgeSAM/                      EdgeSAM 源码 clone（gitignored）
```

---

## v1 pipeline（`tools/test_mp.py`，保留作对照）

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

## v2 pipeline（`test_edge_sam.py`，2026-08-13 简化版）

```
img/N.jpg
  │
  ├─ [1] YOLO26m → person_mask（算法=可视化用同一个，单 mask）
  ├─ [2] MediaPipe Pose (原图+翻转) → 4 个端点 (15,16,31,32)
  └─ [3] EdgeSAM encoder (一次性 image embedding)
        ↓
  对每个端点 (4 次)：
    ┌─────────────────────────────────────────────────────────┐
    │ segment_endpoint(px, py, body_cx, body_cy, person_mask) │
    │   input: box (端点±80px, 高分辨率自适应放大)              │
    │          + 1 正点 + 6 手工负点 + 8 person 边缘负点        │
    │   EdgeSAM → 4 mask 候选 → 筛选 score/面积/距端点近       │
    │   排序 (-score, +dist, -area) 选最佳                     │
    │   mask -= person_mask (单一 mask，无保护圈)              │
    │   mask 减 person 后像素<50 → "非岩点"                    │
    │   _keep_main_component (含端点域优先, 否则最大)          │
    │   _sample_color                                         │
    │     路径 A: 过半像素 S<35 AND V>140 → 白                  │
    │     路径 B: mask S 中位 < 75 AND m_s/r_s<0.60 → 白       │
    │     网格 5×5 patch 圆形 hue 均值 → (h, s, v)            │
    │   颜色名：暗 / 白 / else → hue_to_color_name             │
    └─────────────────────────────────────────────────────────┘
        ↓
  4 端点投票 → 多数票 = 线路色 + 置信度
        ↓
  draw_results (mask 半透明覆盖 + header + 端点 + 标签)
        ↓
result_edgesam_N.jpg
```

### v2 简化要点（2026-08-13）

对比之前累积的 hack 版本（7077d45），简化砍掉了：
- **保护圈机制**（PERSON_MASK_PROTECT_RADIUS=18 圆盘）→ 直接减 person，mask 减完 < 50 px → 非岩点
- **伪块检测**（专门针对 13.jpg 圆盘的 hack）→ 不需要（没圆盘就没伪块）
- **white_route_mode 白路线分支**（额外负点 + 收紧面积 + 降 score）→ 统一一套参数
- **双 person mask**（膨胀版给负点、未膨胀版给 subtract）→ 统一一个 mask
- **取色区域 <20% 判非岩点**（9.jpg 调优）→ 改为减 person 后像素 < 50 简单判断
- **候选双层筛选**（正常 + soft 兜底）→ 单层
- **主连通域高饱和亮色优先**（5.jpg 调优）→ 含端点优先，否则最大
- **DARK 黑色 hold 检测**（11.jpg dead code）→ 砍掉

保留的通用逻辑：
- **box prompt 自适应**（高分辨率图 box 按 h/20 放大，13.jpg 案例）→ 通用，非单图 hack
- **相对白路径 B**（环境环饱和度比，白 hold 在暖光/阴影下的通用处理）→ 9.jpg 不是单图特例
- **网格采样 5×5**（取色时避开边缘/墙像素）→ route_color 也用的通用方案

### v2 端点处理内循环细节（简化版）

**输入准备**（一次性）：
- `person_mask` = YOLO26m 攀岩者人体 mask（**单一版本**，算法=可视化共用）
- 4 个端点 (px, py) = 左手腕 15 / 右手腕 16 / 左脚尖 31 / 右脚尖 32

**EdgeSAM prompt**：
- box = 端点周围 box_half*2 px 方框
  - 基础：`box_half = max(80, h // 20)`，上限 250
  - **dwpose_exp 实验扩展**：`compute_box_half()` 额外参考 person_mask bbox 短边 × 0.6 上限
    — 攀岩者一只手/脚约占人体短边 40-60%，box 比人体大太多会切到远处无关 hold
    （line1_white frame02 实例：box=160 切到远处红色三角 hold a=14574 压过手边白 hold；
    box=128 后手边白 hold 候选 score 0.881 胜出）。生产 test_edge_sam.py 未启用此扩展。
- positive point = 端点
- negative points = 6 手工（1 身体方向 + 5 均匀围绕）+ 8 person mask 边缘外点
- `num_multimask_outputs=4`，4 个候选 mask

**候选筛选（单层）**：
- 面积 `100 < area < img_area * 0.3`
- `score >= 0.4`
- mask 距端点 ≤ DIST_MAX(60)px
- 排序：`(-score, +dist_to_endpoint, -area)` 选 best

**mask 后处理**：
- `mask -= person_mask`（单一 mask，无保护圈）
- mask 减 person 后像素 < COLOR_MIN_PIXELS(50) → "非岩点"
- `_keep_main_component`：含端点的连通域；缺失时取最大连通域

**颜色统计**（`_sample_color`）：
- 路径 A 绝对白：mask 内 S<WHITE_SAT_MAX(35) AND V>WHITE_VAL_MIN(140) 像素 > 50%
- 路径 B 相对白：mask S 中位 < WHITE_REL_SAT_MAX(75) AND m_s/r_s<WHITE_REL_RATIO(0.60)
- bbox 内 5×5 网格采 25 个点 × 5×5 patch；S<WHITE_SAT_MAX OR V<COLOR_VAL_MIN(40) 算"灰白"跳过
- 圆形 hue 均值聚合 → (h, s, v)

**颜色名判定**：
| 条件 | 颜色名 |
|---|---|
| V<60 AND S<40 | 未知(暗)（衣服/手套/手表） |
| S<WHITE_SAT_MAX(35) | 白（不要求 V>140） |
| else | `route_color.hue_to_color_name(h, s)` |

`HUE_BINS`：`(118,150)=紫`, `(150,170)=粉红`（按色相拆分；4.jpg 深粉红 H=163 走粉红 bin）

### 简化后结果对比（2026-08-13）

| 图 | 真值 | 简化前 | 简化后 |
|---|---|---|---|
| 1 | 紫 | 紫 75% | 紫 75% |
| 2 | 黄 | 黄 67% | 黄 75% |
| 3 | 绿 | 绿 100% | 绿 50% |
| 4 | 粉红 | 粉红 100% | 红 67% |
| 5 | 橙 | 橙 67% | 橙 50% |
| 6 | 黄 | 白 100% | 白 100%（EdgeSAM 切割问题）|
| 7 | 红 | 红 67% | 红 50% |
| 8 | 蓝 | 紫 100% | 紫 100%（蓝 hold 被 person 吞）|
| 9 | 白 | 白 100% | 白 75% |
| 10-15 | - | 略 | 略 |

**回退**：3/4/5/7 票数变散导致置信度下降（线路色仍多对），主要因为去掉双 person mask 让部分端点切到无关 mask；用户接受此回退以换取代码简洁通用。
**修复**：9.jpg 仍判白（之前相对白路径 B 砍了又加回）。

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
| (当前) | **排序 key 改成 `(-score, +dist_to_endpoint, -area)`** —— score 同分时优先选距端点近的 mask，避免选离端点远的"碰巧大 mask" | 8/9 无回归（潜在更稳） |
| `29eeb6a` (dwpose_exp) | **box 按 person_mask 短边 × 0.6 限制**（`compute_box_half()` 函数）；BOX_HALF_MIN 80→50；sys.path 顺序 bug 修复（11 脚本统一改为 `sys.path[0:0] = [...]` 切片插入，dwpose_exp 本地副本之前从未真正加载） | DWPose 全量 **10/10 命中** ✅，总耗时 523.6s（每帧 8.06s） |

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
| `tools/test_mp.py` | v1 main entry | 保留作对照，不动 |
| `route_color.py` | v1 核心逻辑 | `detect_holds` / `detect_route_color` 等 |
| `test_edge_sam.py` | v2 main entry | 调 3 个模型：YOLO26 / EdgeSAM / MediaPipe |
| `edgesam_holds.py` | v2 核心逻辑 | `EdgeSamHoldSegmenter` 类 |
| `yolo_person_mask.py` | v2 子 | `PersonMasker` / `dilate_mask` / `sample_edge_points` |
| `edgesam_stub.py` | v2 子 | stub mmdet / mmcv / loralib / projects.EfficientDet |
| `edgesam_stub.py` 修复 | 注入 stub 必须 | `import edgesam_stub` 必须在 `from edge_sam` 之前 |
| `test_box_prompt_9.py` | 实验：box + detect_holds | 失败案例（已废弃） |
| `test_box_v2_9.py` | 实验：box 不依赖 detect_holds | 验证后并入 production（`use_box_prompt=True`） |
