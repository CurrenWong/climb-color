"""从主仓库 LESSONS_LEARNED.md 抽取第 10-12 章 + 追加第 13 章 -> mediapipe2/LESSONS.md"""
import io

SRC = r"D:\Project\climbing_wall_search\LESSONS_LEARNED.md"
DST = r"D:\Project\climbing_wall_search\mediapipe2\LESSONS.md"

with io.open(SRC, encoding='utf-8') as f:
    lines = f.read().split('\n')

# 第 10 章从第 562 行（1-based）开始
tail = '\n'.join(lines[561:]).rstrip() + '\n'

header = """# LESSONS — EdgeSAM / DWPose 路线工程经验（mediapipe2）

> 本文件收录 **mediapipe2 项目**（EdgeSAM 攀岩线路色识别）的工程经验，
> 对应主仓库 `LESSONS_LEARNED.md` 的第 10-13 章。
> 其它路线（VLM、MediaPipe 传统 CV、FastSAM、MobileSAM/V2）的经验见主仓库。
>
> 章节：
> - 第 10 章 环形带黑修正（核心创新）
> - 第 11 章 YOLO26 pose 替代实验（否决）
> - 第 12 章 DWPose 替代实验
> - 第 13 章 DWPose ONNX NMS 性能 bug（12x 加速）

---

"""

chap13 = """
---

## 第十三章：DWPose ONNX NMS 性能 bug —— 单类 NMS 带来 12x 加速 (2026-08-13)

### 现象

做 ablation 实验时发现 DWPose 全量回归从 8.06 s/帧 变成 **48.8 s/帧**（慢 6 倍）。
同版本代码、同机器、CPU 占用仅 13-24%（16 逻辑核不饱和）。

### 诊断链（三步定位）

| 步骤 | 脚本 | 发现 |
| --- | --- | --- |
| 1. 分阶段计时 | `debug/bench_frame.py` | `DWPose.run = 43.1 s`，而 EdgeSAM 4 端点仅 2.4 s、person mask 1.8 s、set_image 1.1 s（都正常） |
| 2. 裸 session 对照 | `debug/bench_ort.py` | onnxruntime 直接跑 yolox_l：**0.36 s** → 不是推理慢 |
| 3. detect 内部拆解 | `debug/bench_detect_stages.py` | **`_multiclass_nms = 41.93 s`**，输出 160075 个检测框 |

### 根因

`dwpose_onnx.DWPoseONNX.detect()` 调用 `_multiclass_nms(boxes, scores, nms_thr, score_thr)`，
该函数对 **80 个 COCO 类别逐个**执行：

```python
for cls_ind in range(num_classes):        # 80 次
    valid = cls_scores > score_thr        # 0.1
    keep = self._nms(vboxes, vscores, nms_thr)   # 纯 Python while 循环 NMS
```

问题在于 **阈值 0.1 太低 + yolox 输出 sigmoid 后概率普遍 > 0.1**，每类过滤后
仍剩数千框，80 类的 Python `while` NMS 累积 41.9 s，输出 16 万框。

### 修复

我们只关心 person（COCO class 0）。改为只对 class 0 做一次 NMS：

```python
cls0 = 1.0 / (1.0 + np.exp(-predictions[:, 5]))   # person logit → prob
scores0 = obj * cls0
valid = scores0 > DET_SCORE_THR
keep = self._nms(boxes_xyxy[valid], scores0[valid], DET_NMS_THR)
final_boxes = boxes_xyxy[valid][keep]
final_scores = scores0[valid][keep]
```

**结果**：`detect` 34.5 s → **3.39 s**（12x），全量 65 帧 496.5 s（7.64 s/帧），
与修复前命中率完全一致（10/10）。

### 为什么之前没发现

这个性能坑被 **sys.path 顺序 bug**（第九章后续）掩盖：DWPose 脚本此前实际加载
根目录的 `dwpose_onnx.py`，修好 sys.path 让 `dwpose_exp/` 本地副本真正生效后才暴露。

### 可复用经验

1. **性能回归先问"加载的是哪份代码"**：多份同名副本 + sys.path 顺序问题会让
   一个改动看似"引入性能退化"，实际只是加载路径变了。
2. **分阶段计时 + 裸算子对照**：`bench_frame`（阶段）→ `bench_ort`（裸 kernel）
   → `bench_detect_stages`（函数内部）三步可定位到具体函数行。
3. **通用检测器只取所需类别**：COCO 80 类的通用后处理在只需 1 类时是纯粹浪费；
   ultralytics/mmdet 内部都有"按类过滤后 NMS"的优化，手写实现时要注意。
4. **纯 Python NMS 在数千框规模上不可用**：O(n²) 的 while 循环在 2000+ 框时
   即达秒级；生产代码应使用 `cv2.dnn.NMSBoxes` 或向量化实现。
"""

with io.open(DST, 'w', encoding='utf-8') as f:
    f.write(header + tail + chap13)

print(f"WROTE {DST}")
print(f"  source lines 562-end: {len(lines) - 561} lines")
