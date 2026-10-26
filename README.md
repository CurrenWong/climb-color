# ClimbColor

[简体中文](README_zh.md) | **English**

> Recognise the colour of the route a climber is currently attempting, from climbing
> video frames (purple / yellow / green / blue / red / pink / white / black / cyan).
> Body-anchored visual prompting + EdgeSAM semantic segmentation + pixel-level colour voting.

[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python](https://img.shields.io/badge/python-3.12-blue.svg)]()
[![EdgeSAM](https://img.shields.io/badge/EdgeSAM-v0.1-orange.svg)](https://github.com/chongzhou96/EdgeSAM)

---

## ✨ Highlights

- **Body-Anchored Visual Prompting** — the person mask guides EdgeSAM's sampling box and
  negative-point layout, so the target hold is still isolated cleanly when the climber's
  hand or foot fuses visually with it.
- **Ring-Based Black Vote Correction** — zero extra forward passes; fixes dark holds that
  the mask misses because the pose endpoint drifted. If black pixels make up ≥ 18 % of a
  20–70 px ring just outside the mask, the colour is corrected to black.
- **Person-Bbox-Adaptive Box** — the sampling-box half is capped at 0.6 × the person
  bounding-box short side, which prevents grabbing distant, unrelated holds.
- **Pixel-Level Color Voting** — per-pixel binning instead of mean-HSV sampling, so
  low-saturation pixels are no longer systematically discarded.

**Result**: 10/10 walls on 65 frames from two real climbing videos; two colour-space
baselines score 4/10 and 6/10 on the same walls. The frame-level bootstrap expectation
is 8.2/10 — see the [paper](paper/) for the full analysis.

---

## 📸 Demo

Left column: single-image recognition. Right column: video-frame recognition.
In the right column, cyan/orange outlines are the EdgeSAM hold masks, white circles are
the ring used for the black vote, and the pink outline is the person mask.

| Single image | Video frame |
|---|---|
| ![purple](docs/demo/result_1_purple.jpg) | ![blue](docs/demo/wall_blue.jpg) |
| ![pink](docs/demo/result_4_pink.jpg) | ![white](docs/demo/wall_white.jpg) |
| ![orange](docs/demo/result_5_orange.jpg) | ![yellow](docs/demo/wall_yellow.jpg) |

More examples in [`docs/demo/`](docs/demo/).

> **Image sources.** The demo images were collected from public platforms such as
> Xiaohongshu and Baidu Images, and are used for research and demonstration only.
> The climber's face in the first video-frame image has been pixelated. If you hold
> the rights to any of these images and would like it removed, please open an issue.

---

## 🚀 Quick Start

### 1. Environment

```bash
# Python 3.12+ is required (EdgeSAM's torch/timm wheels)
conda create -n climbing python=3.12 -y
conda activate climbing

# Base dependencies
pip install opencv-python numpy mediapipe==1.0.0

# EdgeSAM inference (CPU build)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install timm yacs
```

### 2. Download weights

```bash
python download_weights.py                    # everything
python download_weights.py edgesam yolo26     # or pick what you need
```

Or download manually into `weights/`:

- `edge_sam.pth` (~40 MB) — [EdgeSAM](https://github.com/chongzhou96/EdgeSAM)
- `yolo26m-seg.pt` (~53 MB) — ultralytics v8.4.0 release
- `pose_landmarker_*.task` (~5–29 MB) — [MediaPipe Pose](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)
- `yolox_l.onnx` (206 MB) + `dw-ll_ucoco_384.onnx` (128 MB) — [DWPose](https://github.com/TH14GZI-J/DWPose-onnx) (optional, for the DWPose experiments)

### 3. Clone the EdgeSAM source

```bash
git clone --depth 1 https://github.com/chongzhou96/EdgeSAM.git
```

### 4. Run

```bash
# Single/multiple images (--img takes a DIRECTORY; defaults to img/)
python test_edge_sam.py --img img

# Aggregate over video frames of a wall (route-colour recognition)
python test_edge_sam.py --aggregate-walls
```

> ⚠️ **MediaPipe's TFLite teardown is slow** (~42 s per process). The scripts end with
> `os._exit(0)` to skip the GC stall; `debug/test_no_os_exit_mp.py` verifies that a normal
> return also works.

---

## 📊 Results

### Single-image test set (15 images)

| Scene | Truth | Predicted | Hit | Note |
|---|---|---|---|---|
| Purple hold | purple | purple 75 % | ✅ | |
| Yellow hold | yellow | yellow 67 % | ✅ | |
| Green hold | green | green 100 % | ✅ | |
| Camouflaged wall (pink wall + pink holds) | pink | pink 100 % | ✅ | one of the core hard cases |
| Orange hold | orange | orange 67 % | ✅ | |
| Small distant yellow hold | yellow | white 100 % | ❌ | target too small; EdgeSAM snaps to the wall |
| Red hold | red | red 67 % | ✅ | |
| Blue hold occluded by the body | blue | purple 100 % | ❌ | residual pixels shift hue |
| White hold | white | white 100 % | ✅ | relative-white path fires |

**8/9 hits.** The mechanism behind both failures is analysed in Section 5.3 of the paper.

### Full video regression (2 climbing videos, 65 frames / 10 walls)

| Wall | Truth | Predicted (votes) |
|---|---|---|
| 2X wall_01 | black | black 14/26 ✅ |
| 2X wall_02 | blue | blue 19/30 ✅ |
| 2X wall_03 | pink | pink 10/23 ✅ |
| 2X wall_04 | cyan | cyan 7/21 ✅ |
| 3O wall_01 | blue | blue 10/30 ✅ |
| 3O wall_02 | red | red 13/28 ✅ |
| 3O wall_03 | yellow | yellow 6/12 ✅ |
| 3O wall_04 line1 | white | white 6/16 ✅ |
| 3O wall_04 line2 | pink | pink 4/15 ✅ |
| 3O wall_05 | green | green 8/24 ✅ |

**10/10 walls**, at 7.2 s/frame on CPU. For comparison, on the same walls:
HSV-only **4/10**, HSV connected components **6/10**, and a general vision-language
model (Qwen-VL) **5/10**.

---

## 🏗️ Project layout

```
climb-color/
├── test_edge_sam.py            # entry point: per-frame recognition / wall aggregation
├── edgesam_holds.py            # EdgeSAM wrapper (box prompt, candidate filtering, ring correction)
├── route_color.py              # HSV colour classification + endpoint-neighbourhood pixel voting
├── yolo_person_mask.py         # YOLO26m person segmentation
├── edgesam_stub.py             # stubs mmdet/mmcv so EdgeSAM imports for pure inference
├── download_weights.py         # weight downloader
├── docs/demo/                  # curated demo images referenced by this README
├── paper/                      # paper (EN + ZH, figures/ scripts/ data/, DOCX export)
├── baselines/                  # paper baselines (HSV-only 4/10, detect_holds 6/10)
├── tools/                      # manual inspection & calibration tools
├── dwpose_exp/                 # DWPose experiment sub-project (decoupled from production)
├── vlm_exp/                    # vision-language-model baseline (Qwen-VL 5/10)
├── retrieval_exp/              # joint wall retrieval (DINOv2+ALIKED/RANSAC: photo → wall → color)
├── debug/                      # exploratory scripts (kept by convention)
│   └── rhand_0005s/            #   └ how the ring black-vote fix was found (exp3 series)
├── weights/                    # all model weights (gitignored)
├── EdgeSAM/                    # cloned EdgeSAM source (gitignored)
├── PIPELINE.md                 # full pipeline details (Chinese)
├── LESSONS.md                  # engineering notes (Chinese)
└── CHANGELOG.md  LICENSE  README.md  README_zh.md
```

> All experiment code lives in `baselines/ tools/ debug/ dwpose_exp/ vlm_exp/
> retrieval_exp/`; the repository root keeps only the production pipeline and
> documentation.

---

## 🧪 Key design

### 1. EdgeSAM box + point prompt (the core)

- **box** = adaptive half-size around the endpoint (capped at 0.6 × person-bbox short side)
- **1 positive point** (the endpoint) + **14 negative points** (1 toward the body,
  5 evenly spaced around the endpoint, 8 on the person-mask boundary)
- `num_multimask=4` candidates → ranked by `(-score, +dist, -area)`, best one wins

### 2. Ring-Based Black Vote Correction

- Motivation: when the endpoint drifts (measured up to 49 px), EdgeSAM may not include the
  dark hold in the mask at all.
- Rule: if the mask colour is not black but black pixels make up ≥ 18 % of a 20–70 px ring
  outside the mask, correct the colour to black.
- Why directional: colourful holds in the ring are the norm (no information), whereas a dark
  hold on a colourful wall is *scarce* (a strong signal).

### 3. Pixel-Level Color Voting

- Replaces mean-HSV sampling: every pixel is binned (black / white / red / orange / …) and
  the bins vote.
- Low-saturation pixels are no longer discarded; a separate "relative white" path handles
  white holds under warm lighting.

Details in [`PIPELINE.md`](PIPELINE.md).

---

## 📚 Documentation

- **[paper/](paper/)** — paper (bilingual), figures, experiment scripts
- **[PIPELINE.md](PIPELINE.md)** — full pipeline details
- **[LESSONS.md](LESSONS.md)** — engineering notes (EdgeSAM integration, pose-model comparison, threshold derivation)
- **[dwpose_exp/README.md](dwpose_exp/README.md)** — DWPose experiment sub-project
- **[docs/wechat/](docs/wechat/)** — WeChat article source (markdown + figures) for the Chinese write-up

---

## 🔬 Citation

```bibtex
@misc{climbcolor_2026,
  title={ClimbColor: Pose-Anchored Hold Segmentation for Climbing Route Color Recognition},
  author={Curren},
  year={2026},
  url={https://github.com/CurrenWong/climb-color}
}
```

Built on:

- [EdgeSAM](https://github.com/chongzhou96/EdgeSAM) — Zhou et al., *Prompt-In-the-Loop Distillation for On-Device Deployment of Segment Anything*
- [MediaPipe](https://developers.google.com/mediapipe) — Lugaresi et al., *MediaPipe: A Framework for Building Perception Pipelines*
- [YOLO26](https://github.com/ultralytics/ultralytics) — Ultralytics
- [DWPose](https://github.com/TH14GZI-J/DWPose-onnx) — Yang et al., *Effective Whole-Body Pose Estimation with Two-Stage Distillation*

---

## 📝 Known limitations

1. **Endpoint localisation is the dominant error source.** The wrist keypoint is not the
   grip point: with fingers wrapped around a hold, the hold centre can sit 30–50 px away
   from the wrist — further than the hold's own radius. Post-processing can mitigate this
   but cannot fix it.

2. **Small and densely packed holds are a hard boundary for segmentation.** When the target
   hold is very small (distant) or surrounded by many similar holds, EdgeSAM's encoder
   features decay and the mask snaps to the wall or a neighbouring hold.

3. **Body occlusion causes unrecoverable information loss.** When the climber's body covers
   a hold, subtracting the person mask may leave only a few dozen pixels and the colour
   statistics become unreliable. No amount of tuning fixes this — the information simply is
   not in the frame.

4. **Mind the statistics.** Wall-level conclusions rely on multi-frame aggregation; at a 5 s
   sampling interval each wall has only 4–9 frames, so a handful of noisy votes can decide
   the outcome. Re-extracting the three weakest walls at 1 s raised the mean margin from
   +1.7 to +14.3 votes — most fragility comes from sampling density rather than the method
   itself, but one wall barely moved, which is a systematic bias. (Denser sampling is more
   stable but costs linearly more.)

5. **CPU inference is slow.** About 7.2 s/frame (EdgeSAM encode + 4 endpoint queries + two
   ONNX models + YOLO26m); on GPU this should drop below 1 s. All reported numbers are CPU.

---

## 🤝 Contributing

Issues and PRs are welcome.

- Please open an issue before sending a large change.
- Make sure `python -m py_compile` passes.
- Put new debug scripts under `debug/`.

## 📄 License

MIT — see [LICENSE](LICENSE).
