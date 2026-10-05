# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-08-13

First public release.

### Added
- EdgeSAM-based climbing route color recognition (v2 pipeline)
  - YOLO26m person mask → MediaPipe Pose 4 endpoints → EdgeSAM box+point prompt
  - Pixel-level HSV color voting with multi-path white detection
  - Cross-frame aggregation for video route color stability
- Ring-based black vote correction (zero extra EdgeSAM predict)
  - Detects dark holds missed by endpoint-anchored segmentation
  - 7/7 → 9/10 → 10/10 wall-level accuracy improvement on test set
- Person-bbox-adaptive box prompt (`compute_box_half()`)
  - Caps box half by person mask bbox short side × 0.6
  - Prevents cutting irrelevant distant holds when endpoint is near climber body
- DWPose endpoint alternative (`dwpose_exp/`)
  - 133-point COCO-WholeBody (includes big_toe for foot holds)
  - Side-by-side comparison with MediaPipe; both validated

### Validated
- 15 single-image test set: 8/9 correct on the most challenging cases
- 2 sport climbing videos (65 frames, 10 walls): 10/10 wall-level hit rate
- Per-frame latency: ~8.06s (EdgeSAM + 2 ONNX + YOLO26m + MediaPipe, CPU only)

### Documentation
- `PIPELINE.md` — full technical pipeline spec
- `PRODUCT.md` — 9-image diagnostic catalog (project origin)
- `LESSONS.md` — 12 chapters of engineering lessons learned
- `docs/demo/` — curated output examples for README

### Repository
- `debug/` — exploratory scripts kept for traceability
- `dwpose_exp/` — DWPose experiment subset (split for independent evolution)
- `EdgeSAM/` — submodule-style source clone (download separately)
- `download_weights.py` — fetch all required model checkpoints
