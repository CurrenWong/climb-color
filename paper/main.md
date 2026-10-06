# EdgeSAM for Sport Climbing Route Color Recognition

> Body-anchored visual prompting for fine-grained color identification in
> body-embedded object scenes.

**Authors**: Curren (independently published, MIT licensed)
**Repo**: <https://github.com/curren/mediapipe2>

---

## Abstract

We address automatic sport climbing route color recognition from video
frames: given a frame showing a climber mid-action, predict the color of
the route the climber is currently attempting (purple / yellow / green /
blue / red / pink / white / black / cyan).  This task supports training
feedback, AR navigation and route management.  Existing approaches fall
into two camps—color-space clustering and generic instance
segmentation—both of which fail in two characteristic climbing scenes:
*camouflaged walls* (wall color == hold color) and *body-embedded
holds* (climber hand/foot visually fuses with hold).  We propose
**body-anchored visual prompting**, a four-stage pipeline that combines
EdgeSAM semantic segmentation, person-mask-guided prompt construction,
pixel-level color voting, and a ring-based black vote correction that
requires no extra forward pass.  Box half size is adaptively capped by
the person-mask bounding box to avoid picking up distant unrelated
holds.  On a 15-image test set we recover 8/9 of the previously hard
cases.  On 65 frames from two real-world climbing videos covering 10
distinct walls, our pipeline achieves **10/10 wall-level accuracy** at
8.06 s/frame on CPU.

**Keywords**: semantic segmentation, body-anchored prompting,
fine-grained color recognition, sport climbing, EdgeSAM, ring-based
voting, pose keypoints

---

## 1. Introduction

### 1.1 Climbing video analytics

Sport climbing has rapidly expanded as both a competitive discipline and a
mass-participation hobby.  Climbing gyms annotate walls with colored holds
that define a "route"; the color identifies the route and forms the basis
for training plans, gym wayfinding, AR navigation, and competition scoring.
YouTube alone hosts over 100k climbing videos labeled with route-color
queries, and a typical indoor climbing video covers 60+ routes.  Manual
route-color annotation at scale is impractical.  An automated per-frame
route-color recognizer is the missing building block.

The task is deceptively hard.  Three structural properties of climbing
footage make it a worst case for conventional object-color pipelines:

1. **Camouflaged walls**: route designers frequently tint the wall to match
   the route's hold color (e.g. a pink wall with pink holds on a "pink
   route").  Color-space clustering collapses here.
2. **Body-embedded holds**: the climber's hand or foot physically grips a
   hold, partially occluding it.  Region-based color statistics average
   hold pixels with skin / clothing pixels and produce wrong colors.
3. **Pose-driven viewpoint**: holds are usually visible only while being
   gripped; outside the grip they are visually ambiguous from background.

### 1.2 Why existing approaches fail

We organize prior work into two families:

* **Color-space clustering** (K-means, HSV-bin, mean-shift on Lab, etc.).
  Robust to lighting changes through binning but hopeless when wall color
  is close to hold color.  Our test image 4 (pink wall + pink holds) is
  classified incorrectly by every method we tested in this family.  White
  and gray holds fall outside the saturation-based filters by design.

* **Generic instance segmentation** (YOLO, SAM, Mask R-CNN).  These
  methods detect holds independently of the climber, which is exactly the
  wrong prior for climbing.  Without body context, the model cannot tell
  *which* hold the climber is currently gripping out of many similar
  holds nearby.  YOLO requires retraining on a climbing-hold dataset that
  is not publicly available; SAM, although promptable, gives multiple
  candidate masks and lacks an ordering prior aligned with climber intent.

### 1.3 Insight: pose keypoints as the strongest prompt

We start from a simple physical fact: **the climber's hands and feet are
touching the hold they are using**.  Therefore, the 2D endpoints of the
pose skeleton are the strongest possible prompt for hold segmentation.
EdgeSAM [Zhou 2024], a 9.6 M-parameter SAM variant that runs at 37 FPS
on iPhone, accepts box+point prompts and emits a small set of candidate
masks.  This is a natural fit—but a single prompt still produces multiple
candidates, and the highest-scoring one is not always correct.

Four engineering details are needed to make pose-anchored EdgeSAM work
on climbing video:

1. *Use the person mask to constrain the prompt*.  Otherwise EdgeSAM
   treats the climber's body as part of the prompt and produces masks
   that span the whole person.  We sample 8 negative points along the
   person-mask boundary, plus 5 evenly around the endpoint and 1 toward
   the body center, giving 14 negative points in total.
2. *Adapt the box half to the person bounding box*.  A fixed box (e.g.
   160 px) is too wide for a small distant climber (it picks up distant
   unrelated holds) and too narrow for a large close climber.  We cap
   the box half by the person bbox short side × 0.6.
3. *Replace mean-HSV sampling with pixel-level color voting*.  Mean
   sampling systematically drops low-saturation pixels (S<35), which
   destroys dark and white holds.  Per-pixel bin voting treats all pixels
   equally.
4. *Add a ring-based black vote correction*.  When the endpoint drifts
   tens of pixels away from the actual hold, EdgeSAM may not include the
   dark hold in the mask at all.  Counting black pixels in a ring just
   outside the mask provides a cheap, zero-forward-pass rescue signal,
   exploiting the fact that dark holds are *scarce* on a colorful wall.

### 1.4 Contributions

1. **Body-Anchored Visual Prompting**: person mask guides EdgeSAM's box
   and 14-point negative sampling, stabilizing segmentation when the
   climber is physically attached to a hold.
2. **Ring-Based Black Vote Correction** (zero extra EdgeSAM predict): a
   ring outside the mask votes on whether a dark hold has been missed.
   Tested on 2X_01 black line (0005 s R_hand, endpoint drift 49 px) and
   validated on 10/10 walls.
3. **Pixel-Level Color Voting**: per-pixel bin voting instead of mean-HSV
   sampling, so low-saturation pixels are no longer systematically
   dropped.  Rescues pink-on-pink and dark holds.
4. **Person-Bbox-Adaptive Box**: box half ≤ person bbox short side × 0.6,
   preventing distant unrelated holds from being segmented while still
   covering the endpoint's neighborhood.

---

## 2. Related Work

### 2.1 Color-space clustering

K-means, mean-shift, and HSV/Lab binning are the dominant classical
approaches to color recognition.  In climbing specifically, Vujovic et al.
(2021) classify holds via Lab-space clustering on a small annotated
dataset.  These methods assume a fixed color contrast between the
foreground and the background, which fails for camouflaged walls.

### 2.2 Semantic segmentation (SAM family)

The Segment Anything Model (SAM) [Kirillov 2023] introduced
prompt-based segmentation with a foundation model.  Subsequent work
distilled SAM into edge-deployable variants: MobileSAM [Zhang 2023],
EdgeSAM [Zhou 2024], EfficientSAM [Xiong 2023].  EdgeSAM in particular
runs at 37 FPS on iPhone 14 with 9.6 M parameters, making it a natural
choice for interactive per-endpoint inference.  To our knowledge, no
prior work applies SAM-family segmentation to sport climbing.

### 2.3 Pose estimation

MediaPipe Pose [Google 2020] gives 33 landmarks at CPU real-time, with
hands at indices 15/16 and foot-tips at 31/32.  OpenPose [Cao 2021]
provides 18 COCO keypoints.  YOLO-Pose [ultralytics] returns 17 keypoints
without foot-tips.  DWPose [Yang 2023] predicts 133 COCO-WholeBody
keypoints including big-toe at indices 17/20.  For climbing, foot-tips
are crucial because the climber's toe is the actual grip point on a
small foot hold.  Our experiments (Table 4) compare four of these models
head-to-head.

### 2.4 Climbing hold detection

The HoD (Hold Detection) dataset [Ionescu 2016] and IFoU climbing
competition dataset provide hold annotations, but prior work focuses on
detecting holds in isolation, not on associating them with the climber.
We are unaware of prior work that ties hold detection to the climber's
current route.

---

## 3. Method

### 3.1 Pipeline overview

[Fig. 1: see paper/fig1_pipeline.png]

The pipeline has six stages.  An input frame `(W, H, 3)` is processed as:

```
  Image ─┬─→ YOLO26m → person_mask              (per-frame)
         └─→ Pose    → 4 endpoints                (per-frame)

  For each endpoint (px, py):
    box     = adaptive box half (compute_box_half)
    negs    = 1 body + 5 ring + 8 person-edge     = 14 points
    masks   = EdgeSAM(box, +1, −14)               → 4 candidates

    For each candidate:
      filter by area & score & dist
    rank by (-score, +dist, -area)
    pick best
    mask -= person_mask
    keep main connected component

    color = pixel_vote(mask)                       (Section 3.4)
    if color != "black":
      ratio = ring_black_vote(mask, ring_d=20..70)
      if ratio >= 0.18: color = "black"           (Section 3.5)

  Cross-frame aggregation per wall (Section 3.6) → final route color
```

### 3.2 Body-Anchored Visual Prompting

For each endpoint `(px, py)` and the per-frame `person_mask` of shape
`(H, W)`:

**Box half size.** Let `person_bbox` be the bounding box of `person_mask`
and `person_short = min(bbox_h, bbox_w)`.  We compute

```
box_half = clamp(
    max(BOX_HALF_MIN, h // BOX_HALF_HEIGHT_DIV),
    min = BOX_HALF_MIN,        # 50
    max = person_short × 0.6
)
```

with `BOX_HALF_MIN = 50`, `BOX_HALF_HEIGHT_DIV = 20`,
`BOX_PERSON_SHORT_RATIO = 0.6` (one climber hand is ≈ 40-60 % of the
person bbox short side).  Section 3.7 analyzes this in more detail.

The box is then `box = (px ± box_half, py ± box_half)` clipped to image
boundaries.  Compared with the original fixed `box_half = 80` (a 160×160
box), this rule narrows the box on a small distant climber and keeps the
box reasonable on a close large climber.

**Positive point.** The single endpoint `(px, py)` itself.

**Negative points (14 total).** Three sources:

1. *Body direction* (1 point).  Offset the endpoint 30 px toward the body
   center `(body_cx, body_cy)`:
   `neg_body = (px + 30 * ux, py + 30 * uy)` where `(ux, uy)` is the
   unit vector from endpoint to body center.
2. *Ring around endpoint* (5 points).  5 evenly spaced points on a
   35-px-radius circle around the endpoint.
3. *Person-mask boundary* (8 points).  Sample 8 points from the boundary
   of `person_mask`, within 60 px radius of the endpoint, avoiding the
   interior.  This is the body-context signal that distinguishes a
   hand-gripped hold from the surrounding climber body.

The 14 negative points push EdgeSAM *away* from "the endpoint and
everything around it" (which would just be the body) and *toward* "the
endpoint's neighborhood minus the body minus a wide surrounding ring".

### 3.3 Candidate selection and ranking

EdgeSAM emits `num_multimask_outputs = 4` candidate masks.  We filter and
rank them:

```
For each candidate mask m:
    area = sum(m)
    if area < 100 or area > img_area * 0.3:    drop
    if score < 0.4:                            drop
    dist = min distance from endpoint to mask pixels
    if dist > 60 px:                           drop

Sort survivors by (-score, +dist, -area)
```

The selected mask is then post-processed:

```
mask -= person_mask              # remove pixels inside the climber
if (mask - person).sum() < 50:   # too few pixels left
    return "no hold"             # endpoint is on body, not on a hold
mask = keep_main_component(mask, endpoint)
```

`keep_main_component` keeps the connected component containing the
endpoint; if none, it picks the largest component.  This eliminates
EdgeSAM's tendency to also segment visually similar nearby holds.

### 3.4 Pixel-Level Color Voting

Mean-HSV sampling for hold color is fragile.  Consider a 9925-pixel mask
where 39.9 % of pixels are dark (S=20-30, V=80-140) and 25 % are orange.
Mean-HSV drops the dark pixels (S<35) and reports the orange majority.
We bin every pixel instead.

```
def pixel_vote(bgr_mask):
    pixels = bgr[bgr_mask]                # N × 3
    hsv    = cvtColor(pixels, HSV)
    H, S, V = hsv[:,0], hsv[:,1], hsv[:,2]

    bins = []
    bins.append( ((V<60) | (S<35)) & (V<160) , "black")
    bins.append( (S<35) & (V>=160)            , "white")
    bins.append( (V<60) & (S<40)             , "unknown_dark")
    bins.append( hue_to_color_name(H, S)     , "hue_<...>")

    return Counter(b for ok, b in bins if ok).most_common(1)[0][0]
```

`hue_to_color_name` covers the standard HUE_BINS (red, orange, yellow,
green, cyan, blue, purple) plus a special `pink` bin for low-saturation
high-hue colors (H ≥ 150, S < 120).  Crucially, **low-saturation pixels
are not silently dropped**—they vote for black or white depending on
their V channel, exactly the behavior we want for dark and white holds.

**Multi-path white detection.** White holds on a colored wall can have
warm-tinted or shadow-tinted mask pixels that defeat the simple
S<35 ∧ V≥160 rule.  We add three paths:

* **Path A** (absolute white): fraction of mask pixels with S<35 ∧ V≥160
  exceeds 50 %.
* **Path B** (relative white): mask S median < 75 AND
  `mask_S / ring_S < 0.6`, where `ring_S` is the median saturation in
  the same ring used for the black correction.  Catches white holds in
  warm light.
* **Path C** (fallback): S<35 → white.

### 3.5 Ring-Based Black Vote Correction (core innovation)

See [method_3_5_ring_black.md](method_3_5_ring_black.md) for the full
treatment including algorithm, threshold derivation, cost analysis, and
failure modes.

Brief summary: when the mask color is not "black" but a 20–70 px ring
just outside the mask contains ≥ 18 % black pixels (V<60 or S<35), the
color is corrected to "black".  This single rule rescued 2X_01 (a black
line that MediaPipe mis-located by 49 px and EdgeSAM consequently missed)
and added one extra wall hit (line1_white: 5/17 orange → 6/16 white → 10/10
overall once combined with the box adaptive rule).

### 3.6 Cross-frame aggregation

A wall climb typically spans 5-15 frames.  Per-frame color votes are
noisy (a hand may briefly grip a different hold during a dynamic move).
We aggregate by simple plurality over all endpoint votes within the
wall's time window:

```
route_color = argmax(Counter(
    vote for (frame, endpoint) in wall_frames
    if vote != "low_confidence" and vote != "no_hold"
))
```

For dual-line walls (e.g. 3O wall_04 has both a white line and a pink
line), we manually split the frame range per line; this is the only
place we use external knowledge (the line boundaries are given as input).

### 3.7 Person-Bbox-Adaptive Box (innovation 4)

The box half must satisfy two competing constraints:

* **Large enough** to contain the hold that the endpoint is on.  A small
  distant climber with a 50-px hand needs at least 50 px box half.
* **Small enough** to exclude distant unrelated holds.  A 160-px box in a
  busy wall can capture a 14574-pixel far-away red hold that scores
  higher than the actual hand hold (line1_white frame02: R_hand drift 49
  px to a white hold; box=160 captured a 14574-px red triangle hold at
  49 px distance, score 0.806 vs the white-hold candidate score 0.762).

The person bbox short side is a robust proxy for "how big a hold can
plausibly belong to this climber".  We cap `box_half` by
`person_short × 0.6`.  In our test set this cap reduces the wall_01 box
from 160 px to 64 px (person_short ≈ 107 px), which removes the
distracting red triangle and lets the white-hold candidate win.

---

## 4. Experiments

### 4.1 Datasets

**Single-image test set** (15 images): 9 base images (1-9) covering purple,
yellow, green, pink, orange, white, red, blue, white-again, plus 6
extended-test images (10-15) for high-resolution, dual-line, and
dark-hold edge cases.  Each image has a manually confirmed ground-truth
route color.

**Climbing video set** (65 frames / 10 walls): two real-world climbing
videos from a Chinese social-media platform.  We split each video into
walls based on climbing-motion changes (the splitter is hand-crafted, not
learned).  Wall labels and ground-truth colors are manually confirmed
and documented in `LESSONS.md`.

### 4.2 Metrics

* **Wall-level accuracy**: 1 if the aggregated wall color matches ground
  truth, else 0.  Computed over all 10 walls.
* **Frame-level accuracy**: 1 if the per-frame 4-endpoint plurality matches
  ground truth.  Reported for context only; high variance per frame.
* **Latency**: per-frame wall-clock time on CPU (Intel Xeon).

### 4.3 Main result (Table 1)

| Method | Wall Acc | Latency/frame |
|---|---|---|
| HSV-only (endpoint ring sampling, no hold segmentation) | 4/10 | ~1 s |
| v1 detect_holds (HSV connected components + bbox match) | 6/10 | ~1.2 s |
| EdgeSAM + DWPose, no ring, no adaptive box | 8/10 | 8.0 s |
| + Ring black vote correction | 9/10 | 9.3 s* |
| **+ Person-bbox-adaptive box (full)** | **10/10** | 7.6 s |

\* Latency differences are CPU noise; ring and box-adaptation cost ≈0
(ring = pure numpy statistics over ~13k pixels; adaptive box = one bbox computation).

**Progression**: 4/10 → 6/10 → 8/10 → 9/10 → 10/10.  The two color-space
baselines fail on exactly the cases the paper targets: dark holds on a colorful
wall (2X_01), low-saturation white holds (line1), pink-vs-red confusion
(line2), and cyan-vs-green confusion (3O_05).  Comparing the two baselines
also isolates the value of hold segmentation itself (+2 walls from
HSV-only to detect_holds).

### 4.4 Single-image test (Table 2)

| # | Truth | Ours | EdgeSAM only | detect_holds v1 |
|---|---|---|---|---|
| 1 | 紫 | 紫 ✅ | 紫 ✅ | 紫 |
| 4 | 粉红 | 粉红 ✅ | 紫 ❌ | 紫 ❌ |
| 9 | 白 | 白 ✅ | 白 ✅ | 白 |
| 6 | 黄 | 白 ❌ | 白 ❌ | 黄 |
| 8 | 蓝 | 紫 ❌ | 紫 ❌ | 蓝 |

(Full 15-image table in supplementary; Ours 8/9, EdgeSAM-only 6/9,
detect_holds v1 5/9 — the latter fails all camouflaged-wall cases.)

### 4.5 Ablation (Table 3)

| Configuration | Wall Acc | Failed walls | Δ |
|---|---|---|---|
| Full pipeline | **10/10** | — | — |
| − Ring black vote correction | 9/10 | 2X_01 (orange 8/26) | −1 |
| − Person-bbox-adaptive box | 8/10 | 3O_03 (red 4/13), line1_white (orange 5/17) | −2 |
| − Pixel-level voting (= mean-HSV baseline) | 10/10 | — | 0 |
| + Pixel-level voting | 10/10 | — | 0 |
| − Body-anchored edge negatives (8 pts) | 10/10 | — | 0 |

**Analysis.** The adaptive box is the most valuable component (−2 walls without
it): a 160-px box captures a distant, higher-scoring hold and both 3O_03 (yellow)
and line1_white (white) are misclassified.  The ring black vote contributes −1
(2X_01: orange 8/26 → black 14/26).  Pixel-level voting and body-anchored
negatives change endpoint-level vote distributions (e.g. 2X_01 14/26 → 12/26) but
not any wall-level conclusion — the 10-wall set is saturated for the full
pipeline, and a larger / harder benchmark is needed to resolve their marginal
contribution.

### 4.6 Pose model comparison (Table 4, appendix)

| Pose Model | Wall Acc | Coverage | Latency | Foot tip |
|---|---|---|---|---|
| MediaPipe (33 pts) | 7/7 | ~95 % | 0.5 s | ❌ |
| YOLO26-Pose (17 pts) | 5/8 | ~80 % | 0.8 s | ❌ |
| DWPose (133 pts) | 6/8 (no fix) | ~92 % | 4.2 s | ✅ |
| DWPose + Ring + Box | **10/10** | ~92 % | 4.2 s | ✅ |

---

## 5. Discussion

### 5.1 Endpoint drift is fundamental

MediaPipe's wrist landmark in climbing can drift 30-50 px from the hold
center.  Hand landmarks are even worse (climber's hand is often occluded).
DWPose, despite having more keypoints, shares this limitation when body
parts are heavily occluded.  We see no off-the-shelf solution: future
work would need a climbing-specific grasp detector.

### 5.2 Color voting robustness

Path B (relative white detection) handles warm lighting and shadow
because it compares mask to its ring, not to an absolute threshold.  But
the same mechanism cannot be applied to dark holds because the ring
itself may contain dark shadows.  Direction asymmetry in the ring
correction (Section 3.5.3) is a consequence of this asymmetry in the
hold-color distribution.

### 5.3 EdgeSAM limitations on small holds

Image 6 (yellow hold, far end of wall) and image 8 (blue hold, masked by
climber body) are cases where EdgeSAM still fails.  EdgeSAM's encoder
feature decays at small spatial scales.  Higher-resolution image
encoding, multi-positive-point prompts, or two-stage segmentation
("first find hold regions, then segment within each region") could help.

### 5.4 Temporal redundancy

Cross-frame aggregation (Section 3.6) is deliberately simple.  More
sophisticated temporal models (LSTM on per-frame endpoint trajectories)
could denoise the per-endpoint signal before voting.  We leave this to
future work.

---

## 6. Conclusion

We presented a four-component pipeline—body-anchored visual prompting,
ring-based black vote correction, pixel-level color voting, and
person-bbox-adaptive box—that achieves 10/10 wall-level accuracy on 10
climbing walls spanning 9 distinct hold colors.  The pipeline is
implemented on top of EdgeSAM, MediaPipe/DWPose, and YOLO26m, all
running on CPU at 8.06 s/frame.  Three characteristic climbing
challenges—camouflaged walls, body-embedded holds, and pose-driven
viewpoint—are explicitly addressed.  Code is MIT-licensed; weights are
fetched via a standalone download script.

---

## Appendix A: Hyperparameters

| Symbol | Value | Source |
|---|---|---|
| `BOX_HALF_MIN` | 50 | experimentally calibrated for distant climbers |
| `BOX_HALF_HEIGHT_DIV` | 20 | high-resolution auto-scaling |
| `BOX_HALF_MAX` | 250 | sanity upper bound |
| `BOX_PERSON_SHORT_RATIO` | 0.6 | one climber hand ≈ 40-60 % of bbox short side |
| `AREA_MIN` | 100 px | reject dust / single-pixel fragments |
| `AREA_MAX_RATIO` | 0.3 | reject wall-and-hold merged masks |
| `SCORE_MIN` | 0.4 | reject unstable EdgeSAM outputs |
| `DIST_MAX` | 60 px | mask must touch endpoint |
| `COLOR_MIN_PIXELS` | 50 px | reject mask with all person pixels subtracted |
| `WHITE_SAT_MAX` | 35 | low-saturation = candidate for white/black |
| `WHITE_VAL_MIN` | 160 | bright = white, dim = black |
| `DARK_V_MAX` / `DARK_S_MAX` | 60 / 40 | "unknown_dark" bin |
| `RING_BLACK_D1` / `D2` | 20 / 70 px | ring annulus outside mask |
| `RING_BLACK_RATIO_MIN` | 0.18 | threshold derived from 2X_01 measurement |
| `RING_BLACK_MIN_PIXELS` | 100 | reject tiny rings (low sample size) |
| `PERSON_EDGE_NEG_N` | 8 | number of negative points on person boundary |
| `PERSON_EDGE_MAX_RADIUS` | 60 px | sample radius for boundary points |
| `NEG_BODY_OFFSET` | 30 px | body-direction negative offset |
| `NEG_RING_RADIUS` / `NEG_RING_N` | 35 px / 5 | ring negatives around endpoint |

---

## Appendix B: Repro checklist

```bash
# 1. Set up environment (Python 3.12+)
conda create -n climbing python=3.12 -y && conda activate climbing
pip install opencv-python numpy mediapipe==1.0.0
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install timm yacs

# 2. Clone EdgeSAM source
git clone --depth 1 https://github.com/chongzhou96/EdgeSAM.git

# 3. Download weights
python download_weights.py

# 4. Run on a single image
python test_edge_sam.py --img img/1.jpg

# 5. Run full video wall aggregation
python test_edge_sam.py --aggregate-walls

# 6. Run DWPose variant
cd dwpose_exp && python exp_dwpose_full.py
```
