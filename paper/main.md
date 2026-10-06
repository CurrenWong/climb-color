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
distinct walls, our pipeline achieves **10/10 wall-level accuracy**
(7.2 s/frame on CPU), against **4/10** and **6/10** for two color-space
baselines evaluated on the same walls.  Because five of the ten walls
are decided by ≤2 votes, we additionally report a frame-level bootstrap
that puts the expectation at **8.2/10** and makes the 10/10 single run
itself only 12 % likely — the honest reading is that the pipeline is
reliable but not yet saturated.  Re-extracting the three weakest walls at
1 s intervals (4.4× more frames) raises the mean margin from +1.7 to
+14.3 votes and the mean bootstrap accuracy from 68 % to 90 %: most of the
fragility is a sampling artefact, while one wall barely moves, exposing a
systematic endpoint-drift bias that density cannot fix.  Ablations show the
person-bbox-adaptive box is the single most valuable component (−2 walls
without it), followed by the ring black vote (−1); the two remaining
components change per-endpoint votes but not wall-level outcomes on this
benchmark.

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
FastSAM, EdgeSAM [Zhou 2024], and EfficientSAM [Xiong 2023].  EdgeSAM in
particular runs at 37 FPS on iPhone 14 with 9.6 M parameters, making it a
natural choice for interactive per-endpoint inference.

We did not pick a backbone on reputation.  During development we
evaluated FastSAM, MobileSAMV2, and EdgeSAM as the per-endpoint mask
generator under our prompt scheme (Section 3.2).  FastSAM's point-prompt
masks were too coarse to isolate a single hold (2/9 → 3/9 on our
single-image test set), which is why we abandoned the "segment the hold
entity directly" route in favour of endpoint-anchored prompting.
MobileSAMV2 produced usable masks and reached comparable wall-level
accuracy inside the full pipeline, but cost 6.8 s/frame against
EdgeSAM's 1.07 s encoder + 2.40 s for four endpoint queries (~3.5 s).
EdgeSAM gave the best accuracy–latency trade-off and was adopted.

To our knowledge, no prior work applies SAM-family segmentation to sport
climbing, and no prior work uses pose keypoints as the *segmentation
prompt* — the closest analogue is pose-conditioned human parsing, where
keypoints guide part segmentation of a person rather than of an
interacted-with object.

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

### 3.0 Notation and constants

All symbols used below are collected here, together with the numeric
constants of the full pipeline.  Constants were fixed once on a
development subset and never re-tuned per wall.

| Symbol | Meaning |
|---|---|
| `H`, `W` | frame height and width (720, 1280 in our videos) |
| `I` | input frame, BGR |
| `M_person` | binary person mask from YOLO26m |
| `p = (p_x, p_y)` | a pose endpoint: one wrist or one ankle |
| `K` | the four endpoints {L_hand, R_hand, L_foot, R_foot} |
| `s_p` | keypoint confidence of endpoint `p` |
| `B_p` | square sampling box centred on `p`, half-size `h_p` |
| `h_p` | adaptive box half-size (Eq. 1) |
| `c_p` | colour label assigned to endpoint `p` |
| `r_p` | ring black-vote ratio at endpoint `p` |
| `w` | a wall: the ordered frame range assigned to one route |
| `ρ(c, w)` | share of endpoint votes equal to colour `c` in wall `w` |

| Constant | Value | Role |
|---|---|---|
| `BOX_HALF_HEIGHT_DIV` | 20 | base box half = `H / 20` |
| `BOX_HALF_MIN` / `MAX` | 50 / 250 | adaptive box bounds |
| `BOX_PERSON_SHORT_RATIO` | 0.6 | box half ≤ 0.6 × person-bbox short side |
| `NEG_RING_RADIUS` / `NEG_RING_N` | 35 px / 5 | negative points around the endpoint |
| `N_BODY` / `N_EDGE` | 1 / 8 | negative points toward body / on person edge |
| `RING_BLACK_D1` / `D2` | 20 px / 70 px | ring annulus inner / outer radius |
| `RING_BLACK_RATIO_MIN` | 0.18 | ring black-share threshold (Eq. 2) |
| `RING_BLACK_MIN_PIXELS` | 100 | minimum ring size for a valid vote |
| `V_BLACK` / `S_BLACK` | 60 / 35 | "dark pixel" tests in HSV |
| `S_LOW` | 35 | low-saturation cut in the mean-HSV baseline |

### 3.1 Pipeline overview

[Fig. 1 — pipeline overview: person mask + pose → EdgeSAM prompt →
 candidate selection → pixel voting → ring correction → cross-frame
 aggregation.  See `paper/fig1_pipeline.png`.]

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

[Fig. 2 — prompt geometry: 1 positive point (endpoint) + 14 negative points
(1 toward body, 5 around the endpoint, 8 on the person-mask boundary), with
the adaptive box.  See `paper/fig2_prompt_design.png`.]

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

[Fig. 3 — geometry and decision rule; see `paper/fig3_ring_black.png`.]

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

[Fig. 4 — before/after comparison on 3O wall_04 line1 frame 0105 s:
with a fixed 160-px box the R_hand candidate set contains a distant red
hold scoring 0.806 (area 14574) and wins; with the adaptive 128-px box
the nearby white hold scores 0.881 and wins.  See
`paper/fig4_box_adaptive.jpg`.]

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

**Climbing video set** (65 frames / 10 walls): two real-world climbing videos
sourced from Xiaohongshu (a Chinese climbing-video community), 1280×720,
sampled at 5-second intervals (8 frames per wall on average).  Each video
covers a single session on one wall section, with the climber's route
changing over time; we split each into "walls" by hand using the video
timeline (e.g. 3O wall_04 contains a white line over 0100-0120 s and a
pink line over 0125-0140 s, split at 0122 s).  Wall boundaries and
ground-truth colors are manually confirmed and documented in `LESSONS.md`.
This splitter is hand-crafted rather than learned, and we treat automatic
wall segmentation as out of scope (see Section 5.4).

For every wall we report the plurality of all endpoint votes across its
frames; per-frame votes are also logged for diagnosis but are not the
reported metric, since a climber legitimately touches different holds
while moving through a route.

### 4.2 Metrics

* **Wall-level accuracy**: 1 if the aggregated wall color matches ground
  truth, else 0.  Computed over all 10 walls.
* **Frame-level accuracy**: 1 if the per-frame 4-endpoint plurality matches
  ground truth.  Reported for context only; high variance per frame.
* **Latency**: per-frame wall-clock time on CPU (Intel Core i5-13500H).
  Full breakdown in Table 5 (appendix).

### 4.3 Main result (Table 1)

| Method | Wall Acc | Latency/frame |
|---|---|---|
| HSV-only (endpoint ring sampling, no hold segmentation) | 4/10 | ~1 s |
| v1 detect_holds (HSV connected components + bbox match) | 6/10 | ~1.2 s |
| EdgeSAM + DWPose, no ring, no adaptive box | 8/10 | 7.2 s |
| + Ring black vote correction | 9/10 | 7.2 s |
| **+ Person-bbox-adaptive box (full)** | **10/10** | 7.2 s |

\* Latency is the full-set mean (7.17 s/frame over 65 frames); per-stage
breakdown in Table 5.  Ring and box-adaptation cost ≈0 (ring = pure numpy
statistics over ~13k pixels; adaptive box = one bbox computation), so the
last three rows share the same latency within measurement noise.

**Progression**: 4/10 → 6/10 → 8/10 → 9/10 → 10/10.  The two color-space
baselines fail on exactly the cases the paper targets: dark holds on a colorful
wall (2X_01), low-saturation white holds (line1), pink-vs-red confusion
(line2), and cyan-vs-green confusion (3O_05).  Comparing the two baselines
also isolates the value of hold segmentation itself (+2 walls from
HSV-only to detect_holds).

The 10/10 figure is a **single-run** result; five of the ten walls are
decided by a margin of two votes or less, and Section 4.7 quantifies how
much of it is structural (bootstrap expectation 8.2/10).  We report both
numbers throughout and do not present the saturated single run as
evidence of a solved problem.

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

### 4.7 Stability of the wall-level result (Table 6)

The full pipeline scores 10/10, but the margin behind five of those ten
walls is only two votes or fewer (Table 6).  To separate structural
accuracy from lucky draws we ran a frame-level bootstrap: for each wall
we resampled its frames with replacement 2000 times, re-aggregated the
endpoint votes, and took the plurality.

**Table 6. Per-wall detail for the full pipeline, with frame-level
bootstrap stability.**

| Wall | Truth | Pred | Votes | Share | Runner-up | Margin | Bootstrap acc. |
|---|---|---|---|---|---|---|---|
| 2X_wall_01 | black | black | 14/26 | 53.8 % | orange 6 | +8 | 97.4 % |
| 2X_wall_02 | blue | blue | 19/30 | 63.3 % | orange 2 | +17 | 100.0 % |
| 2X_wall_03 | pink | pink | 10/23 | 43.5 % | black 6 | +4 | 90.3 % |
| 2X_wall_04 | cyan | cyan | 7/21 | 33.3 % | orange 5 | +2 | 72.5 % |
| 3O_wall_01 | blue | blue | 10/30 | 33.3 % | orange 8 | +2 | 65.9 % |
| 3O_wall_02 | red | red | 13/28 | 46.4 % | black 4 | +9 | 99.9 % |
| 3O_wall_03 | yellow | yellow | 6/12 | 50.0 % | blue 2 | +4 | 97.2 % |
| 3O_wall_04_line1 | white | white | 6/16 | 37.5 % | orange 4 | +2 | 61.9 % |
| 3O_wall_04_line2 | pink | pink | 4/15 | 26.7 % | black 3 | **+1** | 71.0 % |
| 3O_wall_05 | green | green | 8/24 | 33.3 % | black 6 | +2 | 68.2 % |

**Findings.**  Per-wall bootstrap accuracy averages **82.4 %**, so a
single run would be expected to get **8.2 / 10** walls right, and the
probability that all ten walls come out correct in one run is only
**12.2 %**.  The reported 10/10 therefore contains a substantial
lucky-draw component concentrated in the five tight walls.

This estimate is deliberately conservative.  Frames within a wall are
strongly temporally correlated — our sampling interval is 5 s and
neighbouring frames differ mainly by small climber movements — whereas
the bootstrap treats frames as exchangeable and thus overstates the
variance of the plurality vote.  The true stability lies between the
single-run number (10/10) and the bootstrap expectation (8.2/10); we
report both rather than the flattering one alone.

The tight walls share a single root cause: endpoint drift occasionally
lands the sampling point on an adjacent hold (Section 5.1).  The clearest
case is 3O_wall_04_line2_pink (margin +1, bootstrap 71 %), whose
runner-up is the red hold the same climber grabs a second earlier in the
timeline.  Denser temporal sampling, per-endpoint trajectory smoothing,
or a climbing-specific grasp detector would all attack this variance
directly.  We test the first of these below.

### 4.8 Sampling density (Table 7)

If tight margins are caused by sparse sampling — only 4-9 frames per wall,
so a handful of noisy votes decides the plurality — then sampling more
densely should widen them.  We tested this directly: the three
lowest-bootstrap walls were re-extracted from the source video at **1 s**
intervals (4.4× more frames) and pushed through the identical pipeline and
the identical bootstrap.

**Table 7. Effect of sampling density on the three weakest walls.**

| Wall | Frames (5 s → 1 s) | Margin (5 s → 1 s) | Bootstrap acc. (5 s → 1 s) |
|---|---|---|---|
| 3O_wall_01 | 8 → 36 | +2 → +6 | 66.3 % → 70.7 % |
| 3O_wall_04_line2_pink | 4 → 16 | +1 → **+18** | 70.5 % → **100.0 %** |
| 3O_wall_05 | 7 → 31 | +2 → **+19** | 68.2 % → **98.5 %** |
| **Mean** | 6.3 → 27.7 | **+1.7 → +14.3** | **68.3 % → 89.7 %** |
| P(all three walls correct) | 31.8 % → **68.2 %** | | |

**Findings.**  Denser sampling raises the mean margin by almost an order of
magnitude and nearly doubles the joint probability of getting all three
walls right.  Two walls become effectively deterministic (100 % and
98.5 %), which confirms that their 5 s results were a sampling artefact
rather than a colour-classification failure.

The exception is the most informative result.  3O_wall_01 improves only
marginally (66.3 % → 70.7 %) despite 4.5× more frames.  Its vote
distribution stays bimodal even at 1 s (blue 36 vs orange 30), meaning the
endpoint-anchored masks land on the orange hold for a large minority of
frames no matter how densely we sample.  That is a **systematic** bias —
endpoint drift, exactly the mechanism of Section 5.1 — not noise.
Sampling density removes variance; it cannot remove bias.  Distinguishing
the two is what makes this experiment worth reporting, and it identifies
the bias case as the one that genuinely requires a better grasp detector
rather than more data.

---

## 5. Discussion

### 5.1 Endpoint drift is fundamental

The single largest source of error in our pipeline is not segmentation or
color classification but the pose endpoint itself.  In climbing, the wrist
landmark is not the grip point: the climber's fingers curl around a hold
whose center may lie 30-50 px away from the wrist.  We measured 49 px on
2X_01 frame 0005 s (wrist at (291, 672) vs the black hold at (250, 646)) —
a distance that exceeds the hold's own radius.  MediaPipe is also
systematically less confident on hands (0.1-0.26) than on feet, because
climbing hands are frequently occluded by the body, chalk dust, or the
wall itself.

Switching to a denser pose model does not remove the problem.  DWPose
(133 COCO-WholeBody keypoints) improves foot accuracy through its
big-toe landmarks, but its hand keypoints degrade under the same
occlusions, and its aggregate score without our corrections (6/8 on our
earlier 8-wall subset) was *worse* than MediaPipe's.  The ring-based
black vote correction mitigates one specific consequence of drift — a
dark hold falling entirely outside the endpoint-anchored mask — but it
is a directional heuristic, not a general solution.  A climbing-specific
grasp detector, or a hand landmark model trained on gripped poses, would
be the principled next step; we leave this to future work.

### 5.2 Color voting robustness and its asymmetry

The relative-white path (Section 3.4, path B) illustrates why local
contrast beats absolute thresholds: it compares the mask's median
saturation to that of the surrounding ring rather than to a fixed
constant, so a white hold under warm gym lighting is still recognized.
The same trick cannot be applied symmetrically to dark holds, because
the ring itself may contain dark wall shadows; a symmetric "mask is
black and ring is colorful → recolor" rule would misfire on genuinely
black holds — our L_hand counter-example (mask correctly black, ring
58 % yellow) would be broken immediately.  The directional asymmetry in
Section 3.5 is therefore not an implementation shortcut but a
consequence of the underlying color distribution: dark holds are scarce
on a colorful wall, so their presence in a ring is informative, whereas
colorful holds are common, so their presence is not.

Pixel-level voting inherits the same philosophy: rather than deciding
which pixels are "worth counting" a priori (as mean-HSV does by
dropping S < 35), it lets every pixel vote and resolves the color at the
bucket level.  On our saturated 10-wall benchmark this does not change
any wall-level outcome, but it removes a systematic bias against dark
and white holds that would matter on a harder set.

### 5.3 EdgeSAM limitations on small and occluded holds

Two single-image cases remain unsolved (Fig. 5).  In image 6 the route
hold is a small yellow hold at the far end of a predominantly white
wall; EdgeSAM's encoder features decay at this spatial scale and all
four candidate masks latch onto the white wall instead, yielding white
100 % against a ground truth of yellow.  In image 8 the climber's body
covers much of the blue hold; after subtracting the person mask only a
few dozen pixels survive, and the remaining fragment is closer in HSV to
the purple holds nearby, yielding purple 100 % against blue.

These two failure modes bracket the practical limits of the approach:
one is a scale problem in the segmentation backbone, the other is an
occlusion problem that no amount of post-processing can recover from,
because the information is genuinely absent from the frame.  Plausible
remedies include higher-resolution image encoding, multi-positive-point
prompts derived from hold candidates, a two-stage "detect hold regions
first, then segment within each region" scheme, and — for the occlusion
case — temporal accumulation across frames in which the hand moves.

### 5.4 Temporal redundancy and its limits

Cross-frame plurality aggregation (Section 3.6) is deliberately simple,
and on our videos it is the difference between a usable and an unusable
system: single-frame votes fluctuate because the climber's grip changes
during a dynamic move, while the per-wall plurality is stable.  The
gain is not free, however — it requires the wall boundaries as input.
Automatic wall segmentation (detecting where one climb ends and the next
begins) is a separate problem we did not address, and in gyms where
climbers rest on the wall between attempts it is non-trivial.  More
expressive temporal models (per-endpoint trajectory smoothing, or an
LSTM over frame-level color distributions) could replace the plurality
vote, at the cost of additional supervision we did not have.

### 5.5 Summary of limitations

The limitations surfaced above can be stated compactly, roughly in order
of how much they constrain the claims:

1. **Statistical.**  Wall-level accuracy is measured on ten walls.  The
   single-run 10/10 has a bootstrap expectation of 8.2/10 (Section 4.7),
   and the dense-sampling test (Section 4.8) shows that much of that
   fragility is attributable to the 5 s sampling interval rather than to
   the method itself.  A larger benchmark — hundreds of walls across many
   gyms — is still needed before this should be quoted as a rate rather
   than a case study.
2. **Causal.**  The dominant error source is endpoint drift (30-50 px),
   and no post-processing step fully removes it; the ring correction
   patches one of its consequences and the adaptive box another.
3. **Scope.**  Wall boundaries are given as input; automatic wall
   segmentation is not addressed.  The pipeline also assumes a single
   climber on the wall and a fairly complete pose track.
4. **Segmentation.**  EdgeSAM fails on very small holds and on holds
   largely occluded by the climber's body (Fig. 5).  The occlusion case
   is information-theoretically unrecoverable from the frame alone.
5. **Hardware.**  All measurements are CPU-only; the GPU figure of
   0.8 s/frame is extrapolated from component timings, not measured.

---

## 6. Conclusion

We presented a four-component pipeline—body-anchored visual prompting,
ring-based black vote correction, pixel-level color voting, and
person-bbox-adaptive box—that achieves **10/10 wall-level accuracy** on ten
climbing walls spanning nine distinct hold colors, with a frame-level
bootstrap expectation of 8.2/10 and a 12 % single-run probability of the
saturated result (Section 4.7).  On the same walls, two color-space
baselines score 4/10 (endpoint ring sampling) and 6/10 (HSV connected
components), so the progression from pure color space to pose-anchored
semantic segmentation is 4 → 6 → 8 → 9 → 10 as components are added.  The
pipeline runs on CPU at 7.2 s/frame (0.8 s estimated on GPU), and the two
components with measurable accuracy impact — the adaptive box and the
ring-based black vote — each cost essentially nothing at inference time.
The three characteristic climbing challenges (camouflaged walls,
body-embedded holds, pose-driven viewpoint) are explicitly addressed.
Code is MIT-licensed; weights are fetched via a standalone download
script; wall-level annotations and ablation logs are released with the
repository.

---

## References

1. **Kirillov et al. (2023).** Segment Anything. *ICCV 2023.*
2. **Zhang et al. (2023).** Faster Segment Anything: Towards Lightweight SAM
   for Mobile Applications (MobileSAM). *arXiv:2306.14289.*
3. **Zhou et al. (2024).** EdgeSAM: Prompt-In-the-Loop Distillation for
   On-Device Deployment of Segment Anything in Real-Time. *arXiv:2312.06640.*
4. **Xiong et al. (2023).** EfficientSAM: Leveraged Masked Image Pretraining
   for Efficient Segment Anything. *CVPR 2024.*
5. **Lugaresi et al. (2019).** MediaPipe: A Framework for Building Perception
   Pipelines. *arXiv:1906.08172.*
6. **Cao et al. (2021).** OpenPose: Realtime Multi-Person 2D Pose Estimation
   Using Part Affinity Fields. *IEEE TPAMI 43(1).*
7. **Yang et al. (2023).** Effective Whole-Body Pose Estimation with Two-Stage
   Distillation (DWPose). *ICCVW 2023.*
8. **Ultralytics (2024).** YOLO26 / YOLOv8 segmentation and pose models.
   <https://github.com/ultralytics/ultralytics>
9. **Ionescu et al. (2016).** Hold-detection datasets for climbing imagery.
10. **Vujovic et al. (2021).** Classification of climbing holds by color in
    Lab space.

> Entries 9-10 are representative of the sparse climbing-specific literature;
> the absence of a standard benchmark is one reason we release wall-level
> annotations together with our videos.

---

## Table 5 (appendix): Per-frame latency breakdown

CPU: Intel Core i5-13500H (12C/16T), PyTorch + onnxruntime CPU execution
providers.  Stage timings measured on 3 frames of 2X_wall_01 (one warmup
frame excluded) via `dwpose_exp/dbg_profile_frame.py`; the full-set mean
is over all 65 frames of the benchmark.

| Component | Time | Note |
|---|---|---|
| DWPose — yolox_l person detection | 2.76 s | after the single-class NMS fix (see below) |
| DWPose — RTMPose 133-keypoint | 0.40 s | 288×384 input, top-down |
| YOLO26m person mask | 1.79 s | Ultralytics, CPU |
| EdgeSAM `set_image` (encoder) | 1.02 s | once per frame |
| EdgeSAM `segment_endpoint` × 4 | 1.86 s | ≈0.56 s each; includes pixel vote + ring statistics |
| **Sum of components** | **7.83 s** | 3-frame profiling sample (3.3 valid endpoints/frame) |
| **Full-set mean** | **7.17 s** | 65 frames; frames with fewer valid endpoints are faster |

The single-class NMS fix is worth noting: the vendored ONNX person
detector originally ran 80-class NMS, costing 34.5 s per frame; skipping
the per-class loop for our single-person case reduces it to 3.4 s, a
**12× speedup** on the largest single component.

GPU estimate (not measured): DWPose ~0.4 s + YOLO ~0.2 s + EdgeSAM ~0.15 s
≈ **~0.8 s/frame**.

The two components with measurable accuracy impact — the adaptive box and
the ring black vote — add essentially zero latency (one bbox computation
and one numpy pass over ~13k pixels respectively).

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
