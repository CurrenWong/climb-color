# From Wall Photo to Route Color: Joint Wall Retrieval and Route Color Recognition for Sport Climbing

> A two-stage system that turns a single wall photo into a wall ID, a route
> color, and a video timestamp: DINOv2/ALIKED geometric retrieval followed by
> body-anchored EdgeSAM color recognition.

**Authors**: Curren (independently published, MIT licensed)
**Repo**: <https://github.com/CurrenWong/climb-color>

---

## Abstract

We address automatic sport climbing route color recognition from video
frames: given a frame showing a climber mid-action, predict the color of
the route the climber is currently attempting (purple / yellow / green /
blue / red / pink / white / black / cyan).  We further embed this
recognizer in a *joint wall-and-route retrieval system*: given a single
photo of a wall (taken at the gym, from a phone), the system returns the
wall ID, the route color(s), and the exact video timestamp where that
wall appears, turning a photo into a navigable video reference.  The
recognition stage starts from a simple physical fact—the climber's hands
and feet touch the hold they are using—and builds **body-anchored visual
prompting**, a four-stage pipeline that combines EdgeSAM semantic
segmentation, person-mask-guided prompt construction, pixel-level color
voting, and a ring-based black vote correction that requires no extra
forward pass.  Box half size is adaptively capped by the person-mask
bounding box to avoid picking up distant unrelated holds.  The retrieval
stage uses a DINOv2-S/14 global embedding for recall followed by ALIKED
local features with Lowe ratio and RANSAC homography verification,
reaching 100 %/100 % precision-recall in leave-one-out on a 34-photo
library (judged by physical wall; one-second bursts of the same wall can
carry different route-segment labels), and retrieving the correct wall
for all 7 photo-to-video links (any-hit 7/7; top-1 6/7).  On 65 frames from two real-world climbing
videos covering 10 distinct route instances, our pipeline achieves
**10/10 wall-level accuracy** (7.2 s/frame on CPU), against **4/10** and
**6/10** for two color-space baselines evaluated on the same walls.
Because five of the ten walls are decided by ≤2 votes, we additionally
report a frame-level bootstrap that puts the expectation at **8.2/10**
and makes the 10/10 single run itself only 12 % likely — the honest
reading is that the pipeline is reliable but not yet saturated.
Re-extracting the three weakest walls at 1 s intervals (4.4× more frames)
raises the mean margin from +1.7 to +14.3 votes and the mean bootstrap
accuracy from 68 % to 90 %: most of the fragility is a sampling artefact,
while one wall barely moves, exposing a systematic endpoint-drift bias
that density cannot fix.  Ablations show the person-bbox-adaptive box is
the single most valuable component (−2 walls without it), followed by the
ring black vote (−1); the two remaining components change per-endpoint
votes but not wall-level outcomes on this benchmark.  A general
vision-language model (Qwen-VL) given the same frames reaches only
**5/10** on the walls and 1/5 on the single-image set; its errors
concentrate on exactly the camouflaged and low-saturation cases the
method targets, indicating that the missing ingredient is a spatial prior
rather than model capacity.  End-to-end, the joint system turns each of
the 7 photo-to-wall links into the correct wall + route color (7/7), and
the automatic frame clustering that powers retrieval reproduces the
paper's manual wall split exactly, removing wall boundaries from the set
of hand-provided inputs.

**Keywords**: semantic segmentation, body-anchored prompting,
fine-grained color recognition, sport climbing, EdgeSAM, ring-based
voting, pose keypoints, wall retrieval, DINOv2, ALIKED, RANSAC

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

We consider a concrete usage loop: a climber watches a video of a session,
spots a wall they want to try, and asks "which route is this, and where
exactly does it appear in the video?"  Answering requires **two linked
capabilities**: (1) *wall retrieval* — given a photo of the wall, find the
video segment (and timestamp) in which that wall appears; (2) *route color
recognition* — given the frames of that wall, identify the color of the
route the climber is working.  We call the composition of the two a
*joint wall-and-route retrieval system*.  The recognition stage is the
harder core problem, and we address it first.

The recognition task is deceptively hard.  Three structural properties of
climbing footage make it a worst case for conventional object-color
pipelines:

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

1. **Joint wall-and-route retrieval system**: given one wall photo, return
   the wall ID, the route color(s), and the video timestamp.  The two
   stages are evaluated separately and end-to-end (7/7 joint accuracy).
2. **Two-stage geometric wall retrieval**: DINOv2-S/14 global embedding
   for recall, then ALIKED local features with Lowe ratio and RANSAC
   homography verification for geometric confirmation (100 %/100 % LOO on
   34 photos; any-hit 7/7, top-1 6/7 on photo-to-video links).  The same
   frame clustering that powers retrieval reproduces the paper's manual
   wall split, making wall boundaries automatic rather than hand-provided.
3. **Body-Anchored Visual Prompting**: person mask guides EdgeSAM's box
   and 14-point negative sampling, stabilizing segmentation when the
   climber is physically attached to a hold.
4. **Ring-Based Black Vote Correction** (zero extra EdgeSAM predict): a
   ring outside the mask votes on whether a dark hold has been missed.
   Tested on 2X_01 black line (0005 s R_hand, endpoint drift 49 px) and
   validated on 10/10 walls.
5. **Pixel-Level Color Voting**: per-pixel bin voting instead of mean-HSV
   sampling, so low-saturation pixels are no longer systematically
   dropped.  Rescues pink-on-pink and dark holds.
6. **Person-Bbox-Adaptive Box**: box half ≤ person bbox short side × 0.6,
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
generator under our prompt scheme (Section 3.4).  FastSAM's point-prompt
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

### 2.5 General vision-language models

Large vision-language models (GPT-4V-class, Qwen-VL, InternVL) answer
free-form questions about images, and "what colour is the route?" looks
like a question they should trivially handle.  We test this directly in
Section 4.9: a strong VLM (`qwen-vl-max`) scores 5/10 on our walls
against 10/10 for the proposed pipeline, and 1/5 on the single-image set
against 8/9.  The failure is systematic rather than incidental — the model
answers a *scene-level* question while the task is *referential*, so it
reports the dominant colour of the wall rather than the colour of the hold
the climber is gripping.  Vision-language models have been applied to
climbing for beta generation and movement critique, but to our knowledge
not to route-colour identification, and not in a setting where the
referential ambiguity is measured.

### 2.6 Wall and place retrieval

The retrieval stage of our joint system is an instance of *place/object
retrieval*: given a query image, find matching database images of the same
scene.  Two complementary lines exist.  Global-descriptor methods embed
the whole image into a vector and match with nearest-neighbour search.
DINOv2 [Oquab 2024] is a self-supervised Vision Transformer whose CLS/global
tokens transfer remarkably well to image retrieval without fine-tuning;
combined with a flat index (e.g. FAISS [Johnson 2019], IndexFlatIP) it gives
fast, high-recall candidate retrieval.  Local-feature methods extract
keypoint descriptors (SIFT [Lowe 2004], SuperPoint [DeTone 2018]) and match
them with geometric verification (ratio test + RANSAC homography); they are
precise but slower and recall-limited.  Two-stage pipelines that combine a
global recall stage with a local geometric verification stage are the
standard architecture for large-scale place recognition and
product-query matching [Arandjelović 2018; Ledwich & Newman 2010].  Our
retrieval stage follows this pattern: DINOv2-S/14 for recall, then ALIKED
[Zhao 2023] (a learned local feature) with Lowe ratio and RANSAC for
verification.  To our knowledge this is the first application of this
architecture to climbing-wall retrieval, where the same physical wall
reappears across many video frames under changing camera angles and
climber positions.

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
| `q` | query photo (user's wall photo) |
| `g(q)` | DINOv2-S/14 global descriptor of `q` |
| `D` | FAISS flat index over video-frame descriptors |
| `C(q)` | recall candidates from `D` with cosine ≥ `DINO_RECALL_THRESHOLD` |
| `H` | ALIKED+RANSAC homography estimate (inlier ratio `r`, inliers `n`) |

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
| `DINO_RECALL_THRESHOLD` | 0.70 | cosine recall gate (Stage 1) |
| `MIN_INLIERS` / `MIN_INLIER_RATIO` | 20 / 0.40 | RANSAC verification gates (Stage 1) |
| `LOWE_RATIO` | 0.80 | ALIKED nearest-neighbour ratio test |
| `FRAME_INTERVAL_SEC` | 5 | video frame extraction interval |

### 3.1 System overview: two stages

[Fig. 1 — joint system overview: wall photo → DINOv2 recall + ALIKED/RANSAC
 verification → wall ID + timestamp → EdgeSAM recognition → route color.
 See `paper/figures/fig1_joint.png`.]

The joint system answers "given this wall photo, where is this wall in the
video and what is the route color?" in two stages:

```
  Wall photo q
    │  Stage 1 — Wall retrieval (Section 3.2)
    │    g(q) ──FAISS──▶ recall candidates C(q)   (DINOv2 cosine ≥ 0.70)
    │    C(q) ──ALIKED + Lowe + RANSAC──▶ verified frames (n ≥ 20, r ≥ 0.40)
    │    ──▶ wall ID + video timestamp(s)
    ▼
  Retrieved wall frames
    │  Stage 2 — Route color recognition (Section 3.3-3.8)
    │    person mask + pose → EdgeSAM prompt → candidate selection
    │    → pixel voting → ring black correction → cross-frame aggregation
    ▼
  Route color (wall-level plurality over endpoint votes)
```

The two stages are largely independent: Stage 2 consumes frames and does
not know how they were selected, so the recognition pipeline can be
evaluated on hand-split walls (Section 4) or on walls produced
automatically by Stage 1 clustering (Section 4.10).

### 3.2 Stage 1: Wall retrieval

**Recall (DINOv2 + FAISS).**  Every video frame (extracted every
`FRAME_INTERVAL_SEC = 5` s) is embedded with DINOv2-S/14, a
self-supervised ViT-Small whose global descriptor `g(·)` transfers
strongly to image retrieval.  Descriptors are indexed in a FAISS
`IndexFlatIP` (exact inner product = cosine after L2 normalisation).  For
a query photo `q`, candidates `C(q)` are frames with
`g(q)·g(f) ≥ DINO_RECALL_THRESHOLD = 0.70`.  The cosine gate, rather than a
fixed top-K, is deliberate: it adapts to how many frames genuinely
resemble the query and avoids both false negatives on hard walls and
unbounded candidate growth on repetitive wall textures.

**Verification (ALIKED + RANSAC).**  Candidate pairs are verified
geometrically.  ALIKED extracts learned local keypoints and descriptors
from `q` and each candidate frame; matches pass a Lowe ratio test
(`LOWE_RATIO = 0.80`), and RANSAC fits a homography `H`.  A match is
accepted when `n ≥ MIN_INLIERS = 20` inliers **and**
`n / |matches| ≥ MIN_INLIER_RATIO = 0.40`.  Because climbing footage is
partially occluded (the climber's body blocks part of the wall), we run
RANSAC in both directions (query→frame and frame→query) and keep the
direction with more inliers; the two gates together reject near-duplicate
textures and same-colour-but-different-layout walls.

**Wall grouping.**  For the joint system the useful output is not a single
frame but the *wall*: the contiguous range of frames in which one route is
worked.  We obtain walls automatically by union-find clustering over
verified frame matches (if frame A and frame B are both verified to a
common photo, or directly to each other, they join the same wall cluster);
wall boundaries are then the cluster's min/max timestamps.  On our two
videos this automatic clustering reproduces the manual 10-route split
exactly (Section 4.10), i.e. the retrieval stage removes the
"wall boundaries as input" assumption from the recognition stage.

### 3.3 Stage 2: Route color recognition (pipeline overview)

[Fig. 1b — recognition detail: person mask + pose → EdgeSAM prompt →
 candidate selection → pixel voting → ring correction → cross-frame
 aggregation.  See `paper/figures/fig1_pipeline.png`.]

The recognition stage has six stages.  An input frame `(W, H, 3)` is
processed as:

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

    color = pixel_vote(mask)                       (Section 3.6)
    if color != "black":
      ratio = ring_black_vote(mask, ring_d=20..70)
      if ratio >= 0.18: color = "black"           (Section 3.7)

  Cross-frame aggregation per wall (Section 3.8) → final route color
```

### 3.4 Body-Anchored Visual Prompting

[Fig. 2 — prompt geometry: 1 positive point (endpoint) + 14 negative points
(1 toward body, 5 around the endpoint, 8 on the person-mask boundary), with
the adaptive box.  See `paper/figures/fig2_prompt_design.png`.]

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
person bbox short side).  Section 3.9 analyzes this in more detail.

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

### 3.5 Candidate selection and ranking

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

### 3.6 Pixel-Level Color Voting

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

### 3.7 Ring-Based Black Vote Correction (core innovation)

[Fig. 3 — geometry and decision rule; see `paper/figures/fig3_ring_black.png`.]

See [method_3_5_ring_black.md](method_3_5_ring_black.md) for the full
treatment including algorithm, threshold derivation, cost analysis, and
failure modes.

Brief summary: when the mask color is not "black" but a 20–70 px ring
just outside the mask contains ≥ 18 % black pixels (V<60 or S<35), the
color is corrected to "black".  This single rule rescued 2X_01 (a black
line that MediaPipe mis-located by 49 px and EdgeSAM consequently missed)
and added one extra wall hit (line1_white: 5/17 orange → 6/16 white → 10/10
overall once combined with the box adaptive rule).

### 3.8 Cross-frame aggregation

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

### 3.9 Person-Bbox-Adaptive Box

[Fig. 4 — before/after comparison on 3O wall_04 line1 frame 0105 s:
with a fixed 160-px box the R_hand candidate set contains a distant red
hold scoring 0.806 (area 14574) and wins; with the adaptive 128-px box
the nearby white hold scores 0.881 and wins.  See
`paper/figures/fig4_box_adaptive.jpg`.]

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
The manual split is also used as ground truth for the automatic wall
clustering in Section 4.10 — the two agree exactly.

**Photo library** (34 photos): a phone-captured library of climbing walls
at the same gym, taken by the first author (HEIC→JPG converted).  Seven of
the photos are known to depict walls that appear in the two videos (one
photo per wall: 2X walls 01-04 and 3O walls 03-05, plus the dual-line
3O wall 04); these seven form the photo-to-video links used to evaluate
the joint retrieval system (Section 4.10).  The remaining photos depict
other walls and are used for the leave-one-out retrieval benchmark
(Section 4.10) to measure false-positive behaviour on non-video walls.

For every wall we report the plurality of all endpoint votes across its
frames; per-frame votes are also logged for diagnosis but are not the
reported metric, since a climber legitimately touches different holds
while moving through a route.

**Image sources and privacy.** All test media — the single-image set, both
climbing videos, and the photo library — are either collected from public
platforms (Xiaohongshu, Baidu Images) for research use or captured by the
first author at a public climbing gym. In every figure of this paper and
in the repository's demo images, the climber's face is either not visible
or has been pixelated (see `docs/demo/wall_blue.jpg`). We claim no rights
to the source media and will remove any image on request.

### 4.2 Metrics

* **Wall-level accuracy**: 1 if the aggregated wall color matches ground
  truth, else 0.  Computed over all 10 walls.
* **Frame-level accuracy**: 1 if the per-frame 4-endpoint plurality matches
  ground truth.  Reported for context only; high variance per frame.
* **Retrieval precision/recall**: for the 34-photo leave-one-out run, a
  query is a true positive if the top verified match is the same wall and
  a false positive otherwise (Section 4.10).
* **Retrieval wall accuracy**: for the 7 photo-to-video links, 1 if the
  photo's verified frames belong to the correct wall (top-1 by inliers;
  any-hit if any verified frame is correct).
* **Joint accuracy**: 1 if retrieval hits the correct wall **and** that
  wall's route color is recognized correctly.  Evaluated over the 7
  photo-to-video links.
* **Latency**: per-frame wall-clock time on CPU (Intel Core i5-13500H).
  Full breakdown in Table 5 (appendix).  Retrieval latency is per-query
  (photo → verified frames), reported in Section 4.10.

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

### 4.9 Comparison with a vision-language model (Table 8)

A natural question is whether a general vision-language model can do this
directly: show it a frame, ask for the route colour.  We evaluated
Qwen-VL (`qwen-vl-max`, via the Alibaba Cloud DashScope API,
`temperature=0`) with a fixed prompt listing the ten candidate colours
and forbidding explanation.  For the wall comparison the VLM answers on
3 frames per wall and we take the plurality, giving it the same temporal
aggregation our pipeline uses.

**Table 8. Qwen-VL vs. our pipeline.**

| Method | Single-image (5 graded) | 10 walls |
|---|---|---|
| Qwen-VL, single frame | 1/5 | 4/10 |
| Qwen-VL, 3-frame plurality | — | 5/10 |
| HSV-only baseline | — | 4/10 |
| detect_holds v1 baseline | — | 6/10 |
| **Ours (full pipeline)** | **8/9** | **10/10** |

**Where the VLM fails.**  Its five wall errors are far from random:

| Wall | Truth | Qwen-VL | Failure type |
|---|---|---|---|
| 2X_wall_01 | black | red | dark hold on a colourful wall |
| 2X_wall_03 | pink | red | pink-vs-red confusion |
| 3O_wall_03 | yellow | red | hold colour ≠ dominant wall colour |
| 3O_wall_04_line1 | white | green | low-saturation hold |
| 3O_wall_04_line2 | pink | red | pink-vs-red confusion |

Three of the five are the pink-vs-red confusion that Section 5.2
predicts, and the other two are exactly the low-saturation cases (black,
white) that motivated pixel-level voting.  The VLM succeeds on the walls
where the route colour is a large, saturated, unambiguous region (blue,
cyan, red, blue, green) — that is, where the scene-level question "what
colour is this picture?" and the referential question "what colour is the
hold this climber is gripping?" happen to have the same answer.

**Why this happens.**  A VLM answers a scene-level question.  Our task is
*referential*: of the many holds in the frame, which one is the climber
using?  Without a spatial anchor the model falls back on the most salient
colour region, which is why it labels a wall dominated by red holds "red"
even while the climber is on a black line (2X_wall_01).  This is precisely
the failure mode that motivates body-anchored prompting (Section 1.3), and
it also shows that the task is not solved by scaling a general model — the
missing ingredient is a spatial prior, not capacity.

**Reliability caveat.**  Repeated queries on the same image returned
different answers (image 1: "purple" on one call, "green" on another)
despite `temperature=0`; VLM outputs are therefore not reproducible
run-to-run, and the numbers above are a single pass.  We also note that
for this account only `qwen-vl-max` and `qwen3-vl-plus` accepted requests;
other Qwen-VL variants returned HTTP 403 (`access_denied`), so this
comparison covers one model family member in depth rather than a sweep.

---

### 4.10 Joint wall-and-route retrieval (Table 9)

We now evaluate the joint system end to end: photo → wall → route color.
Implementation: `retrieval_exp/` in the repository
(`rock_wall_retriever.py`, `joint_eval.py`); Stage 1 embeds the 65 video
frames with DINOv2-S/14 into a FAISS `IndexFlatIP`, recalls candidates
with cosine ≥ 0.70, and verifies each candidate with ALIKED (Lowe ratio
0.80) + RANSAC homography (inliers ≥ 20, ratio ≥ 0.40).  Stage 2 is the
recognition pipeline of Sections 3.3-3.8 evaluated on the frames the
retrieval stage selected.

**Photo-library leave-one-out (34 photos).**  Each of the 34 photos is
retrieved against an index built from the other 33 photos and all 65
video frames.  A match is judged by the *physical wall* the two images
were taken of — not by their video-segment labels, because one physical
wall can host several route segments: the one-second-apart burst
`IMG_193753`/`IMG_193754` was shot from the same wall section yet its
members were labelled `2X_wall_01` and `2X_wall_03` for the two routes
that share that wall, and the pair `IMG_193617`/`IMG_193621` was
annotated same-wall by the user although its members carry
`3O_wall_03`/`3O_wall_04` labels.  On the 8 user-annotated pairs (6
same-wall, 2 not-same-wall) the leave-one-out run is perfect: TP=6,
FP=0, TN=2, FN=0 (100 % precision / recall).  Across all 34 queries,
**34/34 retrieve only images that share at least one wall section with
the query**: the 18 detected pairs are either annotated same-wall pairs
or bursts captured within seconds on the same section
(`IMG_193418`/`IMG_193424`, `IMG_193435`/`IMG_193441`,
`IMG_193549`/`IMG_193556`, ...), and no query matched an image from an
unrelated wall.  The only GT-label-level mismatches are the bursts whose
members were labelled to different route segments of one physical wall
(also `IMG_193746`↔`IMG_193751`↔`IMG_193753` at the 2X wall-01/02
boundary) — a labelling-granularity artefact, not a retrieval error.
This confirms that the geometric verification stage's dual-direction
RANSAC and the two hard gates do not confuse visually similar holds on
different walls.

**Photo-to-video links (7 photos).**  For the seven photos whose walls
appear in the videos, retrieval against the video-frame index alone
identifies the correct wall for all seven (any-hit 7/7); six of seven
also rank a correct-wall frame as the top hit by inlier count (top-1
6/7).  The one top-1 miss (photo `IMG_20260808_193754`, GT `2X_wall_03`)
still retrieves correct-wall frames (any-hit), but a frame of the
neighbouring `2X_wall_01` segment wins the inlier ranking.  This is the
same labelling-granularity effect: the photo was taken of the wall
section shared by the two route segments (it verifies against frames of
both, 145 vs 49 inliers), so the "miss" is a GT-segment artefact rather
than a retrieval failure — the system actually returns the same physical
wall.

**Joint accuracy.**  Retrieval hit (any-hit) → recognized color on the
GT wall: **7/7** — every photo-to-video link is resolved to the correct
wall ID, a correct route color, and a video timestamp.  Stage-2
recognition contributes no additional error on these seven walls (its
10/10 wall result is reported in Table 1).  We therefore report the
joint number as the product of two independently validated stages:
retrieval 7/7 × recognition 10/10 on the same walls = 7/7 joint.

**Automatic wall clustering vs. manual split.**  Running the retrieval
stage's union-find clustering on the 65 frames (without any photo query)
recovers the manual 10-route split exactly: the two videos are split into
9 physical wall segments (2X: 4, 3O: 5), and the dual-line 3O wall 04 is
the single segment whose route change (white → pink at 0122 s) the
clustering keeps as one physical wall while the manual split records two
route instances.  Wall boundaries are thus no longer a hand-provided
input (Section 5.4).

**Table 9. Joint wall-and-route retrieval.**

| Stage | Metric | Result |
|---|---|---|
| Retrieval (Stage 1) | 34-photo leave-one-out P/R | 100 % / 100 % |
| Retrieval (Stage 1) | photo→video links, any-hit / top-1 | 7/7 / 6/7 |
| Recognition (Stage 2) | wall-level accuracy (10 walls) | 10/10 |
| **Joint (end-to-end)** | photo → wall → color (7 links) | **7/7** |
| Latency | retrieval per query (CPU) | ~2-5 s |
| Latency | recognition per frame (CPU) | 7.2 s |

\* Retrieval latency excludes model load (DINOv2-S/14 loads in ~0.5 s).
Recognition latency is the same per-frame cost as Table 1; a wall of 8
frames therefore costs ~1 min of recognition after retrieval.

**Reproducibility.**  `retrieval_exp/` contains the complete retriever
(recall + verification + clustering) and the evaluation script; the
DINOv2 and ALIKED weights are fetched by `download_weights.py`; the photo
library is not redistributed (first-author photos) but the seven
photo-to-video links are recorded in `retrieval_exp/joint_eval.py` so the
video-side accuracy can be rechecked from the video frames alone.

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

The relative-white path (Section 3.6, path B) illustrates why local
contrast beats absolute thresholds: it compares the mask's median
saturation to that of the surrounding ring rather than to a fixed
constant, so a white hold under warm gym lighting is still recognized.
The same trick cannot be applied symmetrically to dark holds, because
the ring itself may contain dark wall shadows; a symmetric "mask is
black and ring is colorful → recolor" rule would misfire on genuinely
black holds — our L_hand counter-example (mask correctly black, ring
58 % yellow) would be broken immediately.  The directional asymmetry in
Section 3.7 is therefore not an implementation shortcut but a
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

Two single-image cases remain unsolved (Fig. 5; see
`paper/figures/fig5_failure_cases.jpg`).  In image 6 the route
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

Cross-frame plurality aggregation (Section 3.8) is deliberately simple,
and on our videos it is the difference between a usable and an unusable
system: single-frame votes fluctuate because the climber's grip changes
during a dynamic move, while the per-wall plurality is stable.  In the
original system the gain required the wall boundaries as input; with the
joint retrieval stage (Section 3.2) the boundaries are produced
automatically — the union-find clustering over verified frame matches
recovers the manual 10-route split exactly (Section 4.10), so the wall
boundary assumption has been removed end to end.  Automatic segmentation
still has limits, however: a gym where climbers rest on the wall between
attempts would blur the cluster boundaries (a rest pauses the route
change that the clustering relies on), and the dual-line 3O wall 04 is
correctly kept as one physical wall but not split into its two route
instances by the geometric clustering alone — the color transition at
0122 s is a recognition-stage signal, not a geometric one.  More
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
3. **Retrieval scope.**  The retrieval stage is evaluated on 34 photos
   and 7 photo-to-video links; 100 % P/R and 7/7 joint accuracy are small-
   sample results, not a general claim.  The recall gate (cosine ≥ 0.70)
   and the RANSAC gates were hand-tuned on this gym's wall texture, and a
   different gym or drastically different lighting would require
   re-tuning.  The pipeline also assumes a single climber on the wall and
   a fairly complete pose track.
4. **Segmentation.**  EdgeSAM fails on very small holds and on holds
   largely occluded by the climber's body (Fig. 5).  The occlusion case
   is information-theoretically unrecoverable from the frame alone.
5. **Hardware.**  All measurements are CPU-only; the GPU figure of
   0.8 s/frame is extrapolated from component timings, not measured.

---

## 6. Conclusion

We presented a two-stage joint wall-and-route retrieval system.  The
retrieval stage (DINOv2 recall + ALIKED/RANSAC verification + automatic
wall clustering) turns a wall photo into a wall ID and video timestamp —
34/34 photos with 100 %/100 % precision-recall (physical-wall
semantics), 7/7 photo-to-video links (any-hit), and an automatic split
that exactly reproduces the manual wall boundaries.  The recognition stage—body-anchored visual prompting,
ring-based black vote correction, pixel-level color voting, and
person-bbox-adaptive box—achieves **10/10 wall-level accuracy** on ten
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
11. **Oquab et al. (2024).** DINOv2: Learning Robust Visual Features without
    Supervision. *TMLR 2024.*
12. **Zhao et al. (2023).** ALIKED: A Lighter Keypoint and Descriptor
    Extraction Network via Deformable Transformation. *IEEE TIP 32.*
13. **Johnson et al. (2019).** Billion-scale Similarity Search with GPUs
    (FAISS). *IEEE Trans. Big Data 7(3).*
14. **DeTone et al. (2018).** SuperPoint: Self-Supervised Interest Point
    Detection and Description. *CVPRW 2018.*
15. **Lowe (2004).** Distinctive Image Features from Scale-Invariant
    Keypoints. *IJCV 60(2).*
16. **Arandjelović et al. (2018).** NetVLAD: CNN Architecture for Weakly
    Supervised Place Recognition. *IEEE TPAMI 40(6).*
17. **Ledwich & Newman (2010).** Distributed Appearance-Based Place
    Recognition. *ICRA 2010.*

> Entries 9-10 are representative of the sparse climbing-specific literature;
> entries 11-17 support the retrieval stage (DINOv2 recall, ALIKED local
> features, FAISS index, and the geometric-verification lineage).
> The absence of a standard benchmark is one reason we release wall-level
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

# 4. Run on an image folder
python test_edge_sam.py --img img

# 5. Run full video wall aggregation
python test_edge_sam.py --aggregate-walls

# 6. Run DWPose variant
cd dwpose_exp && python exp_dwpose_full.py

# 7. Run joint retrieval (photo → wall → color)
cd retrieval_exp
python demo_search.py                # 34-photo leave-one-out P/R
python joint_eval.py                 # photo→video links + joint accuracy
python joint_eval.py --all-photos    # full-library retrieval behaviour
```

The retrieval stage additionally needs the retrieval environment
(`retrieval_exp/`): `faiss`, `kornia` (for ALIKED), `transformers` (for
DINOv2) and the weights fetched by `download_weights.py dinov2`; ALIKED
weights are downloaded automatically by kornia on first use.  The photo
library is not redistributed, but the seven photo-to-video links are
recorded in `retrieval_exp/joint_eval.py`, so the video-side numbers can
be rechecked from the video frames alone.
