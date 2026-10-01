# Master methodology for the article — YOLO26-seg vs. U-Net vs. SAM 3 on ISIC 2018 Task 1

> **What this document is.** A single, citable description of how the three experimental arms
> (`sandbox_yolo26`, `sandbox_unet`, `sandbox_sam3`) were built, trained, evaluated and compared, written so that the
> Methods, Results and Limitations sections of the thesis/article can be drafted directly from it.
>
> **Sources.** It was synthesised from the code (the authoritative source: `common.py`, the phase scripts and
> `segmentation_metrics.py` of each repository), the three READMEs and the commit history (which records the
> *why* of every protocol decision, with the numbers observed in the first full runs). The per-repository
> `METHODOLOGY_NOTES_FOR_ARTICLE.md` files are listed in each `.gitignore` ("My files") and were never pushed, so
> they could not be read here: **merge any point from your private notes that is missing below.**
>
> **Placeholders.** Everything written as **[RESULT]** must be filled in from `article_outputs/` after the final run
> (produced by `Article_Figures_and_Tables.ipynb`). No number below is a result of the final run; numbers quoted
> from earlier runs are labelled as such.

---

## Contents

1. [Research question and design at a glance](#1-research-question-and-design-at-a-glance)
2. [Dataset handling (Phase 0)](#2-dataset-handling-phase-0)
3. [The unified 5-phase pipeline and why it is a fair comparison](#3-the-unified-5-phase-pipeline-and-why-it-is-a-fair-comparison)
4. [Why early stopping is disabled in every model](#4-why-early-stopping-is-disabled-in-every-model)
5. [Training configuration of each arm](#5-training-configuration-of-each-arm)
6. [Model-specific quirks that must be reported](#6-model-specific-quirks-that-must-be-reported)
7. [Evaluation metrics and statistics (Phase 5a)](#7-evaluation-metrics-and-statistics-phase-5a)
8. [Real-world efficiency measurement (Phase 5b)](#8-real-world-efficiency-measurement-phase-5b)
9. [Cross-architecture aggregation (root notebook)](#9-cross-architecture-aggregation-root-notebook)
10. [Reproducibility, determinism and software environment](#10-reproducibility-determinism-and-software-environment)
11. [Threats to validity and limitations](#11-threats-to-validity-and-limitations)
12. [Ready-to-adapt wording for the Methods section](#12-ready-to-adapt-wording-for-the-methods-section)
13. [Artefact map: which file feeds which table or figure](#13-artefact-map-which-file-feeds-which-table-or-figure)
14. [Committee-compliance checklist and audit changelog](#14-committee-compliance-checklist-and-audit-changelog)

---

## 1. Research question and design at a glance

**Question.** For binary skin-lesion segmentation (ISIC 2018 Task 1), how does a fine-tuned **foundation model**
(SAM 3, 840.5 M parameters) compare with **lightweight specialised models** (YOLO26-seg n/s/m/l/x, 2.7–63 M
parameters; a classic U-Net, 2.16 M parameters) in **accuracy** *and* in **measured deployment cost** (latency,
throughput, memory) — not only in proxies such as parameters and GFLOPs?

**Design.** Three independent repositories implement the *same* five-phase protocol with the *same* data, splits,
cross-validation folds, ground truth, metric code, evaluation resolution, test isolation, efficiency profiler and
output schema. Only the model, its native input size, its default recipe and its tuning space differ.

| | YOLO26-seg (n, s, m, l, x) | U-Net | SAM 3 |
|---|---|---|---|
| Family | one-stage instance segmentation (CNN + PSA attention) | encoder–decoder CNN (Keras baseline, ported to PyTorch) | promptable vision–language foundation model (ViT) |
| Parameters (fused) | 2.7 / 10.4 / 23.6 / 28.0 / 62.8 M (from the `.yaml`; final values in Table 2) | 2,158,705 (2,161,649 unfused) | 840,509,750 (all trainable) |
| Native input | 640 px (letterbox) | 256 × 256 | 1008 × 1008, prompt "skin lesion" |
| Training code | Ultralytics 8.4.21 | own (bit-exact resumable) | Meta's official SAM 3 trainer, wrapped |
| Default recipe (Baseline) | Ultralytics defaults | original Keras baseline | Meta's official fine-tuning recipe |
| HPO | Ultralytics genetic tuner (seeded), 30 trials × 30 epochs | Optuna TPE (seeded), 30 × 30 | Optuna TPE (seeded), 10 × 10 |
| Training budget (Ph 1, 2, 4) | 120 epochs, no early stopping | 120 epochs, no early stopping | 30 epochs, no early stopping |
| Model selection (`best.pt`) | Ultralytics fitness (box + mask mAP50-95) on val | per-image mean JSI on val | per-image mean JSI on val |

---

## 2. Dataset handling (Phase 0)

### 2.1 Source and split

* **Source:** the **raw official ISIC 2018 Task 1 release** (JPEG dermoscopy images + PNG binary lesion masks),
  mounted read-only (`../datasets/ISIC2018_Raw`).
* **Split:** the **official** partition, asserted on input *and* output: **2,594 training / 100 validation /
  1,000 test** images (3,694 in total); every image paired with its mask; no ISIC ID in two splits.
* **Why not the earlier Roboflow export.** The export used in exploratory work had silently dropped 47 training and
  6 test images (2,547 / 100 / 994) and stretched every image to 640 × 640 (aspect ratio destroyed). It was
  abandoned; Phase 0 now rebuilds everything from the raw release. *(Disclose if any earlier result is quoted.)*

### 2.2 One shared dataset, three model-specific views

YOLO26's Phase 0 (`yolo26_seg/prepare_dataset.py`) is the **single source of truth**; the U-Net and SAM 3 Phase 0
scripts read *its* output (mounted read-only), so all arms see byte-identical images, masks and IDs.

| Step | What is done | Why |
|---|---|---|
| Working resolution | longer side ≤ **1,024 px**, aspect ratio preserved, no upscaling; images decoded **without EXIF re-orientation**, stored as PNG (lossless) | official images reach several thousand px; 1,024 bounds compute while keeping lesion borders; no distortion |
| Ground truth | the **official mask** resized to the working resolution and stored as `masks/<ISIC_ID>.png` | the evaluation ground truth of **all three arms** (`segmentation_metrics.ground_truth_mask`) |
| Resampling check | Dice between the resized mask and the full-resolution official mask recorded per image (mean 0.998–0.999) | quantifies the (negligible) error introduced by the working resolution |
| YOLO labels (training only) | polygons from `cv2.findContours` (RETR_CCOMP); holes bridged into the outer contour; fragments/holes < 0.1 % of the lesion area left out of the *training label only*; every label rasterised back and checked against the mask (**Dice ≥ 0.98 asserted**; achieved ≥ 0.989, mean 0.9999) | YOLO needs polygons; the fidelity check bounds label error; evaluation never uses the polygons |
| U-Net view | 256 × 256 arrays (area interpolation), **strictly binary** masks (area resize + 0.5 threshold), ISIC-ID manifests, SHA-256 provenance | fixes the soft (non-binary) masks of the old Keras `.npy` data |
| SAM 3 view | COCO dataset: one instance per image whose **RLE mask is the official mask** (lossless; holes and fragments exact); images hard-linked/copied; per-split and per-fold annotation files | the official SAM 3 loaders consume COCO RLE |

### 2.3 Cross-validation folds

* **Pool:** training ∪ validation = **2,694** images; the test set is **excluded and verified** (by path and by file
  name) before any fold is built.
* **Algorithm:** NumPy `RandomState(0)` shuffle + contiguous folds — bit-identical to scikit-learn's
  `KFold(n_splits=5, shuffle=True, random_state=0)` without the dependency; ≈ 539 held-out images per fold.
* **Identical folds in the three repositories:** each repository rebuilds the pool in YOLO26's order and verifies the
  fold fingerprints (`splits_manifest.json`); a re-run with different folds is refused.

### 2.4 Task 2 (lesion attributes)

Phase 0 of all three repositories can also build the multi-label ISIC 2018 Task 2 dataset (five attributes). It is
**not** used in Phases 1–5 of this study (Task 1 only); mention it, if at all, as future work.

---

## 3. The unified 5-phase pipeline and why it is a fair comparison

### 3.1 The phases

| Phase | Purpose | Data touched | Output used in the article |
|---|---|---|---|
| **0 — Dataset** | build and verify the shared dataset (§ 2) | raw release | dataset description |
| **1 — Baseline** | train with the **base setup + default hyperparameters** | train / val | Baseline model (`best.pt`) |
| **2 — Baseline CV** | 5-fold CV of the Phase 1 configuration; pixel metrics of each held-out fold at dataset resolution | train ∪ val (test excluded) | variance of the architecture, generalisation check (Table 6) |
| **3 — HPO** | seeded, fault-tolerant hyperparameter search | train / val | tuned hyperparameters |
| **4 — Optimized** | train with the **base setup + tuned hyperparameters** | train / val | Optimized model (`best.pt`) |
| **5 — Test** | 5a accuracy, 5b efficiency, 5c consolidated report — Baseline **and** Optimized, FP32 **and** FP16 | **test (only here, once)** | Tables 1–4 and 7, Figures 1–6 |

Every step is orchestrated by one script per repository (`run_pipeline.sh`, `run_pipeline_unet.sh`,
`run_pipeline_sam3.sh`), is idempotent and resumable, and writes into an isolated folder `logs/<pipeline-name>/`.

### 3.2 The fairness controls (what is identical across the three arms)

1. **Same images, split, folds and ground truth** — one Phase 0 output, read-only, addressed by ISIC ID (§ 2).
2. **Same metric code** — `segmentation_metrics.py` is a **byte-identical copy** in the three repositories (checked by
   checksum); DSC, JSI, ISIC JSI, sensitivity, specificity, accuracy, Boundary IoU, NSD and HD95 are computed by the
   same function for every model.
3. **Same evaluation resolution** — every prediction is brought to the image's dataset resolution and scored against
   the same official mask (U-Net: bilinear upsampling of the 256 × 256 probability map; YOLO26: `retina_masks=True`;
   SAM 3: the official post-processor's upsampling). Native input sizes differ by design (§ 11).
4. **Same handling of failures** — an empty prediction on a lesion image scores 0 (HD95: image diagonal); nothing is
   skipped, so a model cannot improve its mean by failing silently.
5. **Test isolation** — the test split is read only in Phase 5, evaluated once per (variant, precision); the CV pool
   is verified test-free; HPO and model selection see only train/val.
6. **Same protocol shape** — Baseline = base setup + defaults, Optimized = base setup + tuned hyperparameters; the base
   setup (budget, precision, input, batch, seed, …) is **protected**: a tuned file or a search space that tries to
   change it is rejected. The HPO effect is therefore isolated in every arm (Table 3).
7. **Same training policy** — FP32 training, fixed seeds (0), **no early stopping** in any phase (§ 4).
8. **Same variant compared** — the cross-architecture comparison uses the variant fixed *a priori* (Optimized,
   Phase 4) for every model, not the better of the two after looking at the test set.
9. **Same efficiency profiler** — one benchmark design (batch 1, single GPU, fresh process per configuration, same
   timers, warm-up, statistics, memory accounting, contention check) derived from one script (§ 8).
10. **Same software stack** — PyTorch 2.5.1 / CUDA 12.1 in every Docker image (§ 10).
11. **Same statistics** — per-image means with seeded bootstrap 95 % CI; paired tests on the same 1,000 images (§ 7).

### 3.3 Phase details that matter for the text

* **Best epoch.** Every validation number reported for a training run is read at the epoch that produced `best.pt`,
  so reported validation metrics describe exactly the checkpoint evaluated in Phase 5.
* **Phase 2 metrics** are reported as mean ± **sample** SD (ddof = 1) over the 5 folds.
* **Phase 3** checkpoints its state before every trial; a failed trial (crash, OOM, NaN) is retried with identical
  hyperparameters, then recorded with fitness 0; GPU/driver failures exit with code 75 and are retried by the
  orchestrator; a change of search space, protocol, seed, data or library version refuses to resume.
* **Phase 5 caching** — results are keyed by the SHA-256 of the weights, the test list and the measurement settings
  (with a method version), so they are recomputed exactly when something relevant changes.

---

## 4. Why early stopping is disabled in every model

**Decision.** `patience = epochs` in every training phase of every arm: 120 / 120 for YOLO26 and U-Net (Phases 1, 2
and 4), 30 / 30 per HPO trial; 30 / 30 for SAM 3 (10 / 10 per HPO trial). Every run therefore completes its full
budget and learning-rate schedule. `best.pt` is still the epoch with the best validation score (§ 3.3).

**The problem: the official validation split has only 100 images.** Early stopping and model selection both compare
validation scores between epochs. With 100 images, the epoch-to-epoch fluctuation of the validation score is of the
same size as — or larger than — the real improvements late in training, so "no improvement for *p* epochs" happens
by chance while the model is still improving. *(Illustration: if per-image JSI has a standard deviation σ, the
standard error of a 100-image mean is σ/10; with σ = 0.15 that is 0.015, against ≈ 0.0065 on a 539-image CV fold.)*

**Evidence from the first full runs (`logs/pipeline_final_v1`, before the change):**

| Arm | Observation with early stopping | Consequence |
|---|---|---|
| YOLO26 (patience 25) | Median epoch-to-epoch change of the validation fitness: **0.04–0.06** on the 100-image split vs. **0.014** on a 530-image CV fold. **Every** Phase 1 run stopped between epochs **50 and 82** of 120. | Training stopped at a still-high learning rate, *before* the cosine decay and the final `close_mosaic` epochs, and `best.pt` sat on a noise spike. |
| U-Net (patience 25) | Baseline **and** Optimized stopped at epoch **81** of 120 (best epoch 56); **10 of 30** HPO trials stopped before their 30 epochs; meanwhile **4 of 5** CV folds (≈ 530 validation images) ran the full budget and kept improving until epochs **96–118**. | The HPO winner (val JSI **0.811** vs. **0.782** for the defaults) **did not transfer**: the Optimized U-Net was *worse* on the test set (ΔDSC = **−0.0052**, Wilcoxon p = 7.5 × 10⁻⁹) — the signature of selecting on validation noise. |
| SAM 3 | Not run with early stopping; disabled pre-emptively for the same reason (same 100-image split). | — |

**Why disabling it makes the comparison fairer.** (i) Every model of an arm receives the same, complete compute
budget, so differences are not artefacts of *when* noise stopped a run; (ii) schedules that depend on the epoch
count (cosine LR, warm-up, `close_mosaic`) run as designed; (iii) Baseline and Optimized differ only by their
hyperparameters, not by their stopping epoch; (iv) HPO trials are compared at equal budget.

**What it does not remove (state it as a limitation).** `best.pt` and the HPO fitness are still chosen on the
100-image validation split, so a milder "winner's curse" remains. It is controlled by reporting (a) the 5-fold CV
of the Baseline protocol (Table 6) and (b) the *test-set* paired Optimized − Baseline difference (Table 3), which shows
whether tuning transferred.

**Suggested wording:** *"Early stopping was disabled (patience equal to the epoch budget) in all phases and for all
models. In preliminary runs, early stopping on the 100-image official validation split was dominated by
epoch-to-epoch noise — all YOLO26 baseline runs stopped between epochs 50 and 82 of 120, before the end of the
learning-rate schedule, and the U-Net stopped at epoch 81 while cross-validation folds with ≈ 530 validation images
kept improving until epochs 96–118 — and the hyperparameters selected under early stopping did not transfer to the
test set. All models therefore train for their full budget; the reported checkpoint is the epoch with the best
validation score."*

---

## 5. Training configuration of each arm

### 5.1 YOLO26-seg (`yolo26_seg/common.py`)

| Setting | Value | Ultralytics default | Reason |
|---|---|---|---|
| `epochs` / `patience` | 120 / 120 (HPO trials 30 / 30) | 100 / 100 | § 4 |
| `amp` | **False** (FP32) | True | xlarge overflowed in FP16 (NaN in the cls-loss at epoch 42 of Phase 1); uniform numerics across sizes |
| `optimizer` | **MuSGD** | `auto` | `auto` picks AdamW or MuSGD from the iteration count and then **ignores** `lr0`/`momentum`; explicit optimiser makes default and tuned values effective |
| `cos_lr` | True | False | one LR schedule in every phase |
| pinned defaults | `batch=16`, `nbs=64`, `imgsz=640`, `workers=8`, `close_mosaic=10`, `seed=0`, `deterministic=True` | same | pinned so an upstream default change cannot alter the protocol |

* **HPO:** Ultralytics' genetic (mutation) tuner with `SeededTuner` (`np.random.default_rng([seed, trial])` instead of the
  clock), 30 trials × 30 epochs per model, **refined** space of 15 hyperparameters (`lr0` 1e-3–4e-3, `lrf`, `momentum`,
  `weight_decay` 1e-6–1e-4, `warmup_epochs`, loss gains `cls`, `dfl`, colour `hsv_h/s/v`, `translate`, `flipud`,
  `mosaic`, `mixup`, `copy_paste`). The refined space was narrowed from an earlier wide search on YOLO26-s
  (hyperparameters with |r| < 0.05 against fitness pinned). Note: the defaults `lr0 = 0.01` and
  `weight_decay = 5e-4` lie **outside** the refined bounds, so the Baseline's own values are not candidates.
* **Fitness / `best.pt`:** Ultralytics segmentation fitness = **box mAP50-95 + mask mAP50-95** (verified against the
  checkpoint), not JSI (§ 6.1).

### 5.2 U-Net (`unet/common.py`)

| Base setup (fixed) | Value |
|---|---|
| Architecture | Keras baseline U-Net: 16 base filters, ReLU, BatchNorm, 4 levels + bottleneck |
| Optimiser / schedule | AdamW (decoupled weight decay ≡ Keras `Adam(weight_decay)`), ε = 1e-7, β₂ = 0.999, **constant LR** |
| Loss | BCE + soft Dice (the original `bce_dice_loss`) |
| Input / batch / budget | 256 × 256, batch 16, 120 epochs, patience 120 (HPO 30 / 30) |
| Numerics / selection | FP32, seed 0, deterministic algorithms; `best.pt` = best validation per-image JSI |

| Tuned (Phase 3) | Default | Range |
|---|---|---|
| `lr0` | 1e-3 | [1e-4, 1e-2] log |
| `weight_decay` | 0 | [1e-6, 1e-3] log (default 0 outside → first trial clipped to 1e-6) |
| `beta1` | 0.9 | [0.80, 0.95] |
| `dropout` | 0.1 | [0.10, 0.40] |
| `degrees` / `translate` / `scale` | 15 / 0.1 / 0.1 | [0, 45] / [0, 0.2] / [0, 0.3] |
| `fliplr` / `flipud` | 0.5 / 0.5 | [0, 0.5] each |

HPO: Optuna TPE, 30 trials (10 start-up), a fresh sampler seeded per proposal (`seed`, number of completed trials),
so proposals are reproducible and re-verified on resume.

### 5.3 SAM 3 (`sam3_seg/common.py`)

| Decision | Value | Reason |
|---|---|---|
| Model | official SAM 3 image model, **all 840.5 M parameters trainable** (text encoder not frozen) | full fine-tuning, as Meta's recipe |
| Prompt | `"skin lesion"` (single category) | clinically neutral (most ISIC lesions are benign) |
| Input | 1008 × 1008 | fixed by the architecture |
| Precision / batch | FP32, batch 2, official activation checkpointing | memory probe: without checkpointing OOM at 30.9 GiB; with it 12.8 / 16.5 GiB (allocated / device), 5.0 s per step, ≈ 106 min per epoch on a V100S |
| Budget | **30 epochs**, patience 30 (no early stopping) | one FP32 epoch ≈ 1.8 h; 120 epochs would exceed 8 days per run |
| HPO | **10 trials × 10 epochs**, Optuna TPE (5 start-up trials), seeded per proposal | compute; a foundation model starts from strong weights |
| Search space | `lr_scale` [0.01, 0.2] log (default 0.1; scales LR 8e-4 transformer / 2.5e-4 vision / 5e-5 language), `weight_decay` [0.01, 0.2] log (0.1), `lrd_vision_backbone` [0.6, 1.0] (0.9), `scheduler_warmup` [1, 1000] log (2), `hflip_p` [0, 0.5] (0.5), `resize_min_size` [320, 1008] step 16 (480) | learning dynamics and the recipe's own augmentations only |
| Selection | `best.pt` = best validation per-image JSI, with the shared metric code | as the U-Net |

Worst-case compute (V100S, FP32): Phase 1 ≈ 55 h, Phase 2 ≈ 5 × 47 h, Phase 3 ≈ 10 × 18 h, Phase 4 ≈ 55 h —
about three weeks on one GPU (Phases 2 and 3 can run in parallel on two GPUs).

---

## 6. Model-specific quirks that must be reported

### 6.1 YOLO26-seg

* **Requested vs. actual batch size.** The protocol micro-batch is 16. In FP32 at 640 px on a 32 GB V100S the training
  peak is ≈ 5.4 / 10.7 / 21.6 / 24.7 GB for n / s / m / l, and **xlarge does not fit (~40 GB)**: Ultralytics catches
  the out-of-memory error in the first epoch and **silently halves the batch to 8**. Every run now records the batch
  it really used (`batch_effective`, read from `best.pt`); a mismatch is a warning in `summary/phase*_val.json` and in
  `final_results.json` (and is printed by the root notebook). HPO trials use the same per-model micro-batch
  (`MICRO_BATCH`: 16, xlarge 8). Because the nominal batch `nbs = 64` is held constant, Ultralytics accumulates
  gradients (`accumulate = round(nbs / batch)`: 4 at batch 16, 8 at batch 8), so **the effective optimisation batch
  is 64 for every size and phase**; only BatchNorm batch statistics differ for xlarge.
  *Suggested wording:* "The nominal batch size (nbs = 64) is held constant across all phases and model sizes, so the
  effective optimisation batch size is identical (64) throughout the study; the micro-batch is 16, except for the
  largest variant (8), which does not fit at 16 in FP32 on a 32 GB GPU."
* **Selection criterion differs from the other arms:** `best.pt` and the HPO fitness use Ultralytics' fitness (box +
  mask mAP50-95 on the validation split), whereas the U-Net and SAM 3 select on validation per-image JSI. This is the
  native criterion of each framework; the test evaluation is identical.
* **Two confidence thresholds:** instance metrics (mAP, P, R, F1) use the standard `conf = 0.001` mAP protocol; the
  pixel masks (DSC, JSI, boundary metrics) use the union of instances with `conf ≥ 0.25` (Ultralytics' predict
  default). Only YOLO26 reports instance metrics; they are `NaN` for the other arms (SAM 3's validation/CV tables carry
  its official COCO mAP50-95 instead).
* **Resume is not bit-identical** (the dataloader RNG state is not checkpointed); resumed runs are logged and flagged.
* **Checkpoint on disk is FP16** (Ultralytics `strip_optimizer`), so `size_mb_disk` ≈ the FP16 size.
* **GFLOPs counter:** thop (Ultralytics' `get_flops`). YOLO26 contains PSA attention blocks whose batched matmuls thop
  cannot see; measured against `FlopCounterMode` the under-count is small — **+1.33 % (n), +0.71 % (s), +0.20 % (m),
  +0.26 % (l), +0.18 % (x)** at 640 px (measured on the architecture definitions, 2026-10).

### 6.2 U-Net

* **Faithful port of the original Keras model** (`unet/model.py`): Keras BatchNorm momentum/ε, `he_normal` /
  `glorot_uniform` initialisation, Keras "same" alignment of the transposed convolutions. With weights copied from the
  Keras model the PyTorch output is identical (max abs. difference 0.0) and the parameter count matches (2,161,649).
* **Deterministic, bit-exact behaviour.** `last.pt` stores the model, optimiser, epoch and *all* RNG states; data order
  and augmentation are pure functions of (seed, epoch, sample). A run killed at any point and resumed is
  **bit-identical** to an uninterrupted one (verified on CPU and GPU, also with 0 vs. 8 dataloader workers);
  `results.csv` is rebuilt from the checkpoint. This is the strongest reproducibility guarantee of the three arms.
* **Augmentation fix vs. Keras:** the Keras `ImageDataGenerator` interpolated masks bilinearly into soft labels; the
  port warps masks with nearest-neighbour interpolation, so they stay binary.
* **Resolution ceiling:** the U-Net predicts at 256 × 256. An oracle (perfect 256 × 256 prediction pushed through the
  same upsampling) scored DSC 0.9967 (min 0.965) on the 994 test images of the earlier export —
  **[RESULT: re-measure on the official 1,000-image test set]**. Part of any U-Net deficit at the dataset resolution
  is this ceiling; report it.
* **GFLOPs** at the native 256 px (6.40) and, for a resolution-matched comparison with YOLO26, at 640 px (40.0);
  thop and `FlopCounterMode` agree exactly for this pure CNN (6.401 / 40.003 GFLOPs, measured).
* **Constant learning rate** (no schedule) — the original recipe; HPO tunes its value.

### 6.3 SAM 3

* **Compute profile and the thop limitation.** One forward pass at 1008 px costs ≈ **6,057 GFLOPs**: image encoder
  5,614, detector 424, text encoder 20; by operation: linear layers 4,747, **attention products (QKᵀ, AV) 956 —
  15.8 % of the total**, convolutions 354. thop counts FLOPs through *module hooks* and cannot see
  `torch.nn.functional.scaled_dot_product_attention` or functional matmuls, so it would **miss at least these 956
  GFLOPs (≈ 16 %)**. SAM 3's GFLOPs are therefore measured natively with `torch.utils.flop_counter.FlopCounterMode`
  (same 2 × MAC convention as thop; element-wise ops, normalisation and interpolation uncounted by both). Technical
  notes: counting runs under `torch.no_grad` (under `inference_mode` the dispatcher bypasses Python dispatch modes and
  nothing is counted) and with `nn.MultiheadAttention`'s fused fast path disabled (no FLOP formula); the counted graph
  is otherwise the benchmarked one. For the CNNs the two counters agree (§ 6.1, § 6.2), so GFLOPs are comparable
  across arms. Report `attention_flop_share` and the per-stage split (notebook 02, Figure 8 of `sandbox_sam3`).
* **Parameters by component:** vision backbone 454.0 M, text encoder 353.7 M, transformer 21.0 M, geometry encoder
  8.2 M, segmentation head 2.3 M, scoring 1.2 M; without the text encoder 486.8 M. With a fixed prompt the text
  features can be cached: `forward_cached_text` measures that deployment form.
* **Reduced budget** (30 epochs; HPO 10 × 10) — disclosed in `final_results.json` → `protocol_notes`.
* **Determinism caveat:** seeds, cuDNN deterministic, a **deterministic `grid_sample`** replacement
  (`sam3_seg/determinism.py`) and warn-only deterministic algorithms; the ViT memory-efficient attention backward
  remains nondeterministic. PyTorch's strict mode was verified bit-exact but costs +33 % step time (6.67 vs. 5.01 s),
  so it is off; repeated or resumed runs differ by ≤ ~1e-4 in the weights (5th digit of val JSI).
* **FP16 = `torch.autocast(float16)` with FP32 weights** (SAM 3's official mixed-precision path), whereas YOLO26 and
  the U-Net convert the weights (`.half()`). Consequently SAM 3's `vram_weights_mb` is the FP32 footprint in both rows.
* **Prediction rule:** union of the instances with score ≥ 0.5 (score = sigmoid(logit) × presence), produced by each
  run's own validation pipeline rebuilt from its `config.yaml` (verified to reproduce the trainer's validation JSI to
  the last digit).
* **Storage:** a finished run deletes its 9.4 GB resume checkpoint; HPO trials keep metrics only.

---

## 7. Evaluation metrics and statistics (Phase 5a)

### 7.1 Per-image metrics (`segmentation_metrics.py`, identical in the three repositories)

From the confusion counts TP / FP / FN / TN at dataset resolution:

| Metric | Definition | Notes |
|---|---|---|
| DSC | 2TP / (2TP + FP + FN) | Dice / pixel F1 |
| JSI | TP / (TP + FP + FN) | Jaccard / IoU |
| ISIC JSI$_{0.65}$ | JSI if JSI ≥ 0.65 else 0 | official ISIC 2018 Task 1 score |
| Sensitivity, specificity, accuracy | TP/(TP+FN), TN/(TN+FP), (TP+TN)/N | |
| **Boundary IoU** | IoU of the boundary bands (pixels within *d* of each mask's own contour), *d* = 2 % of the image diagonal (≈ 26 px at 1024 × 768) | Cheng et al., CVPR 2021 |
| **NSD** (surface Dice) | fraction of both contours within τ of the other contour, τ = 1 % of the diagonal (≈ 13 px) | Nikolov et al., 2021; recommended by Metrics Reloaded |
| **HD95** | max of the two directed 95th percentiles of contour-to-contour distances, **pixels** at dataset resolution (lower is better) | MONAI / DeepMind `surface-distance` convention; added in this audit |

Empty masks: both empty → overlap and boundary scores 1, HD95 0; exactly one empty → 0, and HD95 = the image diagonal
(Metrics Reloaded: penalise, never drop, missing predictions). Contours are 1-pixel inner boundaries; the image border
counts as background. HD95 was verified against a brute-force pairwise-distance reference (max error 4 × 10⁻⁶ px), and
adding it left every previous metric bit-identical.

**Why boundary metrics** (committee / Metrics Reloaded): overlap scores are dominated by the lesion interior and are
insensitive to boundary errors on large lesions; contour adherence matters clinically (excision margins, border
irregularity is a melanoma criterion). Boundary IoU and NSD measure the *fraction* of boundary within a tolerance;
HD95 measures the *size* of the boundary errors.

### 7.2 Aggregation and confidence intervals

* **Primary figure:** per-image (macro) mean over the 1,000 test images; also SD (ddof = 1), median, IQR, and the
  pooled (micro) DSC/JSI.
* **95 % CI:** seeded percentile bootstrap of the mean (2,000 resamples, seed 0) — in every `accuracy/*.json`,
  `summary/test_accuracy.csv` and Table 1.
* **CV:** mean ± sample SD over the 5 folds.

### 7.3 Paired statistical comparisons

Every model is scored on the same images, and per-image scores are stored (`phase5_test/per_image/*.csv`, keyed by
ISIC ID), so all comparisons are **paired**:

* **Within an arm (HPO effect, `summary/hpo_gain.csv`, Table 3):** Optimized − Baseline per image, mean difference
  with paired bootstrap 95 % CI, two-sided Wilcoxon signed-rank p, number of images improved / worsened (direction-aware:
  for HD95 lower is better) — for DSC, JSI, Boundary IoU, NSD and HD95.
* **Across architectures (root notebook, Table 7, Figure 4):** every pair of systems and every metric — mean paired
  difference with seeded bootstrap 95 % CI, two-sided Wilcoxon signed-rank test, matched-pairs rank-biserial
  correlation (effect size), wins/losses, and **Holm-adjusted** p-values within each metric (family = all pairs).
  Omnibus: **Friedman test** across all systems per metric with **Kendall's W** and mean ranks.
* **Reporting rule:** always give the CI of the difference and the effect size, not only p — with n = 1,000,
  practically negligible differences can be statistically significant.

---

## 8. Real-world efficiency measurement (Phase 5b)

The committee's point — *"GFLOPs and parameters are proxies and do not prove real-time or on-device capability"* — is
answered by measured latency, throughput and memory of the deployed models.

| Aspect | How it is measured (identical design in the three repositories) |
|---|---|
| Batch | **1** image (the clinical / on-device use case), single GPU, `cudnn.benchmark = True` (fixed input shape, as in deployment) |
| Isolation | every (variant, size, precision) runs in a **fresh process**: no allocator, cache or autotuning leaks between models |
| `forward` | network only, pre-processed input at the native size; `torch.cuda.Event` pairs + synchronise per iteration; **50 warm-up + 500 timed** |
| `end_to_end` | the deployed pipeline from a **decoded image in host memory to the binary mask at dataset resolution in host memory** (pre-processing, inference, post-processing, upsampling, threshold, device→host copy) — YOLO26 `predict(conf=0.25, retina_masks=True)` + union mask; U-Net resize → forward → bilinear upsampling → 0.5; SAM 3 Meta's `Sam3Processor`; `perf_counter` around synchronised calls; **20 warm-up + 200 timed** on one test image |
| `end_to_end_dataset` | the same pipeline once on each of the **first 100 test images sorted by ISIC ID** (the same images for every model), after one untimed pass (cuDNN autotuning of every input shape) — input-dependent spread (image size, number of instances) |
| SAM 3 only | `forward_cached_text`: forward with the prompt's text features precomputed (fixed-prompt deployment) |
| Statistics | mean, SD, **median**, P90, **P95**, P99, min, max; **FPS** = 1000 / mean (also 1000 / median) |
| **Peak VRAM** | (a) **allocator peak** — `torch.cuda.max_memory_allocated` during the timed loops, counters reset *after* warm-up (cuDNN autotuning workspaces excluded and reported separately) = the model's own footprint; (b) **process peak** — device memory held by the benchmark process as reported by the driver (`nvidia-smi` delta: CUDA context, kernels and allocator cache included) = what a deployment GPU must provide; plus the CUDA-context size and the VRAM of the weights alone |
| Host RAM | RSS after loading/benchmarking and peak RSS |
| Contention | GPU utilisation sampled before/after each run; a busy GPU marks the result `contended` (re-run before quoting) |
| Model size | parameters (fused and unfused), GFLOPs at the native input (+ at 640 px for the U-Net), checkpoint size, theoretical FP32/FP16 weight size |

**Real-time criterion used in the article:** a model is real-time at *F* FPS if its **P95 end-to-end latency over the
100 distinct test images** is ≤ 1000 / *F* ms (default *F* = 30 → 33.3 ms), at batch 1 on the reported GPU. Using
P95 rather than the mean guarantees the frame budget for 95 % of frames.

**Report with every latency:** the GPU model, driver, CUDA/cuDNN and PyTorch versions (stored in each efficiency
JSON) and the precision. Never quote the `infer_ms` column of the Phase 5a per-image CSVs: it is a by-product of the
accuracy loop with arm-specific scopes, not a benchmark.

---

## 9. Cross-architecture aggregation (root notebook)

`Article_Figures_and_Tables.ipynb` (driver) and `article_aggregator.py` (logic, also runnable headless) read only the
Phase 5 summaries and per-image files of the three pipelines and write `article_outputs/`:

* **Tables (LaTeX `booktabs` + CSV):** 1 accuracy with 95 % CI (best per column in bold); 2 efficiency with the
  real-time criterion; 3 HPO effect; 4 FP16 vs. FP32; 5 training cost per phase; 6 CV vs. test; 7 pairwise paired
  tests on DSC (other metrics in `data/paired_tests_all_metrics.csv`).
* **Figures (PDF + PNG):** 1 accuracy (DSC, JSI) vs. latency / FPS / parameters with the Pareto front; 2 training cost
  and inference latency (median → P95, FP32/FP16) vs. the real-time budget; 3 per-image distributions; 4 pairwise ΔDSC
  matrix with Holm-adjusted significance (+ JSI, Boundary IoU, HD95 versions); 5 boundary metrics with CI; 6 peak VRAM
  (allocator vs. process).
* **Key numbers** for the Results section (best accuracy, fastest model, which models meet the real-time criterion,
  SAM 3 vs. the best lightweight model with CI/p/effect size and cost ratios, Friedman test).

Colour encodes the architecture consistently in every figure (validated colour-vision-deficiency-safe palette);
YOLO26 sizes are distinguished by direct labels (n, s, m, l, x).

---

## 10. Reproducibility, determinism and software environment

* **Pinned environment (Docker):** `nvidia/cuda:12.1.0-devel-ubuntu22.04`, `torch==2.5.1`, `torchvision==0.20.1`,
  `torchaudio==2.5.1` (cu121) in all three images; YOLO26 `ultralytics==8.4.21`, `pandas==3.0.1`; U-Net
  `optuna==5.0.0`, `ultralytics-thop`, pandas, scipy (exact pins); SAM 3 the vendored official code (`pip install -e
  ".[train, notebooks]"`), `optuna==5.0.0`, `pandas==3.0.6`. The HPO state refuses to resume under another library
  version.
* **Seeds:** `torch`, NumPy, `random`, `PYTHONHASHSEED`; deterministic cuDNN/cuBLAS.
* **Determinism by arm:** U-Net bit-exact (including resume); YOLO26 deterministic up to `deterministic=True`
  (warn-only) and DDP reduction order — HPO proposals are bit-reproducible given identical fitness values, resumed
  runs are not bit-identical; SAM 3 near-deterministic (≤ ~1e-4, § 6.3).
* **Provenance:** SHA-256 of datasets, splits, weights and settings in every output; Phase 5 results recomputed exactly
  when weights, test list or method version change.
* **Fault tolerance:** idempotent, resumable phases; locks against concurrent writers; `--force` never deletes (old
  outputs moved to `*.bak-<UTC>`).

---

## 11. Threats to validity and limitations

1. **Native input resolution differs** (256 / 640 / 1008 px). Evaluation is at the same dataset resolution against
   the same masks, but part of the accuracy *and* cost differences is resolution. Report the U-Net's 640-px GFLOPs and
   its resolution ceiling.
2. **Unequal training budgets for SAM 3** (30 epochs, HPO 10 × 10 vs. 120 epochs, HPO 30 × 30). Justified by compute
   (≈ 1.8 h per epoch); bias direction: SAM 3 is, if anything, under-tuned.
3. **Different selection criteria** (YOLO26: box + mask mAP50-95; U-Net and SAM 3: per-image JSI) — each framework's
   native criterion.
4. **Small validation split** (100 images) — model selection and HPO fitness remain noisy (§ 4); mitigated by CV and
   by the paired test-set HPO analysis.
5. **Different HPO algorithms and search spaces** across arms (genetic tuner vs. TPE) — each tunes its own recipe;
   the HPO effect is reported per arm rather than compared.
6. **Hardware dependence of latency** — absolute numbers are for the reported GPU (V100S) and software stack; ratios
   between models are more portable than absolute values. No CPU or embedded-GPU (e.g. Jetson) measurement is part of
   the protocol; claims about such devices must be framed as extrapolations from measured VRAM and latency.
7. **FP16 paths differ** (`.half()` for the CNNs, autocast for SAM 3).
8. **Single dataset** (ISIC 2018 Task 1, dermoscopy); generalisation to other datasets or modalities is not tested.
9. **Single training seed** per configuration; run-to-run variance is estimated through the 5-fold CV, not repeated seeds.

---

## 12. Ready-to-adapt wording for the Methods section

> **Data.** We used the official ISIC 2018 Task 1 release with its official partition (2,594 training, 100
> validation and 1,000 test images). Images were resized so that their longer side was at most 1,024 px (aspect ratio
> preserved, no upscaling), and the official masks at this resolution served as ground truth for all models. The test
> set was used only once, for the final evaluation.

> **Protocol.** All models followed the same five-phase protocol: (1) training with the default hyperparameters of
> each model's reference recipe; (2) 5-fold cross-validation of this configuration on the union of the training and
> validation sets (identical folds for all models); (3) seeded hyperparameter optimisation on the validation set;
> (4) retraining with the tuned hyperparameters; and (5) evaluation of both the default and the tuned model on the
> test set. Training was performed in FP32 without early stopping (justification: see the wording in § 4), and the
> checkpoint with the best validation score was retained.

> **Metrics.** Predictions were evaluated at the dataset resolution against the official masks with a single shared
> implementation of the Dice similarity coefficient (DSC), the Jaccard index (JSI), the ISIC thresholded JSI and,
> following the Metrics Reloaded recommendations, three boundary metrics: Boundary IoU (band of 2 % of the image
> diagonal), the normalised surface distance (tolerance of 1 % of the diagonal) and the 95th-percentile Hausdorff
> distance. Empty predictions were scored as failures and never excluded. We report per-image means with 95 %
> percentile-bootstrap confidence intervals (2,000 resamples). Models were compared pairwise on the same test images
> with the Wilcoxon signed-rank test (Holm correction for multiple comparisons) and paired bootstrap confidence
> intervals of the mean difference, after a Friedman omnibus test.

> **Efficiency.** Inference efficiency was measured on a single [GPU] at batch size 1, each model in a fresh process,
> after warm-up: network latency with CUDA events (500 iterations) and end-to-end latency — from a decoded image in
> host memory to the binary mask at the dataset resolution in host memory — on 100 distinct test images. We report the
> median and 95th-percentile latency, throughput (FPS) and peak GPU memory (both the PyTorch allocator peak and the
> process footprint including the CUDA context). A model was considered real-time if its 95th-percentile end-to-end
> latency did not exceed 33.3 ms (30 FPS). Parameters and FLOPs are reported for completeness (FLOPs counted with
> thop for the convolutional models and with PyTorch's `FlopCounterMode` for SAM 3, whose attention products thop
> cannot count).

---

## 13. Artefact map: which file feeds which table or figure

| Article element | Produced by | Underlying files |
|---|---|---|
| Table 1, Fig. 1, 3, 5 | root notebook | `<repo>/logs/<name>/summary/test_accuracy.csv`, `phase5_test/per_image/*.csv` |
| Table 2, Fig. 1, 2, 6 | root notebook | `summary/efficiency.csv`, `phase5_test/efficiency/*.json` |
| Table 3 | root notebook | `summary/hpo_gain.csv` |
| Table 4 | root notebook | `summary/test_accuracy.csv` + `efficiency.csv` (both precisions) |
| Table 5, Fig. 2a | root notebook | `summary/training_cost.csv` (from every `results.csv`) |
| Table 6 | root notebook | `summary/phase2_cv_pixel.csv` |
| Table 7, Fig. 4 | root notebook | per-image CSVs (paired by ISIC ID) |
| Qualitative figure | `notebooks/01_Segmentation_Visualizer.ipynb` (each repo) | `phase5_test/masks/<variant>_<model>/`, dataset images and official masks |
| Per-architecture figures | `notebooks/02_Metrics_and_Efficiency_Analysis.ipynb` (each repo) | `summary/*` |
| Disclosures | `summary/final_results.json` → `warnings`, `protocol_notes` | printed by the root notebook |

---

## 14. Committee-compliance checklist and audit changelog

| Committee demand | Where it is met |
|---|---|
| Measured real-time / on-device evidence, not only GFLOPs/params | Phase 5b: batch-1 median and P95 latency, FPS, allocator and process peak VRAM, end-to-end over 100 distinct images; real-time criterion on P95 (Table 2, Fig. 2, 6) |
| Metrics Reloaded: boundary metric | Boundary IoU, NSD, HD95 in every test/CV summary (Table 1, Fig. 5) |
| 95 % CIs | seeded bootstrap CI for every per-image metric; paired bootstrap CI for every difference |
| Paired statistical comparisons | Wilcoxon + bootstrap within arms (HPO) and across architectures (Holm, Friedman, effect sizes) |
| Standardised visual artefacts | notebooks 01/02 identical across repositories (shared cells byte-identical) |
| Unified article artefacts | root notebook + aggregator → LaTeX tables and figures |

**Changes made in the final audit (2026-10-01)** — cite these method versions if asked:

* `segmentation_metrics.py` (byte-identical in the three repositories): **HD95** added; previous metrics unchanged
  (verified bit-identical). Evaluation cache version `EVAL_VERSION = 3` in the three repositories.
* `benchmark_efficiency.py`: driver-level **process VRAM** (CUDA context included) and the **`end_to_end_dataset`**
  scope (100 distinct test images, same images in every repository) in all three; YOLO26's `end_to_end` now ends at
  the same artefact as the other arms (full-resolution union mask on the host, `retina_masks=True`, `conf=0.25`).
  `BENCHMARK_VERSION` 3 (YOLO26, U-Net) / 2 (SAM 3).
* `build_final_report.py`: HD95 in the accuracy and HPO-gain tables (direction-aware improvement counts); new
  efficiency columns (`e2e_dataset_*`, `vram_process_peak_mb`, `vram_cuda_context_mb`, `gflops_640`, `input_px`).
* Notebooks 01/02 standardised (ground-truth column, standard figures A–C with DSC/JSI/BIoU and HD95, real-time table,
  readable log axes, corrected stale notes); root notebook + aggregator added (`article/` in each repository).
