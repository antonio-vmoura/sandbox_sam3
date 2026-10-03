# Methodology notes — YOLO26-seg vs. U-Net vs. SAM 3 on ISIC 2018 Task 1

> **What these notes are.** Working lab notes describing how the three experimental arms
> (`sandbox_yolo26`, `sandbox_unet`, `sandbox_sam3`) were built, trained, evaluated and compared — the protocol
> reference from which the Methods, Results and Limitations sections of the thesis are drafted.
>
> **Sources.** (1) The code — the authoritative description of the final protocol (`common.py`, the phase scripts
> and `segmentation_metrics.py` of each repository); (2) the three READMEs and the commit history, which record the
> *why* of each protocol decision with the numbers observed in the first full runs; (3) the authors' three per-arm
> lab notes of YOLO26, U-Net and SAM 3 (kept locally, not versioned), cross-checked line by line against the code.
> Where a note describes an earlier state of the protocol (e.g. the Roboflow export, early stopping with patience 25,
> HPO micro-batch 32), this document follows the **current code** and lists every superseded statement in
> [Appendix A](#appendix-a--statements-in-the-per-arm-notes-that-the-final-protocol-supersedes), so the notes and
> this document never silently disagree.
>
> **Placeholders.** Everything written as **[RESULT]** or **[…]** must be filled in after the final run (most values
> come from `analysis_outputs/`, produced by `04_cross_architecture_results.ipynb`). Numbers quoted from earlier or
> preliminary runs are labelled as such.

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
14. [Committee-compliance checklist, disclosure checklist and audit changelog](#14-committee-compliance-checklist-disclosure-checklist-and-audit-changelog)
- [Appendix A — Statements in the per-arm notes that the final protocol supersedes](#appendix-a--statements-in-the-per-arm-notes-that-the-final-protocol-supersedes)
- [Appendix B — Suggested references](#appendix-b--suggested-references)

---

## 1. Research question and design at a glance

**Question.** For binary skin-lesion segmentation (ISIC 2018 Task 1), how does a fine-tuned **foundation model**
(SAM 3, 840.5 M parameters) compare with **lightweight specialised models** (YOLO26-seg n/s/m/l/x, ≈ 2.7–63 M
parameters; a classic U-Net, 2.16 M parameters) in **accuracy** *and* in **measured deployment cost** (latency,
throughput, memory) — not only in proxies such as parameters and GFLOPs?

The YOLO26 arm answers two questions in sequence, and the other two arms follow the same logic:
(1) *how well does the architecture perform "out of the box"* — default hyperparameters (Baseline) and their
cross-validated generalisation; (2) *what does hyperparameter optimisation (HPO) add, and at what computational
cost* — the Optimized model against the Baseline on a test set used only in the final phase, together with a
hardware-efficiency profile in FP32 and FP16.

**Design.** Three independent repositories implement the *same* five-phase protocol with the *same* data, splits,
cross-validation folds, ground truth, metric code, evaluation resolution, test isolation, efficiency profiler and
output schema. Only the model, its native input size, its default recipe and its tuning space differ.

| | YOLO26-seg (n, s, m, l, x) | U-Net | SAM 3 |
|---|---|---|---|
| Family | one-stage instance segmentation (CNN + PSA attention blocks) | encoder–decoder CNN (Keras baseline, ported to PyTorch) | promptable vision–language ("concept") foundation model (ViT) |
| Parameters (fused) | ≈ 2.7 / 10.4 / 23.6 / 28.0 / 62.8 M (architecture files with the 80-class COCO head; the trained single-class models are slightly smaller, e.g. YOLO26n-seg 2.69 M) — final values in Table 2 | 2,158,705 (2,161,649 unfused) | 840,509,750 (all trainable) |
| Initialisation | COCO-pretrained Ultralytics checkpoints `yolo26{n,s,m,l,x}-seg.pt`; all layers trainable (transfer learning) | random (Keras initialisers), no pretraining | official SAM 3 checkpoint (`facebook/sam3`); full fine-tuning, text encoder included |
| Native input | 640 px (letterbox), single class (`nc = 1`) | 256 × 256 | 1008 × 1008, text prompt "skin lesion" |
| Training code | Ultralytics 8.4.21 | own (bit-exact resumable) | Meta's official SAM 3 trainer, wrapped (`ProtocolTrainer`), vendored code unmodified |
| Default recipe (Baseline) | Ultralytics defaults | original Keras baseline | Meta's official fine-tuning recipe |
| HPO | Ultralytics genetic algorithm (seeded `SeededTuner`), 30 trials × 30 epochs | Optuna TPE (seeded per proposal), 30 × 30 | Optuna TPE (seeded per proposal), 10 × 10 |
| Training budget (Ph 1, 2, 4) | 120 epochs, no early stopping | 120 epochs, no early stopping | 30 epochs, no early stopping |
| Model selection (`best.pt`) | Ultralytics fitness (box + mask mAP50-95) on val; ties → latest epoch | per-image mean JSI on val (at 256 × 256); strict improvement, ties → earliest epoch | per-image mean JSI on val (dataset resolution); strict improvement, ties → earliest epoch |
| Determinism | deterministic algorithms requested (warn-only); resume not bit-identical | **bit-exact** (incl. resume, any worker count) | bit-exact only in strict mode (verified); study uses warn-only → runs differ by ≤ 6.8 × 10⁻⁵ in the weights |

**Computing environment.** Docker image `nvidia/cuda:12.1.0-devel-ubuntu22.04`, Python 3.11, PyTorch 2.5.1 /
torchvision 0.20.1 / torchaudio 2.5.1 (CUDA 12.1 builds) in all three arms (§ 10). Training: [YOLO26: one or two
NVIDIA Tesla V100S 32 GB with DDP — state what the final run used]; U-Net and SAM 3: one V100S. Phase 5 inference
and profiling: a single GPU [Tesla V100S-PCIE-32GB, driver …]; host CPU [Intel Xeon Gold 5220R]. Library versions
and the GPU model are recorded in every output file.

---

## 2. Dataset handling (Phase 0)

### 2.1 Source and split

* **Source:** the **raw official ISIC 2018 Task 1 release** (JPEG dermoscopy images + PNG binary lesion masks),
  mounted read-only (`../datasets/ISIC2018_Raw`).
* **Split:** the **official** partition, asserted on input *and* output: **2,594 training / 100 validation /
  1,000 test** images (3,694 in total); every image paired with its mask; no ISIC ID in two splits.
* The task is single-class lesion segmentation (one lesion region per image; YOLO `nc = 1`, one COCO category).

### 2.2 One shared dataset, three model-specific views

YOLO26's Phase 0 (`yolo26_seg/prepare_dataset.py`) is the **single source of truth**; the U-Net and SAM 3 Phase 0
scripts read *its* output (mounted read-only), so all arms see byte-identical images, masks and IDs, and every
sample is addressed by its ISIC identifier.

| Step | What is done | Why |
|---|---|---|
| Working resolution | longer side ≤ **1,024 px**, aspect ratio preserved, no upscaling; images decoded **without EXIF re-orientation**, stored as PNG (lossless) | official images reach several thousand px; 1,024 bounds compute while keeping lesion borders; no distortion |
| Ground truth | the **official mask** resized to the working resolution, stored as `masks/<ISIC_ID>.png` | the evaluation ground truth of **all three arms** (`segmentation_metrics.ground_truth_mask`) |
| Resampling check | Dice between the resized mask and the full-resolution official mask recorded per image (mean 0.998–0.999) | quantifies the (negligible) error introduced by the working resolution |
| YOLO labels (training only) | polygons from `cv2.findContours` (RETR_CCOMP); holes bridged into the outer contour; fragments/holes < 0.1 % of the lesion area left out of the *training label only*; every label rasterised back and checked against the mask (**Dice ≥ 0.98 asserted**; achieved ≥ 0.989, mean 0.9999) | YOLO needs polygons; the fidelity check bounds label error; **evaluation never uses the polygons** |
| U-Net view | image resized to 256 × 256 with area interpolation (the input size of the original baseline); mask area-resized (each output pixel = fraction of lesion pixels it covers) and **thresholded at 0.5** → strictly binary, boundary at the sub-pixel majority; Phase 0 verifies that masks take only {0, 1} and that no ID occurs in two splits; SHA-256 of every source file and output array; cache rebuilt only when the source or the preprocessing parameters change | fixes the soft (non-binary) masks of the legacy Keras arrays (§ 2.5) |
| SAM 3 view | COCO dataset: images hard-linked (byte-identical; copied across container mounts); one instance per image whose **compressed-RLE mask is the official mask** (lossless; holes and fragments exact); single category named **"skin lesion"** (= the text prompt); per-split and per-fold annotation files; dataset fingerprinted in `meta.json` (SHA-256 of every annotation file) — a run refuses to start on a modified or half-written dataset | the official SAM 3 loaders consume COCO RLE |

### 2.3 Cross-validation folds

* **Pool:** training ∪ validation = **2,694** images, training IDs followed by validation IDs in YOLO26's order;
  images without annotations would be kept as background images. The test set is **excluded and verified** (by
  resolved path and by file name) before any fold is built; a test ID in the pool aborts Phase 2.
* **Algorithm:** indices shuffled with NumPy's legacy Mersenne-Twister (`numpy.random.RandomState(0).shuffle`) and
  split into K = 5 contiguous blocks, the first N mod K blocks receiving one extra image — this reproduces
  `sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=0)` without the scikit-learn dependency.
  With the official pool: **held-out folds of 539 (× 4) and 538 (× 1) images, training folds of 2,155 / 2,156**.
* **Identical folds in the three repositories:** each repository rebuilds the pool in YOLO26's order with the same
  algorithm; the SHA-256 of every fold's ID list is written to `splits_manifest.json` and was verified identical
  across the arms (so per-fold results are **paired across architectures**). Any later execution whose partition
  differs (changed data, K or seed) is refused, so folds from different partitions can never be combined.

### 2.4 Task 2 (lesion attributes)

Phase 0 of all three repositories can also build the multi-label ISIC 2018 Task 2 dataset (five attributes). It is
**not** used in Phases 1–5 (Task 1 only); mention it, if at all, as future work.

### 2.5 History of the data — why earlier results are not used

The final protocol is the third iteration of the data handling; earlier numbers must not be mixed with final ones.

1. **Legacy U-Net arrays (discarded).** The pre-computed NumPy arrays of the original Keras U-Net had three defects:
   (i) the ground-truth masks were **not binary** — all lesion pixels took fractional values (maximum ≈ 0.88,
   thousands of distinct values), so reported IoU/Dice compared thresholded predictions with soft targets and the BCE
   loss was trained on soft labels; (ii) the test-set arrays were missing; (iii) no image identifiers, so the splits
   could not be aligned with the other arms. **All earlier U-Net results were discarded.**
2. **Roboflow export (superseded).** The first aligned runs of all three arms used a YOLO export from Roboflow:
   **2,547 / 100 / 994** images (47 training and 6 test images silently dropped), every image **stretched to
   640 × 640** (aspect ratio lost), ground truth rasterised from the YOLO polygons. The first full runs whose
   early-stopping evidence is quoted in § 4 were made on this export (its CV folds held ≈ 530 images). It was
   abandoned in favour of the raw official release (§ 2.1).
3. **Earlier SAM 3 experiments (superseded).** Before the alignment, SAM 3 was trained on the official split at the
   original image resolutions with the prompt **"skin cancer"** — misleading, since most ISIC lesions are benign; the
   final protocol uses "skin lesion".

---

## 3. The unified 5-phase pipeline and why it is a fair comparison

### 3.1 The phases

| Phase | Purpose | Data touched | Output used in the thesis |
|---|---|---|---|
| **0 — Dataset** | build and verify the shared dataset (§ 2) | raw release | dataset description |
| **1 — Baseline** | train with the **base setup + default hyperparameters** | train (fit), val (checkpoint selection) | Baseline model (`best.pt`) |
| **2 — Baseline CV** | 5-fold CV of the exact Phase 1 configuration; the held-out fold is the per-epoch validation set; instance metrics (YOLO26) and pixel metrics of the fold's `best.pt` (FP32, dataset resolution) | train ∪ val pool (test excluded) | variance of the architecture, generalisation check (Table 6) |
| **3 — HPO** | seeded, fault-tolerant hyperparameter search | train (fit), val (fitness) | tuned hyperparameters |
| **4 — Optimized** | retrain from the pretrained/initial weights with the **base setup + tuned hyperparameters** | train (fit), val (checkpoint selection) | Optimized model (`best.pt`) |
| **5 — Test** | 5a accuracy, 5b efficiency, 5c consolidated report — Baseline **and** Optimized, FP32 **and** FP16 | **test (only here)** | Tables 1–4 and 7, Figures 1–6 |

Every step is orchestrated by one script per repository (`run_pipeline.sh`, `run_pipeline_unet.sh`,
`run_pipeline_sam3.sh`), is idempotent and resumable, and writes into an isolated folder `logs/<pipeline-name>/`.

### 3.2 The fairness controls (what is identical across the three arms)

1. **Same images, split, folds and ground truth** — one Phase 0 output, read-only, addressed by ISIC ID (§ 2).
2. **Same metric code** — `segmentation_metrics.py` is a **byte-identical copy** in the three repositories
   (verified by checksum / `cmp`); every metric is computed by the same function for every model.
3. **Same evaluation resolution** — every prediction is brought to the image's dataset resolution and scored against
   the same official mask (U-Net: bilinear upsampling of the 256 × 256 probability map; YOLO26: `retina_masks=True`;
   SAM 3: the official post-processor's upsampling). Native input sizes differ by design (§ 11).
4. **Same handling of failures** — an empty prediction on a lesion image scores 0 (HD95: image diagonal); nothing is
   skipped, so a model cannot improve its mean by failing silently.
5. **Test isolation** — the test split is read only in Phase 5; the CV pool is verified test-free (by path and file
   name); HPO and checkpoint selection see only train/val.
6. **Same protocol shape** — **Baseline = base setup + default hyperparameters; Optimized = base setup + tuned
   hyperparameters**. The base setup (architecture, optimiser type, loss, input, batch, budget, precision, seed, …) is
   defined once per repository and **protected**: a tuned-hyperparameter file or a search space that tries to change
   a base-setup key is rejected. The HPO effect is therefore isolated in every arm (Table 3), and Baseline and
   Optimized share the architecture and hence the computational cost.
7. **Same training policy** — FP32 training, fixed seeds (0), **no early stopping** in any phase (§ 4).
8. **Same variant compared** — the cross-architecture comparison uses the variant fixed *a priori* (Optimized,
   Phase 4) for every model, not the better of the two after looking at the test set.
9. **Same efficiency profiler** — one benchmark design (batch 1, single GPU, fresh process per configuration, same
   timers, warm-up, statistics, memory accounting, contention check) derived from one script (§ 8), and the same
   deployment transformation (Conv–BN fusion for the CNNs).
10. **Same software stack** — PyTorch 2.5.1 / CUDA 12.1 in every Docker image; the U-Net was ported from
    TensorFlow/Keras to PyTorch precisely so that framework differences (runtime, kernel selection, allocator, FLOP
    counting) cannot be confounded with architectural ones (§ 6.2).
11. **Same statistics and output schema** — per-image means with seeded bootstrap 95 % CI; paired tests on the same
    1,000 images (§ 7); identical summary tables (columns verified), consumed by the same notebooks.

### 3.3 Phase details that matter for the text

* **Best epoch.** Every validation or CV number reported from a training log is the one of the epoch that produced
  `best.pt` (§ 6.1 for how YOLO26 aligns this exactly), so reported validation metrics describe exactly the checkpoint
  evaluated in Phase 5; in Phase 2 both families of fold metrics (instance and pixel) describe the same model.
* **CV aggregation.** Fold-wise values are summarised as $\bar{x} \pm s$ with the **sample** standard deviation
  $s = \sqrt{\tfrac{1}{K-1}\sum_k (x_k - \bar{x})^2}$ (ddof = 1), the appropriate estimator for a small number of folds
  (the population form underestimates it by $\sqrt{(K-1)/K} \approx 0.89$ for K = 5). Because folds share training
  data, fold-wise values are not independent: *s* is a descriptive measure of variability, not the basis of a test.
* **Phase 3** checkpoints its state atomically before every trial; failed trials are retried with identical
  hyperparameters; GPU/driver failures exit with code 75 and are retried by the orchestrator
  (`HPO_MAX_RETRIES` = 5 waits of `HPO_RETRY_WAIT` = 600 s); a change of search space, protocol, seed, data, pretrained
  weights or library version refuses to resume (§ 5).
* **Phase 5 caching** — results are keyed by the SHA-256 of the weights, the test list and the measurement settings
  (with a method version), so they are recomputed exactly when something relevant changes.

---

## 4. Why early stopping is disabled in every model

**Decision.** `patience = epochs` in every training phase of every arm: 120 / 120 for YOLO26 and U-Net (Phases 1, 2
and 4) and 30 / 30 per HPO trial; 30 / 30 for SAM 3 (10 / 10 per HPO trial). Every run therefore completes its full
budget and learning-rate schedule. `best.pt` is still the epoch with the best validation score (§ 3.3).
*Before this change* YOLO26 and the U-Net used patience **25** for training runs and **10** for HPO trials (the values
in the original per-arm notes).

**The problem: the official validation split has only 100 images.** Early stopping and checkpoint selection both
compare validation scores between epochs. With 100 images the epoch-to-epoch fluctuation of the validation score is of
the same size as — or larger than — the real improvements late in training, so "no improvement for *p* epochs" happens
by chance while the model is still improving. *(Illustration only: if per-image JSI has a standard deviation σ, the
standard error of a 100-image mean is σ/10; with σ = 0.15 that is 0.015, against ≈ 0.0065 on a 539-image CV fold.)*

**Evidence from the first full runs (on the Roboflow export of § 2.5, before the change; their outputs were moved
aside by `--force` and are not part of the final `logs/pipeline_final_v1` results):**

| Arm | Observation with early stopping (patience 25; HPO 10) | Consequence |
|---|---|---|
| YOLO26 | Median epoch-to-epoch change of the validation fitness **0.04–0.06** on the 100-image split vs. **0.014** on a 530-image CV fold. **Every** Phase 1 run stopped between epochs **50 and 82** of 120. | Training stopped at a still-high learning rate, *before* the cosine decay and the final `close_mosaic` epochs (mosaic off in the last 10), and `best.pt` sat on a noise spike. |
| U-Net | Baseline **and** Optimized stopped at epoch **81** of 120 (best epoch 56); **10 of 30** HPO trials stopped before their 30 epochs; meanwhile **4 of 5** CV folds (≈ 530 validation images) ran the full budget and kept improving until epochs **96–118**. | The HPO winner (val JSI **0.811** vs. **0.782** for the defaults) **did not transfer**: the Optimized U-Net was *worse* on the test set (ΔDSC = **−0.0052**, Wilcoxon p = 7.5 × 10⁻⁹) — the signature of selecting on validation noise. |
| SAM 3 | Not run with early stopping; disabled pre-emptively for the same reason (same 100-image split). Note that Meta's official recipe validates only every second epoch, skips the first validation and **selects nothing**; the study's wrapper validates every epoch, which per-epoch selection requires (§ 6.3). | — |

**Why disabling it makes the comparison fairer.** (i) Every model of an arm receives the same, complete compute
budget, so differences are not artefacts of *when* noise stopped a run; (ii) schedules that depend on the epoch count
(cosine LR, warm-up, `close_mosaic`) run as designed; (iii) Baseline and Optimized differ only by their
hyperparameters, not by their stopping epoch; (iv) HPO trials are compared at equal budget.

**What it does not remove (state it as a limitation).** `best.pt` and the HPO fitness are still chosen on the
100-image validation split, so a milder "winner's curse" remains, and validation-based figures (Phases 1, 2, 4) are
**optimistically biased**: the same data select the checkpoint and report the metric. Only the Phase 5 test-set
figures are free of this selection bias and should carry the main claims. The bias is further controlled by
reporting (a) the 5-fold CV of the Baseline protocol (Table 6) and (b) the *test-set* paired Optimized − Baseline
difference (Table 3), which shows whether tuning transferred. Tie-breaking differs by framework: Ultralytics
overwrites `best.pt` on an equal fitness (ties → latest epoch), the U-Net and SAM 3 require a strict improvement
(ties → earliest epoch).

**Suggested wording:** *"Early stopping was disabled (patience equal to the epoch budget) in all phases and for all
models. In preliminary runs, early stopping on the 100-image official validation split was dominated by epoch-to-epoch
noise — all YOLO26 baseline runs stopped between epochs 50 and 82 of 120, before the end of the learning-rate
schedule, and the U-Net stopped at epoch 81 while cross-validation folds with ≈ 530 validation images kept improving
until epochs 96–118 — and the hyperparameters selected under early stopping did not transfer to the test set. All
models therefore train for their full budget; the reported checkpoint is the epoch with the best validation score."*

---

## 5. Training configuration of each arm

### 5.1 YOLO26-seg (`yolo26_seg/common.py`, `tune_all_models_v2.py`)

| Setting | Value | Ultralytics 8.4.21 default | Reason |
|---|---|---|---|
| `epochs` / `patience` | 120 / 120 (HPO trials 30 / 30) | 100 / 100 | § 4 |
| `amp` | **False** (FP32) | True | YOLO26x-seg diverged under FP16 training (NaN in the classification loss at epoch 42 of Phase 1), consistent with an overflow the gradient scaler did not prevent. Disabling AMP only for that size would make the comparison asymmetric, so all sizes and phases train in FP32 (also removes a hardware-dependent source of numerical variation). Cost: ≈ 30–40 % more GPU time, ≈ 2× activation memory. |
| `optimizer` | **MuSGD** | `auto` | see below |
| `cos_lr` | True | False (linear) | one LR schedule in every phase |
| `close_mosaic` | 10 | 10 | pinned explicitly — an earlier (legacy) HPO used 15 |
| `workers` | 8 per process | 8 | pinned: the worker count changes how the augmentation random streams are assigned to samples |
| `batch` / `nbs` | 16 / 64 (xlarge: 8 / 64, § 6.1) | 16 / 64 | gradient accumulation `round(nbs / batch)` = 4 (8 for xlarge) → effective optimisation batch 64 everywhere |
| `imgsz`, `seed`, `deterministic` | 640, 0, True | same | pinned so an upstream default change cannot alter the protocol |

**Explicit optimiser.** `optimizer = "auto"` does not designate a fixed optimiser: in Ultralytics 8.4.21 it selects
MuSGD (lr 0.01, momentum 0.9) when a run exceeds 10⁴ optimisation iterations and AdamW (lr = 0.002·5/(4 + nc) = 0.002
for nc = 1, momentum 0.9) otherwise, and in both cases **ignores** the configured `lr0` and `momentum`. Left in place it
would have trained the Baseline with a different optimiser from the Optimized models (which need an explicit optimiser
for the tuned values to take effect), confounding the HPO effect with the optimiser. With MuSGD fixed, the Baseline
trains with the framework defaults **lr0 = 0.01, lrf = 0.01, momentum = 0.937, weight decay = 5 × 10⁻⁴**, the Optimized
models with the HPO values; the training logs were checked to report `optimizer: MuSGD(lr=…, momentum=…)` with the
expected values.

**HPO algorithm (Ultralytics genetic algorithm, re-implemented with a seeded RNG).** The first trial evaluates the
default hyperparameters clipped to the search bounds (lr0 0.01 → 0.004 and weight decay 5 × 10⁻⁴ → 1 × 10⁻⁴, both
defaults lying outside the refined bounds). Each later trial *i*:

1. **Parent selection** — the (up to) nine trials with the highest fitness are kept (stable sort); nine parents are
   drawn from them with replacement, with probability proportional to the fitness shifted to be positive
   ($f_i - \min_j f_j + 10^{-6}$).
2. **Crossover (BLX-α, α = 0.2)** — for each gene, a value drawn uniformly from $[\ell - \alpha s,\, h + \alpha s]$,
   with $\ell$, $h$ the minimum and maximum of that gene among the parents and $s = h - \ell$ (a zero span is replaced
   by a random span in [0.01, 0.1]).
3. **Mutation** — each gene is mutated with probability 0.5 by a factor $\exp(\varepsilon)$,
   $\varepsilon \sim \mathcal{N}(0, (\sigma g_k)^2)$, clipped to [0.25, 4], with $g_k$ a per-gene gain (momentum 0.3,
   others 1) and σ decaying linearly from 0.2 to 0.1 over the first 300 trials (with 30 trials σ only goes from 0.200
   to ≈ 0.190); sampling repeats until at least one gene changes.
4. **Constraint** — each gene is clipped to its bounds and rounded to five decimals.

*Fitness* = Ultralytics segmentation fitness on the validation split,
$F = \mathrm{mAP}^{B}_{50\text{-}95} + \mathrm{mAP}^{M}_{50\text{-}95}$.

**Search space ("refined", 15 hyperparameters):**

| Group | Hyperparameter: range |
|---|---|
| Optimisation | lr0 [1 × 10⁻³, 4 × 10⁻³]; lrf [0.005, 0.05]; momentum [0.85, 0.95] (gain 0.3); weight_decay [1 × 10⁻⁶, 1 × 10⁻⁴]; warmup_epochs [1, 5] |
| Loss gains | cls [0.2, 1.5]; dfl [0.8, 1.5] |
| Colour augmentation | hsv_h [0.005, 0.025]; hsv_s [0.3, 0.9]; hsv_v [0.2, 0.7] |
| Geometric augmentation | translate [0.05, 0.20]; flipud [0.0, 0.10] |
| Mixing augmentation | mosaic [0.7, 1.0]; mixup [0.0, 0.05]; copy_paste [0.0, 0.05] |

The ranges were narrowed from a broader 22-parameter space after an exploratory search on YOLO26s-seg:
hyperparameters with negligible correlation with fitness (|r| < 0.05) were fixed at their defaults and high-signal
ranges were narrowed around the best region. **[State that this exploratory search used only the training/validation
splits — never the test split.]** Because lr0 and weight decay defaults lie outside the bounds, the **Baseline is not
itself a candidate** and the search is not guaranteed to return something at least as good as the Baseline; Phase 5
measures the effect of the selected configuration. *(Alternative before the final run: widen these two bounds to
include the defaults.)*

**Deterministic mutation (`SeededTuner`) — a deviation from the reference implementation.** Upstream, the RNG is
re-seeded with the wall-clock time before every mutation (`np.random.seed(int(time.time()))`) and parent selection
uses Python's global `random`, so two runs — or a resumed run — propose different hyperparameters. `SeededTuner`
re-implements the mutation with **the same operators and constants** but draws every random number from
`numpy.random.default_rng([seed, i])` (seed = 0, *i* = trial index), with a stable parent ranking and the same
fitness-proportional probabilities. The proposal for trial *i* is a deterministic function of (seed, *i*, fitness
history of trials 0 … *i* − 1); a trial re-run after an interruption receives exactly the same hyperparameters. Two
independent executions produced byte-identical trial histories when the fitness values were identical. *Scope:* the
search is reproducible **conditional on the fitness values**; fitness comes from GPU training with deterministic
algorithms requested (`torch.use_deterministic_algorithms(True, warn_only=True)`, deterministic cuDNN, fixed seeds),
and operations without a deterministic implementation and the DDP reduction order may still introduce small
run-to-run differences that propagate to later proposals.

**Fault tolerance of the search** (a full search takes several GPU-days):
* `tune_results.csv` and `hpo_state.json` are written after every trial; state writes are atomic (temporary file,
  `fsync`, atomic rename);
* completion is decided by the number of recorded trials, not by the presence of `best_hyperparameters.yaml` (which
  the reference implementation rewrites after every trial);
* on resumption, a torn history line and the interrupted trial's folder are discarded and the interrupted trial is
  re-run **from its first epoch** with the same hyperparameters (contrast: the U-Net and SAM 3 resume the interrupted
  trial's training from its checkpoint);
* a failed trial (fitness 0: OOM, divergence) is retried with identical hyperparameters up to **2** times
  (`--max-trial-retries`), then recorded as failed; failures that coincide with an unavailable GPU are not counted and
  the search resumes when the device is back;
* resumption is refused if the search space, fixed training configuration, seed, pretrained weights or Ultralytics
  version differ from the original run.

### 5.2 U-Net (`unet/common.py`, `unet/model.py`, `unet/training.py`, `unet/tune_unet.py`)

**Architecture (faithful port of the Keras baseline, § 6.2).** Four encoder levels of two 3 × 3 convolutions with
BatchNorm and ReLU (16, 32, 64, 128 filters), each followed by 2 × 2 max-pooling and dropout; a 256-filter bottleneck;
four decoder levels with a 3 × 3 transposed convolution (stride 2), concatenation with the skip connection, dropout and
a convolution block; a 1 × 1 convolution with sigmoid output. One network width only (16 base filters, the original
baseline) — no width scaling comparable to the YOLO26 sizes.

| Base setup (fixed) | Value |
|---|---|
| Loss | BCE (pixel mean) + (1 − soft Dice), Dice over the whole batch with smoothing 10⁻⁶ — the original Keras `bce_dice_loss`; BCE computed from logits (numerically stable; Keras additionally clips probabilities to [10⁻⁷, 1 − 10⁻⁷]) |
| Optimiser / schedule | AdamW (decoupled weight decay ≡ Keras `Adam(weight_decay=…)`), ε = 10⁻⁷ (Keras default), β₂ = 0.999, **constant learning rate** (as in the original baseline) |
| Input / batch / budget | 256 × 256, batch 16, 120 epochs, patience 120 (HPO 30 / 30) |
| Numerics / selection | FP32, seed 0, deterministic algorithms; after every epoch the per-image mean JSI on the validation set (at 256 × 256, same definitions as § 7) is computed; `best.pt` replaced only by a strictly better epoch (ties keep the earliest) |

**Augmentation (re-implementation of Keras' `ImageDataGenerator`, jointly on image and mask):** rotation ±15°,
independent horizontal/vertical shifts ±10 %, zoom ±10 %, horizontal and vertical flips, reflective borders; affine
transform about the image centre, parameters drawn uniformly with the Keras semantics. **One deliberate difference:**
masks are warped with **nearest-neighbour** interpolation and stay strictly binary (Keras interpolated both image and
mask bilinearly, turning mask borders into soft labels — compounding the soft-mask defect of § 2.5); images are
interpolated bilinearly. Visual inspection confirmed image–mask alignment.

| Tuned (Phase 3) | Default (Baseline = Keras) | Range |
|---|---|---|
| `lr0` | 1e-3 | [1e-4, 1e-2] log |
| `weight_decay` | 0 | [1e-6, 1e-3] log |
| `beta1` (AdamW) | 0.9 | [0.80, 0.95] |
| `dropout` | 0.1 | [0.10, 0.40] |
| `degrees` / `translate` / `scale` | 15 / 0.1 / 0.1 | [0, 45] / [0, 0.20] / [0, 0.30] |
| `fliplr` / `flipud` (probabilities) | 0.5 / 0.5 | [0, 0.50] each |

**Baseline outside the search bounds.** The default weight decay 0 cannot be represented on the logarithmic scale
(lower bound 10⁻⁶); the first trial evaluates the Baseline clipped to the bounds (weight decay 10⁻⁶ — negligible, but
formally a different configuration). As for YOLO26, the Baseline is therefore not itself a candidate. The log scale was
kept because plausible weight-decay values span several orders of magnitude.

**HPO algorithm and reproducibility.** The HPO framework of the original U-Net pipeline was retained: Optuna TPE with
SQLite storage, 30 trials of 30 epochs, the first **10** trials random (Optuna's default start-up), fitness = validation
per-image mean JSI of the trial's best epoch. Because the sampler's random state is not persisted, a search resumed
with a re-created `TPESampler(seed)` would propose different configurations; therefore a **fresh TPE sampler seeded
with (seed × 1,000,003 + i) mod 2³²** is installed before every proposal *i* (*i* = number of completed trials; the first
proposal is fixed to the clipped Baseline). TPE builds its model only from completed trials, so proposal *i* is a
deterministic function of (seed, *i*, completed history); proposals are recorded and a resumed or retried proposal is
verified to be identical (otherwise the search stops). Two independent executions produced identical histories.

**HPO crash recovery (`hpo_state.json`, same schema as YOLO26).** A trial left "running" by a crash is marked
interrupted (not a failure) and the same proposal is asked again — identical parameters and **the same trial folder,
so the interrupted training resumes bit-exactly from its checkpoint**; a trial that raises or returns a non-finite
fitness is retried from a clean folder up to 2 times, then recorded with fitness 0 (a legitimately low but finite
fitness is a valid result, not a failure); GPU-coincident failures are not counted (exit 75, orchestrator retries);
resumption is refused if the search space, base setup, trial budget, seed, data or Optuna/PyTorch versions changed.
Verified by fault injection (kill during a trial, transient and persistent trial failures, GPU failure, configuration
change, extension of the number of trials) and by killing a real search mid-trial: the completed search was identical
to an uninterrupted one.

### 5.3 SAM 3 (`sam3_seg/common.py`, `configs/sam3_base_recipe.yaml`, `tune_sam3.py`)

| Decision | Value | Reason |
|---|---|---|
| Model / recipe | official SAM 3 image model and fine-tuning recipe (frozen copy `sam3_base_recipe.yaml`, never edited; per-run configs written to the run folder), **all 840.5 M parameters trainable** (text encoder not frozen) | full fine-tuning, as Meta's recipe |
| Prompt | `"skin lesion"` (single category) | clinically neutral (most ISIC lesions are benign; earlier experiments used the misleading "skin cancer") |
| Input | 1008 × 1008 | fixed by the architecture |
| Precision / batch | FP32, batch 2, official activation checkpointing, no gradient accumulation | memory probe below |
| Budget | **30 epochs**, patience 30 (no early stopping) | one FP32 epoch ≈ 1.8 h; 120 epochs would take > 8 days per run and the HPO several months |
| HPO | **10 trials × 10 epochs**, Optuna TPE (Optuna 5.0.0, SQLite), **5** start-up trials, seeded per proposal (same formula as the U-Net) | compute; a foundation model starts from strong weights |
| Selection | `best.pt` = best validation per-image JSI (strict improvement, ties earliest), the shared metric code | as the U-Net |
| Prediction rule | **top-1** (highest-score) instance among those with score ≥ 0.5 for Phase 5 and the end-to-end benchmark; the training-time validation JSI that selects `best.pt` (Phases 1–4, including the Phase 2 CV folds) keeps the union rule, so all folds of a phase are selected identically (§ 14.4) (score = sigmoid(logit) × presence; top 100 instances per image as in the official prediction dump) | SAM 3 default threshold |

**Default (official recipe) hyperparameters.** `lr_scale` 0.1, which scales the recipe's base learning rates
8 × 10⁻⁴ (detector transformer), 2.5 × 10⁻⁴ (vision backbone) and 5 × 10⁻⁵ (text encoder) to the **effective defaults
8 × 10⁻⁵, 2.5 × 10⁻⁵ and 5 × 10⁻⁶**; AdamW weight decay 0.1; layer-wise LR decay of the vision trunk 0.9;
inverse-square-root schedule with **2 warm-up optimiser steps**; horizontal-flip probability 0.5; random-resize scale
jitter with minimum size 480 px (maximum 1008).

**Search space** (learning dynamics and the recipe's own augmentations only; architecture, resolution, prompt, losses,
Hungarian matcher, optimiser type, batch, budget and precision belong to the base setup):

| Hyperparameter | Range | Default |
|---|---|---|
| `lr_scale` | 0.01 – 0.2, log | 0.1 |
| `weight_decay` (AdamW) | 0.01 – 0.2, log | 0.1 |
| `lrd_vision_backbone` | 0.6 – 1.0 | 0.9 |
| `scheduler_warmup` (optimiser steps) | 1 – 1000, log, integer | 2 |
| `hflip_p` | 0 – 0.5 | 0.5 |
| `resize_min_size` (px) | 320 – 1008, step 16 | 480 |

* **Warm-up in steps.** One epoch = **1,297** optimiser steps (2,594 images / batch 2); the recipe's 2 steps are
  effectively no warm-up, hence a range up to 1,000 steps (≈ 0.8 epoch).
* **All defaults lie inside the bounds**, so the first trial evaluates the Baseline configuration **itself** — unlike
  YOLO26 and the U-Net, whose Baselines are clipped.
* **Loss weights excluded** — including the focal γ searched by an earlier SAM 3 tuner — for consistency with the
  strict search spaces of the other arms.
* **5 start-up trials (Optuna default 10).** With 10 trials the default would leave no trial to TPE (defaults + 9
  random draws); with 5 the search is the default configuration, 4 random proposals and **5 TPE-guided proposals**.
* **Fault tolerance** as the U-Net (`hpo_state.json` same schema; interrupted trial resumes in the same folder from its
  checkpoint; retries; exit 75; configuration hash; exclusive lock). Verified: a search killed in the middle of its
  second trial and resumed produced `tune_results.csv` and `best_hyperparameters.yaml` identical to an uninterrupted
  search, and `hpo_state.json` recorded the interruption and resumption.

**Memory probe** (`probe_memory.py`, official trainer and recipe, V100S 32 GB, each configuration in a fresh process):

| Configuration | Result | Peak VRAM allocated / reserved / device | Step (median) | Projected epoch (2,594 images) |
|---|---|---|---|---|
| Batch 2, activation checkpointing **disabled** in the ViT and text encoder | **OOM** at 30.9 GiB | — | — | — |
| **Batch 2, official activation checkpointing (used)** | fits | 12.8 / 15.8 / **16.5 GiB** | 5.01 s | ≈ 108 min (1,297 × 5.01 s) |
| Same, strict deterministic algorithms | fits | 12.9 / 15.6 / 16.4 GiB | 6.67 s | ≈ 144 min |

*(The READMEs and the original notes quote ≈ 106 / 141 min, projected for the earlier 2,547-image export.)*
Activation checkpointing discards intermediate activations in the forward pass and recomputes them in the backward
pass: it trades compute for memory and is **numerically neutral** (same values), so it changes neither the optimisation
nor the results. SAM 3's detector encoder and decoder *require* it during training (they assert it), so it can only be
disabled in the two backbones; with it, the official batch of 2 runs in FP32 at about half the GPU memory without
freezing any component and without gradient accumulation.

**Compute.** ≈ 1.8 h per epoch including validation and checkpointing — about two orders of magnitude more than the
U-Net. Worst case (V100S, FP32): Phase 1 ≈ 55 h, Phase 2 ≈ 5 × 47 h, Phase 3 ≈ 10 × 18 h, Phase 4 ≈ 55 h — about three
weeks on one GPU (Phases 2 and 3 are independent and can run in parallel on two GPUs).

---

## 6. Model-specific quirks that must be reported

### 6.1 YOLO26-seg

* **Requested vs. actual batch size.** The protocol micro-batch is 16. In FP32 at 640 px on a 32 GB V100S the training
  peak is ≈ 5.4 / 10.7 / 21.6 / 24.7 GB for n / s / m / l, and **xlarge does not fit (~40 GB)**: Ultralytics catches
  the out-of-memory error in the first epoch and **silently halves the batch to 8**, while the run records still said
  16. Every run now records the batch it really used (`batch_effective`, read from `best.pt`'s training arguments); a
  mismatch is a warning in `summary/phase*_val.json` and `final_results.json` (and is printed by the root notebook).
  HPO trials use the same per-model micro-batch (`MICRO_BATCH`: 16, xlarge 8) instead of a larger one that would be
  halved silently. With `nbs = 64` constant, Ultralytics accumulates gradients (4 micro-batches at 16, 8 at 8), so
  **the effective optimisation batch is 64 for every size and phase**; only BatchNorm batch statistics (and the
  per-step memory) differ for xlarge.
  *Suggested wording:* "The nominal batch size (nbs = 64) is held constant across all phases and model sizes, so the
  effective optimisation batch size is identical (64) throughout the study; the micro-batch is 16, except for the
  largest variant (8), which does not fit at 16 in FP32 on a 32 GB GPU."
* **Selection criterion differs from the other arms:** `best.pt` and the HPO fitness use Ultralytics' fitness (box +
  mask mAP50-95 on the validation split); the U-Net and SAM 3 select on validation per-image JSI. Each framework's
  native criterion; the test evaluation is identical.
* **Best-epoch alignment.** Validation and CV metrics reported from a training log are those of the epoch that
  produced `best.pt`. Because `results.csv` stores metrics rounded to six significant digits whereas the framework
  selects on unrounded values, recomputing the selection from the log could, in rare near-ties, designate another
  epoch. The epoch is therefore identified **from the checkpoint itself** (`best.pt` stores the validation metrics of
  its epoch; the log row with identical values is selected), falling back to recomputing the fitness from the log
  (ties → latest) only if the lookup fails; the fitness stored in the checkpoint is also checked to equal box + mask
  mAP50-95 (within the framework's 10⁻⁵ rounding), which guards against a silent change of the fitness definition. An
  earlier pipeline version read the epoch with the highest *mask* mAP50-95 only; it designated a different epoch from
  `best.pt` in one of twelve test runs, which motivated the alignment.
* **Interrupted runs and the duplicate-epoch issue.** A resumed run restores the model, the EMA weights, the
  optimiser state and the epoch counter, but **not the dataloader RNG**, so it is not bit-identical; every resumption is
  logged (`run_state.json`) and flagged in the report. Within an epoch Ultralytics appends the validation row to
  `results.csv` *before* saving the checkpoint; if a run dies between the two, the log contains an epoch whose weights
  were never saved, and after resumption that epoch is logged twice. Selecting from the raw log could pick the orphaned
  row (metrics of non-existent weights), so only the **last** record of each epoch is kept; `epochs_trained` is the last
  epoch number, and gaps in the log are reported.
* **Two confidence thresholds:** instance metrics (P, R, mAP50, mAP50-95, F1 = 2PR/(P + R), box and mask) use the
  validator on the test split with batch 1 and its mAP defaults (`conf = 0.001`, NMS IoU 0.7); the pixel masks (DSC, JSI,
  boundary metrics) use the **top-1 (highest-confidence) instance among those with `conf ≥ 0.001`**
  (`segmentation_metrics.PIXEL_CONF`, `predicted_top1_mask`). Due to the single-lesion nature of ISIC 2018 Task 1, YOLO26 pixel evaluation utilizes a top-1 confidence selection at conf=0.001. This maximizes lesion recall while strictly preventing the merging of low-confidence background artifacts. (Until 2026-10-03 the
  pixel mask was the union of instances with `conf ≥ 0.25`; see § 14.4.) Only YOLO26 reports instance
  metrics; they are `NaN` for the other arms (SAM 3's validation/CV tables carry its official COCO mAP50-95 instead).
* **Checkpoint on disk is FP16** (Ultralytics `strip_optimizer`), whereas the U-Net and SAM 3 store FP32 — compare the
  theoretical FP32/FP16 weight sizes (fused parameters × 4 / × 2 bytes), not file sizes.
* **FP16 at batch 1 is not always faster.** In a preliminary measurement YOLO26n-seg was slightly *slower* in FP16 than
  in FP32 on a V100S: for small models at batch 1 kernel-launch overhead dominates. FP16 latency is measured, never
  assumed.
* **GFLOPs counter:** thop (Ultralytics' `get_flops`, 2 × MACs). YOLO26 contains PSA attention blocks whose batched
  matmuls thop cannot see; measured against `FlopCounterMode` the under-count is small — **+1.33 % (n), +0.71 % (s),
  +0.20 % (m), +0.26 % (l), +0.18 % (x)** at 640 px (measured on the architecture definitions, COCO head, 2026-10).

### 6.2 U-Net

* **Why a PyTorch port.** The original U-Net was a TensorFlow/Keras model; timing, memory and FLOP counting of a
  TensorFlow model next to PyTorch models would confound architecture with framework (runtime, kernel selection,
  allocator, FLOP counting). It was therefore ported to PyTorch on the same pinned stack.
* **Verified equivalence with the Keras model.** Framework defaults that differ were set to the Keras values:
  BatchNorm momentum 0.99 in Keras' convention (= 0.01 in PyTorch's) and ε = 10⁻³; `he_normal` (truncated normal)
  initialisation for the block convolutions and `glorot_uniform` for the transposed convolutions and the output layer,
  all biases zero; and the spatial alignment of Keras' `Conv2DTranspose(padding="same")`, which corresponds to the
  un-padded PyTorch transposed convolution **cropped to its first 2n rows and columns** — the common PyTorch idiom
  `padding=1, output_padding=1` keeps the *last* 2n and is shifted by one pixel (with identical weights it produced
  completely different outputs). After copying the weights of a randomly initialised Keras model built with the
  original, unmodified Keras code (with randomised BatchNorm statistics), the two outputs were **identical (max. abs.
  difference 0.0)** and the trainable-parameter counts matched (2,161,649); the initialiser statistics were also checked.
* **Deployment form (Conv–BN fusion).** As Ultralytics fuses YOLO26 (`model.fuse()`) before inference, every BatchNorm
  of the U-Net is folded into its convolution (`torch.nn.utils.fusion.fuse_conv_bn_eval`) for profiling: outputs
  unchanged up to rounding (max. abs. logit difference 1.9 × 10⁻⁶), parameters 2,161,649 → 2,158,705.
* **Deterministic, bit-exact behaviour.** The augmentation parameters of sample *i* in epoch *e* and the sample order of
  epoch *e* come from generators seeded with (seed, *e*, *i*) and (seed, *e*), so data order and augmentation are pure
  functions of the seed and the epoch, **independent of the number of dataloader workers**. Each data loader owns a
  dedicated generator, so the global random stream used by dropout is also independent of the worker count (by
  default PyTorch draws worker seeds from the global stream, which made results depend on the worker count). A
  checkpoint written atomically after every epoch stores the model, optimiser, epoch, early-stopping state and all RNG
  states. Hence a run **interrupted at any point and resumed is bit-identical** to an uninterrupted one (verified on CPU
  and GPU, with 0 vs. 8 workers); `results.csv` is rebuilt from the checkpoint. Two identical GPU runs were also
  bit-identical, and PyTorch reported no operation without a deterministic implementation. This is the strongest
  reproducibility guarantee of the three arms.
* **Deliberate differences from the original Keras training:** binary (nearest-neighbour) mask augmentation; a fresh
  shuffle every epoch (Keras' iterator continued across epochs); BCE from logits without Keras' probability clipping.
* **Validation-metric resolution.** Checkpoint selection and the validation/CV metrics written in the training logs
  use the 256 × 256 masks; the Phase 2 pixel metrics (`evaluate_cv_pixels.py`) and all test metrics are computed at the
  dataset resolution against the official masks.
* **Resolution ceiling.** The U-Net predicts at 256 × 256, so boundary detail removed by down-sampling cannot be
  recovered. An **oracle** outputting the reference 256 × 256 mask of each test image through exactly the U-Net
  inference path (bilinear upsampling + threshold 0.5) reached, on the 994 test images of the earlier 640 × 640 export,
  mean **DSC 0.9967** (min 0.9650; mean JSI 0.9935; pooled DSC 0.9977): ≈ 0.003 DSC lost on average, up to ≈ 0.035 for
  small or intricate lesions — small compared with typical inter-model differences, but to be stated. Nearest-neighbour
  upsampling of the mask gives a lower ceiling (DSC 0.9951 on 199 test images), which is why the pipeline upsamples the
  probability map bilinearly. The oracle also verifies the alignment of the inference and scoring path (a one-pixel
  shift or a flip would lower its DSC drastically). **[RESULT: re-measure on the official 1,000-image test set at the
  dataset resolution — the ceiling is expected to be lower now that images keep up to 1,024 px.]**
* **Native vs. resolution-matched compute.** The fused U-Net needs **6.40 GFLOPs** at its native 256 × 256 and **40.0
  GFLOPs** at 640 × 640 (cost grows with the pixel count, (640/256)² = 6.25×) — more than four times YOLO26n-seg at
  640 px (≈ 9.0–9.1 GFLOPs) for a comparable parameter count (2.69 M fused): the U-Net convolves at full input
  resolution in its first and last stages, whereas YOLO26 reduces the resolution early in its backbone. Compare
  computational cost with the resolution-matched figure (`gflops_640`); latency is reported at each model's native
  input. thop and `FlopCounterMode` agree exactly for this pure CNN (6.401 / 40.003 GFLOPs, measured).
* **FP16 at batch 1:** in a preliminary (contended) measurement the U-Net, like YOLO26n, was not faster in FP16 than in
  FP32 — confirm on an idle GPU.

### 6.3 SAM 3

* **The trainer wrapper.** SAM 3 is fine-tuned with **Meta's official trainer** (`sam3.train.trainer.Trainer`), model,
  data pipeline, losses, Hungarian matcher, optimiser, LR schedulers and validation, so the results reflect the model
  as its authors intended to fine-tune it. The study protocol is a subclass, `ProtocolTrainer`, selected through the
  Hydra configuration (`trainer._target_`); **the vendored `sam3` code is not modified**. The subclass replaces only the
  epoch loop: (1) official `train_epoch`; (2) official `val_epoch` (dumps the COCO predictions and computes COCO AP),
  then the study's pixel metrics on those predictions with the shared metric code; (3) selection on the validation
  per-image mean JSI (strict improvement, ties earliest), `best.pt` (weights + metrics) written atomically on
  improvement; (4) **then** the checkpoint, and the epoch appended to `results.csv`. The official recipe validates
  every second epoch, skips the first validation and selects nothing; the wrapper validates every epoch.
* **RNG-complete checkpointing.** The official checkpoint holds the model, optimiser and epoch but **no RNG state**, and
  is written *before* validation. `ProtocolTrainer` checkpoints *after* validation and adds the torch CPU and CUDA,
  NumPy and Python RNG states, the selection state (best value, best epoch, epochs without improvement) and the
  per-epoch history. A run interrupted at any point resumes with exactly the state it would have had: same data order
  (epoch-seeded sampler), same augmentations (worker seeds drawn from the restored global RNG), same dropout masks, same
  selection bookkeeping; `results.csv` is rebuilt from the checkpointed history.
* **Determinism and strict mode — what was verified and what the study uses.** Seeds are fixed for every RNG, cuDNN is
  deterministic **without autotuning** during training, and `torch.use_deterministic_algorithms(True, warn_only=True)` is
  enabled. Two operations are nondeterministic on GPU:
  * `grid_sample`'s backward pass (used by the mask loss's point sampling and the geometry encoder) accumulates gradients
    with atomic additions and has no deterministic CUDA kernel; in the training process it is **replaced** by an
    equivalent bilinear interpolation written as a weighted sum of gathered neighbours, whose backward is a deterministic
    `scatter_add` (`sam3_seg/determinism.py`; forward equal up to floating-point rounding);
  * the **memory-efficient attention backward of the ViT**, which PyTorch makes deterministic only in *strict* mode.
    Strict mode costs **+33 % training time** (6.67 vs. 5.01 s per optimiser step) and is therefore **off**
    (`strict_determinism = False`).

  Two identical 3-epoch runs and a run killed during epoch 2 and resumed were compared: **in strict mode** the three runs
  produced **bit-identical weights** (max. abs. difference 0) and identical `results.csv` — the checkpoint/resume
  mechanism itself is exact. **In the warn-only mode used by the study**, the remaining nondeterministic attention
  backward makes two identical runs differ by **up to 6.8 × 10⁻⁵ in the weights after 3 epochs** and in the 5th decimal of
  the validation JSI. Repeated or resumed SAM 3 runs are therefore **statistically equivalent, not bit-identical**; the
  thesis must not claim bit-exact reproducibility for SAM 3.
* **Run lifecycle and isolation.** Each training run is a separate process (`run_training.py`), so a CUDA error or OOM
  cannot take down the HPO driver; `run_state.json` records status, protocol and protocol hash, data fingerprints and
  events (atomic writes); a completed run is skipped, a changed protocol or dataset is refused, an exclusive POSIX lock
  prevents two processes from training the same run. One checkpoint of model + AdamW state occupies **9.4 GB**, so a
  completed run deletes its resume checkpoint, and HPO trials also delete their weights, keeping only metrics.
* **Prediction = validation path.** Phase 5 rebuilds each run's validation pipeline from its saved `config.yaml`
  (official transforms and post-processor) and applies it to the test split; applied to the validation images it
  reproduces the trainer's own validation JSI to the last digit (0.6277150682843381 in both, in the preliminary check) —
  the test metric is the same function of the model as the selection metric. SAM 3 is an **instance/concept model
  evaluated as a semantic segmenter** (top-1 instance with score ≥ 0.5 at test time; union during model selection —
  § 14.4) — state it. Since `EVAL_VERSION` 4 the test metric uses top-1, so the "same function as the selection
  metric" identity above holds for the union rule only; the two rules differ by ≤ 0.003 JSI on the checked splits.
* **Parameters by component:** vision backbone 454.0 M, text encoder 353.7 M, detector transformer 21.0 M, geometry
  encoder 8.2 M, segmentation head 2.3 M, scoring 1.2 M; without the text encoder 486.8 M. With a fixed prompt the text
  features can be cached: `forward_cached_text` measures that deployment form.
* **Compute profile and the thop limitation.** One batch-1 forward pass (1008 × 1008 test image, prompt "skin lesion")
  costs ≈ **6,057 GFLOPs**, measured natively with `torch.utils.flop_counter.FlopCounterMode` (counts 2 × MACs of matrix
  products and convolutions — the Ultralytics/thop convention):

  | Breakdown | GFLOPs per image | Share |
  |---|---|---|
  | **Total** | **6,057** | 100 % |
  | Image encoder (ViT + neck) | 5,614 | 92.7 % |
  | Detector (transformer, geometry encoder, segmentation head, scoring) | 424 | 7.0 % |
  | Text encoder | 20 | 0.3 % |
  | Linear layers (incl. Q/K/V/output projections and MLPs) | 4,747 | 78.4 % |
  | **Attention products (QKᵀ and AV in scaled-dot-product attention)** | **956** | **15.8 %** |
  | Convolutions | 354 | 5.8 % |

  *Why thop cannot be used.* thop counts FLOPs with forward hooks on known `nn.Module` types (convolutions, linear
  layers, normalisation). Modern attention is computed by **functional calls** —
  `torch.nn.functional.scaled_dot_product_attention` and explicit matrix products inside `forward` — which are not
  modules and are invisible to module hooks. For convolutional networks the two tools agree (§ 6.1, § 6.2), but for a
  transformer thop silently omits the attention products (**15.8 % of SAM 3's compute**) and any functional matmul.
  `FlopCounterMode` works at the dispatcher level and counts them. Two measurement details were necessary: under
  `torch.inference_mode()` PyTorch bypasses Python dispatch modes and `FlopCounterMode` counts **nothing** (an initial
  measurement reported 0.3 GFLOPs), so counting runs under `torch.no_grad()`; and `nn.MultiheadAttention`'s fused
  inference fast path (text encoder, detector) has no FLOP formula, so it is disabled during counting.
  *Interpretation:* the 15.8 % refers to the attention **products** only (the quadratic part); the Q/K/V and output
  projections are linear layers (in the 78.4 %), so the attention mechanism as a whole costs more than 15.8 %.
  Element-wise operations, normalisation and interpolation are counted by neither tool. For scale: the U-Net needs
  6.4 GFLOPs at its native input, i.e. SAM 3 needs **≈ 950×** more compute per image at native inputs **[add the
  YOLO26 ratios from Table 2]**.
* **FP16 = `torch.autocast(float16)` with FP32 weights** (SAM 3's official mixed-precision path), whereas YOLO26 and the
  U-Net convert the weights (`.half()`); SAM 3's `vram_weights_mb` is therefore the FP32 footprint in both rows. In the
  preliminary check FP16 changed the mean validation JSI by +0.0006.
* **Preliminary latency** (smoke test, V100S): ≈ 590 ms per image in FP32 (1.7 FPS), ≈ 196 ms with FP16 autocast;
  3.3 GB of VRAM for the weights, forward peak 4.1 GB **[replace with the final Phase 5 values]**.
* **Reduced budget** (30 epochs; HPO 10 × 10) — disclosed in `final_results.json` → `protocol_notes`.

---

## 7. Evaluation metrics and statistics (Phase 5a)

Every combination variant (Baseline, Optimized) × model × precision (FP32, FP16) is evaluated on the test split with
batch 1. FP32 is the primary result (all models were trained in FP32); FP16 quantifies the accuracy cost of
half-precision deployment and is read with the FP16 efficiency results — its accuracy is measured, not assumed.

### 7.1 Per-image metrics (`segmentation_metrics.py`, identical in the three repositories)

From the confusion counts TP / FP / FN / TN at dataset resolution:

$$
\mathrm{DSC} = \frac{2\,TP}{2\,TP + FP + FN}, \qquad
\mathrm{JSI} = \frac{TP}{TP + FP + FN}, \qquad
\mathrm{JSI}_{0.65} = \begin{cases} \mathrm{JSI} & \mathrm{JSI} \ge 0.65 \\ 0 & \text{otherwise} \end{cases}
$$

| Metric | Definition | Notes |
|---|---|---|
| DSC | 2TP / (2TP + FP + FN) | Dice / pixel F1 |
| JSI | TP / (TP + FP + FN) | Jaccard / IoU |
| ISIC JSI$_{0.65}$ | JSI if JSI ≥ 0.65 else 0 | official ISIC 2018 Task 1 score |
| Sensitivity, specificity, accuracy | TP/(TP+FN), TN/(TN+FP), (TP+TN)/N | |
| **Boundary IoU** | IoU of the boundary bands (pixels within *d* of each mask's own contour), *d* = 2 % of the image diagonal (≈ 26 px at 1024 × 768) | Cheng et al., CVPR 2021 |
| **NSD** (surface Dice) | fraction of both contours within τ of the other contour, τ = 1 % of the diagonal (≈ 13 px) | Nikolov et al., 2021; recommended by Metrics Reloaded |
| **HD95** | max of the two directed 95th percentiles of contour-to-contour distances, **pixels** at dataset resolution (lower is better; image sizes vary, so report the median next to the mean — the diagonal penalty of a missed lesion dominates the mean) | MONAI / DeepMind `surface-distance` convention; added in this audit |

* **Ground truth** = the official mask (`masks/<id>.png`; for SAM 3 its lossless RLE). **Prediction**: YOLO26 — each
  test image alone (batch 1), `conf = 0.001`, instance masks at the original resolution (`retina_masks=True`), binarised
  at 0.5, **top-1 instance only** (never merged — one lesion per image); U-Net — bilinear upsampling of the probability map, threshold 0.5; SAM 3 — § 6.3.
* **Empty masks:** an empty prediction with a non-empty ground truth (or vice versa) scores DSC = JSI = 0 (boundary scores 0,
  HD95 = image diagonal) and is **included** in all averages — missed lesions count as complete failures (Metrics Reloaded:
  penalise, never drop). If both were empty: overlap and boundary scores 1, HD95 0; sensitivity is then undefined and
  excluded from its mean. The number of empty predictions is reported ("Missed" in Table 1).
* Contours are 1-pixel inner boundaries; the image border counts as background. HD95 was verified against a
  brute-force pairwise-distance reference (max error 4 × 10⁻⁶ px), and adding it left every previous metric
  bit-identical.
* The predicted FP32 masks are stored, and the qualitative figures (notebook 01) are drawn from these masks and the same
  ground-truth function, so the visualisations show exactly what was scored. The same definitions are applied to the
  held-out folds of Phase 2.

**Why boundary metrics** (committee / Metrics Reloaded): overlap scores are dominated by the lesion interior and are
insensitive to boundary errors on large lesions; contour adherence matters clinically (excision margins; border
irregularity is a melanoma criterion). Boundary IoU and NSD measure the *fraction* of boundary within a tolerance;
HD95 measures the *size* of the boundary errors.

### 7.2 Aggregation and confidence intervals

* **Primary figure:** per-image (macro) mean over the 1,000 test images; also SD (ddof = 1), median, IQR, and the
  pooled (micro) DSC/JSI computed from TP, FP, FN summed over the test set (secondary).
* **95 % CI:** percentile bootstrap of the mean, 2,000 resamples of the test images, seed 0 — in every
  `accuracy/*.json`, `summary/test_accuracy.csv` and Table 1.
* **CV:** mean ± sample SD over the 5 folds (§ 3.3).

### 7.3 Paired statistical comparisons

Every model is scored on the same images and per-image scores are stored (`phase5_test/per_image/*.csv`, keyed by ISIC
ID), so all comparisons are **paired**; per-fold CV results are paired across architectures as well (identical folds).

* **Within an arm (HPO effect, `summary/hpo_gain.csv`, Table 3):** for each image $d_j = s^{\mathrm{opt}}_j - s^{\mathrm{base}}_j$;
  reported: mean difference $\bar d$ with paired percentile-bootstrap 95 % CI (2,000 resamples, seed 0), two-sided
  Wilcoxon signed-rank p (zero differences discarded), numbers of improved and worsened images (direction-aware: lower
  HD95 is better) — for DSC, JSI, Boundary IoU, NSD and HD95; the change in mask mAP50-95 for YOLO26. These p-values
  are **per model and uncorrected**: if a claim covers several models jointly (e.g. "HPO helped all five YOLO26 sizes"),
  correct them (Holm–Bonferroni).
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
| Batch | **1** image (the clinical / on-device use case), single GPU |
| Isolation | every (variant, size, precision) runs in a **freshly started process**: peak-memory counters, allocator cache and cuDNN autotuning state cannot carry over between models |
| Kernel selection | `torch.backends.cudnn.benchmark = True` (deployment with a fixed input shape); autotuning happens during warm-up, excluded from every measurement |
| Model form | the deployed form: Conv–BN fused (YOLO26 `fuse()`, U-Net `fuse_conv_bn_eval`), parameters without gradients, `torch.inference_mode()` |
| `forward` | network only, at the native input size — YOLO26/U-Net: a fixed tensor (uniform random, seed 0); SAM 3: a pre-processed real test image + prompt — **50 warm-up + 500 timed** iterations, each bracketed by a pair of `torch.cuda.Event`s on the current stream followed by a device synchronisation. CUDA events time execution on the device itself, unaffected by asynchronous kernel launches (more accurate than host timers for GPU work); synchronising after every iteration guarantees that each measurement is exactly one image with no queued work (true batch-1 latency) |
| `end_to_end` | the deployed pipeline from a **decoded image in host memory to the binary mask at dataset resolution in host memory** (pre-processing, inference, post-processing, upsampling, threshold, device→host copy) — YOLO26 `predict(conf=0.25, retina_masks=True)` + union mask; U-Net colour conversion → area resize → scaling → host→device → forward → sigmoid → bilinear upsampling → 0.5 → device→host; SAM 3 Meta's `Sam3Processor`; timed with a monotonic host clock (`time.perf_counter`) between two device synchronisations (it includes host work CUDA events cannot see); **20 warm-up + 200 timed** calls on the first test image by ISIC ID (the same image in every arm) |
| `end_to_end_dataset` | the same pipeline once on each of the **first 100 test images sorted by ISIC ID** (the same images for every model), after one untimed pass (cuDNN autotuning of every input shape) — input-dependent spread (image size, number of instances) |
| SAM 3 only | `forward_cached_text`: forward with the prompt's text features precomputed (fixed-prompt deployment) |
| Statistics | mean, SD, **median** (typical latency), P90, **P95** (the worst-case behaviour relevant for real time), P99, min, max; **FPS** = 1000 / mean latency (1000 / median also reported); raw per-iteration latencies kept for distribution plots |
| **Peak VRAM** | (a) **allocator peak** — `torch.cuda.max_memory_allocated` during the timed loops, counters reset *after* warm-up (with `cudnn.benchmark` the warm-up peak is dominated by cuDNN's algorithm-search workspaces — e.g. ≈ 2.2 GB vs. ≈ 74 MB at steady state for YOLO26n-seg FP32 in a test — and is recorded separately) = the model's own footprint; (b) **process footprint** (`vram_process_peak_mb`) — device memory held by the benchmark process as reported by the driver (`nvidia-smi` delta: CUDA context, kernels and allocator cache included), sampled after each timed scope and maximised over the samples — an approximation of what a deployment GPU must provide, not a continuously tracked peak, and only meaningful on an otherwise idle GPU (the delta includes other processes); plus the CUDA-context size and the VRAM of the weights alone |
| Host RAM | resident set size after model loading and after the benchmark, and its peak over the process lifetime (`getrusage`) |
| Units | all memory and size figures in MiB (2²⁰ bytes) |
| Contention | GPU utilisation, memory in use, SM clock and temperature sampled with `nvidia-smi` before and after each run; utilisation by other processes > 5 % flags the run `contended`; only uncontended measurements are reported (re-run on an idle GPU) |
| Model size | parameters (fused and unfused), GFLOPs at the native input (+ at 640 px for the U-Net), checkpoint size, theoretical FP32/FP16 weight size |
| FP16 | YOLO26/U-Net: weights and input cast to half (`model.half()`, `predict(half=True)`); SAM 3: autocast; identical conditions to FP32. FP16 halves weight memory; its latency benefit depends on whether inference is compute-bound (small models at batch 1 may gain nothing, § 6.1) |

Baseline and Optimized share the architecture and therefore the cost; their efficiency figures coincide within noise
and are reported for the deployed (Optimized) models.

**Real-time criterion used in the thesis:** a model is real-time at *F* FPS if its **P95 end-to-end latency over the
100 distinct test images** is ≤ 1000 / *F* ms (default *F* = 30 → 33.3 ms), at batch 1 on the reported GPU. Using P95
rather than the mean guarantees the frame budget for 95 % of frames.

**Report with every latency:** the GPU model, driver, CUDA/cuDNN and PyTorch versions (stored in each efficiency JSON),
the precision and the clock behaviour; state that all reported latencies are uncontended. Never quote the `infer_ms`
column of the Phase 5a per-image CSVs: it is a by-product of the accuracy loop with arm-specific scopes, not a
benchmark.

---

## 9. Cross-architecture aggregation (root notebook)

`04_cross_architecture_results.ipynb` (driver) and `results_aggregator.py` (logic, also runnable headless) read only the
Phase 5 summaries and per-image files of the three pipelines and write `analysis_outputs/`:

* **Tables (LaTeX `booktabs` + CSV):** 1 accuracy with 95 % CI (best per column in bold); 2 efficiency with the
  real-time criterion; 3 HPO effect; 4 FP16 vs. FP32; 5 training cost per phase; 6 CV vs. test; 7 pairwise paired tests
  on DSC (other metrics in `data/paired_tests_all_metrics.csv`).
* **Figures (PDF + PNG):** 1 accuracy (DSC, JSI) vs. latency / FPS / parameters with the Pareto front; 2 training cost
  and inference latency (median → P95, FP32/FP16) vs. the real-time budget; 3 per-image distributions; 4 pairwise ΔDSC
  matrix with Holm-adjusted significance (+ JSI, Boundary IoU, HD95 versions); 5 boundary metrics with CI; 6 peak VRAM
  (allocator vs. process).
* **Key numbers** for the Results section (best accuracy, fastest model, which models meet the real-time criterion,
  SAM 3 vs. the best lightweight model with CI/p/effect size and cost ratios, Friedman test).

Colour encodes the architecture consistently in every figure (validated colour-vision-deficiency-safe palette); YOLO26
sizes are distinguished by direct labels (n, s, m, l, x).

---

## 10. Reproducibility, determinism and software environment

* **Pinned environment (Docker):** `nvidia/cuda:12.1.0-devel-ubuntu22.04`, Python 3.11, `torch==2.5.1`,
  `torchvision==0.20.1`, `torchaudio==2.5.1` (cu121) in all three images; YOLO26 `ultralytics==8.4.21`,
  `pandas==3.0.1`; U-Net `optuna==5.0.0`, `ultralytics-thop`, pandas, scipy (exact pins); SAM 3 the vendored official
  code (`pip install -e ".[train, notebooks]"`), `optuna==5.0.0`, `pandas==3.0.6`. The HPO state refuses to resume under
  another library version.
* **Seeds:** Python `random`, NumPy, PyTorch (CPU and CUDA), `PYTHONHASHSEED`; deterministic algorithms requested in
  PyTorch and cuDNN.
* **Determinism by arm** (from strongest to weakest):

  | Arm | Guarantee | Mechanism / residual source |
  |---|---|---|
  | U-Net | **bit-exact**: repeated runs and resumed runs identical, independent of the worker count | per-(seed, epoch, sample) generators, per-loader generator, RNG-complete checkpoints; no nondeterministic op reported |
  | SAM 3 | bit-exact **in strict mode** (verified); the study uses warn-only → statistically equivalent runs (≤ 6.8 × 10⁻⁵ in the weights after 3 epochs, 5th decimal of val JSI) | RNG-complete checkpoints after validation, deterministic `grid_sample`; residual: ViT memory-efficient attention backward (strict mode +33 % time) |
  | YOLO26 | deterministic algorithms requested (warn-only); a resumed run is **not** bit-identical | dataloader RNG not checkpointed by Ultralytics; ops without deterministic kernels and DDP reduction order; HPO proposals reproducible conditional on the fitness values (`SeededTuner`) |

* **Provenance:** SHA-256 of datasets, splits, weights and settings in every output; Phase 5 results recomputed exactly
  when weights, test list or method version change; deterministic, fingerprinted K-fold manifests.
* **Fault tolerance:** idempotent, resumable phases; atomic state writes; POSIX `lockf` locks against concurrent writers
  (released when the owning process dies, so orphaned dataloader workers of a killed trainer cannot hold them);
  `--force` never deletes (old outputs moved to `*.bak-<UTC>`).

---

## 11. Threats to validity and limitations

1. **Native input resolution differs** (256 / 640 / 1008 px). Evaluation is at the same dataset resolution against the
   same masks, but part of the accuracy *and* cost differences is resolution. Report the U-Net's 640-px GFLOPs and its
   resolution ceiling.
2. **Unequal training budgets for SAM 3** (30 epochs, HPO 10 × 10 vs. 120 epochs, HPO 30 × 30), justified by compute
   (≈ 1.8 h per epoch) and strong pretrained weights; bias direction: SAM 3 is, if anything, under-tuned.
3. **Different selection criteria** (YOLO26: box + mask mAP50-95; U-Net and SAM 3: per-image JSI, the U-Net at
   256 × 256) — each framework's native criterion.
4. **Small validation split** (100 images) — checkpoint selection and HPO fitness remain noisy, and validation-based
   figures are optimistically biased (§ 4); mitigated by CV and by the paired test-set HPO analysis.
5. **Different HPO algorithms and search spaces** (genetic algorithm vs. TPE; YOLO26 and U-Net 30 × 30, SAM 3 10 × 10) —
   each tunes its own recipe; the HPO effect is reported per arm rather than compared. In YOLO26 and the U-Net the
   Baseline values lie outside the search bounds (not candidates of the search); in SAM 3 the Baseline is the first trial.
6. **Hardware dependence of latency** — absolute numbers are for the reported GPU (V100S) and software stack; ratios
   between models are more portable than absolute values. No CPU or embedded-GPU (e.g. Jetson) measurement is part of
   the protocol; claims about such devices must be framed as extrapolations from measured VRAM and latency.
7. **FP16 paths differ** (`.half()` for the CNNs, autocast for SAM 3).
8. **Model classes differ** — YOLO26 and SAM 3 are instance/concept models evaluated as semantic segmenters (union of
   instances above a score threshold); the U-Net is a single-width network (no size family).
9. **Single dataset** (ISIC 2018 Task 1, dermoscopy); generalisation to other datasets or modalities is not tested.
10. **Single training seed** per configuration; variability is estimated by the 5-fold CV (whose folds are not
    independent), not by repeated seeds.
11. **Reproducibility differs by arm** (§ 10): bit-exact (U-Net), statistically equivalent (SAM 3), resumable but not
    bit-identical (YOLO26).

---

## 12. Ready-to-adapt wording for the Methods section

> **Data.** We used the official ISIC 2018 Task 1 release with its official partition (2,594 training, 100
> validation and 1,000 test images). Images were resized so that their longer side was at most 1,024 px (aspect ratio
> preserved, no upscaling), and the official masks at this resolution served as ground truth for all models. The test
> set was used only once, for the final evaluation; the absence of test images from the cross-validation pool was
> verified programmatically.

> **Protocol.** All models followed the same five-phase protocol: (1) training with the default hyperparameters of
> each model's reference recipe; (2) 5-fold cross-validation of this configuration on the union of the training and
> validation sets (identical folds for all models); (3) seeded hyperparameter optimisation on the validation set;
> (4) retraining with the tuned hyperparameters; and (5) evaluation of both the default and the tuned model on the
> test set. A fixed base configuration (architecture, optimiser type, loss, input size, batch size, training budget and
> numerical precision) was shared by all phases of a model, so that the default and tuned models differ only in the
> tuned hyperparameters. Training was performed in FP32 without early stopping (justification: see the wording in § 4),
> and the checkpoint with the best validation score was retained.

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

> **Reproducibility (per arm).** *"The U-Net training is bit-exactly reproducible, including after interruption. SAM 3
> training is deterministic except for the backward pass of the vision transformer's memory-efficient attention;
> repeated runs differ by at most 7 × 10⁻⁵ in the weights (bit-exactness was verified in PyTorch's strict deterministic
> mode, which was not used for the experiments because it increases training time by 33 %). YOLO26 training uses the
> framework's deterministic mode; interrupted runs are resumable but not bit-identical."*

---

## 13. Artefact map: which file feeds which table or figure

| Thesis element | Produced by | Underlying files |
|---|---|---|
| Table 1, Fig. 1, 3, 5 | root notebook | `<repo>/logs/<name>/summary/test_accuracy.csv`, `phase5_test/per_image/*.csv` |
| Table 2, Fig. 1, 2, 6 | root notebook | `summary/efficiency.csv`, `phase5_test/efficiency/*.json` |
| Table 3 | root notebook | `summary/hpo_gain.csv` |
| Table 4 | root notebook | `summary/test_accuracy.csv` + `efficiency.csv` (both precisions) |
| Table 5, Fig. 2a | root notebook | `summary/training_cost.csv` (from every `results.csv`) |
| Table 6 | root notebook | `summary/phase2_cv_pixel.csv` |
| Table 7, Fig. 4 | root notebook | per-image CSVs (paired by ISIC ID) |
| Qualitative figure | `analysis/02_segmentation_visualizer.ipynb` (each repo) | `phase5_test/masks/<variant>_<model>/`, dataset images and official masks |
| Per-architecture figures | `analysis/03_metrics_and_efficiency.ipynb` (each repo) | `summary/*` |
| SAM 3 compute breakdown | notebook 02 of `sandbox_sam3` (Figure 8) | `phase5_test/efficiency/*.json` → `gflops_by_component`, `gflops_by_op` |
| Disclosures | `summary/final_results.json` → `warnings`, `protocol_notes`; `run_state.json`; `hpo_state.json` | printed by the root notebook |

---

## 14. Committee-compliance checklist, disclosure checklist and audit changelog

### 14.1 Committee demands

| Committee demand | Where it is met |
|---|---|
| Measured real-time / on-device evidence, not only GFLOPs/params | Phase 5b: batch-1 median and P95 latency, FPS, allocator and process VRAM footprint, end-to-end over 100 distinct images; real-time criterion on P95 (Table 2, Fig. 2, 6) |
| Metrics Reloaded: boundary metric | Boundary IoU, NSD, HD95 in every test/CV summary (Table 1, Fig. 5) |
| 95 % CIs | seeded bootstrap CI for every per-image metric; paired bootstrap CI for every difference |
| Paired statistical comparisons | Wilcoxon + bootstrap within arms (HPO) and across architectures (Holm, Friedman, effect sizes) |
| Standardised visual artefacts | notebooks 01/02 identical across repositories (shared cells byte-identical) |
| Unified analysis artefacts | root notebook + aggregator → LaTeX tables and figures |

### 14.2 Points to verify or disclose before submission (merged from the three per-arm notes)

1. **Resumed training runs** — which runs were resumed (`run_state.json`, flagged in the report); YOLO26 resumes are not
   bit-identical, SAM 3 resumes are statistically equivalent.
2. **Baseline outside the search bounds** — YOLO26 (lr0, weight decay) and U-Net (weight decay 0 → 10⁻⁶); SAM 3's
   Baseline is the first trial. Origin of YOLO26's refined bounds; **state that the exploratory search used train/val
   only**.
3. **Best-epoch alignment** — reported validation metrics are those of the selected checkpoint (§ 6.1).
4. **Optimistic bias of validation-based figures** — main claims rest on the Phase 5 test figures.
5. **HPO bookkeeping** — completed, failed and accepted-failure trials per model (`hpo_state.json`).
6. **Search-algorithm asymmetry** — genetic algorithm (YOLO26) vs. TPE (U-Net, SAM 3); SAM 3's 10 × 10 budget with 5
   start-up trials.
7. **Effective batch** — YOLO26x micro-batch 8 (`batch_effective`), effective batch 64 everywhere.
8. **Contended efficiency runs** — none may be quoted; report GPU, driver, clock behaviour.
9. **Different input resolutions and the U-Net resolution ceiling** (re-measured on the official test set).
10. **Validation-metric resolution of the U-Net** (256 × 256 for selection).
11. **Deliberate differences from the original Keras training** (binary mask augmentation, per-epoch reshuffle, BCE
    from logits).
12. **Reduced SAM 3 budget; FP16 = autocast for SAM 3; GFLOPs counters differ** (state in the efficiency table).
13. **SAM 3 evaluated as a semantic segmenter** — prompt "skin lesion", top-1 instance with score ≥ 0.5 at test time
    (union of instances with score ≥ 0.5 for checkpoint selection; § 14.4).
14. **Reproducibility statement per arm** (§ 10, § 12) — never claim bit-exactness for SAM 3 or YOLO26.
15. **Multiple comparisons** — Holm for joint claims across models (automatic in the cross-architecture tables; to be
    applied by hand to the per-model HPO p-values if they are claimed jointly).
16. **Superseded numbers** — no figure from the Roboflow-export runs, the legacy Keras arrays or the "skin cancer"
    SAM 3 experiments may be mixed with final results (§ 2.5, Appendix A).

### 14.3 Changes made in the final audit (2026-10-01) — cite these method versions if asked

* `segmentation_metrics.py` (byte-identical in the three repositories): **HD95** added; previous metrics unchanged
  (verified bit-identical). Evaluation cache version `EVAL_VERSION = 3` in the three repositories.
* `benchmark_efficiency.py`: driver-level **process VRAM** (CUDA context included) and the **`end_to_end_dataset`**
  scope (100 distinct test images, same images in every repository) in all three; YOLO26's `end_to_end` now ends at
  the same artefact as the other arms (full-resolution union mask on the host, `retina_masks=True`, `conf=0.25`;
  superseded by § 14.4: top-1 mask, `conf=0.001`).
  `BENCHMARK_VERSION` 3 (YOLO26, U-Net) / 2 (SAM 3).
* `build_final_report.py`: HD95 in the accuracy and HPO-gain tables (direction-aware improvement counts); new
  efficiency columns (`e2e_dataset_*`, `vram_process_peak_mb`, `vram_cuda_context_mb`, `gflops_640`, `input_px`).
* Notebooks 01/02 standardised (ground-truth column, standard figures A–C with DSC/JSI/BIoU and HD95, real-time table,
  readable log axes, corrected stale notes); root notebook + aggregator added (`analysis/` in each repository).
* This document: cross-checked against the three per-arm notes (2026-10-02); their missing details merged, superseded
  statements listed in Appendix A.

### 14.4 YOLO26 pixel-mask rule changed to top-1 at conf 0.001 (2026-10-03)

* **Change:** `segmentation_metrics.evaluate_images` now scores `predicted_top1_mask` (the single highest-confidence
  instance) instead of `predicted_union_mask`, and the pixel `--conf` default of `evaluate_cv_pixels.py`,
  `evaluate_test_set.py` and `benchmark_efficiency.py` is `PIXEL_CONF = 0.001` (was 0.25). `EVAL_VERSION` 3 → 4 and
  YOLO26 `BENCHMARK_VERSION` 3 → 4, so no cached union-rule result is reused. Due to the single-lesion nature of ISIC 2018 Task 1, YOLO26 pixel evaluation utilizes a top-1 confidence selection at conf=0.001. This maximizes lesion recall while strictly preventing the merging of low-confidence background artifacts.
* **Why:** CPU diagnostic of the five Phase 1 `best.pt` on the 100-image validation split
  (`logs/pipeline_final_v1/phase1_pixel_eval_cpu/`). Union at 0.25: 3–7 empty masks per size, JSI 0.774–0.786.
  Union at 0.001: no empty masks, but background instances merged into the mask (specificity 0.81–0.89), JSI
  0.706–0.799. Top-1 at 0.001: no empty masks, JSI 0.793–0.817 (nano 0.817 / DSC 0.891).
* **Disclosure:** the rule was chosen after seeing these validation results, but **before** any Phase 2 pixel-level
  or Phase 5 test-set evaluation. Confirm on the CV folds before reporting. `segmentation_metrics.py` is therefore no
  longer byte-identical across the three repositories (YOLO26 only: `PIXEL_CONF`, `predicted_top1_mask`); the
  U-Net (dense, no instances) is unaffected.
* **SAM 3 aligned (2026-10-03):** `inference.evaluate_annotations` (Phase 5) and the `end_to_end` benchmark now keep
  the **top-1 instance among those with score ≥ 0.5** (SAM 3's own threshold is unchanged). `EVAL_VERSION` 3 → 4 and
  SAM 3 `BENCHMARK_VERSION` 2 → 3. **Not changed:** `protocol_trainer.pixel_metrics_from_dump`, the training-time
  validation JSI that selects `best.pt` and that provides SAM 3's Phase 2 CV metrics, because the Phase 2 CV was
  running when the rule was adopted and changing it would have selected fold 0 and folds 1–4 under different rules.
  Measured impact (CPU, from the saved prediction dumps): Phase 1 validation (n = 100; 3 images with > 1 instance)
  union JSI 0.8553 = top-1 0.8553; Phase 2 fold 0 (n = 539; 23 images with > 1 instance) union 0.8415 vs top-1
  0.8439 (+0.0024). The asymmetry is therefore negligible but must be disclosed: SAM 3's CV row uses union,
  its test row top-1.

---

## Appendix A — Statements in the per-arm notes that the final protocol supersedes

The per-arm notes were written at different stages of the protocol. When writing, use the **current** value.

| # | Note(s) | The note says | The final protocol (current code) |
|---|---|---|---|
| 1 | YOLO26 § 3.1, § 5; U-Net § 1, § 3.3, § 4.2 | early-stopping patience **25** (training) and **10** (HPO trials) | patience = epochs: 120 / 120 and 30 / 30 (SAM 3: 30 / 30 and 10 / 10) — § 4 |
| 2 | YOLO26 § 3.1 | HPO trial micro-batch **32** (16 for YOLO26x-seg, owing to FP32 + DDP memory) | per-model micro-batch `MICRO_BATCH` = **16** (xlarge **8**) in every phase incl. HPO; effective batch 64 — § 6.1 |
| 3 | U-Net § 2.1, § 5, § 7.1, § 9.3; SAM 3 § 1.2 | data = Roboflow YOLO export, **2,547 / 100 / 994** images at **640 × 640** (aspect ratio not preserved); "split differs from the official ISIC split" | data = raw official release, **2,594 / 100 / 1,000** images, longer side ≤ 1,024 px, aspect ratio preserved — § 2 |
| 4 | YOLO26 § 6.2; U-Net § 2.1, § 2.3; SAM 3 § 1.2 | ground truth = YOLO polygons rasterised at the image resolution | ground truth = the **official mask** (`masks/<id>.png`; SAM 3: its lossless RLE); polygons only for YOLO training labels |
| 5 | U-Net § 2.3, § 6, § 7.1, § 9.1 | evaluation and end-to-end benchmark at "the original resolution (640 × 640)" | at each image's **dataset resolution** (variable, ≤ 1,024 px) |
| 6 | SAM 3 § 1.3 | folds of 2,117 / 530 (× 2) and 2,118 / 529 (× 3) | pool 2,694 → **2,155 / 539 (× 4) and 2,156 / 538 (× 1)** |
| 7 | SAM 3 § 3.1, § 3.4; README | one epoch = **1,273** optimiser steps; ≈ 106 min (strict 141 min) | **1,297** steps (2,594 / 2); projected ≈ 108 min (strict ≈ 144 min) at 5.01 / 6.67 s per step |
| 8 | YOLO26 § 3.3 | epochs trained "computed over unique epochs" | `epochs_trained` = last epoch number; gaps in `results.csv` reported |
| 9 | YOLO26 § 7.2, § 7.3; U-Net § 6 | end-to-end = `predict()` on the first test image (YOLO26 without `retina_masks`); VRAM excludes the CUDA context | end-to-end ends at the scored full-resolution mask on the host for every arm; first image **by ISIC ID**; plus `end_to_end_dataset` (100 images) and process-level VRAM incl. CUDA context — § 8 |
| 10 | all three | metrics: DSC, JSI, JSI$_{0.65}$, sensitivity, specificity, accuracy | **+ Boundary IoU, NSD, HD95**; cross-architecture paired tests with Holm and Friedman — § 7 |
| 11 | U-Net § 7.1 | resolution ceiling DSC 0.9967 (994 images, 640 × 640) | still valid for the old export only — **re-measure** on the official test set |
| 12 | SAM 3 § 1.4, § 5 | validation-JSI reproduction value 0.6277150682843381; FP16 Δ JSI +0.0006 | preliminary checks on the earlier data — keep as verification evidence, not as results |
| 13 | YOLO26 § 1.1, § 1.3 | [N_train/N_val/N_test]; training with DDP on two V100S | 2,594 / 100 / 1,000; GPU count of the final run to be stated (the launcher `wait_gpu.sh` now trains on one GPU by default) |

---

## Appendix B — Suggested references

*(Merged from the three per-arm notes plus the references of the boundary metrics and statistics added in the audit.
Verify every bibliographic detail before use.)*

**Data and task**
* N. Codella et al., "Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin
  Imaging Collaboration (ISIC)," arXiv:1902.03368, 2019.
* P. Tschandl, C. Rosendahl, H. Kittler, "The HAM10000 dataset, a large collection of multi-source dermatoscopic images
  of common pigmented skin lesions," *Scientific Data* 5, 180161, 2018.

**Models and frameworks**
* Ultralytics YOLO (software), version 8.4.21, https://github.com/ultralytics/ultralytics.
* O. Ronneberger, P. Fischer, T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," MICCAI 2015.
* Meta AI, "SAM 3: Segment Anything with Concepts," 2025 **[complete authors, venue/arXiv identifier]**.
* A. Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library," NeurIPS 2019.
* S. Ioffe, C. Szegedy, "Batch Normalization," ICML 2015.
* K. He, X. Zhang, S. Ren, J. Sun, "Delving Deep into Rectifiers," ICCV 2015 (He initialisation).
* X. Glorot, Y. Bengio, "Understanding the difficulty of training deep feedforward neural networks," AISTATS 2010.
* I. Loshchilov, F. Hutter, "Decoupled Weight Decay Regularization," ICLR 2019 (AdamW).

**Hyperparameter optimisation**
* T. Akiba, S. Sano, T. Yanase, T. Ohta, M. Koyama, "Optuna: A Next-generation Hyperparameter Optimization Framework,"
  KDD 2019.
* J. Bergstra, R. Bardenet, Y. Bengio, B. Kégl, "Algorithms for Hyper-Parameter Optimization," NeurIPS 2011 (TPE).
* L. J. Eshelman, J. D. Schaffer, "Real-coded genetic algorithms and interval-schemata," in *Foundations of Genetic
  Algorithms 2*, 1993 (BLX-α crossover).

**Metrics**
* L. R. Dice, "Measures of the amount of ecologic association between species," *Ecology* 26(3), 1945.
* P. Jaccard, "The distribution of the flora in the alpine zone," *New Phytologist* 11(2), 1912.
* L. Maier-Hein, A. Reinke et al., "Metrics reloaded: recommendations for image analysis validation," *Nature Methods*
  21, 2024.
* B. Cheng, R. Girshick, P. Dollár, A. C. Berg, A. Kirillov, "Boundary IoU: Improving Object-Centric Image Segmentation
  Evaluation," CVPR 2021.
* S. Nikolov et al., "Clinically applicable segmentation of head and neck anatomy for radiotherapy: deep learning
  algorithm development and validation study," *J. Med. Internet Res.* 23(7), 2021 (surface Dice / NSD).

**Statistics**
* F. Wilcoxon, "Individual comparisons by ranking methods," *Biometrics Bulletin* 1(6), 1945.
* B. Efron, R. J. Tibshirani, *An Introduction to the Bootstrap*, Chapman & Hall, 1993.
* S. Holm, "A simple sequentially rejective multiple test procedure," *Scandinavian Journal of Statistics* 6(2), 1979.
* M. Friedman, "The use of ranks to avoid the assumption of normality implicit in the analysis of variance," *JASA*
  32(200), 1937; M. G. Kendall, B. Babington Smith, "The problem of m rankings," *Ann. Math. Stat.* 10(3), 1939
  (Kendall's W).
* D. S. Kerby, "The simple difference formula: an approach to teaching nonparametric correlation," *Comprehensive
  Psychology* 3, 2014 (matched-pairs rank-biserial correlation).
