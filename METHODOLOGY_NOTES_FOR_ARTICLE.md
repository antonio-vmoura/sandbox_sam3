# Methodology Notes for the Article — SAM 3 on ISIC 2018 Task 1 (aligned with YOLO26-seg and U-Net)

> **Purpose.** Technical foundation for the *Materials and Methods* section of the thesis and article, for the
> SAM 3 arm of the study. It describes the protocol exactly as implemented in this repository
> (`run_pipeline_sam3.sh`, `sam3_seg/`) and how it was aligned with the YOLO26-seg (`sandbox_yolo26`) and U-Net
> (`sandbox_unet`) pipelines, each documented in its own `METHODOLOGY_NOTES_FOR_ARTICLE.md`. Items in
> **[brackets]** must be completed with values from the final run. Section 6 lists points to verify or disclose
> before submission.

---

## 1. Pipeline architecture and alignment with YOLO26 / U-Net

### 1.1 The five-phase protocol

SAM 3 (Meta's *Segment Anything with Concepts* image model, 840.5 M parameters) is the foundation-model arm of
the comparison. It follows the **same five-phase protocol** as YOLO26-seg and the U-Net:

| Phase | Content | Data |
|---|---|---|
| 0 — Dataset | COCO-format dataset derived from the YOLO26 dataset | train / val / test |
| 1 — Baseline | Full fine-tuning with the base setup and the official recipe's hyperparameters | train / val |
| 2 — Baseline CV | 5-fold cross-validation with the Phase 1 configuration | train ∪ val pool (test excluded) |
| 3 — HPO | Seeded, fault-tolerant Optuna TPE search over learning dynamics and augmentation | train / val |
| 4 — Optimised | Same base setup + the Phase 3 hyperparameters | train / val |
| 5 — Test set | Baseline **and** Optimised: pixel accuracy (FP32 + FP16) and batch-1 efficiency | **test** (only here) |

The protocol is defined once (`sam3_seg/common.py`): **Baseline = base setup + default (official recipe)
hyperparameters; Optimised = base setup + tuned hyperparameters**, so the tuned values are the only difference
between the two. Tuned files that try to override a base-setup key are rejected.

**Base setup (identical in every phase):** official SAM 3 image model and fine-tuning recipe (frozen copy in
`sam3_seg/configs/sam3_base_recipe.yaml`); all parameters trainable, text encoder included; input 1008 × 1008
(native); text prompt `"skin lesion"`; batch 2 with the official activation checkpointing (Section 4); FP32
(no mixed precision); seed 0; deterministic kernels (Section 2.4); model selection and early stopping on the
validation per-image mean Jaccard index (no early stopping: patience = epoch budget); predicted mask at test time = the top-1 (highest-score) instance with score ≥ 0.5 (model selection: union of those instances).

**Default hyperparameters (official recipe):** `lr_scale` 0.1 (learning rates 8e-5 for the detector, 2.5e-5 for
the vision backbone, 5e-6 for the text encoder), AdamW weight decay 0.1, layer-wise LR decay of the vision trunk
0.9, inverse-square-root schedule with 2 warm-up optimiser steps, horizontal-flip probability 0.5, random-resize
scale jitter with minimum size 480 px.

### 1.2 Shared data (Phase 0)

The SAM 3 data are **derived from the YOLO26 dataset itself**, which is the single source of truth of the study:

* the same images and the same split — 2547 train / 100 validation / 994 test images at 640 × 640, one lesion
  instance per image; images are hard-linked (byte-identical), and every image carries its ISIC identifier;
* each YOLO polygon is rasterised with the **same function and convention** used by the shared metric code and
  stored as a COCO compressed RLE mask, so the ground truth SAM 3 trains and is evaluated on is pixel-identical to
  the ground truth used for YOLO26 and the U-Net;
* the single COCO category is named `"skin lesion"`, which becomes the text prompt (the earlier SAM 3 experiments
  used `"skin cancer"`, a misleading prompt since most ISIC lesions are benign);
* the dataset is fingerprinted (SHA-256 of every annotation file in `meta.json`); a training run refuses to start
  on a modified or half-written dataset.

Before this alignment, the SAM 3 experiments used the official ISIC split (2594 / 100 / 1000 images) at the
original image resolutions; the realignment makes the three architectures directly comparable.

### 1.3 Identical cross-validation folds (Phase 2)

The CV pool is the train IDs followed by the validation IDs, in the YOLO26 order; the folds are produced with
YOLO26's algorithm (`numpy.random.RandomState(0)` shuffle followed by contiguous folds, equivalent to
`sklearn.model_selection.KFold(shuffle=True, random_state=0)`), giving folds of 2117 / 530 (×2) and
2118 / 529 (×3) images. The SHA-256 of each fold's validation ID list is **identical** to the U-Net's (verified),
and therefore to YOLO26's. Phase 2 re-derives the partition and refuses to run if a fold file disagrees, or if
any test ID appears in the pool.

### 1.4 Identical pixel metrics

`sam3_seg/segmentation_metrics.py` is a **byte-identical copy** of YOLO26's (verified with `cmp`): per image,
Dice (DSC), Jaccard (JSI), ISIC thresholded Jaccard (JSI < 0.65 → 0), sensitivity, specificity and accuracy,
computed at the original 640 × 640 resolution; empty predictions score 0 and are never skipped; aggregates are the
per-image mean, sample standard deviation, median, IQR, a seeded bootstrap 95 % confidence interval and the pooled
DSC/JSI. Cross-validation summaries use the sample standard deviation (ddof = 1).

The SAM 3 prediction is converted to a binary lesion mask exactly as it is during model selection: SAM 3's
official postprocessor scores each instance as sigmoid(logit) × presence score and upsamples its mask logits
bilinearly to the original image size (threshold 0.5); the predicted lesion mask is the **highest-scoring
instance (top-1) among those with score ≥ 0.5** (top 100 per image, as in the official prediction dump). Due to the
single-lesion nature of ISIC 2018 Task 1, lower-ranked instances are never merged, the same rule as YOLO26. The
training-time validation JSI that selects `best.pt` (including the cross-validation folds) keeps the union of the
instances with score ≥ 0.5, because the rule was adopted (2026-10-03) while the cross-validation was running; on
the checked splits the two rules differ by at most 0.0024 JSI. Phase 5 rebuilds each run's
validation pipeline from its saved configuration and applies it to the test split; with the union rule, applied
to the validation images, it reproduced the trainer's own validation JSI to the last digit (0.6277150682843381 in
both).

The Ultralytics instance metrics of YOLO26 (box/mask precision, recall, F1, mAP50) do not exist for SAM 3 and are
reported as NaN so the output tables keep an identical schema; the validation and CV tables carry the COCO
mAP50-95 of SAM 3's official evaluator in the corresponding columns.

---

## 2. The trainer wrapper

### 2.1 Design

SAM 3 is fine-tuned with **Meta's official trainer** (`sam3.train.trainer.Trainer`), model, data pipeline,
losses, Hungarian matcher, optimiser, learning-rate schedulers and validation, so that the results reflect the
model as its authors intended to fine-tune it. The study-specific protocol is added by a subclass,
`ProtocolTrainer` (`sam3_seg/protocol_trainer.py`), selected through the Hydra configuration
(`trainer._target_`); **the vendored `sam3` code is not modified**. The subclass replaces only the epoch loop:

1. train one epoch (official `train_epoch`);
2. validate (official `val_epoch`, which dumps the COCO predictions and computes COCO AP), then compute the
   study's pixel metrics on those predictions with the shared metric code (Section 1.4);
3. model selection and early stopping on the validation **per-image mean JSI** — strict improvement, ties keep
   the earliest epoch, stop after `patience` epochs without improvement; `best.pt` (model weights + metrics) is
   written atomically on improvement;
4. **then** checkpoint (Section 2.2), and append the epoch to `results.csv`.

The official recipe validates every second epoch, skips the first validation and selects nothing; the wrapper
validates every epoch, which is required for per-epoch model selection.

### 2.2 RNG-complete checkpointing

The official checkpoint contains the model, optimiser and epoch counter, but no random-number-generator state,
and it is written **before** validation. `ProtocolTrainer` instead checkpoints after validation and adds the
torch CPU and CUDA RNG states, the NumPy and Python RNG states, the early-stopping state (best value, best epoch,
epochs without improvement) and the per-epoch history to the official checkpoint. A run interrupted at any point
therefore resumes with exactly the same state it would have had: the same data order (epoch-seeded sampler), the
same augmentations (DataLoader worker seeds drawn from the restored global RNG), the same dropout masks and the
same early-stopping bookkeeping; `results.csv` is rebuilt from the checkpointed history.

**Bit-exactness — what was verified and what the study uses.** Two identical 3-epoch runs and a run killed
during epoch 2 and resumed were compared:

* in PyTorch's *strict* deterministic mode (`torch.use_deterministic_algorithms(True)`, plus the deterministic
  `grid_sample` of Section 2.4) the three runs produced **bit-identical weights** (maximum absolute difference 0)
  and identical `results.csv` files — the checkpoint/resume mechanism itself is exact;
* the study, however, trains in *warn-only* mode (Section 2.4) for speed. The single remaining nondeterministic
  operation — the backward pass of the ViT's memory-efficient attention — makes two identical runs differ by up to
  6.8 × 10⁻⁵ in the weights after 3 epochs and in the 5th decimal of the validation JSI. Repeated or resumed runs
  are therefore **statistically equivalent, not bit-identical**, and the article should not claim bit-exact
  reproducibility for SAM 3.

### 2.3 Run lifecycle and isolation

Each training run is a separate process (`run_training.py`) launched by `training.train_or_resume`, so a CUDA
error or an out-of-memory failure cannot take down the caller (e.g. the HPO driver). The run configuration is
generated from the frozen recipe plus the protocol and written to the run directory, never to the source tree.
`run_state.json` records the lifecycle (status, protocol and protocol hash, data fingerprints, events) with
atomic writes; a completed run is skipped, a changed protocol or dataset is refused, and an exclusive POSIX lock
prevents two processes from training the same run. Because one checkpoint of model + AdamW state occupies 9.4 GB,
a completed run deletes its resume checkpoint (a completed run is never resumed), and HPO trials also delete their
weights, keeping only their metrics.

### 2.4 Determinism

Seeds are fixed for every RNG, cuDNN runs in deterministic mode without autotuning, and
`torch.use_deterministic_algorithms(True, warn_only=True)` is enabled. Two operations of SAM 3 training are
nondeterministic on GPU:

* `grid_sample`'s backward pass (used by the mask loss's point sampling and the geometry encoder) accumulates
  gradients with atomic additions and has no deterministic CUDA kernel; it is replaced in the training process by
  an equivalent bilinear interpolation written as a weighted sum of gathered neighbours, whose backward pass is a
  deterministic `scatter_add` (`sam3_seg/determinism.py`; forward result equal up to floating-point rounding);
* the memory-efficient attention backward of the ViT, which PyTorch makes deterministic only in strict mode. Strict
  mode was measured to cost **+33 % training time** (6.67 vs 5.01 s per optimiser step on the V100S) and was
  therefore not used (protocol key `strict_determinism = False`).

---

## 3. Fault-tolerant hyperparameter optimisation (Phase 3)

### 3.1 Search space

Only learning dynamics and the recipe's own augmentations are searched; architecture, resolution, prompt, losses,
matcher, optimiser type, batch, budget and precision belong to the base setup. Loss weights (including the focal
γ searched by an earlier SAM 3 tuner) are deliberately excluded, for consistency with the strict search spaces of
the other arms.

| Hyperparameter | Range | Default (Baseline) |
|---|---|---|
| `lr_scale` | 0.01 – 0.2, log | 0.1 |
| `weight_decay` (AdamW) | 0.01 – 0.2, log | 0.1 |
| `lrd_vision_backbone` (layer-wise LR decay) | 0.6 – 1.0 | 0.9 |
| `scheduler_warmup` (optimiser steps) | 1 – 1000, log, integer | 2 |
| `hflip_p` | 0 – 0.5 | 0.5 |
| `resize_min_size` (scale-jitter lower bound, px) | 320 – 1008, step 16 | 480 |

The warm-up is counted in optimiser steps (one epoch = 1273 steps); the recipe's default of 2 steps is
effectively no warm-up, which is why the range spans up to ~0.8 epoch. All defaults lie inside the bounds, so
the first trial evaluates the Baseline configuration itself.

### 3.2 Sampler and reproducibility

The search uses Optuna's TPE sampler (Optuna 5.0.0) with SQLite storage. A **fresh TPE sampler is installed
before every proposal**, seeded from the global seed and the proposal index; TPE builds its model only from
completed trials, so proposal *i* is a pure function of the seed, *i* and the completed history. The first
proposal is fixed to the default hyperparameters. Every proposal is recorded, and a resumed or retried proposal
is **verified to be identical** before it is used.

**Start-up trials reduced to 5.** TPE draws random proposals until `n_startup_trials` trials have completed
(Optuna default: 10). With a budget of 10 trials, the default would leave no trial to TPE — the search would
reduce to the defaults plus 9 random draws. With 5 start-up trials the search consists of the default
configuration, 4 random proposals and **5 TPE-guided proposals**.

### 3.3 Fault tolerance (`hpo_state.json`)

* `hpo_state.json` (same schema as YOLO26's and the U-Net's) is checkpointed atomically before every trial; the
  search is complete when the number of completed trials reaches the target.
* A trial left running by a crash is marked as interrupted (not counted as a failure); the same proposal is
  asked again, receives identical parameters and the **same trial folder**, so the interrupted training itself
  resumes from its checkpoint.
* A trial that fails or returns a non-finite fitness is retried with the same parameters (from a clean folder) up
  to a maximum number of times, then recorded with fitness 0. A failure that coincides with an unhealthy GPU is
  not counted; the script exits with code 75 and the orchestrator retries after a waiting period.
* Resuming with a changed search space, base setup, seed, dataset or library version is refused (configuration
  hash); an exclusive lock prevents two processes from tuning the same model.
* **Verified:** a search killed in the middle of its second trial and resumed with the same command produced a
  `tune_results.csv` and `best_hyperparameters.yaml` identical to an uninterrupted search, and its
  `hpo_state.json` recorded the interruption and resumption.

Fitness = validation per-image mean JSI of the trial's best epoch (the same criterion that selects checkpoints in
every phase). Trials train on the train split and are scored on the validation split; the test split is never
used.

### 3.4 Budget (10 × 10)

| | YOLO26 / U-Net | SAM 3 |
|---|---|---|
| Phases 1, 2, 4 | 120 epochs, no early stopping | **30 epochs, no early stopping** (patience 30) |
| Phase 3 | 30 trials × 30 epochs, no early stopping | **10 trials × 10 epochs, no early stopping** (patience 10) |

One FP32 training epoch of SAM 3 on the full train split takes ≈ 106 min of optimiser steps (≈ 1.8 h including
validation and checkpointing) on a V100S, about two orders of magnitude more than the U-Net. At YOLO26's budget a
single 120-epoch run would take more than 8 days and the HPO several months. The reduced budget is a **disclosed
limitation**: it is justified by the compute cost and by the fact that SAM 3 starts from strong pretrained
weights, but it means SAM 3 receives fewer epochs and a smaller search than the specialised architectures.
Estimated worst-case cost (no early stop): Phase 1 ≈ 55 h, Phase 2 ≈ 5 × 47 h, Phase 3 ≈ 10 × 18 h,
Phase 4 ≈ 55 h — about three weeks on one GPU; Phases 2 and 3 are independent and can run on two GPUs.

---

## 4. Memory probe and computational cost

### 4.1 Memory probe

Full FP32 fine-tuning of an 840.5 M-parameter model (454.0 M vision backbone, 353.7 M text encoder, 21.0 M
detector transformer, 8.2 M geometry encoder, 2.3 M segmentation head, 1.2 M scoring) was profiled with the
official trainer and recipe on a Tesla V100S (32 GB), each configuration in a fresh process
(`sam3_seg/probe_memory.py`):

| Configuration | Result | Peak VRAM (allocated / reserved / device) | Step time (median) | Projected epoch |
|---|---|---|---|---|
| Batch 2, activation checkpointing **disabled** in the ViT and text encoder | **out of memory** at 30.9 GiB | — | — | — |
| **Batch 2, official activation checkpointing (used)** | fits | 12.8 / 15.8 / **16.5 GiB** | 5.01 s | ≈ 106 min |
| Same, strict deterministic algorithms | fits | 12.9 / 15.6 / 16.4 GiB | 6.67 s | ≈ 141 min |

Activation checkpointing discards intermediate activations in the forward pass and recomputes them during the
backward pass. It trades compute for memory and is **numerically neutral** (the recomputed activations are the
same values), so it changes neither the optimisation nor the results. Without it the ViT and text-encoder
activations exceed the 32 GB of the GPU; with it the official batch size of 2 runs at about half the GPU memory,
in FP32 and without freezing any component (in particular, the text encoder did not need to be frozen). SAM 3's
detector encoder and decoder require activation checkpointing during training (they assert it), so it can only be
disabled in the two backbones. Gradient accumulation (batch 1 × 2) was therefore not needed.

### 4.2 GFLOPs measured with `FlopCounterMode`

The computational cost of one batch-1 forward pass (1008 × 1008 test image, prompt `"skin lesion"`) was measured
natively with PyTorch's `torch.utils.flop_counter.FlopCounterMode`, which intercepts every ATen operation and
counts 2 × multiply-accumulates of matrix products and convolutions (the convention of Ultralytics/thop GFLOPs):

| Breakdown | GFLOPs per image | Share |
|---|---|---|
| **Total** | **6,057** | 100 % |
| Image encoder (ViT + neck) | 5,614 | 92.7 % |
| Detector (transformer, geometry encoder, segmentation head, scoring) | 424 | 7.0 % |
| Text encoder | 20 | 0.3 % |
| Linear layers (incl. Q/K/V/output projections and MLPs) | 4,747 | 78.4 % |
| **Attention products (QKᵀ and AV in scaled-dot-product attention)** | **956** | **15.8 %** |
| Convolutions | 354 | 5.8 % |

For comparison, the U-Net requires 6.4 GFLOPs at its native 256 × 256 input (40.0 at 640 × 640) — SAM 3 needs
about **950 times** more compute per image than the U-Net at their native inputs [add YOLO26 values from its
efficiency table]. Batch-1 latency on the V100S (preliminary, from the pipeline's smoke test): ≈ 590 ms per image
in FP32 (1.7 FPS) and ≈ 196 ms with FP16 autocast; 3.3 GB of VRAM for the weights [replace with final Phase 5
values].

**Why `thop` cannot be used for SAM 3.** `thop` (used by Ultralytics and by the U-Net pipeline) counts FLOPs with
forward hooks registered on known `nn.Module` types (convolutions, linear layers, normalisation). Attention in
modern PyTorch is computed by **functional calls** — `torch.nn.functional.scaled_dot_product_attention` and
explicit matrix products inside `forward` methods — which are not modules and are invisible to module hooks. For
convolutional networks the two tools agree, but for a transformer `thop` silently omits the attention products
(15.8 % of SAM 3's compute here) and any functional matmul. `FlopCounterMode` works at the dispatcher level and
counts them. Two measurement details were necessary:

* under `torch.inference_mode()` PyTorch bypasses Python dispatch modes and `FlopCounterMode` counts **nothing**
  (an initial measurement reported 0.3 GFLOPs); the count is therefore taken under `torch.no_grad()`;
* `nn.MultiheadAttention` (text encoder and detector) uses a fused inference fast path that has no FLOP formula;
  the fast path is disabled during counting so the attention inside those modules is counted as well.

**Interpretation for the article.** The 15.8 % share refers to the attention *products* only (the quadratic part,
QKᵀ and the attention-weighted sum of values). The Q/K/V and output projections of every attention block are
linear layers and fall in the 78.4 %; the cost of the attention mechanism as a whole is therefore larger than
15.8 %. Element-wise operations, normalisation and interpolation are counted by neither tool.

---

## 5. FP16 evaluation

FP16 results use SAM 3's official mixed-precision path (`torch.autocast` with float16, FP32 weights) rather than
converting the weights with `.half()` as for YOLO26 and the U-Net; the weight memory is therefore the FP32
footprint in both precisions. On the validation images used for testing, FP16 changed the mean JSI by +0.0006.

---

## 6. Points to verify or disclose before submission

1. **Reduced budget** (30 epochs; HPO 10 trials × 10 epochs; no early stopping) versus 120 epochs and 30 × 30 for
   YOLO26 / U-Net (Section 3.4).
2. **No early stopping.** The first YOLO26 and U-Net runs showed that early stopping on the 100-image validation
   split is noise-driven (YOLO26: median epoch-to-epoch fitness change 0.04–0.06 versus 0.014 on a 530-image fold;
   every Phase 1 run stopped at epochs 50–82 of 120; U-Net: Baseline and Optimised stopped at epoch 81 while CV
   folds kept improving to epochs 96–118). All three pipelines therefore train for the full budget (patience =
   epochs, also in HPO trials); `best.pt` is still the epoch with the best validation JSI.
3. **Reproducibility statement:** deterministic except the ViT attention backward; repeated or resumed runs differ
   by ≤ 7 × 10⁻⁵ in the weights (5th decimal of the validation JSI); bit-exactness was verified only in strict mode
   (Section 2.2).
4. **Prompt** `"skin lesion"` and the top-1 rule (score ≥ 0.5; union for model selection) define the SAM 3 prediction; SAM 3 is
   an instance/concept model evaluated as a semantic segmenter.
5. **FP16 = autocast** for SAM 3 versus `.half()` for YOLO26 / U-Net (Section 5).
6. **GFLOPs counters differ:** `FlopCounterMode` for SAM 3, thop for YOLO26 / U-Net; equivalent for convolutional
   networks, but state it in the efficiency table.
7. **[Final numbers]:** Phase 1–5 results, latency, VRAM, and the YOLO26 / U-Net efficiency values for the
   cross-architecture comparison.
