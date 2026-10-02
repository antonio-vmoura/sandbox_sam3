# SAM 3 — Skin Lesion Segmentation (ISIC 2018 Task 1)

This repository contains the SAM 3 arm of the study that compares segmentation architectures (YOLO26-seg,
U-Net, SAM 3) on ISIC 2018 Task 1. SAM 3 is fine-tuned with **Meta's official trainer and recipe** (vendored in
`sam3/`, unmodified by the pipeline) and run through the **same 5-phase protocol, metrics, evaluation
resolution, efficiency profiling and output schema as `sandbox_yolo26` and `sandbox_unet`**, so the three
pipelines can be compared directly and analysed with the same notebooks.

The pipeline lives in **`sam3_seg/`**; the protocol is defined once in **`sam3_seg/common.py`** and everything is
orchestrated by **`run_pipeline_sam3.sh`**.

---

## What makes the comparison fair

| Aspect | How it is guaranteed |
|---|---|
| **Same data** | Phase 0 builds a COCO dataset **from the YOLO26 dataset itself**: same images (hard-linked), same split (2,594 / 100 / 1,000, the official split), each YOLO polygon rasterised with the shared convention and stored as RLE. Every image carries its ISIC ID. |
| **Same CV folds** | YOLO26's K-Fold algorithm (NumPy `RandomState(0)`) over the same pool order → **identical folds** (verified against the U-Net manifest; Phase 2 re-derives and checks them). |
| **Same metrics, ground truth and resolution** | `sam3_seg/segmentation_metrics.py` is a **byte-identical copy** of YOLO26's. The predicted mask (union of the instances with score ≥ 0.5) is scored at the original dataset resolution against the ground truth of the same polygons. |
| **Same selection criterion as the U-Net** | Every phase selects `best.pt` on the validation **per-image mean JSI** (no early stopping: patience = epochs), computed with that same code (YOLO26 selects on the Ultralytics fitness, box + mask mAP50-95). Phase 5 runs the **identical** validation pipeline on the test set (verified: it reproduces the trainer's validation JSI to the last digit). |
| **Same profiling** | `benchmark_efficiency.py` is derived from YOLO26's: `torch.cuda.Event` timing, statistics, steady-state VRAM, contention checks and JSON schema; same `torch==2.5.1`. |

## SAM 3-specific protocol decisions (disclosed in the report)

| Decision | Value | Why |
|---|---|---|
| Model | Official SAM 3 image model, **all 840.5 M parameters trainable** (text encoder not frozen) | full fine-tuning, as Meta's recipe |
| Prompt | `"skin lesion"` (single COCO category) | clinically neutral (most ISIC lesions are benign) |
| Input | 1008 × 1008 (native) | fixed by the architecture |
| Precision / batch | FP32, batch 2, official activation checkpointing | memory probe below |
| **Budget, Phases 1, 2, 4** | **30 epochs, no early stopping** (patience 30; YOLO26 / U-Net: 120 epochs, no early stopping) | one FP32 epoch ≈ 1.8 h on a V100S; early stopping on the 100-image validation split proved noise-driven in YOLO26 and the U-Net |
| **HPO** | **10 trials × 10 epochs, no early stopping** (patience 10; YOLO26 / U-Net: 30 × 30); Optuna TPE with 5 start-up trials | compute; a foundation model starts from strong weights |
| Search space | `lr_scale` [0.01, 0.2] log, `weight_decay` [0.01, 0.2] log, `lrd_vision_backbone` [0.6, 1.0], `scheduler_warmup` [1, 1000] steps log, `hflip_p` [0, 0.5], `resize_min_size` [320, 1008] step 16 | learning dynamics + the recipe's own augmentations only (no loss weights, no `focal_gamma`) |
| Determinism | seeds, cuDNN deterministic, **deterministic `grid_sample`** (`sam3_seg/determinism.py`), warn-only deterministic algorithms | the ViT memory-efficient attention backward stays nondeterministic: PyTorch's strict mode would cost +33 % time (verified bit-exact, `strict_determinism` in `common.py`). Repeated / resumed runs differ by ≤ ~1e-4 in the weights (5th digit of val JSI). |
| FP16 (Phase 5) | `torch.autocast(float16)`, FP32 weights | SAM 3's official mixed-precision path |

**Memory / throughput probe** (`sam3_seg/probe_memory.py`, V100S 32 GB, FP32, official recipe):

| Configuration | Result | Peak VRAM (alloc / device) | Step (median) | Epoch (2594 images) |
|---|---|---|---|---|
| batch 2, activation checkpointing **off** (ViT + text encoder) | OOM at 30.9 GiB | — | — | — |
| **batch 2, official activation checkpointing (used)** | fits | 12.8 / 16.5 GiB | 5.0 s | ≈ 106 min |
| same, strict deterministic algorithms | fits | 12.9 / 16.4 GiB | 6.7 s | ≈ 141 min |

A finished run deletes its 9.4 GB resume checkpoint (`checkpoint.pt`, weights + AdamW state); HPO trials also
delete their weights and keep only metrics.

---

## The 5-phase protocol

| Phase | What | Data | Script(s) |
|---|---|---|---|
| **0 — Dataset** | COCO dataset from the YOLO26 dataset; per-split and per-fold annotation files; SHA-256 provenance | train / val / test | `prepare_dataset.py` |
| **1 — Baseline** | Base setup + **official recipe** hyperparameters | train / val | `train_baseline_sam3.py` |
| **2 — Baseline CV** | 5-fold CV with the Phase 1 configuration; pixel metrics per fold (dataset resolution) | train ∪ val pool (**test excluded and verified**) | `train_cv_sam3.py`, `consolidate_cv_results_sam3.py` |
| **3 — HPO** | Optuna TPE, **seeded per proposal**, fault-tolerant (`hpo_state.json`, exit 75 → retried) | train / val | `tune_sam3.py`, `check_hpo_validity.py` |
| **4 — Optimised** | Same base setup + Phase 3 hyperparameters | train / val | `train_optimized_sam3.py` |
| **5 — Test set** | Baseline **and** Optimised: DSC, JSI, ISIC thresholded JSI, sensitivity, specificity, Boundary IoU, NSD, HD95 with bootstrap 95 % CI (FP32 + FP16); batch-1 efficiency — median/P95 latency, FPS, peak VRAM (FP32 + FP16); final report | **test** (only here) | `evaluate_test_set.py`, `benchmark_efficiency.py`, `build_final_report.py` |

**How the official trainer is wrapped.** `ProtocolTrainer` (`sam3_seg/protocol_trainer.py`) subclasses
`sam3.train.trainer.Trainer` and only replaces the epoch loop: train one epoch (official) → validate (official
`val_epoch` + the study's pixel metrics on its dumped predictions) → early stopping / `best.pt` on val JSI →
checkpoint **after** validation, including every RNG state and the early-stopping history. Configs are generated
per run from a frozen copy of the recipe (`sam3_seg/configs/sam3_base_recipe.yaml`) and written to the run
directory, never into the source tree. Each run is a separate process (`run_training.py`), so a crash or OOM
cannot take the HPO driver down.

**Fault tolerance.** Every step is idempotent: re-running the same command skips completed runs and resumes an
interrupted one from the end of its last epoch (exact RNG / optimiser / early-stopping state). Verified by
killing a training mid-epoch and an HPO search mid-trial: the resumed HPO produced identical
`tune_results.csv` / `best_hyperparameters.yaml`. A changed protocol, dataset or library version is refused
(config hashes); `--force` moves old outputs to `*.bak-<UTC>`.

---

## ISIC 2018 Task 2 (lesion attributes) — Phase 0

`prepare_dataset.py --task 2` builds a COCO dataset with **five categories whose names are the text prompts**
("pigment network", "negative network", "streaks", "milia-like cyst", "globules") instead of the single
"skin lesion" prompt; every attribute present in an image is one instance (its official mask as RLE), so overlapping
attributes coexist and images without attributes have no instance. Input: the Task 2 YOLO26 dataset (built with
YOLO26's `prepare_dataset.py --task 2`; mount it at `/workspace/yolo26_dataset_task2`); output:
`datasets/isic_2018_task2_sam3`; same 2,594 / 100 / 1,000 images and CV folds.

```bash
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$(pwd)/datasets:/workspace/datasets" -v "$(pwd)/sam3_seg:/workspace/sam3_seg" \
    -v "$(pwd)/../sandbox_yolo26/datasets/isic2018_task2_official:/workspace/yolo26_dataset_task2:ro" \
    -w /workspace/sam3_seg --entrypoint python sam3_ft prepare_dataset.py --task 2
```

`--task 1` is the default everywhere (the orchestrator and `wait_gpu_sam3.sh` run Task 1); **Phases 1–5 currently
implement Task 1 only** (the training recipe, prediction rule and metrics assume one prompt and one binary mask).

## Running the pipeline

### Build the image

```bash
docker build -t sam3_ft .
```

The SAM 3 checkpoint is gated on Hugging Face (`facebook/sam3`): download it once into `sam3_cache/huggingface`
with your token (`export HUGGING_FACE_HUB_TOKEN=...`); the pipeline then runs with `HF_HUB_OFFLINE=1`.

### Full run (all phases)

```bash
GPU=1                                  # host GPU index
PIPELINE_NAME="pipeline_final_v1"

mkdir -p "logs/${PIPELINE_NAME}"     # the terminal log goes inside the pipeline folder
docker run --gpus "\"device=${GPU}\"" -it --rm --ipc=host \
    --user "$(id -u):$(id -g)" \
    -e HF_HOME=/workspace/cache/huggingface -e HF_HUB_OFFLINE=1 -e HOME=/workspace/cache \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e GPU_DEVICE=0 -e PIPELINE_NAME="${PIPELINE_NAME}" \
    -v "$(pwd)/sam3:/workspace/sam3" -v "$(pwd)/sam3_seg:/workspace/sam3_seg" \
    -v "$(pwd)/sam3_cache:/workspace/cache" -v "$(pwd)/datasets:/workspace/datasets" \
    -v "$(pwd)/logs:/workspace/logs" \
    -v "$(pwd)/run_pipeline_sam3.sh:/workspace/run_pipeline_sam3.sh:ro" \
    -v "$(pwd)/../sandbox_yolo26/datasets/isic2018_task1_official:/workspace/yolo26_dataset:ro" \
    -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \
    --entrypoint bash sam3_ft \
    /workspace/run_pipeline_sam3.sh --yolo-data /workspace/yolo26_dataset/data.yaml \
    2>&1 | tee "logs/${PIPELINE_NAME}/terminal_$(date -u +%Y%m%dT%H%M%SZ).log"
```

* The YOLO26 dataset is mounted **read-only** (it is the source of truth); Phase 0 writes
  `datasets/isic_2018_task1_sam3/`. Run on the host (outside Docker) Phase 0 hard-links the images; across
  container mounts it copies them.
* `/etc/passwd` and `/etc/group` are mounted because PyTorch's compile cache needs the user to resolve.
* Inside the container the selected GPU is index `0` (hence `GPU_DEVICE=0`).
* If the run is interrupted for any reason, **run the same command again** — it resumes.

### Common variations (arguments after `run_pipeline_sam3.sh`)

```bash
--phases "3 4 5"                # a subset of phases
--dry-run                       # print the commands only
--epochs 3 --patience 2         # SMOKE TEST ONLY (applied to Phases 1, 2 and 4 together)
--bench-device 0                # GPU used for Phase 5 (default: --device)
--pipeline-name pipeline_final_v2   # a fresh, isolated study
--force                         # start the selected phases over (old outputs → *.bak-<UTC>)
```

Environment overrides (defaults): `CV_K_FOLDS=5`, `CV_SEED=0`, `HPO_ITERATIONS=10`, `HPO_EPOCHS_PER_TRIAL=10`,
`HPO_PATIENCE=10`, `HPO_MAX_RETRIES=5`, `HPO_RETRY_WAIT=600`, `EVAL_PRECISIONS="fp32 fp16"`, `YOLO_DATA_YAML`,
`DATA_DIR`, `LOGS_ROOT`, `PROJECT`.

Exit codes: `0` success · `75` the HPO gave up after repeated GPU failures (fix the driver and re-run to
resume) · otherwise the exit code of the failing step (log in `pipeline_runs/<UTC>/`).

**Compute budget (worst case, no early stop, V100S FP32):** Phase 1 ≈ 55 h, Phase 2 ≈ 5 × 47 h, Phase 3 ≈
10 × 18 h, Phase 4 ≈ 55 h — about 3 weeks on one GPU. Phases 2 and 3 are independent: running them on two GPUs
(two containers, `--phases 2` and `--phases 3`, same `PIPELINE_NAME`) saves about a week.

### Waiting for an idle GPU

```bash
GPU_DEVICE=1 ./wait_gpu_sam3.sh     # polls nvidia-smi, then launches the docker command above
GPU_DEVICE=1 ./wait_gpu_sam3.sh --phases "1 2"   # extra arguments go to run_pipeline_sam3.sh
```

---

## Phase 5 — what exactly is measured

**Accuracy (`evaluate_test_set.py`)** — test split only, batch 1, FP32 (primary) and FP16. Each run's own
validation pipeline is rebuilt from its `config.yaml` (official transforms and postprocessor, prompt
`"skin lesion"`), applied to `annotations/test.json`; per image the union of the instances with score ≥ 0.5 is
scored at dataset resolution: DSC, JSI, ISIC thresholded JSI (`JSI < 0.65 → 0`), sensitivity, specificity, accuracy (empty
prediction → 0, never skipped), Boundary IoU, NSD and HD95 (as YOLO26). Same aggregates and JSON/CSV schema as YOLO26; the Ultralytics-only instance
metrics are present as `NaN` (the CV and validation tables carry SAM 3's official COCO mAP50-95 in
`map5095_m` / `map5095_b`).

**Efficiency (`benchmark_efficiency.py`)** — batch 1, one GPU, each configuration in a fresh process:

* `forward`: image encoder + text encoder + detector on a pre-processed real test image (1×3×1008×1008) with
  `torch.cuda.Event` (50 warm-up + 500 timed); `forward_cached_text`: the same with the prompt's text features
  precomputed (fixed-prompt deployment); `end_to_end`: Meta's deployment API `Sam3Processor` (test image at
  dataset resolution → resize → forward → mask upsampling → threshold → host) with `perf_counter` (20 + 200);
* mean, SD, median, P90/P95/P99, FPS = 1000 / mean; steady-state peak VRAM after warm-up, weight VRAM, host RAM;
* **Driver-level VRAM**: `vram_process_peak_mb` = device memory held by the benchmark process at the end of the
  forward / end-to-end loops (CUDA context, kernels and allocator cache included; `nvidia-smi` delta) and
  `vram_cuda_context_mb` — the memory a deployment GPU must provide, next to the allocator peak (the model).
* **`end_to_end_dataset`**: the end-to-end pipeline once on each of the first 100 test images sorted by ISIC ID
  (the same images in the three repositories; `--e2e-images`), after one untimed pass — median/P95 over real,
  varying inputs. The real-time criterion in the notebooks uses its P95.
* parameters — 840,509,750 in total: vision backbone 454.0 M, text encoder 353.7 M, transformer 21.0 M,
  geometry encoder 8.2 M, segmentation head 2.3 M, scoring 1.2 M (`params_without_text` = 486.8 M);
* **GFLOPs measured natively with `torch.utils.flop_counter.FlopCounterMode`** (2 × MACs, the YOLO26 / thop
  convention; unlike thop it sees `scaled_dot_product_attention` and functional matmuls) — ≈ 6,057 GFLOPs per
  image: image encoder 5,614, detector 424, text encoder 20; by operation: linear layers 4,747, **attention
  products (QKᵀ, AV) 956 (15.8 %)**, convolutions 354. SAM 3 resizes every input to 1008, so `gflops_640` =
  `gflops`.

  For reference (same GPU, smoke test): FP32 forward ≈ 590 ms (1.7 FPS), FP16 autocast ≈ 196 ms (5.1 FPS);
  weights 3.3 GB VRAM, forward peak 4.1 GB.

* contended runs (another process using the GPU) are flagged — re-run them on an idle GPU.

Note for the comparison: YOLO26 and the U-Net report thop GFLOPs. For convolutional networks thop and
`FlopCounterMode` agree on convolutions and linear layers; thop cannot count SAM 3's attention products, which
is why SAM 3 uses `FlopCounterMode`.

---

## Output layout

```
logs/pipeline_final_v1/
├── phase1_baseline/sam3_baseline/          config.yaml, checkpoints/best.pt, results.csv, run_state.json,
│                                           protocol_final.json, train.log, dumps/, logs/, tensorboard/
├── phase2_cv_baseline/sam3/                splits_manifest.json, runs/fold_<k>/, metrics_per_fold.csv,
│                                           metrics_summary.json
├── phase3_hpo/tune_sam3/                   tune_results.csv, best_hyperparameters.yaml, hpo_state.json,
│                                           optuna_study.db, trials/trial_<i>/ (metrics only)
├── phase4_optimized/sam3_optimized/        as Phase 1 + tuned_hyperparameters.yaml
├── phase5_test/{accuracy, per_image, masks, efficiency}/
├── summary/                                phase1_val, phase2_cv_baseline, phase2_cv_pixel, phase4_val,
│                                           test_accuracy, efficiency, hpo_gain, final_results (.csv/.json)
├── figures/  tables/                       written by the notebooks
└── pipeline_runs/<UTC>/                    pipeline.log + one log per step
```

The `summary/` files have **the same columns as YOLO26's** (SAM 3 adds columns, e.g. the cached-text latency,
per-component parameters/GFLOPs and `attention_flop_share`); `final_results.json` lists the protocol deviations
under `protocol_notes`.

## Project structure

```
sandbox_sam3/
├── run_pipeline_sam3.sh   wait_gpu_sam3.sh   Dockerfile
├── sam3/                      # Meta's SAM 3 code (vendored; trainer, model, data, eval)
├── sam3_seg/
│   ├── common.py              # base setup, default HPs, budgets, protocols, paths, locks
│   ├── prepare_dataset.py     # Phase 0
│   ├── data.py                # verified access to the Phase 0 dataset
│   ├── configs/sam3_base_recipe.yaml   # frozen official recipe
│   ├── protocol_trainer.py    # official trainer + val-JSI selection, early stopping, RNG-complete checkpoints
│   ├── run_training.py  training.py   # one run per process; resumable run lifecycle
│   ├── determinism.py         # deterministic grid_sample
│   ├── inference.py           # the validation pipeline applied to the test set
│   ├── segmentation_metrics.py  # byte-identical to YOLO26's
│   ├── probe_memory.py        # memory / throughput probe
│   ├── train_baseline_sam3.py  train_cv_sam3.py  consolidate_cv_results_sam3.py
│   ├── tune_sam3.py  check_hpo_validity.py  train_optimized_sam3.py  collect_phase_metrics_sam3.py
│   ├── evaluate_test_set.py  benchmark_efficiency.py  build_final_report.py
│   └── legacy/                # previous pipeline scripts (not used)
├── notebooks/
│   ├── 01_Segmentation_Visualizer.ipynb
│   └── 02_Metrics_and_Efficiency_Analysis.ipynb
├── utils/examples/            # Meta's SAM 3 example notebooks
├── utils/legacy/              # earlier conversion / metric scripts (kept as a backup)
├── notebooks/legacy/          # earlier analysis notebooks (kept as a backup)
├── sam3/train/configs/custom/legacy/   # earlier training / HPO / CV configs
└── datasets/  logs/  sam3_cache/   # not versioned
```

Meta's original README is kept as `README_OFC.md`; `README_TRAIN.md` documents the official training code.

## Analysis notebooks

Same notebooks as YOLO26 and the U-Net, adapted to SAM 3 (they read only the pipeline outputs; no GPU needed):
`01_Segmentation_Visualizer` (ground truth green/solid vs. prediction red/dashed, Baseline vs. Optimised) and
`02_Metrics_and_Efficiency_Analysis` (DSC/JSI across phases, paired HPO gain, accuracy vs. size, latency vs.
FPS, latency distribution, memory, accuracy–latency trade-off, **compute breakdown** — stage and attention vs.
linear vs. convolution — and a **cross-architecture comparison** read from `../sandbox_yolo26` and
`../sandbox_unet` when their summaries exist; LaTeX tables; standard figures A–C shared with YOLO26 and the U-Net).
The full cross-architecture article notebook is in `analysis/` (see `analysis/README.md`).

```bash
docker run --rm -it -p 8888:8888 --user "$(id -u):$(id -g)" -e HOME=/workspace/cache \
    -v "$(pwd)/..:/projects" -w /projects/sandbox_sam3 \
    sam3_ft jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --notebook-dir=/projects/sandbox_sam3
```

---

## Running on a remote server

```bash
screen -S sam3_ft        # start; run the docker command above
# Ctrl + A, then D       # detach
screen -r sam3_ft        # reattach
```

Copy the results:

```bash
rsync -avz --progress -e "ssh -p 13508" \
    antoniovinicius@164.41.75.221:/home/antoniovinicius/projects/sandbox_sam3/logs/pipeline_final_v1 \
    /home/avmoura_linux/Documents/unb/SANDBOX_SAM3/logs/
```

Hardware monitoring: `nvidia-smi`, `nvtop`.
