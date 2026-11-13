# pipeline_final_v1 — SAM 3: fold manifests, dataset manifests and result files

This folder holds the fold manifests, the dataset split manifests and the result files of the pipeline_final_v1 experiments for the SAM 3 arm. The experiments use the official ISIC 2018 Task 1 partition (2,594 training, 100 validation and 1,000 test images). Five-fold cross-validation runs on the 2,694-image pool (training + validation), with seed 0. The five phases are 1 Baseline, 2 cross-validation, 3 hyperparameter optimisation (HPO), 4 Optimized retraining and 5 test evaluation and efficiency benchmark. They are run by `run_pipeline_sam3.sh` with the code in `sam3_seg/`. The other two arms (`sandbox_unet`, `sandbox_yolo26`) have a folder with the same layout and the same `folds/folds.csv`.

Model name(s) in file names: `sam3`.

Every file except `README.md`, `CHECKSUMS.sha256` and `folds/folds.csv` is a copy of a file the pipeline wrote under `logs/pipeline_final_v1/` (not versioned) or under `datasets/isic_2018_task1_sam3/` (Phase 0). The only change to the copies is the path mapping described in [Path mapping](#path-mapping), applied to text files that contained container paths.

## Layout

```
pipeline_final_v1/
├── README.md
├── CHECKSUMS.sha256             SHA-256 of every other file of this folder
├── folds/                       cross-validation fold manifests
├── dataset/                     split manifests of the prepared dataset (no image, no mask)
└── results/
    ├── phase1_baseline/         Phase 1: training curves and run state; per-image validation scores
    ├── phase2_cv/               Phase 2: per-fold metrics, fold training curves
    ├── phase3_hpo/              Phase 3: trial history, best hyperparameters, trial curves
    ├── phase4_optimized/        Phase 4: training curves, run state, hyperparameters used
    ├── phase5_test/             Phase 5: per-image test scores and accuracy JSON per variant/precision
    ├── efficiency/              Phase 5: efficiency benchmark JSON per variant/precision
    └── summary/                 summary tables written by build_final_report.py and the collect steps
```

Placeholders used below: `<k>` fold 0-4; `<variant>` `baseline` or `optimized`; `<precision>` `fp32` or `fp16`; `<NNN>` trial number 000-009.

## folds/

| File | Content | Produced by |
|---|---|---|
| `folds/splits_manifest.json` | Fold fingerprint written by the Phase 2 training code: `k`, `seed`, `pool_size`, `test_size_excluded`, `pool_sha256` and, per fold, `n_train`, `n_val`, `val_sha256` (SHA-256 of the ordered validation list). It contains no image ids. Source: `logs/pipeline_final_v1/phase2_cv_baseline/sam3/splits_manifest.json` | `sam3_seg/train_cv_sam3.py` (Phase 2) |
| `folds/folds.csv` | One row per (fold, image): columns `fold` (0-4), `role` (`train` or `val`), `image_id` (ISIC id). 5 x 2,694 = 13,470 rows. Every image of the pool is `val` in exactly one fold | see below |

`splits_manifest.json` stores hashes, not ids. So `folds/folds.csv` was **regenerated** with this repository's own code: `verified_folds` and `build_splits_manifest` of `sam3_seg/train_cv_sam3.py` (which also checks the Phase 0 fold files `datasets/isic_2018_task1_sam3/folds/fold_<k>/{train,val}.json`), with `k` = 5 and `seed` = 0 read from the manifest.

`folds/folds.csv` is byte-identical in the three repositories. The fold algorithm is the same in all three: `numpy.random.RandomState(0)` shuffle of the pool (training ids followed by validation ids, in the dataset's sorted order), then five contiguous blocks, the first four with 539 images and the last with 538.

## dataset/

Copied from `datasets/isic_2018_task1_sam3/` (Phase 0, `sam3_seg/prepare_dataset.py`). These files contain no image or mask.

| File | Content |
|---|---|
| `dataset/manifest_train.csv`, `dataset/manifest_val.csv`, `dataset/manifest_test.csv` | One row per image of the split |
| `dataset/meta.json` | Preparation parameters, fold sizes and SHA-256 of the annotation and fold files |

Columns: `index` row; `id` ISIC id; `file_name` image file name; `image`, `label` paths in the YOLO26 dataset; `width`, `height` size of the image in the YOLO26 dataset, i.e. the dataset resolution (long side at most 1,024 px). This is not the size in the official release; that size is in `sandbox_yolo26`'s manifest (`orig_w`, `orig_h`). `n_instances` lesion instances in the COCO annotation (1). The split is the file name.

## results/

Sources are under `logs/pipeline_final_v1/` of the run. Training-curve CSVs (`results.csv`) have one row per epoch.

### results/phase1_baseline/ (Phase 1)

| File | Source | Produced by |
|---|---|---|
| `results/phase1_baseline/sam3/results.csv` | `phase1_baseline/sam3_baseline/results.csv` | `sam3_seg/train_baseline_sam3.py` |
| `results/phase1_baseline/sam3/run_state.json` | same folder | `sam3_seg/train_baseline_sam3.py`: status, best epoch and its metrics, protocol, events |
| `results/phase1_baseline/sam3/config.yaml` | same folder | `sam3_seg/training.py`: the trainer configuration of the run |
| `results/phase1_baseline/sam3/protocol_final.json` | same folder | `sam3_seg/protocol_trainer.py`: best epoch, its selection value, epochs trained |
| `results/phase1_baseline/sam3/val_per_image.csv`, `results/phase1_baseline/sam3/val_per_image_meta.json` | `phase1_val_masks/sam3/per_image.csv`, `meta.json` | `sam3_seg/export_phase1_val_masks.py --source best` (run outside `run_pipeline_sam3.sh`) |

`val_per_image.csv` scores the Phase 1 `best.pt` on the 100 official validation images with the Phase 5 rule. Its columns are the per-image columns below, with `id` (ISIC id) added first. `val_per_image_meta.json` records the weights' SHA-256, the prediction rule and the means of `dsc` and `jsi`.

### results/phase2_cv/ (Phase 2)

| File | Source | Produced by |
|---|---|---|
| `results/phase2_cv/sam3/pixel_metrics_per_fold.csv` | `phase2_cv_baseline/sam3/` | `sam3_seg/evaluate_cv_pixels.py` |
| `results/phase2_cv/sam3/pixel_metrics_summary.json` | `phase2_cv_baseline/sam3/` | `sam3_seg/evaluate_cv_pixels.py`: per-fold values, mean and sample SD (ddof = 1), weights' SHA-256 in `settings_hash` |
| `results/phase2_cv/sam3/metrics_per_fold.csv` | `phase2_cv_baseline/sam3/` | `sam3_seg/train_cv_sam3.py`: metrics of each fold's training run at its best epoch |
| `results/phase2_cv/sam3/metrics_summary.json` | `phase2_cv_baseline/sam3/` | `sam3_seg/train_cv_sam3.py`: same, with mean and SD over folds |
| `results/phase2_cv/sam3/fold_<k>/results.csv`, `results/phase2_cv/sam3/fold_<k>/run_state.json` | `phase2_cv_baseline/sam3/runs/fold_<k>/` | `sam3_seg/train_cv_sam3.py` |
| `results/phase2_cv/sam3/fold_<k>/config.yaml`, `results/phase2_cv/sam3/fold_<k>/protocol_final.json` | `phase2_cv_baseline/sam3/runs/fold_<k>/` | `sam3_seg/training.py`, `sam3_seg/protocol_trainer.py` |

**Per-image cross-validation scores are not part of this package.** The pipeline did not write them. `sam3_seg/evaluate_cv_pixels.py` scores every held-out image of fold *k* with that fold's `best.pt`, using the Phase 5 rule. It keeps those per-image rows in memory and writes only the per-fold aggregates `pixel_metrics_per_fold.csv` and `pixel_metrics_summary.json`. These two files are exactly what the pipeline wrote. To regenerate per-image scores, run that same script on the fold weights (`phase2_cv_baseline/sam3/runs/fold_<k>/checkpoints/best.pt`, not included). The rows are produced inside `evaluate_model()` before aggregation; saving them to a file needs a change to the script.

**Aggregation rule.** In `pixel_metrics_per_fold.csv`, each fold's value is the mean of the non-NaN per-image scores of that fold's held-out images (`segmentation_metrics.aggregate_scores`). `n_images` is the fold size; the five sizes add up to 2,694. The cross-validation value in `pixel_metrics_summary.json` is `statistics.mean` over the five fold values: a plain mean of the fold means, with no weighting by fold size and no pooling over images. `build_final_report.py` copies this value unchanged into `results/summary/phase2_cv_pixel.csv` and `results/summary/final_results.*`. The SD is `statistics.stdev` over the folds (ddof = 1). The per-fold values are stored at full double precision (Python `repr`, as written by `csv.DictWriter`).

### results/phase3_hpo/ (Phase 3)

| File | Source | Produced by |
|---|---|---|
| `results/phase3_hpo/sam3/tune_results.csv` | `phase3_hpo/tune_sam3/` | `sam3_seg/tune_sam3.py`: one row per completed trial, in trial order |
| `results/phase3_hpo/sam3/best_hyperparameters.yaml` | `phase3_hpo/tune_sam3/` | `sam3_seg/tune_sam3.py`: hyperparameters of the best trial |
| `results/phase3_hpo/sam3/hpo_state.json` | `phase3_hpo/tune_sam3/` | `sam3_seg/tune_sam3.py`: checkpoint of the search (configuration, history, best trial, failures, proposals) |
| `results/phase3_hpo/sam3/optuna_study.db` | `phase3_hpo/tune_sam3/` | Optuna (SQLite storage of the study: trials, parameters, distributions, values) |
| `results/phase3_hpo/sam3/trials/trial_<NNN>/results.csv`, `results/phase3_hpo/sam3/trials/trial_<NNN>/run_state.json`, `results/phase3_hpo/sam3/trials/trial_<NNN>/config.yaml`, `results/phase3_hpo/sam3/trials/trial_<NNN>/protocol_final.json` | `phase3_hpo/tune_sam3/trials/trial_<NNN>/` | the training run of each trial |

Search: Optuna TPE, 10 trials x 10 epochs (5 TPE start-up trials); the first trial uses the Baseline hyperparameters. `fitness` = validation `val_jsi` of the trial's best epoch. The other columns of `tune_results.csv` are the hyperparameters of the trial.

### results/phase4_optimized/ (Phase 4)

| File | Source | Produced by |
|---|---|---|
| `results/phase4_optimized/sam3/results.csv`, `results/phase4_optimized/sam3/run_state.json` | `phase4_optimized/sam3_optimized/` | `sam3_seg/train_optimized_sam3.py` |
| `results/phase4_optimized/sam3/tuned_hyperparameters.yaml` | same folder | `sam3_seg/train_optimized_sam3.py`: the Phase 3 hyperparameters used |
| `results/phase4_optimized/sam3/config.yaml`, `results/phase4_optimized/sam3/protocol_final.json` | same folder | `sam3_seg/training.py`, `sam3_seg/protocol_trainer.py` |

### results/phase5_test/ (Phase 5, accuracy)

| File | Source | Produced by |
|---|---|---|
| `results/phase5_test/per_image/<variant>_sam3_<precision>.csv` | `phase5_test/per_image/` | `sam3_seg/evaluate_test_set.py`: one row per test image (1,000 rows) |
| `results/phase5_test/accuracy/<variant>_sam3_<precision>.json` | `phase5_test/accuracy/` | `sam3_seg/evaluate_test_set.py`: aggregates (mean, SD, median, IQR, seeded bootstrap 95 % CI, pooled DSC/JSI), instance metrics (NaN: not defined for this architecture), SHA-256 of the weights and of the test-id list |

Every `<variant>` x `<precision>` combination is present: Baseline and Optimized, FP32 and FP16. The aggregate means in `test_accuracy.csv` are the means of the non-NaN values of these per-image columns.

### results/efficiency/ (Phase 5, efficiency)

| File | Source | Produced by |
|---|---|---|
| `results/efficiency/<variant>_sam3_<precision>.json` | `phase5_test/efficiency/` | `sam3_seg/benchmark_efficiency.py`: batch-1 latency of each scope with its raw per-iteration timings, memory, model size, parameters, GFLOPs, environment (`env`), GPU state before and after (`gpu_before`, `gpu_after`) |

**The `contended` flag.** These JSON files are copied unchanged, and their `contended` field (top level and inside `gpu_before` / `gpu_after`) is the one stored when the benchmark ran. It came from an earlier rule based on GPU utilisation. `build_final_report.py` does not use that stored field. It recomputes the flag with the current rule, `CONTENTION_MEM_MIB` = 100: a run is contended when `gpu_before.mem_used_mb` or `gpu_after.mem_used_mb` (memory held on the GPU by other processes) exceeds 100 MiB. That recomputed value is the `contended` column of `results/summary/efficiency.csv`. So the stored field and the summary column can disagree for the same run; the summary column is the one the current code produces.

### results/summary/

| File | Produced by |
|---|---|
| `results/summary/phase1_val.csv`, `results/summary/phase1_val.json` | `sam3_seg/collect_phase_metrics_sam3.py --phase phase1`: validation metrics of each Phase 1 run at its best epoch |
| `results/summary/phase4_val.csv`, `results/summary/phase4_val.json` | `sam3_seg/collect_phase_metrics_sam3.py --phase phase4`: same for Phase 4 |
| `results/summary/phase2_cv_baseline.csv`, `results/summary/phase2_cv_baseline.json` | `sam3_seg/consolidate_cv_results_sam3.py`: mean and SD over folds of the training-time metrics (from `metrics_summary.json`) |
| `results/summary/phase2_cv_pixel.csv` | `sam3_seg/build_final_report.py`: cross-validation pixel metrics (from `pixel_metrics_summary.json`) |
| `results/summary/test_accuracy.csv` | `sam3_seg/build_final_report.py`: one row per variant x model x precision |
| `results/summary/hpo_gain.csv` | `sam3_seg/build_final_report.py`: Optimized minus Baseline on the test set (FP32), per model |
| `results/summary/efficiency.csv` | `sam3_seg/build_final_report.py`: one row per variant x model x precision |
| `results/summary/training_cost.csv` | `sam3_seg/build_final_report.py`: epochs and wall-clock training time per phase |
| `results/summary/final_results.csv` | `sam3_seg/build_final_report.py`: every value of the tables above, in long format |
| `results/summary/final_results.json` | `sam3_seg/build_final_report.py`: the same tables nested (`tables`), HPO bookkeeping (`hpo`), `warnings`, `protocol_notes` |

## CSV columns

### Per-image scores (`results/phase5_test/per_image/*.csv`, `val_per_image.csv`)

| Column | Meaning |
|---|---|
| `image` | Path of the image file used for the evaluation (mapped as in [Path mapping](#path-mapping)); its file stem is the ISIC id |
| `height`, `width` | Image size in pixels at the dataset (working) resolution, the resolution of the evaluation |
| `n_pred` | Instances with score >= 0.001 |
| `max_conf` | Highest instance score (0 if none) |
| `infer_ms` | Wall-clock time (ms) of the prediction for this image during the accuracy run. It is not the efficiency benchmark (see `results/efficiency/`) |
| `tp`, `fp`, `fn`, `tn` | Pixel confusion counts between the predicted and the reference mask |
| `gt_px`, `pred_px` | Foreground pixels of the reference mask and of the predicted mask |
| `dsc` | Dice: 2TP / (2TP + FP + FN) |
| `jsi` | Jaccard: TP / (TP + FP + FN) |
| `jsi_thr` | `jsi` if `jsi` >= 0.65, else 0 (the ISIC 2018 Task 1 thresholded Jaccard) |
| `sensitivity`, `specificity`, `accuracy` | TP / (TP + FN), TN / (TN + FP), (TP + TN) / all pixels |
| `biou` | Boundary IoU; boundary band width = 2 % of the image diagonal |
| `nsd` | Normalised surface distance (surface Dice); tolerance = 1 % of the image diagonal |
| `hd95` | 95th-percentile symmetric Hausdorff distance between the contours, in pixels at the dataset resolution |
| `empty_gt`, `empty_pred`, `both_empty` | Flags for an empty reference mask, an empty prediction, or both |

Empty masks follow `segmentation_metrics.py`: both empty gives DSC = JSI = 1 (sensitivity undefined, NaN); exactly one
empty gives DSC = JSI = 0, boundary scores 0 and `hd95` = image diagonal. NaN values are excluded from the means.

Prediction rule: prompt "skin lesion", input 1008 px; the highest-score instance with score >= 0.001, at the dataset resolution. Reference: the official ISIC mask at the dataset resolution.

### `folds/folds.csv`

`fold` (0-4), `role` (`train` or `val`), `image_id` (ISIC id).

### Training curves (`results.csv` of Phases 1, 2, 3, 4)

`epoch`; `time_s` seconds of the epoch; `train_loss`, `val_loss`; `val_dsc`, `val_jsi`, `val_jsi_thr`, `val_sensitivity`, `val_specificity`, `val_accuracy` (per-image means on the validation split), `val_pooled_dsc`, `val_pooled_jsi` (pooled over all pixels), `val_n_empty_pred` empty predictions; `val_biou`, `val_nsd`, `val_hd95`; `val_coco_ap_segm`, `val_coco_ap_bbox` COCO AP of the official evaluator; `improved` this epoch became the best so far.

The training-time validation scores use the trainer's own rule, the union of instances with score >= 0.5, as recorded in `final_results.json` → `protocol_notes`. Phases 2 (pixel re-score) and 5 use the top-1 rule above.

### `tune_results.csv`

`fitness` (see Phase 3 above), then one column per searched hyperparameter, with the value used in that trial.

### `pixel_metrics_per_fold.csv`

`fold`; `n_images` held-out images; `n_empty_pred` empty predictions; `pooled_dsc`, `pooled_jsi` computed from the confusion counts summed over the fold's images; `dsc`, `jsi`, `jsi_thr`, `sensitivity`, `specificity`, `accuracy`, `biou`, `nsd`, `hd95`: mean over the fold's images of the per-image scores (definitions above).

### `metrics_per_fold.csv`, `phase1_val.csv`, `phase4_val.csv`

Metrics of a training run at its best epoch, on the validation split of the run. `fold` (per-fold file only); `model`, `split`, `resumed` (the run was resumed from a checkpoint), `run_dir` (the run folder, mapped path); `precision_b`, `recall_b`, `map50_b`, `map5095_b`, `f1_b` (box) and `precision_m`, `recall_m`, `map50_m`, `map5095_m`, `f1_m` (mask): Ultralytics instance metrics (NaN for this architecture); `val_*`: the training-curve columns of that epoch; `fitness` the selection value of the run; `best_epoch`, `epochs_trained`; `best_epoch_source` how the best epoch was identified. A file has only the columns its run produced.

### `phase2_cv_baseline.csv`, `phase2_cv_pixel.csv`

`model`, `n_folds`, then `<metric>_mean` and `<metric>_std` (sample SD over folds, ddof = 1) for every metric of `metrics_per_fold.csv` (`phase2_cv_baseline.csv`) or of `pixel_metrics_per_fold.csv` (`phase2_cv_pixel.csv`). For the rule behind `_mean`, see Phase 2 above.

### `test_accuracy.csv`

`variant`, `model`, `precision`, `n_images`; the instance metrics `map50_m`, `map5095_m`, `precision_m`, `recall_m`, `f1_m`, `map50_b`, `map5095_b`, `precision_b`, `recall_b`, `f1_b` (NaN for this architecture); `n_empty_pred`; `pooled_dsc`, `pooled_jsi`; then, for each per-image score (`dsc`, `jsi`, `jsi_thr`, `sensitivity`, `specificity`, `accuracy`, `biou`, `nsd`, `hd95`), `<score>_mean`, `<score>_std` (ddof = 1), `<score>_median`, `<score>_ci95_low`, `<score>_ci95_high` (seeded bootstrap 95 % CI of the mean, seed 0).

### `hpo_gain.csv`

Per model, on the FP32 test files: `n_images`; for each of `dsc`, `jsi`, `biou`, `nsd`, `hd95`: `<score>_baseline`, `<score>_optimized` (means), `delta_<score>` (Optimized - Baseline), `delta_<score>_ci95_low`, `delta_<score>_ci95_high` (paired bootstrap 95 % CI), `wilcoxon_p_<score>` (two-sided Wilcoxon signed-rank test on the per-image pairs), `n_improved_<score>`, `n_worse_<score>` (images that improved or worsened); `delta_map5095_m` (Optimized - Baseline mask mAP50-95, NaN for this architecture).

### `efficiency.csv`

One row per `variant`, `model`, `precision`; `batch` (1); `gpu`; `contended` (see the efficiency section). Latencies in ms:

- `fwd_mean_ms`, `fwd_std_ms`, `fwd_median_ms`, `fwd_p90_ms`, `fwd_p95_ms`, `fwd_p99_ms`, `fwd_fps` (1000 / mean), `fwd_fps_median` (1000 / median): the whole network (image encoder, text encoder, detector) on a fixed pre-processed test image (1 x 3 x 1008 x 1008) and the prompt, timed with CUDA events.
- `fwd_cached_text_mean_ms`, `fwd_cached_text_median_ms`, `fwd_cached_text_p95_ms`, `fwd_cached_text_fps`: the same forward pass with the text encoding computed once and reused.
- `e2e_mean_ms`, `e2e_median_ms`, `e2e_p95_ms`, `e2e_p99_ms`, `e2e_fps`, `e2e_fps_median`: the deployed prediction pipeline on one real test image, from decoded image in host memory to binary mask on the host, timed with `time.perf_counter`.
- `e2e_dataset_n`, `e2e_dataset_mean_ms`, `e2e_dataset_median_ms`, `e2e_dataset_p95_ms`, `e2e_dataset_p99_ms`, `e2e_dataset_fps`: the same pipeline once on each of the first `e2e_dataset_n` test images sorted by ISIC id (the same images in the three repositories).
- Memory (MB): `vram_weights_mb` weights; `vram_peak_allocated_mb`, `vram_peak_reserved_mb` PyTorch allocator peaks during the forward scope after warm-up; `vram_peak_allocated_e2e_mb`, `vram_peak_allocated_e2e_dataset_mb` same for the end-to-end scopes; `vram_cuda_context_mb` CUDA context; `vram_process_peak_mb` driver-level device memory of the process (`nvidia-smi`); `ram_peak_rss_mb`, `ram_model_delta_mb` host memory.
- Model: `size_mb_disk` size of `best.pt`; `size_mb_fp32_theoretical`, `size_mb_fp16_theoretical` fused parameters x 4 or x 2 bytes; `params`, `params_fused`; `gflops` at the native input; `gflops_640` at 640 x 640 (equal to `gflops`, since every input is resized to 1008); `input_px` native input side.
- SAM 3 only: `params_without_text` and `params_<component>` (`dot_prod_scoring`, `geometry_encoder`, `other`, `segmentation_head`, `text_encoder`, `transformer`, `vision_backbone`) parameter counts; `gflops_without_text`, `gflops_image_encoder`, `gflops_text_encoder`, `gflops_detector` GFLOPs per component; `gflops_op_attention`, `gflops_op_convolution`, `gflops_op_linear` GFLOPs per operation type; `attention_flop_share` = `gflops_op_attention` / `gflops` (counted with `torch.utils.flop_counter.FlopCounterMode`, 2 x MACs).

### `training_cost.csv`

`phase`, `model`; `runs` number of `results.csv` files of the phase; `epochs` epochs summed over them; `train_hours` wall-clock training time summed over them; `sec_per_epoch` = seconds / epochs.

### `final_results.csv`

Long format, one row per value: `phase`, `variant`, `model`, `split` (`val`, `cv`, `test`), `precision`, `metric`, `value`, `std`, `ci95_low`, `ci95_high` (empty where not defined), `n` (images, or folds for `cv`).

## Path mapping

The pipeline ran in Docker, with the repository's folders mounted under `/workspace` (see the `docker run` command of the top-level README). In the copied text files the container paths were replaced as follows. Nothing else was changed:

| Container path | Written as |
|---|---|
| `/workspace/yolo26_dataset` | `../sandbox_yolo26/datasets/isic2018_task1_official` (the YOLO26 dataset, mounted read-only) |
| `/workspace/cache` | `sam3_cache` |
| `/workspace/<path>` | `<path>`, relative to the repository root |
| `/workspace` (alone, `root_dir` of `config.yaml`) | `.` |

Paths under `logs/` point to the location in the checkout that ran the pipeline. That folder is not versioned; the files of this package are the copies.

## Regenerating

**Folds.** The folds depend only on the prepared dataset and the seed. Prepare the dataset (Phase 0, `sam3_seg/prepare_dataset.py`, see the top-level README) and run Phase 2:

```bash
bash /workspace/run_pipeline_sam3.sh --phases "2"        # inside the sam3_ft container, as in the top-level README
```

`sam3_seg/train_cv_sam3.py` builds the folds and writes `splits_manifest.json`. If an existing manifest does not match, it refuses to continue. To check a dataset against this package without training, rebuild the partition with the functions named in [folds/](#folds) and compare the result with `folds/splits_manifest.json` and `folds/folds.csv`.

**Evaluation.** The evaluation needs the trained weights, which are not included. With weights from a run of this code (same paths under `logs/pipeline_final_v1/`):

```bash
bash /workspace/run_pipeline_sam3.sh --phases "5"        # evaluate_test_set.py, benchmark_efficiency.py, build_final_report.py
python sam3_seg/evaluate_cv_pixels.py --protocol baseline --project /workspace/logs/pipeline_final_v1 --device 0
python sam3_seg/export_phase1_val_masks.py --source best
```

## Not included

| Output | Produced by |
|---|---|
| Model weights and checkpoints (`best.pt`, `last.pt` and trial weights) | the training scripts of every phase |
| Images, reference masks and predicted masks (`phase5_test/masks/`, `phase1_val_masks/*/masks/`) | `evaluate_test_set.py`, `export_phase1_val_masks.py` |
| Raw console logs (`pipeline_runs/*/*.log`, `terminal_*.log`) | `run_pipeline_sam3.sh` and the terminal log of the `docker run` command in the top-level README |
| Figures (`figures/*.png`, `figures/*.pdf`) | the notebooks in `analysis/` (with `analysis/results_aggregator.py`) |
| LaTeX tables (`tables/*.tex`) | the notebooks in `analysis/` (with `analysis/results_aggregator.py`) |
| Lock files (`*.lock`) | the training scripts (run locks, empty) |
| Prose note on Phase 1 (`sam3_phase1_analysis.md`) | written by hand |
| Phase 1 validation scores computed from the trainer's last-epoch prediction dump instead of `best.pt` (`phase1_val_masks/sam3_dump_epoch30/`) | `sam3_seg/export_phase1_val_masks.py --source dump` |
| COCO prediction dumps of every run (`dumps/coco_predictions_segm.json`: predicted masks, RLE) | the SAM 3 trainer (`sam3_seg/training.py`) |
| TensorBoard event files (`tensorboard/`) and trainer logs (`train.log`, `logs/log.txt`) | the SAM 3 trainer (`sam3_seg/training.py`) |
| Per-image cross-validation scores | never written by the pipeline (see Phase 2 above) |

## Checksums

```bash
cd reproducibility/pipeline_final_v1 && sha256sum --check CHECKSUMS.sha256
```
