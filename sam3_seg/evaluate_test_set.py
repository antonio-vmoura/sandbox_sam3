"""Phase 5a — Accuracy of the Baseline and Optimised SAM 3 on the held-out TEST set.

Mirror of YOLO26's ``yolo26_seg/evaluate_test_set.py`` and the U-Net's: same
metrics, same ground truth, same resolution and the same output schema, so
the three architectures can be compared directly. For every ``variant ∈
{baseline, optimized}`` × ``model`` × ``precision ∈ {fp32, fp16}`` the final
``best.pt`` is evaluated on the ``test`` split — the only phase that ever
touches it:

* **Pixel metrics** (:mod:`inference`) — per image, batch = 1, through the
  run's own validation pipeline (the one that selected ``best.pt``): official
  transforms (1008 × 1008) and postprocessor, prompt ``"skin lesion"``; the
  predicted mask is the top-1 (highest-score) instance with score >= 0.5, at the
  original dataset resolution; scored against the **same official ISIC mask**
  as YOLO26 and the U-Net (stored losslessly as RLE in Phase 0) with
  :func:`segmentation_metrics.pixel_scores`: DSC, JSI, ISIC thresholded JSI
  (0.65), sensitivity, specificity, accuracy, Boundary IoU, NSD, HD95.
  Empty predictions score 0 (never skipped). Aggregates: per-image mean,
  sample std, median, IQR, seeded bootstrap 95 % CI, pooled DSC/JSI
  (:func:`segmentation_metrics.aggregate_scores`).
* **Instance metrics** — YOLO26's Ultralytics box/mask P, R, F1 and mAP are not
  produced by SAM 3; the keys are kept (value NaN) so the JSON schema is
  identical. COCO mAP50-95 is the subject of the official evaluator and is not
  recomputed here.

FP32 is the primary result (training used FP16 AMP with FP32 master weights); FP16 (SAM 3's official
``torch.autocast`` path) quantifies the accuracy cost of half precision.

Outputs (per variant/model/precision), as YOLO26::

    <project>/phase5_test/accuracy/<variant>_<model>_<precision>.json
    <project>/phase5_test/per_image/<variant>_<model>_<precision>.csv
    <project>/phase5_test/masks/<variant>_<model>/<image stem>.png   # FP32 only

Each JSON records the SHA-256 of the weights and of the test-ID list; up-to-date
results are skipped.

Usage:
    python sam3_seg/evaluate_test_set.py --project /workspace/logs/pipeline_final_v1 --device 0
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from common import (
    DEFAULT_DATA_DIR,
    DEFAULT_ORDER,
    DEFAULT_PIPELINE_ROOT,
    PROMPT,
    RESOLUTION,
    SEED,
    PipelinePaths,
    atomic_write_json,
    config_hash,
    parse_device,
    read_json,
    seed_everything,
    sha256_file,
    utc_now_iso,
)
from data import CocoData, ids_fingerprint
from inference import EVAL_VERSION, SCORE_THRESHOLD, Predictor, evaluate_annotations
from segmentation_metrics import aggregate_scores
from training import RUN_STATE_FILE

#: YOLO26 instance-metric keys (not produced by SAM 3's pixel evaluation → NaN).
YOLO_INSTANCE_KEYS: tuple[str, ...] = (
    "precision_b", "recall_b", "map50_b", "map5095_b",
    "precision_m", "recall_m", "map50_m", "map5095_m", "f1_b", "f1_m",
)

VARIANTS: tuple[str, ...] = ("baseline", "optimized")
PRECISIONS: tuple[str, ...] = ("fp32", "fp16")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Phase 5a."""
    p = argparse.ArgumentParser(description="Phase 5a — SAM 3 test-set accuracy (pixel metrics, YOLO26 schema).")
    p.add_argument("--models", nargs="+", default=DEFAULT_ORDER, choices=DEFAULT_ORDER)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    p.add_argument("--precisions", nargs="+", default=list(PRECISIONS), choices=PRECISIONS)
    p.add_argument("--data", default=DEFAULT_DATA_DIR, help="Phase 0 dataset directory.")
    p.add_argument("--device", default="0", help="Single GPU id (default: 0).")
    p.add_argument("--workers", type=int, default=4, help="DataLoader workers (default: 4).")
    p.add_argument("--project", default=DEFAULT_PIPELINE_ROOT, help="Pipeline root.")
    p.add_argument("--no-save-masks", action="store_true", help="Do not save predicted masks.")
    p.add_argument("--force", action="store_true", help="Re-evaluate even if results are up to date.")
    return p.parse_args()


def _require_trained(paths: PipelinePaths, variant: str, model: str) -> Path:
    """Weights of a completed training run, or raise."""
    weights = paths.best_pt(variant, model)
    state = read_json(weights.parent.parent / RUN_STATE_FILE)   # <run>/checkpoints/best.pt
    if state is None or state.get("status") != "complete" or not weights.exists():
        raise RuntimeError(f"{variant}/{model}: training run not complete ({weights.parent.parent})")
    return weights


def evaluate_one(variant: str, model_name: str, precision: str, args, device: torch.device,
                 paths: PipelinePaths, data: CocoData, test_ids: list[str]) -> dict[str, Any]:
    """Evaluate one ``(variant, model, precision)`` on the test set (or skip if current)."""
    weights = _require_trained(paths, variant, model_name)
    run_dir = weights.parent.parent
    out_json = paths.phase5_accuracy_json(variant, model_name, precision)
    settings = {
        "weights_sha256": sha256_file(weights), "test_list_sha256": ids_fingerprint(test_ids),
        "conf": SCORE_THRESHOLD, "imgsz": RESOLUTION, "prompt": PROMPT, "precision": precision,
        "eval_version": EVAL_VERSION, "dataset": data.fingerprint(),
        "run_config_sha256": sha256_file(run_dir / "config.yaml"),
    }
    previous = read_json(out_json)
    if previous and previous.get("settings_hash") == config_hash(settings) and not args.force:
        return {"tag": out_json.stem, "skipped": True, "payload": previous}

    tag = paths.phase5_tag(variant, model_name, precision)
    print(f"\n=== {tag}  ({len(test_ids)} test images, weights={weights})")
    t0 = time.perf_counter()
    predictor = Predictor(run_dir, weights, device)
    save_masks = precision == "fp32" and not args.no_save_masks
    rows = evaluate_annotations(predictor, data.image_dir, data.split_json("test"), precision=precision,
                                mask_dir=paths.phase5_mask_dir(variant, model_name) if save_masks else None,
                                workers=args.workers)
    del predictor
    torch.cuda.empty_cache()
    pixel = aggregate_scores(rows, seed=SEED)
    print(f"  pixel   : DSC={pixel['dsc']['mean']:.4f} (95% CI {pixel['dsc']['ci95_low']:.4f}-"
          f"{pixel['dsc']['ci95_high']:.4f})  JSI={pixel['jsi']['mean']:.4f}  "
          f"JSI_thr={pixel['jsi_thr']['mean']:.4f}  empty_pred={pixel['n_empty_pred']}")

    per_image_csv = paths.phase5_per_image_csv(variant, model_name, precision)
    per_image_csv.parent.mkdir(parents=True, exist_ok=True)
    with per_image_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    mean_ms = sum(r["infer_ms"] for r in rows) / len(rows)
    payload = {
        "variant": variant, "model": model_name, "precision": precision, "split": "test",
        "weights": str(weights), "data": str(data.meta.get("source_data_yaml")), "n_images": len(test_ids),
        "settings": settings, "settings_hash": config_hash(settings),
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "instance_metrics": {k: math.nan for k in YOLO_INSTANCE_KEYS},
        "instance_conf": None,
        "fp16_mode": "torch.autocast(float16), FP32 weights" if precision == "fp16" else None,
        "pixel_metrics": pixel,
        "pixel_conf": SCORE_THRESHOLD,
        "ultralytics_speed_ms": {"preprocess": math.nan, "inference": mean_ms, "postprocess": math.nan},
        "per_image_csv": str(per_image_csv),
        "mask_dir": str(paths.phase5_mask_dir(variant, model_name)) if save_masks else None,
        "elapsed_min": round((time.perf_counter() - t0) / 60, 2),
        "created_at": utc_now_iso(),
    }
    atomic_write_json(out_json, payload)
    return {"tag": tag, "skipped": False, "payload": payload}


def main() -> int:
    """Evaluate every requested combination.

    Returns:
        ``0`` on success, ``1`` if any combination failed.
    """
    args = parse_args()
    seed_everything()
    device = torch.device(f"cuda:{parse_device(args.device)}")
    paths = PipelinePaths(Path(args.project))
    data = CocoData(args.data)
    test_ids = data.ids(data.split_json("test"))
    print(f"Phase 5a — test set: {len(test_ids)} images | variants={args.variants} "
          f"models={args.models} precisions={args.precisions} device={device}")
    failures = 0
    for variant in args.variants:
        for m in args.models:
            for precision in args.precisions:
                try:
                    r = evaluate_one(variant, m, precision, args, device, paths, data, test_ids)
                    if r["skipped"]:
                        print(f"  [skip] {r['tag']} up to date")
                except Exception:
                    failures += 1
                    print(f"  [fail] {variant}/{m}/{precision}:\n{traceback.format_exc()}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
