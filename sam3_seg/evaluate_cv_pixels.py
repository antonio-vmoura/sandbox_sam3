"""Phase 2 (post-step) — pixel metrics of every CV fold's ``best.pt`` on its held-out fold, with the TEST rule.

During Phase 2 training the per-epoch validation JSI (the checkpoint-selection
metric) is computed with the training-time rule — union of the instances with
score >= 0.5 — from the official prediction dump of each epoch. The Phase 5 test
evaluation scores the **top-1 instance with score >= PIXEL_CONF** (the rule of
YOLO26's pixel evaluation). This step re-scores each fold's ``best.pt`` on its
held-out fold through exactly the Phase 5 path (:func:`inference.evaluate_annotations`),
so SAM 3's CV row uses the same prediction rule as its test row and as the
YOLO26 / U-Net CV rows (their ``evaluate_cv_pixels.py``). Checkpoint selection is
untouched. The test set is never used here.

Outputs, in the YOLO26 / U-Net format (read by ``build_final_report.py``)::

    <project>/phase2_cv_<protocol>/<model>/
    ├── pixel_metrics_per_fold.csv    # per-fold means of DSC, JSI, ...
    └── pixel_metrics_summary.json    # mean ± sample std (ddof=1) across folds

Up-to-date results (same fold weights, folds and method version) are skipped.

Usage:
    python sam3_seg/evaluate_cv_pixels.py --project /workspace/logs/pipeline_final_v1 --device 0
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import traceback
from pathlib import Path

import torch

from common import (
    DEFAULT_DATA_DIR,
    DEFAULT_ORDER,
    DEFAULT_PIPELINE_ROOT,
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
from data import CocoData
from inference import EVAL_VERSION, Predictor, evaluate_annotations
from segmentation_metrics import PIXEL_CONF, SCORE_KEYS, aggregate_scores
from training import RUN_STATE_FILE


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Pixel metrics of each SAM 3 CV fold on its held-out fold (test rule).")
    p.add_argument("--models", nargs="+", default=DEFAULT_ORDER, choices=DEFAULT_ORDER)
    p.add_argument("--protocol", choices=["baseline", "optimized"], default="baseline")
    p.add_argument("--data", default=DEFAULT_DATA_DIR, help="Phase 0 dataset directory.")
    p.add_argument("--device", default="0", help="Single GPU id (default: 0).")
    p.add_argument("--workers", type=int, default=4, help="DataLoader workers (default: 4).")
    p.add_argument("--project", default=DEFAULT_PIPELINE_ROOT, help="Pipeline root.")
    p.add_argument("--force", action="store_true", help="Re-evaluate even if up to date.")
    return p.parse_args()


def evaluate_model(model_name: str, args, device: torch.device, paths: PipelinePaths, data: CocoData) -> None:
    """Score every fold of one model and write the per-fold CSV + summary JSON."""
    cv_root = paths.cv_model_dir(model_name, args.protocol)
    manifest = read_json(cv_root / "splits_manifest.json")
    if manifest is None:
        raise RuntimeError(f"{cv_root}: no splits_manifest.json — run Phase 2 training first")

    folds = []
    for k in range(int(manifest["k"])):
        run = cv_root / "runs" / f"fold_{k}"
        state = read_json(run / RUN_STATE_FILE)
        weights = run / "checkpoints" / "best.pt"
        if state is None or state.get("status") != "complete" or not weights.exists():
            raise RuntimeError(f"{run}: fold training not complete")
        folds.append((k, run, weights, data.fold_json(k, "val")))

    settings = {"weights_sha256": [sha256_file(w) for _, _, w, _ in folds], "conf": PIXEL_CONF,
                "manifest": manifest, "eval_version": EVAL_VERSION, "dataset": data.fingerprint(),
                "val_json_sha256": [sha256_file(v) for _, _, _, v in folds]}
    out_json = cv_root / "pixel_metrics_summary.json"
    previous = read_json(out_json)
    if previous and previous.get("settings_hash") == config_hash(settings) and not args.force:
        print(f"  [skip] {model_name}: pixel metrics up to date")
        return

    per_fold = []
    for k, run, weights, val_json in folds:
        print(f"  [{model_name}] fold {k}: {len(data.ids(val_json))} held-out images")
        predictor = Predictor(run, weights, device)
        rows = evaluate_annotations(predictor, data.image_dir, val_json, precision="fp32", workers=args.workers)
        del predictor
        torch.cuda.empty_cache()
        agg = aggregate_scores(rows, seed=SEED)
        per_fold.append({"fold": k, "n_images": agg["n_images"], "n_empty_pred": agg["n_empty_pred"],
                         "pooled_dsc": agg["pooled_dsc"], "pooled_jsi": agg["pooled_jsi"],
                         **{key: agg[key].get("mean", float("nan")) for key in SCORE_KEYS}})
        print(f"    DSC={per_fold[-1]['dsc']:.4f}  JSI={per_fold[-1]['jsi']:.4f}")

    with (cv_root / "pixel_metrics_per_fold.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_fold[0].keys()))
        w.writeheader()
        w.writerows(per_fold)
    summary = {}
    for key in (*SCORE_KEYS, "pooled_dsc", "pooled_jsi"):
        vals = [r[key] for r in per_fold]
        summary[key] = {"mean": statistics.mean(vals), "std": statistics.stdev(vals) if len(vals) > 1 else 0.0}
    atomic_write_json(out_json, {
        "model": model_name, "protocol": args.protocol, "split": "cv_heldout_fold",
        "resolution": "original (dataset resolution)", "n_folds": len(per_fold), "std_ddof": 1,
        "conf": PIXEL_CONF, "rule": "top-1 instance with score >= conf (Phase 5 path)",
        "per_fold": per_fold, "summary": summary,
        "settings_hash": config_hash(settings), "created_at": utc_now_iso(),
    })
    print(f"  [{model_name}] DSC={summary['dsc']['mean']:.4f}±{summary['dsc']['std']:.4f}  "
          f"JSI={summary['jsi']['mean']:.4f}±{summary['jsi']['std']:.4f}")


def main() -> int:
    """Run the CV pixel evaluation for every requested model.

    Returns:
        ``0`` on success, ``1`` if any model failed.
    """
    args = parse_args()
    seed_everything()
    device = torch.device(f"cuda:{parse_device(args.device)}")
    paths = PipelinePaths(Path(args.project))
    data = CocoData(args.data)
    print(f"Phase 2 pixels (protocol={args.protocol}) for {args.models}; top-1 rule, conf={PIXEL_CONF}")
    failures = 0
    for m in args.models:
        try:
            evaluate_model(m, args, device, paths, data)
        except Exception:
            failures += 1
            print(f"  [fail] {m}:\n{traceback.format_exc()}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
