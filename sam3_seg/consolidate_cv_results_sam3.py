"""Consolidate Phase 2 (cross-validation) results into a paper-ready CSV + JSON.

Reads ``<project>/phase2_cv_<protocol>/<model>/metrics_summary.json`` written
by :mod:`train_cv_sam3` and writes, in the YOLO26 format, under
``<project>/summary/``:

* ``phase2_cv_<protocol>.csv`` — one row per model with ``mean`` and ``std``
  (sample std, ddof=1) of every metric: the YOLO26 instance-metric columns
  (COCO mAP50-95 from SAM 3's evaluator in ``map5095_m`` / ``map5095_b``,
  NaN for the metrics SAM 3 does not produce) followed by SAM 3's validation
  pixel metrics.
* ``phase2_cv_<protocol>.json`` — per-fold metrics and the aggregate.

Usage:
    python sam3_seg/consolidate_cv_results_sam3.py --project /workspace/logs/pipeline_final_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from common import DEFAULT_ORDER, DEFAULT_PIPELINE_ROOT, PipelinePaths, atomic_write_json
from train_cv_sam3 import VAL_KEYS, YOLO_NAN_KEYS

#: Reported metrics: YOLO26's list first (same column order), then SAM 3's.
REPORT_METRICS: list[str] = [
    "map50_b", "map5095_b", "precision_b", "recall_b", "f1_b",
    "map50_m", "map5095_m", "precision_m", "recall_m", "f1_m",
    "best_epoch", "epochs_trained", *VAL_KEYS,
]
assert set(YOLO_NAN_KEYS) <= set(REPORT_METRICS)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Consolidate Phase 2 CV results into CSV + JSON.")
    p.add_argument("--models", nargs="+", default=DEFAULT_ORDER, choices=DEFAULT_ORDER)
    p.add_argument("--project", default=DEFAULT_PIPELINE_ROOT, help="Pipeline root.")
    p.add_argument("--protocol", choices=["baseline", "optimized"], default="baseline")
    return p.parse_args()


def main() -> int:
    """Consolidate CV summaries.

    Returns:
        ``0`` if every requested model has a summary, ``1`` otherwise.
    """
    args = parse_args()
    paths = PipelinePaths(Path(args.project).resolve())
    paths.summary_dir.mkdir(parents=True, exist_ok=True)
    per_model, missing = [], []
    for m in args.models:
        path = paths.cv_model_dir(m, args.protocol) / "metrics_summary.json"
        if not path.exists():
            print(f"  [warn] CV summary not found for {m}: {path}")
            missing.append(m)
            continue
        payload = json.loads(path.read_text())
        if payload.get("std_ddof") != 1:
            print(f"  [warn] {m}: summary is not ddof=1 — re-run Phase 2 to refresh it")
        per_model.append({"model": m, "n_folds": payload["n_folds"], "summary": payload["summary"],
                          "per_fold": payload["per_fold"], "source": str(path)})
        s = payload["summary"]
        print(f"  {m:<8} : k={payload['n_folds']} | val DSC={s['val_dsc']['mean']:.4f}±{s['val_dsc']['std']:.4f}  "
              f"JSI={s['val_jsi']['mean']:.4f}±{s['val_jsi']['std']:.4f}")

    csv_path = paths.summary_dir / f"phase2_cv_{args.protocol}.csv"
    if per_model:
        with csv_path.open("w", newline="") as f:
            fields = ["model", "n_folds"] + [f"{k}_{s}" for k in REPORT_METRICS for s in ("mean", "std")]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for e in per_model:
                row = {"model": e["model"], "n_folds": e["n_folds"]}
                for k in REPORT_METRICS:
                    v = e["summary"].get(k, {})
                    row[f"{k}_mean"], row[f"{k}_std"] = v.get("mean", ""), v.get("std", "")
                w.writerow(row)
    atomic_write_json(paths.summary_dir / f"phase2_cv_{args.protocol}.json",
                      {"protocol": args.protocol, "models": per_model, "missing": missing})
    print(f"\nConsolidated: {csv_path}")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
