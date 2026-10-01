"""Consolidate single-split validation metrics of Phase 1 or Phase 4 into CSV + JSON.

For each model this script reads the run's validation metrics at the epoch
that produced ``best.pt`` (:func:`training.best_metrics`) and writes, in the
YOLO26 format::

    <project>/summary/<phase>_val.csv
    <project>/summary/<phase>_val.json

The YOLO26 columns ``map5095_m`` / ``map5095_b`` hold SAM 3's official COCO
mAP50-95 (segm / bbox); the other Ultralytics instance metrics (P, R, F1,
mAP50) are not produced and are written as NaN. SAM 3's pixel metrics are
reported in the ``val_*`` columns (640 × 640, same ground truth as YOLO26).
These are **validation-split** metrics; test-set metrics come from Phase 5.

Usage:
    python sam3_seg/collect_phase_metrics_sam3.py --phase phase1 --project /workspace/logs/pipeline_final_v1
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from common import DEFAULT_ORDER, DEFAULT_PIPELINE_ROOT, PipelinePaths, atomic_write_json, read_json
from train_cv_sam3 import fold_row
from training import RUN_STATE_FILE, best_metrics


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Collect single-split validation metrics (Phase 1 or 4).")
    p.add_argument("--phase", choices=["phase1", "phase4"], required=True)
    p.add_argument("--models", nargs="+", default=DEFAULT_ORDER, choices=DEFAULT_ORDER)
    p.add_argument("--project", default=DEFAULT_PIPELINE_ROOT, help="Pipeline root.")
    return p.parse_args()


def run_dir(paths: PipelinePaths, phase: str, model: str) -> Path:
    """Training run directory of ``model`` in ``phase``."""
    if phase == "phase1":
        return paths.phase1_dir / paths.phase1_run_name(model)
    return paths.phase4_dir / paths.phase4_run_name(model)


def collect_row(paths: PipelinePaths, phase: str, model: str) -> dict | None:
    """One consolidated row, or ``None`` if the run is missing/incomplete."""
    rd = run_dir(paths, phase, model)
    state = read_json(rd / RUN_STATE_FILE)
    if state is None or state.get("status") != "complete":
        print(f"  [warn] {model}: run not complete ({rd})")
        return None
    metrics = best_metrics(rd)
    resumed = any(e.get("event") == "resume" for e in state.get("events", []))
    print(f"  {model:<8} : val DSC={metrics['val_dsc']:.4f} JSI={metrics['val_jsi']:.4f} "
          f"JSI_thr={metrics['val_jsi_thr']:.4f} (best epoch {metrics['best_epoch']})")
    return {"model": model, "split": "val", "resumed": resumed, "run_dir": str(rd), **fold_row(metrics)}


def main() -> int:
    """Collect per-model metrics for one phase and write CSV + JSON.

    Returns:
        ``0`` if every requested model was collected, ``1`` otherwise.
    """
    args = parse_args()
    paths = PipelinePaths(Path(args.project))
    paths.summary_dir.mkdir(parents=True, exist_ok=True)
    rows, missing = [], []
    for m in args.models:
        row = collect_row(paths, args.phase, m)
        (rows.append(row) if row else missing.append(m))
    csv_out = paths.summary_dir / f"{args.phase}_val.csv"
    json_out = paths.summary_dir / f"{args.phase}_val.json"
    if rows:
        with csv_out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    atomic_write_json(json_out, {"phase": args.phase, "split": "val", "models": rows, "missing": missing})
    print(f"\nGenerated: {csv_out}\n           {json_out}")
    if missing:
        print(f"  [warn] missing/incomplete: {missing}")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
