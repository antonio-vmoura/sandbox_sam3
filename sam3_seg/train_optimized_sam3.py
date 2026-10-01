"""Phase 4 — Optimised fine-tuning of SAM 3 on ISIC 2018 Task 1.

Fine-tunes SAM 3 once on the standard train/val split with the
hyperparameters selected in Phase 3, via :func:`common.optimized_protocol`:
the **same base setup** as the Baseline (official recipe, FP32, batch 2,
budget, seed, prompt) + the tuned learning dynamics / augmentation. Tuned
files that try to change a base-setup key are rejected.

A model is only trained when its HPO is complete (``hpo_state.json`` status
``complete``); pass ``--allow-incomplete-hpo`` to override. The YAML used is
copied into the run directory for provenance.

Outputs:
    ``<project>/phase4_optimized/sam3_optimized/{checkpoints/best.pt, results.csv,
    run_state.json, tuned_hyperparameters.yaml, ...}``

Usage:
    python sam3_seg/train_optimized_sam3.py --project /workspace/logs/pipeline_final_v1 --device 0
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import traceback
from pathlib import Path

from common import (
    DEFAULT_DATA_DIR,
    DEFAULT_ORDER,
    DEFAULT_PIPELINE_ROOT,
    TRAIN_EPOCHS,
    TRAIN_PATIENCE,
    PipelinePaths,
    optimized_protocol,
    parse_device,
    seed_everything,
)
from data import CocoData
from training import load_tuned_hp, print_phase_summary, require_complete_hpo, train_or_resume

PHASE: str = "phase4_optimized"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Phase 4."""
    p = argparse.ArgumentParser(description="Phase 4 — Optimised fine-tuning of SAM 3 with the Phase 3 HPs.")
    p.add_argument("--models", nargs="+", default=DEFAULT_ORDER, choices=DEFAULT_ORDER)
    p.add_argument("--data", default=DEFAULT_DATA_DIR, help="Phase 0 dataset directory.")
    p.add_argument("--device", default="0", help="Single GPU id (default: 0).")
    p.add_argument("--project", default=DEFAULT_PIPELINE_ROOT, help="Pipeline root.")
    p.add_argument("--epochs", type=int, default=TRAIN_EPOCHS,
                   help=f"Training budget (default: {TRAIN_EPOCHS}). Must match Phases 1 and 2.")
    p.add_argument("--patience", type=int, default=TRAIN_PATIENCE,
                   help=f"Early-stopping patience (default: {TRAIN_PATIENCE}).")
    p.add_argument("--allow-incomplete-hpo", action="store_true",
                   help="Train even if the Phase 3 search did not reach its target trials.")
    p.add_argument("--force", action="store_true", help="Retrain from scratch (old run → *.bak-<UTC>).")
    return p.parse_args()


def main() -> int:
    """Run Phase 4 for the requested models.

    Returns:
        ``0`` on success (including skipped models), ``1`` if any model failed.
    """
    args = parse_args()
    seed_everything()
    device = parse_device(args.device)
    paths = PipelinePaths(Path(args.project))
    data = CocoData(args.data)
    train_json, val_json = data.split_json("train"), data.split_json("val")

    print(f"Phase 4 (Optimised) for models: {args.models}")
    print(f"  device = {device}   output = {paths.phase4_dir}")
    print(f"  budget = {args.epochs} epochs, patience {args.patience} (val JSI), FP32, seed 0")

    summary: list[dict] = []
    t0 = time.perf_counter()
    for m in args.models:
        print("\n" + "=" * 80 + f"\n=== PHASE 4 (OPTIMISED): {m}\n" + "=" * 80)
        try:
            if not args.allow_incomplete_hpo:
                require_complete_hpo(paths.phase3_state(m))
            hp_yaml = paths.phase3_best_yaml(m)
            tuned = load_tuned_hp(hp_yaml)
            print(f"  HP source: {hp_yaml}")
            for k, v in sorted(tuned.items()):
                print(f"    {k:20s} = {v}")
            protocol = optimized_protocol(device, tuned, args.epochs, args.patience)
            run_dir = paths.phase4_dir / paths.phase4_run_name(m)
            summary.append(train_or_resume(
                phase=PHASE, model_name=m, protocol=protocol, image_dir=data.image_dir,
                train_json=train_json, val_json=val_json,
                project=paths.phase4_dir, name=paths.phase4_run_name(m), force=args.force,
            ))
            shutil.copy2(hp_yaml, run_dir / "tuned_hyperparameters.yaml")
        except Exception as e:
            print(f"  [fail] {m}:\n{traceback.format_exc()}", file=sys.stderr)
            summary.append({"model": m, "failed": True, "reason": str(e)})

    print_phase_summary("PHASE 4 (OPTIMISED)", summary, (time.perf_counter() - t0) / 60)
    return 1 if any(s.get("failed") for s in summary) else 0


if __name__ == "__main__":
    sys.exit(main())
