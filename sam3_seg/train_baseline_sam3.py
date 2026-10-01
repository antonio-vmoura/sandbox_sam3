"""Phase 1 — Baseline fine-tuning of SAM 3 on ISIC 2018 Task 1.

Fine-tunes SAM 3 (all 840.5 M parameters, text encoder included) with Meta's
official trainer and recipe, the fixed **base setup** shared by every phase
and the **default hyperparameters** (the official recipe values) via
:func:`common.baseline_protocol`: FP32, batch 2, official activation
checkpointing, text prompt ``"skin lesion"``, seed 0, deterministic kernels
(except the ViT attention backward, see ``run_training.py``). Baseline and Optimised (Phase 4) share the identical base setup and
differ only in the tuned hyperparameters.

Data: the Phase 0 COCO dataset (same images and splits as YOLO26 / U-Net):
``train`` for fitting, ``val`` for model selection / early stopping (per-image
mean JSI). The ``test`` split is never touched.

Budget: 30 epochs, no early stopping (patience 30; SAM 3-specific, see :data:`common.TRAIN_EPOCHS`).

The run is fault-tolerant and resumable (see :mod:`training`); a
completed run is skipped unless ``--force`` is passed.

Outputs:
    ``<project>/phase1_baseline/sam3_baseline/{checkpoints/best.pt, results.csv, run_state.json, ...}``

Usage:
    python sam3_seg/train_baseline_sam3.py --project /workspace/logs/pipeline_final_v1 --device 0
"""

from __future__ import annotations

import argparse
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
    baseline_protocol,
    parse_device,
    seed_everything,
)
from data import CocoData
from training import print_phase_summary, train_or_resume

PHASE: str = "phase1_baseline"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Phase 1."""
    p = argparse.ArgumentParser(description="Phase 1 — Baseline fine-tuning (base setup + default HPs) of SAM 3.")
    p.add_argument("--models", nargs="+", default=DEFAULT_ORDER, choices=DEFAULT_ORDER)
    p.add_argument("--data", default=DEFAULT_DATA_DIR, help="Phase 0 dataset directory.")
    p.add_argument("--device", default="0", help="Single GPU id (default: 0).")
    p.add_argument("--project", default=DEFAULT_PIPELINE_ROOT, help="Pipeline root.")
    p.add_argument("--epochs", type=int, default=TRAIN_EPOCHS,
                   help=f"Training budget (default: {TRAIN_EPOCHS}). Must match Phases 2 and 4.")
    p.add_argument("--patience", type=int, default=TRAIN_PATIENCE,
                   help=f"Early-stopping patience on val JSI (default: {TRAIN_PATIENCE}).")
    p.add_argument("--force", action="store_true", help="Retrain from scratch (old run → *.bak-<UTC>).")
    return p.parse_args()


def main() -> int:
    """Run Phase 1 for the requested models.

    Returns:
        ``0`` on success (including skipped models), ``1`` if any model failed.
    """
    args = parse_args()
    seed_everything()
    device = parse_device(args.device)
    paths = PipelinePaths(Path(args.project))
    data = CocoData(args.data)
    train_json, val_json = data.split_json("train"), data.split_json("val")
    protocol = baseline_protocol(device, args.epochs, args.patience)

    print(f"Phase 1 (Baseline) for models: {args.models}")
    print(f"  device = {device}   data = {args.data}   output = {paths.phase1_dir}")
    print(f"  data   = {len(data.ids(train_json))} train / {len(data.ids(val_json))} val images (test untouched)")
    print(f"  budget = {args.epochs} epochs, patience {args.patience} (val JSI), FP32, seed 0")
    print("  setup  = official SAM 3 recipe, 1008 px, batch 2, act. checkpointing, prompt 'skin lesion'")
    print("  HPs    = official recipe defaults (lr_scale 0.1, wd 0.1, lrd 0.9, warmup 2, hflip 0.5, resize min 480)")

    summary: list[dict] = []
    t0 = time.perf_counter()
    for m in args.models:
        print("\n" + "=" * 80 + f"\n=== PHASE 1 (BASELINE): {m}\n" + "=" * 80)
        try:
            summary.append(train_or_resume(
                phase=PHASE, model_name=m, protocol=protocol, image_dir=data.image_dir,
                train_json=train_json, val_json=val_json,
                project=paths.phase1_dir, name=paths.phase1_run_name(m), force=args.force,
            ))
        except Exception as e:
            print(f"  [fail] {m}:\n{traceback.format_exc()}", file=sys.stderr)
            summary.append({"model": m, "failed": True, "reason": str(e)})

    print_phase_summary("PHASE 1 (BASELINE)", summary, (time.perf_counter() - t0) / 60)
    return 1 if any(s.get("failed") for s in summary) else 0


if __name__ == "__main__":
    sys.exit(main())
