"""Export the Phase 1 (Baseline) validation masks of SAM 3 for the side-by-side figure.

Two sources, both scored with the Phase 5 rule (top-1 instance with score >= ``PIXEL_CONF``) against the
official masks with :func:`segmentation_metrics.pixel_scores`:

* ``--source best`` (GPU): the Phase 1 ``best.pt`` through the Phase 5 path
  (:func:`inference.evaluate_annotations`) — the selected checkpoint;
* ``--source dump`` (CPU): the official prediction dump the trainer wrote at its **last** validation epoch
  (``dumps/coco_predictions_segm.json``, epoch 30 for a finished run). It is not the selected checkpoint
  (``best.pt`` is the best epoch), so the figure labels it with its epoch. Use it while every GPU is busy.

Writes::

    <project>/phase1_val_masks/sam3/masks/<ISIC_ID>.png   # 0/255, dataset resolution
    <project>/phase1_val_masks/sam3/per_image.csv         # id + every pixel_scores key
    <project>/phase1_val_masks/sam3/meta.json

Read by ``analysis/results_aggregator.py`` (``fig_phase1_side_by_side``). Validation data only.

Usage:
    python sam3_seg/export_phase1_val_masks.py --source dump      # CPU
    python sam3_seg/export_phase1_val_masks.py --source best      # GPU
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from common import DEFAULT_DATA_DIR, DEFAULT_PIPELINE_ROOT, SEED, PipelinePaths, seed_everything, sha256_file
from data import CocoData
from protocol_trainer import union_masks
from segmentation_metrics import PIXEL_CONF, aggregate_scores, pixel_scores


def from_dump(run_dir: Path, val_json: Path, mask_dir: Path) -> tuple[list[dict], dict]:
    """Top-1 masks from the last-epoch prediction dump (CPU)."""
    dump = run_dir / "dumps" / "coco_predictions_segm.json"
    gt = json.loads(val_json.read_text())
    preds: dict[int, list] = {}
    for r in json.loads(dump.read_text()):
        preds.setdefault(r["image_id"], []).append(r)
    gt_by: dict[int, list] = {}
    for a in gt["annotations"]:
        gt_by.setdefault(a["image_id"], []).append(a["segmentation"])
    epochs = [int(line.split(",")[0]) for line in (run_dir / "results.csv").read_text().splitlines()[1:] if line]
    mask_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for im in gt["images"]:
        h, w = im["height"], im["width"]
        kept = [r for r in preds.get(im["id"], []) if r["score"] >= PIXEL_CONF]
        pred = union_masks([max(kept, key=lambda r: r["score"])["segmentation"]] if kept else [], h, w)
        cv2.imwrite(str(mask_dir / f"{Path(im['file_name']).stem}.png"), pred.astype(np.uint8) * 255)
        rows.append({"image": im["file_name"], "height": h, "width": w, "n_pred": len(kept),
                     "max_conf": max((r["score"] for r in kept), default=0.0),
                     **pixel_scores(union_masks(gt_by.get(im["id"], []), h, w), pred)})
    return rows, {"source": f"Phase 1 prediction dump, last validation epoch ({max(epochs)})",
                  "dump_sha256": sha256_file(dump), "epoch": max(epochs)}


def from_best(run_dir: Path, weights: Path, data: CocoData, val_json: Path, mask_dir: Path,
              device: str) -> tuple[list[dict], dict]:
    """Top-1 masks of the selected Phase 1 checkpoint (GPU)."""
    import torch

    from inference import Predictor, evaluate_annotations

    predictor = Predictor(run_dir, weights, torch.device(f"cuda:{device}"))
    rows = evaluate_annotations(predictor, data.image_dir, val_json, precision="fp32", mask_dir=mask_dir)
    best = json.loads((run_dir / "run_state.json").read_text()).get("best_epoch")
    return rows, {"source": f"Phase 1 best.pt (epoch {best})", "weights": str(weights),
                  "weights_sha256": sha256_file(weights), "epoch": best}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", choices=["dump", "best"], default="dump")
    p.add_argument("--project", default=DEFAULT_PIPELINE_ROOT)
    p.add_argument("--data", default=DEFAULT_DATA_DIR, help="Phase 0 dataset directory.")
    p.add_argument("--device", default="0", help="GPU id for --source best.")
    args = p.parse_args()
    seed_everything(SEED)
    data = CocoData(args.data)
    val_json = data.split_json("val")
    weights = PipelinePaths(Path(args.project)).phase1_best_pt("sam3")
    run_dir = weights.parent.parent
    out = Path(args.project) / "phase1_val_masks" / "sam3"
    rows, src = (from_dump(run_dir, val_json, out / "masks") if args.source == "dump"
                 else from_best(run_dir, weights, data, val_json, out / "masks", args.device))
    assert len(rows) == 100, f"expected the 100 official validation images, got {len(rows)}"
    with (out / "per_image.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", *rows[0].keys()])
        w.writeheader()
        w.writerows({"id": Path(r["image"]).stem, **r} for r in rows)
    agg = aggregate_scores(rows, seed=SEED)
    label = "SAM 3" if args.source == "best" else f"SAM 3 (epoch {src['epoch']})"
    (out / "meta.json").write_text(json.dumps({
        "arch": "sam3", "model": "sam3", "label": label, "split": "val (official, n=100)", **src,
        "rule": f"top-1 instance, score >= {PIXEL_CONF}", "jsi_mean": agg["jsi"]["mean"],
        "dsc_mean": agg["dsc"]["mean"]}, indent=2))
    print(f"sam3 ({src['source']}): JSI={agg['jsi']['mean']:.4f} DSC={agg['dsc']['mean']:.4f} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
