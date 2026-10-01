"""The official SAM 3 trainer with the study's protocol added (no vendored code changed).

:class:`ProtocolTrainer` subclasses ``sam3.train.trainer.Trainer`` and is
selected through the Hydra config (``trainer._target_``). It keeps Meta's model,
data pipeline, losses, matcher, optimiser, schedulers and validation exactly as
they are, and only changes the *epoch loop*:

1. train one epoch (official ``train_epoch``);
2. validate (official ``val_epoch``: predictions dumped to
   ``dumps/coco_predictions_segm.json`` + COCO AP), then compute the study's
   pixel metrics from those predictions: for every validation image, the union
   of the predicted masks with score ``>= score_threshold`` is compared with
   the ground truth using :func:`segmentation_metrics.pixel_scores` (the code
   shared with YOLO26 and the U-Net);
3. model selection and early stopping on the validation **per-image mean JSI**
   (strict improvement; ties keep the earliest epoch; stop after ``patience``
   epochs without improvement), ``best.pt`` written on improvement;
4. **then** checkpoint (``checkpoints/checkpoint.pt``, atomic, every epoch) —
   *after* validation, including the torch (CPU + CUDA), NumPy and Python RNG
   states and the early-stopping state / per-epoch history.

Because the checkpoint is written after validation and restores every RNG
state (the official checkpoint has none and is written before validation), a
run interrupted at any point resumes with the same data order (epoch-seeded
sampler), the same augmentation (worker seeds drawn from the restored global
RNG) and the same dropout — **bit-exactly** with ``strict_determinism``
(verified), otherwise up to the attention-backward noise documented in
``run_training.py``. ``results.csv`` is rebuilt from the checkpointed history
on resume.

Extra constructor arguments (all other arguments go to the official trainer):
``patience``, ``score_threshold``, ``val_ann_file`` (COCO file of the
validation images = ground truth) and ``protocol_hash``.
"""

from __future__ import annotations

import csv
import gc
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pycocotools import mask as mask_utils

from sam3.train.trainer import Trainer
from sam3.train.utils.distributed import barrier
from sam3.train.utils.train_utils import makedir

from segmentation_metrics import SCORE_KEYS, aggregate_scores, pixel_scores

#: Columns of ``results.csv`` (one row per epoch).
RESULT_COLUMNS: tuple[str, ...] = (
    "epoch", "time_s", "train_loss", "val_loss",
    "val_dsc", "val_jsi", "val_jsi_thr", "val_sensitivity", "val_specificity", "val_accuracy",
    "val_biou", "val_nsd", "val_pooled_dsc", "val_pooled_jsi", "val_n_empty_pred",
    "val_coco_ap_segm", "val_coco_ap_bbox",
    "improved",
)


def _find(d: dict[str, Any], *needles: str) -> float:
    """First value of ``d`` whose key contains all ``needles`` (NaN if none)."""
    for k, v in d.items():
        if all(n in k for n in needles) and isinstance(v, (int, float)):
            return float(v)
    return math.nan


def union_masks(rles: list[dict[str, Any]], h: int, w: int) -> np.ndarray:
    """Union of COCO RLE masks as a boolean ``h × w`` array (empty if none)."""
    out = np.zeros((h, w), dtype=bool)
    for r in rles:
        out |= mask_utils.decode(r).astype(bool)
    return out


def pixel_metrics_from_dump(pred_file: Path, gt_file: Path, score_threshold: float) -> dict[str, float]:
    """Per-image pixel metrics of dumped COCO predictions against the COCO ground truth."""
    gt = json.loads(Path(gt_file).read_text())
    preds = json.loads(Path(pred_file).read_text()) if Path(pred_file).exists() else []
    gt_by, pr_by = {}, {}
    for a in gt["annotations"]:
        gt_by.setdefault(a["image_id"], []).append(a["segmentation"])
    for p in preds:
        if p.get("score", 0.0) >= score_threshold:
            pr_by.setdefault(p["image_id"], []).append(p["segmentation"])
    rows = []
    for im in gt["images"]:
        h, w = im["height"], im["width"]
        rows.append(pixel_scores(union_masks(gt_by.get(im["id"], []), h, w),
                                 union_masks(pr_by.get(im["id"], []), h, w)))
    agg = aggregate_scores(rows)
    out = {f"val_{k}": agg[k].get("mean", math.nan) for k in SCORE_KEYS}
    out.update(val_pooled_dsc=agg["pooled_dsc"], val_pooled_jsi=agg["pooled_jsi"],
               val_n_empty_pred=agg["n_empty_pred"])
    return out


class ProtocolTrainer(Trainer):
    """Official SAM 3 trainer + early stopping, pixel-metric model selection, exact-state resume."""

    def __init__(self, *args, patience: int, score_threshold: float, val_ann_file: str,
                 protocol_hash: str = "", **kwargs) -> None:
        self.patience = patience
        self.score_threshold = score_threshold
        self.val_ann_file = val_ann_file
        self.protocol_hash = protocol_hash
        self.proto = {"best_value": -math.inf, "best_epoch": 0, "bad_epochs": 0, "history": [], "final": False}
        super().__init__(*args, **kwargs)   # ends with load_checkpoint() → our RNG restore

    # ---- paths ------------------------------------------------------------------
    @property
    def run_dir(self) -> Path:
        return Path(self.logging_conf.log_dir).parent

    @property
    def best_pt(self) -> Path:
        return Path(self.checkpoint_conf.save_dir) / "best.pt"

    # ---- checkpointing -----------------------------------------------------------
    def save_checkpoint(self, epoch, checkpoint_names=None):
        """Official checkpoint + RNG states + protocol state; only ``checkpoint.pt`` is kept."""
        self._extra_ckpt = {
            "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
                    "numpy": np.random.get_state(), "python": random.getstate()},
            "protocol": dict(self.proto), "protocol_hash": self.protocol_hash,
        }
        super().save_checkpoint(epoch, checkpoint_names=["checkpoint"])

    def _save_checkpoint(self, checkpoint, checkpoint_path):
        checkpoint.update(getattr(self, "_extra_ckpt", {}))
        super()._save_checkpoint(checkpoint, checkpoint_path)

    def _load_resuming_checkpoint(self, ckpt_path: str):
        super()._load_resuming_checkpoint(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "protocol" not in ckpt:
            raise RuntimeError(f"{ckpt_path} was not written by ProtocolTrainer; cannot resume its RNG / early-stopping state")
        if self.protocol_hash and ckpt.get("protocol_hash") != self.protocol_hash:
            raise RuntimeError(f"{ckpt_path}: protocol changed since this run started")
        self.proto = ckpt["protocol"]
        rng = ckpt["rng"]
        torch.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state_all(rng["cuda"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
        self._write_results()
        logging_info(f"[protocol] resumed at epoch {self.epoch} (best {self.proto['best_value']:.4f} "
                     f"@ {self.proto['best_epoch']}, bad epochs {self.proto['bad_epochs']}, final={self.proto['final']})")

    # ---- results ---------------------------------------------------------------
    def _write_results(self) -> None:
        path = self.run_dir / "results.csv"
        tmp = path.with_name(".results.csv.tmp")
        with tmp.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
            w.writeheader()
            w.writerows(self.proto["history"])
        os.replace(tmp, path)

    def _save_best(self, epoch: int, metrics: dict[str, Any]) -> None:
        from sam3.train.trainer import unwrap_ddp_if_wrapped
        makedir(str(self.best_pt.parent))
        tmp = self.best_pt.with_name(".best.pt.tmp")
        torch.save({"epoch": epoch, "model": unwrap_ddp_if_wrapped(self.model).state_dict(),
                    "metrics": metrics, "protocol_hash": self.protocol_hash}, tmp)
        os.replace(tmp, self.best_pt)

    # ---- loop --------------------------------------------------------------------
    def run(self):
        assert self.mode == "train", "ProtocolTrainer drives full training runs only"
        while self.epoch < self.max_epochs and not self.proto["final"]:
            t0 = time.perf_counter()
            loader = self.train_dataset.get_loader(epoch=int(self.epoch))
            barrier()
            train_out = self.train_epoch(loader)
            self.logger.log_dict(train_out, self.epoch)
            del loader
            gc.collect()

            val_out = self._validate()
            epoch = int(self.epoch) + 1                          # 1-based number of the finished epoch
            improved = val_out["val_jsi"] > self.proto["best_value"]
            if improved:
                self.proto.update(best_value=val_out["val_jsi"], best_epoch=epoch, bad_epochs=0)
            else:
                self.proto["bad_epochs"] += 1
            row = {"epoch": epoch, "time_s": round(time.perf_counter() - t0, 2),
                   "train_loss": _find(train_out, "Losses/train", "all"),
                   "val_loss": _find(val_out, "Losses/val"),
                   **{k: val_out[k] for k in RESULT_COLUMNS if k.startswith("val_") and k in val_out},
                   "val_coco_ap_segm": _find(val_out, "segm", "AP"),
                   "val_coco_ap_bbox": _find(val_out, "bbox", "AP"),
                   "improved": improved}
            self.proto["history"].append(row)
            self.proto["final"] = self.proto["bad_epochs"] >= self.patience or epoch >= self.max_epochs
            if improved:
                self._save_best(epoch, row)
            self.epoch += 1
            self.save_checkpoint(self.epoch)                     # after validation, with RNG states
            self._write_results()
            logging_info(f"[protocol] epoch {epoch}/{self.max_epochs} val_jsi={val_out['val_jsi']:.4f} "
                         f"dsc={val_out['val_dsc']:.4f}{' *' if improved else ''} "
                         f"(best {self.proto['best_value']:.4f} @ {self.proto['best_epoch']}, "
                         f"bad {self.proto['bad_epochs']}/{self.patience})")
        Path(self.run_dir / "protocol_final.json").write_text(json.dumps(
            {"final": True, "best_epoch": self.proto["best_epoch"], "best_value": self.proto["best_value"],
             "epochs_trained": len(self.proto["history"])}))

    def _validate(self) -> dict[str, Any]:
        """Official validation + the study's pixel metrics on the dumped predictions."""
        dataloader = self.val_dataset.get_loader(epoch=int(self.epoch))
        out = self.val_epoch(dataloader, phase="val")
        del dataloader
        gc.collect()
        if torch.cuda.is_available() and self.empty_gpu_mem_cache_after_eval:
            torch.cuda.empty_cache()
        dump = Path(self.run_dir) / "dumps" / "coco_predictions_segm.json"
        out.update(pixel_metrics_from_dump(dump, Path(self.val_ann_file), self.score_threshold))
        return out


def logging_info(msg: str) -> None:
    import logging

    logging.info(msg)
    print(msg, flush=True)
