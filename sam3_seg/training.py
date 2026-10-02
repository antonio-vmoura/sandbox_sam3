"""Resumable SAM 3 training runs (Phases 1–4) — mirror of the U-Net / YOLO26 ``training.py``.

:func:`train_or_resume` turns a protocol (:mod:`common`) into a run of the
official SAM 3 trainer wrapped by :class:`protocol_trainer.ProtocolTrainer`:

* the config is built from the frozen official recipe
  (``configs/sam3_base_recipe.yaml``) + the protocol (data files, FP32, seed 0,
  budget, early stopping, prompt data, hyperparameters) and written to
  ``<run_dir>/config.yaml`` (generated configs never enter the source tree);
* the run is a separate process (``run_training.py``): a CUDA error or an
  out-of-memory cannot take the caller (e.g. the HPO driver) down;
* ``<run_dir>/run_state.json`` records the lifecycle (status, protocol /
  protocol hash, data fingerprints, events, metrics) with atomic writes; a
  completed run is skipped; a changed protocol or data selection is refused;
  ``--force`` moves the old run to ``*.bak-<UTC>``; a ``lockf`` lock prevents two
  processes from training the same run;
* re-running an interrupted run resumes it from the end of its last epoch
  (official ``checkpoint.pt`` + the RNG / early-stopping state of
  ProtocolTrainer) — bit-exactly under ``strict_determinism``, otherwise up to
  the attention-backward noise documented in ``run_training.py``;
* disk: the resume checkpoint (9.4 GB: weights + AdamW state) is deleted once
  the run is complete (a complete run is never resumed, only skipped);
  ``keep_best=False`` (HPO trials) also deletes ``best.pt`` (3.4 GB) — the
  trial's metrics stay in ``results.csv`` / ``run_state.json``.

Outputs in ``<run_dir>``: ``config.yaml``, ``train.log``, ``results.csv``,
``checkpoints/best.pt`` (``checkpoint.pt`` while running), ``protocol_final.json``,
``run_state.json`` and the official ``logs/``, ``dumps/``, ``tensorboard/``.
"""

from __future__ import annotations

import csv
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from common import (
    atomic_write_json,
    config_hash,
    exclusive_lock,
    read_json,
    sha256_file,
    utc_now_iso,
    utc_stamp,
)

#: File that records the lifecycle of one training run.
RUN_STATE_FILE: str = "run_state.json"

#: Frozen official recipe every config is derived from.
BASE_RECIPE: Path = Path(__file__).resolve().parent / "configs" / "sam3_base_recipe.yaml"

#: Protocol keys that do not affect the result (allowed to differ on resume).
_HASH_EXCLUDED: frozenset[str] = frozenset({"device", "train_workers", "val_workers"})

#: Exit code of run_training.py for CUDA out-of-memory.
EXIT_OOM: int = 3


class TrainingOOM(RuntimeError):
    """The training process ran out of GPU memory."""


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
def _find_transform(transforms, suffix: str):
    """First transform (recursively, inside ComposeAPI) whose ``_target_`` ends with ``suffix``."""
    for t in transforms:
        if str(t.get("_target_", "")).endswith(suffix):
            return t
        if "transforms" in t:
            found = _find_transform(t["transforms"], suffix)
            if found is not None:
                return found
    return None


def build_config(protocol: dict[str, Any], image_dir: Path, train_json: Path, val_json: Path,
                 run_dir: Path, protocol_hash: str):
    """Official recipe + protocol → OmegaConf config for :mod:`run_training`."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(BASE_RECIPE)
    cfg.paths.experiment_log_dir = str(run_dir)
    cfg.launcher.experiment_log_dir = str(run_dir)
    cfg.launcher.gpus_per_node = 1
    cfg.roboflow_train.num_images = protocol.get("limit_images")

    # ---- hyperparameters (tunable) --------------------------------------------
    cfg.scratch.lr_scale = float(protocol["lr_scale"])
    cfg.scratch.wd = float(protocol["weight_decay"])
    cfg.scratch.lrd_vision_backbone = float(protocol["lrd_vision_backbone"])
    cfg.scratch.scheduler_warmup = int(protocol["scheduler_warmup"])
    train_tf = cfg.roboflow_train.train_transforms
    flip = _find_transform(train_tf, "RandomHorizontalFlip")
    resize = _find_transform(train_tf, "RandomResizeAPI")
    if flip is None or resize is None:
        raise RuntimeError("official recipe changed: RandomHorizontalFlip / RandomResizeAPI not found")
    flip.p = float(protocol["hflip_p"])
    resize.sizes.min_size = int(protocol["resize_min_size"])

    # ---- base setup --------------------------------------------------------------
    if protocol["resolution"] != cfg.scratch.resolution:
        raise RuntimeError("resolution is part of the official recipe and must not change")
    cfg.scratch.train_batch_size = int(protocol["batch"])
    cfg.scratch.gradient_accumulation_steps = int(protocol["grad_accum_chunks"])
    if protocol["grad_accum_chunks"] != 1:
        raise RuntimeError("gradient accumulation is not supported (the vendored trainer's path is broken)")
    cfg.scratch.num_train_workers = int(protocol["train_workers"])
    cfg.scratch.num_val_workers = int(protocol["val_workers"])

    t = cfg.trainer
    t._target_ = "protocol_trainer.ProtocolTrainer"
    t.patience = int(protocol["patience"])
    t.score_threshold = float(protocol["score_threshold"])
    t.val_ann_file = str(val_json)
    t.protocol_hash = protocol_hash
    t.max_epochs = int(protocol["epochs"])
    t.seed_value = int(protocol["seed"])
    t.val_epoch_freq = 1
    t.skip_first_val = False
    t.skip_saving_ckpts = False
    t.optim.amp.enabled = bool(protocol["amp"])
    t.cuda = {"cudnn_deterministic": bool(protocol["deterministic"]), "cudnn_benchmark": False}
    cfg.study = {"deterministic": bool(protocol["deterministic"]),
                 "strict_determinism": bool(protocol["strict_determinism"])}   # read by run_training.py
    t.checkpoint.save_freq = 0
    t.checkpoint.save_list = []
    t.data.train.dataset.img_folder = str(image_dir) + "/"
    t.data.train.dataset.ann_file = str(train_json)
    t.data.val.dataset.img_folder = str(image_dir) + "/"
    t.data.val.dataset.ann_file = str(val_json)
    for ev in t.meters.val.isic_val.detection.pred_file_evaluators:
        ev.gt_path = str(val_json)
    return cfg


# ----------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------
def read_results(csv_path: Path) -> list[dict[str, Any]]:
    """Parsed ``results.csv`` rows."""
    with Path(csv_path).open() as f:
        return [{k: (v == "True" if k == "improved" else float(v)) for k, v in r.items()} for r in csv.DictReader(f)]


def best_metrics(run_dir: Path) -> dict[str, Any]:
    """Validation metrics of the epoch stored in ``best.pt`` (from ``protocol_final.json`` + ``results.csv``)."""
    final = read_json(Path(run_dir) / "protocol_final.json")
    rows = {int(r["epoch"]): r for r in read_results(Path(run_dir) / "results.csv")}
    row = rows[int(final["best_epoch"])]
    return {**{k: v for k, v in row.items() if k.startswith("val_")},
            "best_epoch": int(final["best_epoch"]), "epochs_trained": len(rows)}


def backup_dir(path: Path) -> Path | None:
    """Move ``path`` to ``<path>.bak-<UTC>``; return the backup path."""
    if not path.exists():
        return None
    backup = path.with_name(f"{path.name}.bak-{utc_stamp()}")
    path.rename(backup)
    print(f"  [force] moved previous run to {backup}")
    return backup


def _sha_list(items: list[str]) -> str:
    return hashlib.sha256("\n".join(items).encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------
# Resumable training
# ----------------------------------------------------------------------------
def train_or_resume(*, phase: str, model_name: str, protocol: dict[str, Any], image_dir: Path,
                    train_json: Path, val_json: Path, project: Path, name: str,
                    force: bool = False, keep_best: bool = True) -> dict[str, Any]:
    """Train ``project/name`` to completion, resuming or skipping as needed.

    Returns:
        Summary dict with ``model``, ``skipped``, ``resumed``, ``reason``,
        ``elapsed_min`` and ``metrics``.

    Raises:
        TrainingOOM: CUDA out of memory.
        RuntimeError: Protocol/data mismatch, lock held, or training failure.
    """
    with exclusive_lock(Path(project), f".{name}.lock"):
        return _train_locked(phase, model_name, protocol, Path(image_dir), Path(train_json), Path(val_json),
                             Path(project), name, force, keep_best)


def _train_locked(phase, model_name, protocol, image_dir, train_json, val_json, project, name, force, keep_best):
    from omegaconf import OmegaConf

    run_dir = project / name
    state_path, ckpt = run_dir / RUN_STATE_FILE, run_dir / "checkpoints" / "checkpoint.pt"
    data_spec = {"train_json_sha256": sha256_file(train_json), "val_json_sha256": sha256_file(val_json),
                 "image_dir": str(image_dir)}
    phash = config_hash({"protocol": {k: v for k, v in protocol.items() if k not in _HASH_EXCLUDED},
                         "data": data_spec, "recipe_sha256": sha256_file(BASE_RECIPE)})
    if force:
        backup_dir(run_dir)
    state = read_json(state_path)
    if state is not None and state.get("protocol_hash") != phash:
        raise RuntimeError(f"{run_dir} was trained with a different protocol/data "
                           f"({state.get('protocol_hash')} != {phash}). Use --force to retrain.")
    if state is not None and state.get("status") == "complete":
        return {"model": model_name, "skipped": True, "resumed": False, "elapsed_min": 0.0,
                "reason": f"complete ({state_path})", "metrics": best_metrics(run_dir)}
    if state is None and run_dir.exists():
        backup_dir(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if state is None:
        state = {"status": "running", "phase": phase, "model": model_name, "run_dir": str(run_dir),
                 "protocol": {k: v for k, v in protocol.items() if k not in _HASH_EXCLUDED},
                 "protocol_hash": phash, "data": data_spec, "events": []}

    def log(event: str, **info: Any) -> None:
        state["events"].append({"at": utc_now_iso(), "event": event, **info})
        atomic_write_json(state_path, state)

    resumed = ckpt.exists()
    log("resume" if resumed else "start")
    if resumed:
        print(f"  [resume] continuing {run_dir.name} from its last checkpoint")
    OmegaConf.save(build_config(protocol, image_dir, train_json, val_json, run_dir, phash), run_dir / "config.yaml")

    env = dict(os.environ)
    if protocol["device"] != "cpu":
        env.setdefault("CUDA_VISIBLE_DEVICES", str(protocol["device"]))
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(Path(__file__).resolve().parent),
                                                      env.get("PYTHONPATH", "")]))
    env["PYTHONUNBUFFERED"] = "1"     # stream the trainer's lines as they are printed
    t0 = time.perf_counter()
    # The trainer's output goes to train.log AND to our stdout, so progress lines
    # also reach the terminal log (tee'd by wait_gpu_sam3.sh).
    with (run_dir / "train.log").open("a") as logf, \
            subprocess.Popen([sys.executable, str(Path(__file__).resolve().parent / "run_training.py"),
                              str(run_dir / "config.yaml")], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             env=env, text=True, errors="replace", bufsize=1) as proc:
        for line in proc.stdout:
            logf.write(line)
            logf.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
        rc = proc.wait()
    if rc == EXIT_OOM:
        log("oom")
        raise TrainingOOM(f"{run_dir}: CUDA out of memory (see train.log)")
    final = read_json(run_dir / "protocol_final.json")
    if rc != 0 or final is None:
        tail = (run_dir / "train.log").read_text().splitlines()[-15:]
        log("failed", exit_code=rc)
        raise RuntimeError(f"{run_dir}: training failed (exit {rc}):\n" + "\n".join(tail))

    metrics = best_metrics(run_dir)
    best_pt = run_dir / "checkpoints" / "best.pt"
    ckpt.unlink(missing_ok=True)                    # resume state is useless once complete
    if not keep_best:
        best_pt.unlink(missing_ok=True)
    state.update(status="complete", metrics=metrics, epochs_trained=metrics["epochs_trained"],
                 best_epoch=metrics["best_epoch"], best_pt=str(best_pt) if keep_best else None)
    log("complete", elapsed_min=round((time.perf_counter() - t0) / 60, 2))
    return {"model": model_name, "skipped": False, "resumed": resumed, "reason": None,
            "elapsed_min": (time.perf_counter() - t0) / 60, "metrics": metrics}



# ----------------------------------------------------------------------------
# Phase helpers (identical to the U-Net pipeline)
# ----------------------------------------------------------------------------
def load_tuned_hp(path: Path) -> dict[str, Any]:
    """Load the ``best_hyperparameters.yaml`` written by Phase 3.

    Raises:
        ValueError: If the YAML is empty (signals a failed tune).
    """
    import yaml

    with Path(path).open() as f:
        data = yaml.safe_load(f) or {}
    if not data:
        raise ValueError(f"Empty YAML at {path}. Did Phase 3 complete?")
    return data


def require_complete_hpo(state_path: Path) -> None:
    """Raise unless Phase 3's ``hpo_state.json`` reports a complete search."""
    state = read_json(state_path)
    if state is None:
        raise RuntimeError(f"HPO checkpoint not found: {state_path}. Run Phase 3 first.")
    if state.get("status") != "complete":
        raise RuntimeError(
            f"HPO is not complete ({state.get('completed_trials')}/{state.get('target_trials')} "
            f"trials, status={state.get('status')!r}). Re-run Phase 3 to resume it.",
        )


def print_phase_summary(title: str, summary: list[dict], total_min: float) -> None:
    """Per-model summary table shared by the training phases."""
    print("\n" + "=" * 80)
    print(f"=== {title} — SUMMARY")
    print("=" * 80)
    for s in summary:
        m = s.get("metrics") or {}
        score = (f"  val JSI={m['val_jsi']:.4f}  DSC={m['val_dsc']:.4f}  best epoch {m.get('best_epoch')}"
                 if "val_jsi" in m else "")
        if s.get("failed"):
            status = f"FAILED ({s.get('reason')})"
        elif s.get("skipped"):
            status = "skipped (complete)"
        else:
            status = f"ok{' (resumed)' if s.get('resumed') else ''} in {s['elapsed_min']:.1f} min"
        print(f"  {s['model']:<8} : {status}{score}")
    print(f"\nTotal time: {total_min:.1f} min ({total_min / 60:.2f} h)")
