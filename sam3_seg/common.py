"""Shared configuration and helpers for the 5-phase SAM 3 pipeline.

Mirror of ``sandbox_yolo26/yolo26_seg/common.py`` and ``sandbox_unet/unet/common.py``:
constants, the protocol definition, output paths and small I/O helpers live
here, so the experimental protocol is defined **once**. The generic helpers
(:class:`PipelinePaths`, atomic JSON, hashing, the ``lockf`` lock) are
identical to the U-Net's.

SAM 3 specifics
---------------
* The model and its fine-tuning recipe are Meta's official SAM 3 code (the
  vendored ``sam3`` package, Hydra configs + ``sam3.train.trainer.Trainer``);
  the official recipe values are the **default hyperparameters**
  (:data:`DEFAULT_HPS`, taken from the original Phase 1 configuration).
* Text prompt: the single COCO category is named :data:`PROMPT`
  (``"skin lesion"``, clinically neutral — the original data used
  ``"skin cancer"`` although most ISIC lesions are benign).
* Input resolution: SAM 3's native 1008 × 1008.

Baseline = base setup + default HPs; Optimised = base setup + tuned HPs. Only
the learning dynamics and the existing spatial augmentations
(:data:`TUNABLE_KEYS`) may differ; everything else is the base setup.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import random
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Union

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
#: Model names handled by the pipeline (one SAM 3 image model).
DEFAULT_ORDER: list[str] = ["sam3"]

#: YOLO-format dataset — the single source of truth shared with YOLO26 and U-Net.
DEFAULT_YOLO_DATA_YAML: str = "/workspace/datasets/isic_2018_task1_yolo26/data.yaml"

#: Phase 0 output: COCO dataset (shared image pool + per-split / per-fold annotations).
DEFAULT_DATA_DIR: str = "/workspace/datasets/isic_2018_task1_sam3"

#: Default pipeline root; isolates this study from older runs under ``logs/``.
DEFAULT_PIPELINE_ROOT: str = "/workspace/logs/pipeline_final_v1"

#: Text prompt = name of the single COCO category.
PROMPT: str = "skin lesion"

#: Global seed for every RNG in the pipeline.
SEED: int = 0

#: Training budget shared by Phases 1, 2 and 4 — **SAM 3-specific** (YOLO26 and
#: the U-Net use 120 / 25). One FP32 epoch of the 840 M-parameter model takes
#: ~106 min on the V100S (memory probe), so 120 epochs (> 8 days per run) are
#: infeasible; this reduced budget is a disclosed limitation of the study.
TRAIN_EPOCHS: int = 30
TRAIN_PATIENCE: int = 10

#: Phase 3 budget (SAM 3-specific; YOLO26 / U-Net: 30 trials x 30 epochs).
HPO_ITERATIONS: int = 10
HPO_EPOCHS: int = 10
HPO_PATIENCE: int = 5

#: SAM 3 native input resolution.
RESOLUTION: int = 1008

#: Default hyperparameters = the official SAM 3 fine-tuning recipe as used in the
#: original Phase 1 configuration (names of the search space).
DEFAULT_HPS: dict[str, Any] = {
    "lr_scale": 0.10,             # scales lr_transformer 8e-4, lr_vision 2.5e-4, lr_language 5e-5
    "weight_decay": 0.10,         # AdamW weight decay (``scratch.wd``)
    "lrd_vision_backbone": 0.9,   # layer-wise LR decay of the vision trunk
    "scheduler_warmup": 2,        # inverse-sqrt scheduler warm-up steps
    "hflip_p": 0.5,               # RandomHorizontalFlip probability
    "resize_min_size": 480,       # min size of the random-resize scale jitter (max = RESOLUTION)
}

#: The only keys an HPO search space / tuned YAML may contain.
TUNABLE_KEYS: frozenset[str] = frozenset(DEFAULT_HPS)

#: Type alias for the device argument.
DeviceArg = Union[int, str]


# ----------------------------------------------------------------------------
# Device & reproducibility
# ----------------------------------------------------------------------------
def parse_device(arg: str) -> DeviceArg:
    """Parse ``--device``: a single GPU id (``"0"``) or ``"cpu"``.

    Raises:
        ValueError: For a multi-GPU list — SAM 3 is fine-tuned on one GPU.
    """
    if "," in arg:
        raise ValueError("the SAM 3 pipeline uses a single device (e.g. --device 0)")
    return "cpu" if arg == "cpu" else int(arg)


def seed_everything(seed: int = SEED, deterministic: bool = True) -> None:
    """Seed every RNG of the process and request deterministic kernels (call first in ``main``)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


# ----------------------------------------------------------------------------
# Output layout and I/O helpers (identical to the U-Net pipeline)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class PipelinePaths:
    """Canonical output layout of one pipeline run rooted at ``root``."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    # ---- Phase 1 — baseline ------------------------------------------------
    @property
    def phase1_dir(self) -> Path:
        return self.root / "phase1_baseline"

    @staticmethod
    def phase1_run_name(model: str) -> str:
        return f"{model}_baseline"

    def phase1_best_pt(self, model: str) -> Path:
        return self.phase1_dir / self.phase1_run_name(model) / "checkpoints" / "best.pt"

    # ---- Phase 2 — cross-validation ----------------------------------------
    def cv_dir(self, protocol: str = "baseline") -> Path:
        """CV root for a protocol; Phase 2 is ``baseline``, ``optimized`` is optional."""
        return self.root / f"phase2_cv_{protocol}"

    def cv_model_dir(self, model: str, protocol: str = "baseline") -> Path:
        return self.cv_dir(protocol) / model

    # ---- Phase 3 — HPO -----------------------------------------------------
    @property
    def phase3_dir(self) -> Path:
        return self.root / "phase3_hpo"

    @staticmethod
    def phase3_tune_name(model: str) -> str:
        return f"tune_{model}"

    def phase3_tune_dir(self, model: str) -> Path:
        return self.phase3_dir / self.phase3_tune_name(model)

    def phase3_best_yaml(self, model: str) -> Path:
        return self.phase3_tune_dir(model) / "best_hyperparameters.yaml"

    def phase3_state(self, model: str) -> Path:
        return self.phase3_tune_dir(model) / "hpo_state.json"

    # ---- Phase 4 — optimised fine-tune -------------------------------------
    @property
    def phase4_dir(self) -> Path:
        return self.root / "phase4_optimized"

    @staticmethod
    def phase4_run_name(model: str) -> str:
        return f"{model}_optimized"

    def phase4_best_pt(self, model: str) -> Path:
        return self.phase4_dir / self.phase4_run_name(model) / "checkpoints" / "best.pt"

    # ---- Phase 5 & summaries -----------------------------------------------
    def best_pt(self, variant: str, model: str) -> Path:
        """Weights evaluated in Phase 5: ``baseline`` (Phase 1) or ``optimized`` (Phase 4)."""
        if variant == "baseline":
            return self.phase1_best_pt(model)
        if variant == "optimized":
            return self.phase4_best_pt(model)
        raise ValueError(f"unknown variant {variant!r}")

    @property
    def phase5_dir(self) -> Path:
        return self.root / "phase5_test"

    @staticmethod
    def phase5_tag(variant: str, model: str, precision: str) -> str:
        return f"{variant}_{model}_{precision}"

    def phase5_accuracy_json(self, variant: str, model: str, precision: str) -> Path:
        return self.phase5_dir / "accuracy" / f"{self.phase5_tag(variant, model, precision)}.json"

    def phase5_per_image_csv(self, variant: str, model: str, precision: str) -> Path:
        return self.phase5_dir / "per_image" / f"{self.phase5_tag(variant, model, precision)}.csv"

    def phase5_mask_dir(self, variant: str, model: str) -> Path:
        """Predicted test masks (FP32 only) for the visualisation notebook."""
        return self.phase5_dir / "masks" / f"{variant}_{model}"

    def phase5_efficiency_json(self, variant: str, model: str, precision: str) -> Path:
        return self.phase5_dir / "efficiency" / f"{self.phase5_tag(variant, model, precision)}.json"

    @property
    def summary_dir(self) -> Path:
        return self.root / "summary"

    @property
    def pipeline_runs_dir(self) -> Path:
        return self.root / "pipeline_runs"


# ----------------------------------------------------------------------------
# Small I/O helpers (identical to the YOLO26 pipeline)
# ----------------------------------------------------------------------------
def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string (second precision)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def utc_stamp() -> str:
    """Compact UTC timestamp for backup names."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON so ``path`` is never half-written (tmp + fsync + ``os.replace``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path: Path) -> dict[str, Any] | None:
    """Return the parsed JSON at ``path``, or ``None`` if it does not exist."""
    path = Path(path)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def config_hash(payload: Any) -> str:
    """Return a short, stable SHA-256 fingerprint of a JSON-serialisable config."""
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Return the SHA-256 of a file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


@contextmanager
def exclusive_lock(directory: Path, name: str = ".lock") -> Iterator[None]:
    """Hold a non-blocking exclusive POSIX lock (``lockf``) on ``directory/name``.

    ``lockf`` (not ``flock``): a POSIX record lock belongs to the *process* and
    is not inherited by ``fork()``-ed children. With ``flock`` the lock belongs
    to the open file and survives in forked DataLoader workers, so after a
    ``kill -9`` the orphaned workers kept the run locked and an immediate
    restart was refused. The lock is released as soon as the process dies.

    Raises:
        RuntimeError: If another process already holds the lock.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / name).open("w") as fh:
        try:
            fcntl.lockf(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            raise RuntimeError(f"another process is already working in {directory}") from e
        try:
            yield
        finally:
            fcntl.lockf(fh, fcntl.LOCK_UN)


# ----------------------------------------------------------------------------
# Training protocols
# ----------------------------------------------------------------------------
#: Fixed base setup shared by EVERY phase (never searched, never overridden).
#: Memory probe (V100S 32 GB, FP32): batch 2 with the official activation
#: checkpointing peaks at 12.8 GiB allocated / 16.5 GiB on the device.
BASE_SETUP: dict[str, Any] = {
    "resolution": RESOLUTION,
    "prompt": PROMPT,
    "batch": 2,                   # official train_batch_size
    "grad_accum_chunks": 1,       # no gradient accumulation (fits in FP32)
    "act_ckpt": True,             # official setting (numerically neutral; memory/speed only)
    "amp": False,                 # FP32 (protocol): no FP16 mixed precision
    "seed": SEED,
    "deterministic": True,        # seeds + cuDNN deterministic + warn-only deterministic algorithms
                                  # + deterministic grid_sample (see run_training.py)
    "strict_determinism": False,  # True = bit-exact (warn_only=False), +33 % step time (6.67 vs 5.01 s)
    "monitor": "val_jsi",         # model selection / early stopping
    "score_threshold": 0.5,       # instances merged into the predicted mask (SAM 3 default)
    "train_workers": 8,
    "val_workers": 4,
}

#: Keys that define the base setup, budget and reproducibility contract.
PROTECTED_KEYS: frozenset[str] = frozenset({*BASE_SETUP, "epochs", "patience", "device", "limit_images"})


def _check_tunable(hps: dict[str, Any], what: str) -> None:
    """Reject protected or unknown keys in a hyperparameter mapping."""
    clash = PROTECTED_KEYS.intersection(hps)
    if clash:
        raise ValueError(f"{what} must not override base-setup keys: {sorted(clash)}")
    unknown = set(hps) - TUNABLE_KEYS
    if unknown:
        raise ValueError(f"{what} contains unknown keys: {sorted(unknown)} (allowed: {sorted(TUNABLE_KEYS)})")


def baseline_protocol(device: DeviceArg, epochs: int = TRAIN_EPOCHS, patience: int = TRAIN_PATIENCE,
                      limit_images: int | None = None) -> dict[str, Any]:
    """Phase 1 / Phase 2 protocol: base setup + default (official recipe) HPs.

    ``limit_images`` restricts the training set — smoke tests only.
    """
    return {**BASE_SETUP, **DEFAULT_HPS, "epochs": epochs, "patience": patience, "device": device,
            "limit_images": limit_images}


def hpo_trial_protocol(device: DeviceArg, params: dict[str, Any], epochs: int = HPO_EPOCHS,
                       patience: int = HPO_PATIENCE) -> dict[str, Any]:
    """Phase 3 trial protocol: base setup + proposed HPs, trial budget."""
    params = dict(params)
    _check_tunable(params, "HPO proposal")
    return {**baseline_protocol(device, epochs, patience), **params}


def optimized_protocol(device: DeviceArg, tuned_hp: dict[str, Any] | None = None, epochs: int = TRAIN_EPOCHS,
                       patience: int = TRAIN_PATIENCE, limit_images: int | None = None) -> dict[str, Any]:
    """Phase 4 protocol: base setup + tuned HPs (defaults for the rest)."""
    tuned_hp = dict(tuned_hp or {})
    _check_tunable(tuned_hp, "Tuned hyperparameters")
    return {**baseline_protocol(device, epochs, patience, limit_images), **tuned_hp}
