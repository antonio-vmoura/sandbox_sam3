"""Phase 3 — Fault-tolerant, reproducible HPO of SAM 3 with Optuna (TPE).

Mirror of the U-Net's ``tune_unet.py`` (same sampler protocol, checkpoint
schema, retry logic and outputs); each trial is a full fine-tuning run of the
official SAM 3 trainer (:mod:`training`).

The search keeps the framework of the original pipeline (Optuna, TPE sampler,
SQLite storage) and adds the same guarantees as YOLO26's ``SeededTuner``:

Search space (strict "apples-to-apples")
    Only the learning dynamics and augmentation are searched
    (:data:`SEARCH_SPACE`: lr_scale, weight_decay, lrd_vision_backbone,
    scheduler_warmup, and the recipe's two spatial augmentations hflip_p and
    resize_min_size). Architecture, resolution, prompt, losses, matcher,
    optimiser type, batch, budget and precision belong to the base setup and
    cannot be searched; ``focal_gamma`` (a loss parameter) is deliberately not
    searched. The first trial evaluates the default (Baseline) hyperparameters,
    which all lie inside the bounds.

Budget (SAM 3-specific, disclosed): :data:`common.HPO_ITERATIONS` = 10 trials
x :data:`common.HPO_EPOCHS` = 10 epochs (patience 10 = no early stopping) on the full train split —
YOLO26 / U-Net use 30 x 30. With 10 trials, Optuna's default of 10 random
start-up trials would leave no trial to TPE; :data:`TPE_STARTUP_TRIALS` = 5
(the defaults + 4 random proposals, then 5 TPE proposals).

Reproducibility
    A fresh ``TPESampler`` is installed before every proposal, seeded from
    ``(seed, i)`` where *i* is the number of completed trials. TPE builds its
    model only from COMPLETE trials and draws all randomness from that seed,
    so the proposal for trial *i* is a pure function of (seed, i, history of
    completed trials). The proposed parameters are recorded and a resumed or
    retried proposal is **verified** to be identical (else the run stops).

Fault tolerance
    * ``hpo_state.json`` (same schema as YOLO26) is checkpointed atomically
      before every trial; completion is ``completed_trials == --iterations``.
    * Trials left RUNNING by a crash are marked FAIL ("interrupted", not
      counted as a failure). The same proposal is asked again — it receives
      identical parameters and the **same trial folder**, so the interrupted
      training itself resumes from its ``checkpoint.pt``.
    * A trial that raises or returns a non-finite fitness is retried with the
      same parameters (from a clean folder) up to ``--max-trial-retries``
      times, then recorded as a completed trial with fitness 0 (as YOLO26).
      Failures that coincide with an unhealthy GPU are not counted, and the
      script exits with :data:`EXIT_GPU_UNAVAILABLE` (75) so the orchestrator
      retries later.
    * Disk: a finished trial keeps only its metrics (``results.csv``,
      ``run_state.json``, ``train.log``); its weights are deleted.
    * Resuming with a different search space, base setup, seed, data or
      Optuna/torch version is refused (config hash); a lock prevents two
      processes from tuning the same model; ``--force`` moves the previous
      search to ``tune_<model>.bak-<UTC>``.

Fitness: validation per-image mean JSI of the trial's best epoch (the same
criterion that selects checkpoints in every phase). Trials train on ``train``
and are scored on ``val``; the test split is never used.

Outputs (per model)::

    <project>/phase3_hpo/tune_<model>/
    ├── tune_results.csv            # fitness + params per completed trial (YOLO26 format)
    ├── best_hyperparameters.yaml   # consumed by Phase 4
    ├── hpo_state.json              # checkpoint
    ├── optuna_study.db             # Optuna storage (SQLite)
    └── trials/trial_<i>/           # resumable training run of each trial

Exit codes: 0 complete; 1 a model failed; 75 GPU/driver unavailable.

Usage:
    python sam3_seg/tune_sam3.py --project /workspace/logs/pipeline_final_v1 --device 0
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

import optuna
import torch
import yaml
from optuna.distributions import FloatDistribution, IntDistribution
from optuna.trial import TrialState

from common import (
    BASE_SETUP,
    DEFAULT_DATA_DIR,
    DEFAULT_HPS,
    DEFAULT_ORDER,
    DEFAULT_PIPELINE_ROOT,
    HPO_EPOCHS,
    HPO_ITERATIONS,
    HPO_PATIENCE,
    PROTECTED_KEYS,
    SEED,
    TUNABLE_KEYS,
    PipelinePaths,
    atomic_write_json,
    config_hash,
    exclusive_lock,
    hpo_trial_protocol,
    parse_device,
    read_json,
    seed_everything,
    utc_now_iso,
)
from data import CocoData, ids_fingerprint
from training import backup_dir, train_or_resume

# PartialFixedSampler (used for the first trial) has been stable since Optuna 2.4.
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

EXIT_GPU_UNAVAILABLE: int = 75
STATE_SCHEMA: int = 1

#: Search space: name → (low, high, log, step). Learning dynamics + augmentation only.
#: ``step`` is ``None`` for floats; integer parameters use an int step.
SEARCH_SPACE: dict[str, tuple[float, float, bool, int | None]] = {
    "lr_scale":            (0.01, 0.20, True, None),   # range of the original SAM 3 tuner; default 0.1
    "weight_decay":        (0.01, 0.20, True, None),   # range of the original SAM 3 tuner; default 0.1
    "lrd_vision_backbone": (0.60, 1.00, False, None),  # 1.0 = no layer-wise decay; default 0.9
    "scheduler_warmup":    (1, 1000, True, 1),         # optimiser steps (1273 steps / epoch); default 2
    "hflip_p":             (0.0, 0.50, False, None),   # as the U-Net's fliplr; default 0.5
    "resize_min_size":     (320, 1008, False, 16),     # scale-jitter lower bound (px); default 480
}
assert set(SEARCH_SPACE) == TUNABLE_KEYS and not set(SEARCH_SPACE) & PROTECTED_KEYS

#: Random-search trials before TPE modelling starts (Optuna default 10 > the 10-trial budget).
TPE_STARTUP_TRIALS: int = 5

DISTRIBUTIONS = {k: (IntDistribution(int(lo), int(hi), log=log, step=step) if step else
                     FloatDistribution(lo, hi, log=log))
                 for k, (lo, hi, log, step) in SEARCH_SPACE.items()}


class GPUUnavailable(RuntimeError):
    """GPU/driver unhealthy; maps to :data:`EXIT_GPU_UNAVAILABLE`."""


# ----------------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------------
def first_trial_params() -> dict[str, float | int]:
    """Default (Baseline) hyperparameters (all inside the search bounds)."""
    params = {k: (int if step else float)(DEFAULT_HPS[k]) for k, (_, _, _, step) in SEARCH_SPACE.items()}
    outside = [k for k, (lo, hi, _, _) in SEARCH_SPACE.items() if not lo <= params[k] <= hi]
    if outside:
        raise ValueError(f"default hyperparameters outside the search bounds: {outside}")
    return params


def proposal_sampler(seed: int, index: int) -> optuna.samplers.BaseSampler:
    """Sampler for proposal ``index``: seeded TPE (fixed to the defaults for index 0)."""
    tpe = optuna.samplers.TPESampler(seed=(seed * 1_000_003 + index) % 2**32,
                                     n_startup_trials=TPE_STARTUP_TRIALS)
    return optuna.samplers.PartialFixedSampler(first_trial_params(), tpe) if index == 0 else tpe


def completed_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    """COMPLETE trials ordered by proposal index."""
    trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    return sorted(trials, key=lambda t: t.user_attrs["proposal"])


# ----------------------------------------------------------------------------
# Outputs and checkpoint
# ----------------------------------------------------------------------------
def write_outputs(trials: list[optuna.trial.FrozenTrial], tune_dir: Path) -> dict[str, Any]:
    """Write ``tune_results.csv`` and ``best_hyperparameters.yaml``; return progress counters."""
    keys = list(SEARCH_SPACE)
    lines = [",".join(["fitness", *keys])]
    lines += [",".join([repr(float(t.value)), *(repr(t.params[k]) for k in keys)]) for t in trials]
    tmp = tune_dir / ".tune_results.csv.tmp"
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(tune_dir / "tune_results.csv")
    valid = [t for t in trials if not t.user_attrs.get("accepted_failure")]
    progress = {"completed_trials": len(trials), "valid_trials": len(valid),
                "best_fitness": None, "best_trial": None}
    if valid:
        best = max(valid, key=lambda t: (t.value, -t.user_attrs["proposal"]))  # ties → earliest
        progress.update(best_fitness=float(best.value), best_trial=best.user_attrs["proposal"] + 1)
        header = (f"# Phase 3 best of {len(trials)} trial(s): trial {progress['best_trial']}, "
                  f"fitness (val JSI) = {best.value:.6f}\n")
        tmp = tune_dir / ".best_hyperparameters.yaml.tmp"
        tmp.write_text(header + yaml.safe_dump({k: best.params[k] for k in keys}, sort_keys=False))
        tmp.replace(tune_dir / "best_hyperparameters.yaml")
    return progress


class Checkpoint:
    """``hpo_state.json`` (YOLO26 schema); every write is atomic."""

    def __init__(self, path: Path, model: str, config: dict[str, Any], target: int) -> None:
        self.path = path
        self.state = read_json(path) or {
            "schema": STATE_SCHEMA, "model": model, "status": "running", "target_trials": target,
            "completed_trials": 0, "valid_trials": 0, "best_fitness": None, "best_trial": None,
            "in_flight_trial": None, "failed_attempts": {}, "accepted_failures": [],
            "proposals": {}, "config_hash": config_hash(config), "config": config,
            "versions": {"optuna": optuna.__version__, "torch": torch.__version__},
            "created_at": utc_now_iso(), "last_update": utc_now_iso(), "history": [],
        }

    def save(self) -> None:
        self.state["last_update"] = utc_now_iso()
        atomic_write_json(self.path, self.state)

    def log(self, event: str, **info: Any) -> None:
        self.state["history"].append({"at": utc_now_iso(), "event": event, **info})
        self.save()


def gpu_healthy(device) -> bool:
    """``True`` if the NVIDIA driver and ``torch.cuda`` respond (fresh subprocess)."""
    if device == "cpu":
        return True
    probe = ("import sys, torch; "
             "sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 1)")
    try:
        subprocess.run(["nvidia-smi", "-L"], check=True, capture_output=True, timeout=60)
        subprocess.run([sys.executable, "-c", probe], check=True, capture_output=True, timeout=180)
    except (OSError, subprocess.SubprocessError):
        return False
    return True


# ----------------------------------------------------------------------------
# Trials
# ----------------------------------------------------------------------------
def run_trial(index: int, params: dict[str, Any], args, device, data: CocoData, tune_dir: Path) -> float:
    """Train one trial (resumable) and return its fitness (val JSI of the best epoch)."""
    protocol = hpo_trial_protocol(device, params, args.epochs, args.patience)
    result = train_or_resume(phase="phase3_hpo", model_name="sam3", protocol=protocol, image_dir=data.image_dir,
                             train_json=data.split_json("train"), val_json=data.split_json("val"),
                             project=tune_dir / "trials", name=f"trial_{index:03d}", keep_best=False)
    return float(result["metrics"]["val_jsi"])


def _ask(study: optuna.Study, seed: int, index: int, ckpt: Checkpoint) -> optuna.Trial:
    """Ask proposal ``index``; verify it matches any earlier proposal for the same index."""
    study.sampler = proposal_sampler(seed, index)
    trial = study.ask(DISTRIBUTIONS)
    trial.set_user_attr("proposal", index)
    seen = ckpt.state["proposals"].get(str(index))
    if seen is not None and seen != trial.params:
        study.tell(trial, state=TrialState.FAIL)
        raise RuntimeError(f"proposal {index + 1} changed on re-ask ({seen} → {trial.params}); "
                           f"the search is no longer reproducible — use --force to restart")
    ckpt.state["proposals"][str(index)] = trial.params
    return trial


def tune_one_model(model: str, args, device, data: CocoData, paths: PipelinePaths) -> dict:
    """Run (or resume) the HPO of one model until it is complete."""
    tune_dir = paths.phase3_tune_dir(model)
    if args.force:
        backup_dir(tune_dir)
    config = {
        "model": model, "space": {k: list(v) for k, v in SEARCH_SPACE.items()},
        "base_setup": {k: v for k, v in BASE_SETUP.items() if not k.endswith("workers")},
        "trial_budget": {"epochs": args.epochs, "patience": args.patience},
        "seed": args.seed, "sampler": {"type": "TPE", "n_startup_trials": TPE_STARTUP_TRIALS,
                                       "first_trial": "defaults clipped to bounds"},
        "data": {"dataset": data.fingerprint(), "train": ids_fingerprint(data.ids(data.split_json("train"))),
                 "val": ids_fingerprint(data.ids(data.split_json("val")))},
    }
    with exclusive_lock(tune_dir, ".hpo.lock"):
        ckpt = Checkpoint(paths.phase3_state(model), model, config, args.iterations)
        st = ckpt.state
        if st["config_hash"] != config_hash(config):
            raise RuntimeError(f"Configuration changed since this search started ({ckpt.path}); "
                               f"resuming would mix trials. Revert the change or use --force.")
        versions = {"optuna": optuna.__version__, "torch": torch.__version__}
        if st["versions"] != versions:
            msg = f"library versions changed since the search started ({st['versions']} → {versions})"
            if not args.allow_version_change:
                raise RuntimeError(msg + "; pass --allow-version-change to resume anyway")
            print(f"  [warn] {msg}")
        st["target_trials"] = args.iterations

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(study_name=f"{model}_hpo", direction="maximize", load_if_exists=True,
                                    storage=f"sqlite:///{tune_dir / 'optuna_study.db'}")
        for t in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
            study._storage.set_trial_user_attr(t._trial_id, "interrupted", True)
            study.tell(t.number, state=TrialState.FAIL)
            ckpt.log("trial_interrupted", trial=t.user_attrs.get("proposal", -1) + 1)
            print(f"  [resume] trial {t.user_attrs.get('proposal', -1) + 1} was interrupted — it will resume")

        trials = completed_trials(study)
        if st["status"] == "complete" and len(trials) >= args.iterations:
            return {"model": model, "skipped": True, "reason": f"complete ({len(trials)}/{args.iterations} trials)",
                    "elapsed_min": 0.0, "best_fitness": st["best_fitness"], "valid_trials": st["valid_trials"]}
        ckpt.log("resume" if trials else "start", completed_trials=len(trials))
        t0 = time.perf_counter()

        while True:
            trials = completed_trials(study)
            st.update(write_outputs(trials, tune_dir))
            ckpt.save()
            index = len(trials)
            if index >= args.iterations:
                break
            if not gpu_healthy(device):
                ckpt.log("gpu_unavailable")
                raise GPUUnavailable(f"GPU/driver unhealthy before trial {index + 1}")
            trial = _ask(study, args.seed, index, ckpt)
            st["in_flight_trial"] = index + 1
            ckpt.save()
            print(f"\n  [trial {index + 1}/{args.iterations}] "
                  + ", ".join(f"{k}={v:.4g}" for k, v in trial.params.items()), flush=True)
            try:
                fitness = run_trial(index, trial.params, args, device, data, tune_dir)
                if not math.isfinite(fitness):
                    raise RuntimeError(f"non-finite fitness {fitness}")
            except Exception as e:
                study.tell(trial, state=TrialState.FAIL)
                if not gpu_healthy(device):
                    ckpt.log("gpu_unavailable", trial=index + 1)
                    raise GPUUnavailable(f"GPU/driver became unhealthy during trial {index + 1}") from e
                attempts = st["failed_attempts"]
                attempts[str(index)] = attempts.get(str(index), 0) + 1
                ckpt.log("trial_failed", trial=index + 1, attempts=attempts[str(index)], error=str(e)[:300])
                print(f"  [fail] trial {index + 1} (attempt {attempts[str(index)]}): {e}")
                backup_dir(tune_dir / "trials" / f"trial_{index:03d}")
                if attempts[str(index)] > args.max_trial_retries:
                    accepted = _ask(study, args.seed, index, ckpt)
                    accepted.set_user_attr("accepted_failure", True)
                    study.tell(accepted, 0.0)
                    st["accepted_failures"].append(index + 1)
                    ckpt.log("trial_failure_accepted", trial=index + 1)
                continue
            study.tell(trial, fitness)
            print(f"  [trial {index + 1}] fitness (val JSI) = {fitness:.4f}")

        st.update(status="complete", in_flight_trial=None)
        ckpt.log("complete", completed_trials=st["completed_trials"], best_fitness=st["best_fitness"])
    return {"model": model, "skipped": False, "reason": None, "elapsed_min": (time.perf_counter() - t0) / 60,
            "best_fitness": st["best_fitness"], "valid_trials": st["valid_trials"]}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Phase 3."""
    p = argparse.ArgumentParser(description="Phase 3 — fault-tolerant, seeded Optuna HPO of SAM 3.")
    p.add_argument("--models", nargs="+", default=DEFAULT_ORDER, choices=DEFAULT_ORDER)
    p.add_argument("--iterations", type=int, default=HPO_ITERATIONS,
                   help=f"Target number of trials (default: {HPO_ITERATIONS}).")
    p.add_argument("--epochs", type=int, default=HPO_EPOCHS, help=f"Epochs per trial (default: {HPO_EPOCHS}).")
    p.add_argument("--patience", type=int, default=HPO_PATIENCE,
                   help=f"Early-stopping patience per trial (default: {HPO_PATIENCE}).")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--max-trial-retries", type=int, default=2)
    p.add_argument("--data", default=DEFAULT_DATA_DIR, help="Phase 0 dataset directory.")
    p.add_argument("--device", default="0", help="Single GPU id (default: 0).")
    p.add_argument("--project", default=DEFAULT_PIPELINE_ROOT, help="Pipeline root.")
    p.add_argument("--force", action="store_true", help="Start over (old search → tune_<model>.bak-<UTC>).")
    p.add_argument("--allow-version-change", action="store_true",
                   help="Allow resuming a search started with other Optuna/torch versions.")
    return p.parse_args()


def main() -> int:
    """Run Phase 3.

    Returns:
        ``0`` on success, ``1`` if a model failed, ``75`` if the GPU is unavailable.
    """
    args = parse_args()
    seed_everything(args.seed)
    device = parse_device(args.device)
    paths = PipelinePaths(Path(args.project))
    data = CocoData(args.data)
    print(f"Phase 3 (HPO, Optuna {optuna.__version__} TPE) for {args.models}: "
          f"{args.iterations} trials x {args.epochs} ep (patience {args.patience}), seed {args.seed}")
    print("  search space: " + ", ".join(f"{k}∈[{lo:g},{hi:g}]{' log' if log else ''}{' int' if step else ''}"
                                         for k, (lo, hi, log, step) in SEARCH_SPACE.items()))
    code = 0
    for m in args.models:
        print("\n" + "=" * 80 + f"\n=== TUNE {m}\n" + "=" * 80)
        try:
            r = tune_one_model(m, args, device, data, paths)
            print(f"  [{m}] {'skipped — ' + r['reason'] if r['skipped'] else 'done'}; "
                  f"best fitness {r['best_fitness']}, valid trials {r['valid_trials']}")
        except GPUUnavailable as e:
            print(f"  [GPU] {e} — re-run the same command once the host is healthy.", file=sys.stderr)
            return EXIT_GPU_UNAVAILABLE
        except Exception:
            print(f"  [FAIL] {m}:\n{traceback.format_exc()}", file=sys.stderr)
            code = 1
    return code


if __name__ == "__main__":
    sys.exit(main())
