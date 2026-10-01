"""Phase 5c — Consolidate every phase into the tables read by the analysis notebooks (SAM 3).

Derived from YOLO26's ``yolo26_seg/build_final_report.py`` and the U-Net's:
the output files, table layouts and tidy-row schema are **identical** (SAM 3
only adds columns), so the same notebooks — and a cross-architecture
comparison — can read all three pipelines. SAM 3-specific differences: the
Ultralytics instance metrics are NaN except the COCO mAP50-95 of SAM 3's
official evaluator in the validation / CV rows; the CV pixel metrics come
directly from the folds' validation (already at 640 × 640); the efficiency
table adds the cached-text latency, per-component GFLOPs and the attention
FLOP share; ``final_results.json`` records the SAM 3 protocol deviations
(``protocol_notes``).

Inputs (all produced by earlier steps of ``run_pipeline_sam3.sh``)::

    summary/phase1_val.json, summary/phase4_val.json        # single-split val metrics
    phase2_cv_baseline/<m>/metrics_summary.json              # CV validation metrics (640x640)
    phase3_hpo/tune_<m>/hpo_state.json                        # HPO bookkeeping
    phase5_test/accuracy/*.json, phase5_test/per_image/*.csv  # test accuracy
    phase5_test/efficiency/*.json                             # efficiency

Outputs (``<project>/summary/``):

* ``test_accuracy.csv`` — one row per variant × model × precision (instance
  metrics, DSC/JSI mean ± std, median, 95 % CI, pooled, empty predictions).
* ``efficiency.csv`` — one row per variant × model × precision (forward and
  end-to-end latency median/P95/P99, FPS, VRAM, RAM, size, params, GFLOPs).
* ``hpo_gain.csv`` — Optimised − Baseline on the test set per model (FP32):
  ΔDSC, ΔJSI, ΔmAP50-95(M), the paired bootstrap 95 % CI of ΔDSC/ΔJSI and the
  two-sided Wilcoxon signed-rank p-value on per-image DSC/JSI.
* ``phase2_cv_pixel.csv`` — CV DSC/JSI mean ± std (ddof=1) per model.
* ``final_results.csv`` — every number above in tidy long format
  (``phase, variant, model, split, precision, metric, value, std, ci95_low,
  ci95_high, n``) — the single source for the notebooks.
* ``final_results.json`` — the same tables nested, plus HPO bookkeeping and
  ``warnings`` (contended benchmarks, resumed trainings, stale CV std,
  budget mismatch between Baseline and Optimised, missing inputs).

Usage:
    python sam3_seg/build_final_report.py --project /workspace/logs/pipeline_final_v1
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    BASE_SETUP,
    DEFAULT_ORDER,
    DEFAULT_PIPELINE_ROOT,
    HPO_EPOCHS,
    HPO_ITERATIONS,
    HPO_PATIENCE,
    SEED,
    TRAIN_EPOCHS,
    TRAIN_PATIENCE,
    PipelinePaths,
    atomic_write_json,
    read_json,
    utc_now_iso,
)
from train_cv_sam3 import VAL_KEYS
from training import RUN_STATE_FILE

VARIANTS: tuple[str, ...] = ("baseline", "optimized")
PRECISIONS: tuple[str, ...] = ("fp32", "fp16")

#: YOLO26 instance-metric keys (Ultralytics); NaN for SAM 3 except map5095_* in val / CV rows.
INSTANCE_KEYS: tuple[str, ...] = (
    "map50_m", "map5095_m", "precision_m", "recall_m", "f1_m",
    "map50_b", "map5095_b", "precision_b", "recall_b", "f1_b",
)
#: Base-setup keys that must be identical between Baseline and Optimised runs.
BASE_CHECK_KEYS: tuple[str, ...] = (
    "epochs", "patience", "amp", "seed", "batch", "resolution", "prompt", "grad_accum_chunks",
    "act_ckpt", "deterministic", "strict_determinism", "monitor", "score_threshold",
)

#: SAM 3 deviations from the YOLO26 / U-Net protocol (disclosed in the report).
PROTOCOL_NOTES: list[str] = [
    f"Training budget {TRAIN_EPOCHS} epochs / patience {TRAIN_PATIENCE} (YOLO26, U-Net: 120 / 25): "
    "one FP32 epoch of the 840 M-parameter model takes ~1.8 h on a V100S.",
    f"HPO {HPO_ITERATIONS} trials x {HPO_EPOCHS} epochs, patience {HPO_PATIENCE} (YOLO26, U-Net: 30 x 30), "
    "Optuna TPE with 5 start-up trials.",
    "Determinism: seeds, cuDNN deterministic, deterministic grid_sample, warn-only deterministic "
    "algorithms; the ViT memory-efficient attention backward stays nondeterministic (strict mode "
    "costs +33 % time) — repeated runs differ by <= ~1e-4 (weights) / 5th digit (val JSI).",
    "FP16 = torch.autocast(float16) with FP32 weights (SAM 3's official mixed-precision path).",
    f"Prompt '{BASE_SETUP['prompt']}', input {BASE_SETUP['resolution']} px, predicted mask = union of "
    f"instances with score >= {BASE_SETUP['score_threshold']}.",
]
PIXEL_KEYS: tuple[str, ...] = ("dsc", "jsi", "jsi_thr", "sensitivity", "specificity", "accuracy")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Build the consolidated final report of the 5-phase pipeline.")
    p.add_argument("--models", nargs="+", default=DEFAULT_ORDER, choices=DEFAULT_ORDER)
    p.add_argument("--project", default=DEFAULT_PIPELINE_ROOT, help="Pipeline root.")
    return p.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows (union of keys, first-seen order) to CSV; no file when empty."""
    if not rows:
        return
    fields: list[str] = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _read_per_image(path: Path) -> dict[str, dict[str, float]]:
    """Per-image DSC/JSI keyed by image path."""
    with path.open() as f:
        return {r["image"]: {"dsc": float(r["dsc"]), "jsi": float(r["jsi"])} for r in csv.DictReader(f)}


class Report:
    """Accumulates tables, tidy rows and warnings."""

    def __init__(self, paths: PipelinePaths, models: list[str]) -> None:
        self.paths, self.models = paths, models
        self.tidy: list[dict[str, Any]] = []
        self.warnings: list[str] = []
        self.tables: dict[str, list[dict[str, Any]]] = {}

    def add(self, phase: str, variant: str, model: str, split: str, precision: str,
            metric: str, value: Any, std: Any = None, lo: Any = None, hi: Any = None,
            n: Any = None) -> None:
        """Append one tidy row."""
        self.tidy.append({
            "phase": phase, "variant": variant, "model": model, "split": split,
            "precision": precision, "metric": metric, "value": value, "std": std,
            "ci95_low": lo, "ci95_high": hi, "n": n,
        })

    # ---- Phases 1 & 4 (validation split) ---------------------------------
    def single_split(self) -> None:
        """Collect Phase 1 / Phase 4 val metrics and check the shared budget."""
        for phase, variant in (("phase1", "baseline"), ("phase4", "optimized")):
            data = read_json(self.paths.summary_dir / f"{phase}_val.json")
            if data is None:
                self.warnings.append(f"summary/{phase}_val.json missing (run collect_phase_metrics.py)")
                continue
            for row in data["models"]:
                if row["model"] not in self.models:
                    continue
                if row.get("resumed"):
                        self.warnings.append(f"{phase}/{row['model']}: training was resumed after an interruption "
                                         "(not bit-exact: see protocol_notes)")
                for k in INSTANCE_KEYS + ("best_epoch", "epochs_trained") + VAL_KEYS:
                    self.add(phase, variant, row["model"], "val", "fp32", k, row.get(k))
        for m in self.models:
            protos = []
            for run in (self.paths.phase1_dir / self.paths.phase1_run_name(m),
                        self.paths.phase4_dir / self.paths.phase4_run_name(m)):
                state = read_json(run / RUN_STATE_FILE)
                if state:
                    protos.append({k: state["protocol"].get(k) for k in BASE_CHECK_KEYS})
            if len(protos) == 2 and protos[0] != protos[1]:
                self.warnings.append(f"{m}: Baseline and Optimised budgets differ: {protos[0]} vs {protos[1]}")

    # ---- Phase 2 (CV) -------------------------------------------------------
    def cross_validation(self) -> None:
        """Collect CV metrics and the CV pixel metrics (mean ± sample std; already at 640 × 640)."""
        pixel_rows = []
        for m in self.models:
            root = self.paths.cv_model_dir(m, "baseline")
            inst = read_json(root / "metrics_summary.json")
            if inst is None:
                self.warnings.append(f"phase2/{m}: metrics_summary.json missing")
            else:
                if inst.get("std_ddof") != 1:
                    self.warnings.append(f"phase2/{m}: CV std is a population std — re-run Phase 2 to refresh")
                for k in INSTANCE_KEYS + VAL_KEYS:
                    v = inst["summary"].get(k, {})
                    self.add("phase2", "baseline", m, "cv", "fp32", k, v.get("mean"), v.get("std"), n=inst["n_folds"])
                row = {"model": m, "n_folds": inst["n_folds"]}
                for k in PIXEL_KEYS + ("pooled_dsc", "pooled_jsi"):
                    v = inst["summary"][f"val_{k}"]
                    row[f"{k}_mean"], row[f"{k}_std"] = v["mean"], v["std"]
                    self.add("phase2", "baseline", m, "cv", "fp32", k, v["mean"], v["std"], n=inst["n_folds"])
                pixel_rows.append(row)
        self.tables["phase2_cv_pixel"] = pixel_rows

    # ---- Phase 3 (HPO) ------------------------------------------------------
    def hpo(self) -> dict[str, Any]:
        """HPO bookkeeping per model (trials, best fitness, failures)."""
        out = {}
        for m in self.models:
            state = read_json(self.paths.phase3_state(m))
            if state is None:
                self.warnings.append(f"phase3/{m}: hpo_state.json missing")
                continue
            out[m] = {k: state.get(k) for k in (
                "status", "completed_trials", "valid_trials", "target_trials", "best_fitness",
                "best_trial", "accepted_failures", "versions")}
            if state.get("status") != "complete":
                self.warnings.append(f"phase3/{m}: HPO not complete")
        return out

    # ---- Phase 5a (test accuracy) ------------------------------------------
    def test_accuracy(self) -> None:
        """Collect test-set accuracy per variant/model/precision."""
        rows = []
        for variant in VARIANTS:
            for m in self.models:
                for prec in PRECISIONS:
                    acc = read_json(self.paths.phase5_accuracy_json(variant, m, prec))
                    if acc is None:
                        self.warnings.append(f"phase5/{variant}_{m}_{prec}: test accuracy missing")
                        continue
                    pix, inst = acc["pixel_metrics"], acc["instance_metrics"]
                    row = {"variant": variant, "model": m, "precision": prec, "n_images": acc["n_images"],
                           **{k: inst[k] for k in INSTANCE_KEYS},
                           "n_empty_pred": pix["n_empty_pred"],
                           "pooled_dsc": pix["pooled_dsc"], "pooled_jsi": pix["pooled_jsi"]}
                    for k in INSTANCE_KEYS:
                        self.add("phase5", variant, m, "test", prec, k, inst[k], n=acc["n_images"])
                    for k in PIXEL_KEYS:
                        s = pix[k]
                        for stat in ("mean", "std", "median", "ci95_low", "ci95_high"):
                            row[f"{k}_{stat}"] = s.get(stat)
                        self.add("phase5", variant, m, "test", prec, k, s.get("mean"), s.get("std"),
                                 s.get("ci95_low"), s.get("ci95_high"), s.get("n"))
                    for k in ("pooled_dsc", "pooled_jsi", "n_empty_pred"):
                        self.add("phase5", variant, m, "test", prec, k, row[k], n=acc["n_images"])
                    rows.append(row)
        self.tables["test_accuracy"] = rows

    # ---- HPO gain (paired, test set, FP32) ---------------------------------
    def hpo_gain(self) -> None:
        """Paired Optimised − Baseline comparison on the same test images."""
        try:
            from scipy.stats import wilcoxon
        except ImportError:  # pragma: no cover - scipy is pinned in the image
            wilcoxon = None
        rng = np.random.default_rng(SEED)
        rows = []
        for m in self.models:
            files = [self.paths.phase5_per_image_csv(v, m, "fp32") for v in VARIANTS]
            accs = [read_json(self.paths.phase5_accuracy_json(v, m, "fp32")) for v in VARIANTS]
            if not all(f.exists() for f in files) or None in accs:
                continue
            base, opt = (_read_per_image(f) for f in files)
            common_imgs = sorted(set(base) & set(opt))
            if len(common_imgs) != len(base) or len(base) != len(opt):
                self.warnings.append(f"hpo_gain/{m}: baseline and optimised were scored on different images")
            row: dict[str, Any] = {"model": m, "n_images": len(common_imgs)}
            for k in ("dsc", "jsi"):
                b = np.array([base[i][k] for i in common_imgs])
                o = np.array([opt[i][k] for i in common_imgs])
                d = o - b
                boot = d[rng.integers(0, len(d), size=(2000, len(d)))].mean(axis=1)
                p = math.nan
                if wilcoxon is not None and np.any(d != 0):
                    p = float(wilcoxon(o, b, zero_method="wilcox", alternative="two-sided").pvalue)
                row.update({
                    f"{k}_baseline": float(b.mean()), f"{k}_optimized": float(o.mean()),
                    f"delta_{k}": float(d.mean()),
                    f"delta_{k}_ci95_low": float(np.quantile(boot, 0.025)),
                    f"delta_{k}_ci95_high": float(np.quantile(boot, 0.975)),
                    f"wilcoxon_p_{k}": p,
                    f"n_improved_{k}": int((d > 0).sum()), f"n_worse_{k}": int((d < 0).sum()),
                })
                self.add("hpo_gain", "optimized-baseline", m, "test", "fp32", f"delta_{k}",
                         row[f"delta_{k}"], None, row[f"delta_{k}_ci95_low"], row[f"delta_{k}_ci95_high"],
                         len(d))
                self.add("hpo_gain", "optimized-baseline", m, "test", "fp32", f"wilcoxon_p_{k}", p, n=len(d))
            dm = accs[1]["instance_metrics"]["map5095_m"] - accs[0]["instance_metrics"]["map5095_m"]
            row["delta_map5095_m"] = dm
            self.add("hpo_gain", "optimized-baseline", m, "test", "fp32", "delta_map5095_m", dm)
            rows.append(row)
        self.tables["hpo_gain"] = rows

    # ---- Phase 5b (efficiency) ---------------------------------------------
    def efficiency(self) -> None:
        """Collect efficiency per variant/model/precision."""
        rows = []
        for variant in VARIANTS:
            for m in self.models:
                for prec in PRECISIONS:
                    e = read_json(self.paths.phase5_efficiency_json(variant, m, prec))
                    if e is None:
                        self.warnings.append(f"phase5/{variant}_{m}_{prec}: efficiency missing")
                        continue
                    if e.get("contended"):
                        self.warnings.append(f"phase5/{variant}_{m}_{prec}: benchmark ran on a busy GPU")
                    fw, fc, ee = e["forward"], e["forward_cached_text"], e["end_to_end"]
                    mem, mdl = e["memory"], e["model"]
                    row = {
                        "variant": variant, "model": m, "precision": prec, "batch": 1,
                        "gpu": e["env"].get("gpu"), "contended": e.get("contended"),
                        **{f"fwd_{k}": fw[k] for k in ("mean_ms", "std_ms", "median_ms", "p90_ms", "p95_ms", "p99_ms", "fps", "fps_median")},
                        **{f"fwd_cached_text_{k}": fc[k] for k in ("mean_ms", "median_ms", "p95_ms", "fps")},
                        **{f"e2e_{k}": ee[k] for k in ("mean_ms", "median_ms", "p95_ms", "p99_ms", "fps", "fps_median")},
                        "vram_weights_mb": mem["vram_weights_mb"],
                        "vram_peak_allocated_mb": mem["vram_peak_allocated_mb"],
                        "vram_peak_reserved_mb": mem["vram_peak_reserved_mb"],
                        "vram_peak_allocated_e2e_mb": mem["vram_peak_allocated_e2e_mb"],
                        "ram_peak_rss_mb": mem["ram_peak_rss_mb"],
                        "ram_model_delta_mb": mem["ram_model_delta_mb"],
                        "size_mb_disk": mdl["size_mb_disk"],
                        "size_mb_fp32_theoretical": mdl["size_mb_fp32_theoretical"],
                        "size_mb_fp16_theoretical": mdl["size_mb_fp16_theoretical"],
                        "params": mdl["params"], "params_fused": mdl["params_fused"], "gflops": mdl["gflops"],
                        "params_without_text": mdl["params_without_text"],
                        **{f"params_{k}": v for k, v in mdl["params_by_component"].items()},
                        "gflops_without_text": mdl["gflops_without_text"],
                        **{f"gflops_{k}": v for k, v in mdl["gflops_by_component"].items()},
                        **{f"gflops_op_{k}": v for k, v in mdl["gflops_by_op"].items()},
                        "attention_flop_share": mdl["attention_flop_share"],
                    }
                    for k, v in row.items():
                        if k not in ("variant", "model", "precision", "gpu", "batch") and v is not None:
                            self.add("phase5", variant, m, "test", prec, k, v)
                    rows.append(row)
        self.tables["efficiency"] = rows


def main() -> int:
    """Build every table and write CSV + JSON.

    Returns:
        ``0`` if the FP32 test accuracy of both variants exists for every
        requested model, ``1`` otherwise (the partial report is still written).
    """
    args = parse_args()
    paths = PipelinePaths(Path(args.project))
    rep = Report(paths, args.models)
    rep.single_split()
    rep.cross_validation()
    hpo = rep.hpo()
    rep.test_accuracy()
    rep.hpo_gain()
    rep.efficiency()

    out = paths.summary_dir
    for name in ("test_accuracy", "efficiency", "hpo_gain", "phase2_cv_pixel"):
        _write_csv(out / f"{name}.csv", rep.tables.get(name, []))
    _write_csv(out / "final_results.csv", rep.tidy)
    atomic_write_json(out / "final_results.json", {
        "generated_at": utc_now_iso(), "project": str(paths.root), "models": args.models,
        "tables": rep.tables, "hpo": hpo, "protocol_notes": PROTOCOL_NOTES, "warnings": rep.warnings,
    })

    print(f"Final report written to {out}/")
    for name, rows in rep.tables.items():
        print(f"  {name:<16} {len(rows)} row(s)")
    print(f"  final_results    {len(rep.tidy)} tidy row(s)")
    for g in rep.tables.get("hpo_gain", []):
        print(f"  HPO gain {g['model']:<7}: ΔDSC={g['delta_dsc']:+.4f} "
              f"[{g['delta_dsc_ci95_low']:+.4f}, {g['delta_dsc_ci95_high']:+.4f}]  "
              f"p={g['wilcoxon_p_dsc']:.3g}   ΔJSI={g['delta_jsi']:+.4f}")
    if rep.warnings:
        print(f"\n  {len(rep.warnings)} warning(s):")
        for w in rep.warnings:
            print(f"   - {w}")

    have = {(r["variant"], r["model"]) for r in rep.tables.get("test_accuracy", []) if r["precision"] == "fp32"}
    complete = all((v, m) in have for v in VARIANTS for m in args.models)
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
