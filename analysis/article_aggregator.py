"""Master article aggregator — YOLO26-seg vs. U-Net vs. SAM 3 on ISIC 2018 Task 1.

Reads **only** the Phase 5 outputs of the three 5-phase pipelines and writes the
cross-architecture tables (LaTeX + CSV), statistics and figures of the article.
No GPU, no training, no re-evaluation.

Inputs, per pipeline (``<repo>/logs/<pipeline-name>/``, written by the
pipelines' ``build_final_report.py``)::

    summary/test_accuracy.csv      DSC, JSI, BIoU, NSD, HD95 … mean, std, median, bootstrap 95 % CI
    summary/efficiency.csv         batch-1 latency (median/P95), FPS, VRAM, params, GFLOPs
    summary/hpo_gain.csv           paired Optimized − Baseline differences (CI, Wilcoxon p)
    summary/training_cost.csv      GPU hours per phase
    summary/phase2_cv_pixel.csv    5-fold CV DSC/JSI (mean ± SD)
    summary/final_results.json     warnings, protocol notes
    phase5_test/per_image/*.csv    per-image scores (paired cross-architecture tests)

The pipelines are found automatically in the folder that contains the three
repositories (``sandbox_yolo26``, ``sandbox_unet``, ``sandbox_sam3``; the
match is case-insensitive) — the current directory, its parent or its
grandparent, so this file works from that root folder or from a copy in
``<repo>/analysis/``. Override with ``$YOLO26_PIPELINE_DIR``,
``$UNET_PIPELINE_DIR``, ``$SAM3_PIPELINE_DIR`` (each pointing at a pipeline
folder) or ``--root``. A missing pipeline is skipped with a warning.

Statistics (all on the per-image scores of the same 1,000 test images, paired
by ISIC ID):

* every pair of systems and every metric: mean paired difference with a
  seeded percentile-bootstrap 95 % CI, two-sided Wilcoxon signed-rank test,
  matched-pairs rank-biserial effect size, wins / losses, and Holm-adjusted
  p-values within each metric (family = all pairs of that metric);
* across all systems: Friedman test with Kendall's W and mean ranks.

Usage::

    python article_aggregator.py                         # auto-discovery, outputs in <root>/article_outputs
    python article_aggregator.py --root ~/projects --pipeline-name pipeline_final_v1 --variant optimized

Requires numpy, pandas, scipy and matplotlib (no GPU, no deep-learning stack).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Registry of the three architectures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Arch:
    key: str
    label: str
    repo: str
    env: str
    color: str          # categorical slot (validated all-pairs for these three, CVD-safe)
    marker: str


ARCHS: tuple[Arch, ...] = (
    Arch("yolo26", "YOLO26-seg", "sandbox_yolo26", "YOLO26_PIPELINE_DIR", "#2a78d6", "o"),
    Arch("unet", "U-Net", "sandbox_unet", "UNET_PIPELINE_DIR", "#eb6834", "s"),
    Arch("sam3", "SAM 3", "sandbox_sam3", "SAM3_PIPELINE_DIR", "#1baf7a", "D"),
)
ARCH_BY_KEY = {a.key: a for a in ARCHS}

#: Short system names (YOLO26 sizes keep their usual letter).
SYSTEM_NAMES: dict[tuple[str, str], str] = {
    ("yolo26", "nano"): "YOLO26-n", ("yolo26", "small"): "YOLO26-s", ("yolo26", "medium"): "YOLO26-m",
    ("yolo26", "large"): "YOLO26-l", ("yolo26", "xlarge"): "YOLO26-x",
    ("unet", "unet"): "U-Net", ("sam3", "sam3"): "SAM 3",
}
MODEL_ORDER = ["nano", "small", "medium", "large", "xlarge", "unet", "sam3"]

#: Per-image scores compared between systems.
PAIRED_METRICS: tuple[str, ...] = ("dsc", "jsi", "biou", "nsd", "hd95")
METRIC_LABEL = {"dsc": "DSC", "jsi": "JSI", "jsi_thr": "JSI$_{0.65}$", "biou": "Boundary IoU", "nsd": "NSD",
                "hd95": "HD95 (px)", "sensitivity": "Sensitivity", "specificity": "Specificity"}
LOWER_IS_BETTER = {"hd95"}

# Publication style shared with the repositories' notebooks.
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
DIVERGING = ("#e34948", "#f0efec", "#2a78d6")   # red ← neutral grey → blue


# ---------------------------------------------------------------------------
# Discovery and loading
# ---------------------------------------------------------------------------
def _child_ci(parent: Path, name: str) -> Path | None:
    """``parent/name`` matched case-insensitively (``sandbox_yolo26`` = ``SANDBOX_YOLO26``)."""
    if not parent.is_dir():
        return None
    for c in parent.iterdir():
        if c.is_dir() and c.name.lower() == name.lower():
            return c
    return None


def discover_root(start: Path | None = None) -> Path:
    """Folder holding the three repositories: ``start``, its parent or its grandparent (most repos wins)."""
    start = Path(start or Path.cwd()).resolve()
    best, best_n = start, -1
    for cand in (start, start.parent, start.parent.parent):
        n = sum(_child_ci(cand, a.repo) is not None for a in ARCHS)
        if n > best_n:
            best, best_n = cand, n
    return best


@dataclass
class Pipeline:
    arch: Arch
    root: Path
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)

    def table(self, name: str) -> pd.DataFrame:
        return self.tables.get(name, pd.DataFrame())

    def per_image_csv(self, variant: str, model: str, precision: str) -> Path:
        return self.root / "phase5_test" / "per_image" / f"{variant}_{model}_{precision}.csv"


def find_pipelines(root: Path, pipeline_name: str, warnings: list[str]) -> dict[str, Pipeline]:
    """Locate and load every available pipeline."""
    out: dict[str, Pipeline] = {}
    for a in ARCHS:
        cands = [os.environ.get(a.env)]
        repo = _child_ci(root, a.repo)
        if repo is not None:
            cands += [repo / "logs" / pipeline_name]
        pdir = next((Path(c) for c in cands if c and (Path(c) / "summary").is_dir()), None)
        if pdir is None:
            tried = [str(c) for c in cands if c] or [f"{root}/{a.repo}/ (folder not found)"]
            warnings.append(f"{a.label}: no pipeline summary found (tried {tried}) — skipped")
            continue
        p = Pipeline(a, pdir)
        for name in ("test_accuracy", "efficiency", "hpo_gain", "training_cost", "phase2_cv_pixel"):
            f = pdir / "summary" / f"{name}.csv"
            if f.exists():
                p.tables[name] = pd.read_csv(f)
            else:
                warnings.append(f"{a.label}: summary/{name}.csv missing")
        rj = pdir / "summary" / "final_results.json"
        p.report = json.loads(rj.read_text()) if rj.exists() else {}
        for w in p.report.get("warnings", []):
            warnings.append(f"{a.label} pipeline: {w}")
        out[a.key] = p
    return out


def isic_id(image: str) -> str:
    """ISIC ID of a per-image row (``.../ISIC_0012345.png`` → ``ISIC_0012345``)."""
    m = re.search(r"ISIC_\d{7}", str(image))
    return m.group(0) if m else Path(str(image)).stem


# ---------------------------------------------------------------------------
# Unified tables
# ---------------------------------------------------------------------------
def system_frame(pipes: dict[str, Pipeline]) -> pd.DataFrame:
    """One row per (architecture, model, variant, precision): test accuracy ⨝ efficiency."""
    frames = []
    for key, p in pipes.items():
        acc, eff = p.table("test_accuracy"), p.table("efficiency")
        if acc.empty:
            continue
        d = acc.merge(eff, on=["variant", "model", "precision"], how="left", suffixes=("", "_eff")) if len(eff) else acc
        d = d.copy()
        d.insert(0, "arch", key)
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["system"] = [SYSTEM_NAMES.get((a, m), f"{ARCH_BY_KEY[a].label} {m}") for a, m in zip(df.arch, df.model)]
    df["arch_label"] = [ARCH_BY_KEY[a].label for a in df.arch]
    df["order"] = [MODEL_ORDER.index(m) if m in MODEL_ORDER else 99 for m in df.model]
    if "params_fused" in df:
        df["params_M"] = df["params_fused"] / 1e6
    for col in ("e2e_dataset_p95_ms", "e2e_dataset_median_ms"):
        if col not in df:
            df[col] = np.nan
    # Real-time latency figure: over 100 distinct test images when available, else one image repeated.
    df["e2e_p95_rt_ms"] = df["e2e_dataset_p95_ms"].fillna(df.get("e2e_p95_ms"))
    return df.sort_values(["order", "variant", "precision"]).reset_index(drop=True)


def select(df: pd.DataFrame, variant: str, precision: str) -> pd.DataFrame:
    return df[(df.variant == variant) & (df.precision == precision)].sort_values("order").reset_index(drop=True)


def fmt_ci(mean: float, lo: float, hi: float, digits: int = 3) -> str:
    if not np.isfinite(mean):
        return "–"
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def fmt_num(v: float, digits: int = 1) -> str:
    if v is None or not np.isfinite(v):
        return "–"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:.{digits}f}"


def fmt_p(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def p_text(p: float, name: str = "p") -> str:
    """``p < 0.001`` / ``p = 0.012`` for running text."""
    return f"{name} < 0.001" if np.isfinite(p) and p < 0.001 else f"{name} = {fmt_p(p)}"


def accuracy_table(sel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(formatted, numeric) test-accuracy table of the selected variant/precision."""
    keys = [k for k in ("dsc", "jsi", "jsi_thr", "biou", "nsd", "hd95") if f"{k}_mean" in sel]
    num = sel[["system", "arch_label"] + [f"{k}_{s}" for k in keys for s in ("mean", "ci95_low", "ci95_high")]
              + [c for c in ("n_images", "n_empty_pred") if c in sel]].copy()
    fmt = pd.DataFrame({"System": sel.system})
    for k in keys:
        fmt[METRIC_LABEL.get(k, k)] = [fmt_ci(m, lo, hi, 1 if k == "hd95" else 3) for m, lo, hi in
                                       zip(sel[f"{k}_mean"], sel[f"{k}_ci95_low"], sel[f"{k}_ci95_high"])]
    if "n_empty_pred" in sel:
        fmt["Missed"] = sel["n_empty_pred"].map(lambda v: f"{int(v)}" if np.isfinite(v) else "–")
    return fmt, num


EFF_COLS: tuple[tuple[str, str, int], ...] = (
    ("params_M", "Params (M)", 1), ("gflops", "GFLOPs", 1), ("input_px", "Input (px)", 0),
    ("fwd_median_ms", "Fwd median (ms)", 2), ("fwd_p95_ms", "Fwd P95 (ms)", 2), ("fwd_fps", "FPS (fwd)", 1),
    ("e2e_median_ms", "E2E median (ms)", 2), ("e2e_p95_rt_ms", "E2E P95 (ms)", 2), ("e2e_fps", "FPS (E2E)", 1),
    ("vram_peak_allocated_mb", "VRAM alloc. (MB)", 0), ("vram_process_peak_mb", "VRAM process (MB)", 0),
)
LOWER_BETTER_EFF = {"params_M", "gflops", "fwd_median_ms", "fwd_p95_ms", "e2e_median_ms", "e2e_p95_rt_ms",
                    "vram_peak_allocated_mb", "vram_process_peak_mb"}


def efficiency_table(sel: pd.DataFrame, realtime_fps: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [(c, lab, d) for c, lab, d in EFF_COLS if c in sel and sel[c].notna().any()]
    num = sel[["system", "arch_label"] + [c for c, _, _ in cols]].copy()
    num[f"realtime_{int(realtime_fps)}fps"] = sel["e2e_p95_rt_ms"] <= 1000 / realtime_fps
    fmt = pd.DataFrame({"System": sel.system})
    for c, lab, d in cols:
        fmt[lab] = sel[c].map(lambda v, d=d: fmt_num(v, d))
    fmt[f"Real-time ({int(realtime_fps)} FPS)"] = num[f"realtime_{int(realtime_fps)}fps"].map({True: "yes", False: "no"})
    return fmt, num


def precision_table(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    """FP32 vs FP16: accuracy cost and latency gain of half precision."""
    rows = []
    for sysname, g in df[df.variant == variant].groupby("system", sort=False):
        g = g.set_index("precision")
        if not {"fp32", "fp16"} <= set(g.index):
            continue
        a, b = g.loc["fp32"], g.loc["fp16"]
        rows.append({"System": sysname, "DSC FP32": a.dsc_mean, "DSC FP16": b.dsc_mean,
                     "ΔDSC (FP16−FP32)": b.dsc_mean - a.dsc_mean,
                     "Fwd FP32 (ms)": a.fwd_median_ms, "Fwd FP16 (ms)": b.fwd_median_ms,
                     "Speed-up (fwd)": a.fwd_median_ms / b.fwd_median_ms,
                     "E2E FP32 (ms)": a.e2e_median_ms, "E2E FP16 (ms)": b.e2e_median_ms,
                     "Speed-up (E2E)": a.e2e_median_ms / b.e2e_median_ms,
                     "VRAM FP32 (MB)": a.vram_peak_allocated_mb, "VRAM FP16 (MB)": b.vram_peak_allocated_mb})
    return pd.DataFrame(rows)


def hpo_table(pipes: dict[str, Pipeline]) -> pd.DataFrame:
    """Paired Optimized − Baseline differences (test, FP32) from every pipeline's hpo_gain.csv."""
    rows = []
    for key, p in pipes.items():
        g = p.table("hpo_gain")
        for _, r in g.iterrows():
            row = {"System": SYSTEM_NAMES.get((key, r.model), r.model), "n": r.get("n_images")}
            for k in PAIRED_METRICS:
                if f"delta_{k}" in r and pd.notna(r[f"delta_{k}"]):
                    row[f"Δ{METRIC_LABEL[k]}"] = fmt_ci(r[f"delta_{k}"], r[f"delta_{k}_ci95_low"],
                                                        r[f"delta_{k}_ci95_high"], 1 if k == "hd95" else 4)
                    row[f"p ({METRIC_LABEL[k]})"] = fmt_p(r.get(f"wilcoxon_p_{k}"))
            rows.append(row)
    order = {n: i for i, n in enumerate(SYSTEM_NAMES.values())}
    return pd.DataFrame(rows).sort_values("System", key=lambda s: s.map(order)).reset_index(drop=True) if rows else pd.DataFrame()


PHASES = (("phase1_baseline", "Ph 1 baseline"), ("phase2_cv_baseline", "Ph 2 CV (5 folds)"),
          ("phase3_hpo", "Ph 3 HPO"), ("phase4_optimized", "Ph 4 optimized"))


def training_table(pipes: dict[str, Pipeline]) -> pd.DataFrame:
    """GPU hours per phase, total, epochs and seconds per epoch (Phase 1) per system."""
    rows = []
    for key, p in pipes.items():
        c = p.table("training_cost")
        if c.empty:
            continue
        for m, g in c.groupby("model", sort=False):
            g = g.set_index("phase")
            row = {"System": SYSTEM_NAMES.get((key, m), m), "arch": key, "order": MODEL_ORDER.index(m) if m in MODEL_ORDER else 99}
            for ph, lab in PHASES:
                row[f"{lab} (h)"] = g.loc[ph, "train_hours"] if ph in g.index else np.nan
            row["Total (h)"] = float(np.nansum([row[f"{lab} (h)"] for _, lab in PHASES]))
            if "phase1_baseline" in g.index:
                row["Epochs (Ph 1)"] = g.loc["phase1_baseline", "epochs"]
                row["s / epoch (Ph 1)"] = g.loc["phase1_baseline", "sec_per_epoch"]
            rows.append(row)
    return pd.DataFrame(rows).sort_values("order").drop(columns="order").reset_index(drop=True) if rows else pd.DataFrame()


def cv_vs_test_table(pipes: dict[str, Pipeline], sel_base: pd.DataFrame) -> pd.DataFrame:
    """Phase 2 CV (Baseline protocol, mean ± SD over folds) next to the Baseline test score."""
    rows = []
    for key, p in pipes.items():
        cv = p.table("phase2_cv_pixel")
        for _, r in cv.iterrows():
            name = SYSTEM_NAMES.get((key, r.model), r.model)
            t = sel_base[sel_base.system == name]
            rows.append({"System": name,
                         "CV DSC (mean ± SD)": f"{r.dsc_mean:.3f} ± {r.dsc_std:.3f}",
                         "CV JSI (mean ± SD)": f"{r.jsi_mean:.3f} ± {r.jsi_std:.3f}",
                         "Test DSC, Baseline": fmt_ci(*t[["dsc_mean", "dsc_ci95_low", "dsc_ci95_high"]].iloc[0]) if len(t) else "–",
                         "Test JSI, Baseline": fmt_ci(*t[["jsi_mean", "jsi_ci95_low", "jsi_ci95_high"]].iloc[0]) if len(t) else "–"})
    order = {n: i for i, n in enumerate(SYSTEM_NAMES.values())}
    return pd.DataFrame(rows).sort_values("System", key=lambda s: s.map(order)).reset_index(drop=True) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Paired cross-architecture statistics
# ---------------------------------------------------------------------------
def load_per_image(pipes: dict[str, Pipeline], sel: pd.DataFrame, variant: str, precision: str,
                   warnings: list[str]) -> dict[str, pd.DataFrame]:
    """Per-image scores of every selected system, indexed by ISIC ID."""
    out = {}
    for _, r in sel.iterrows():
        f = pipes[r.arch].per_image_csv(variant, r.model, precision)
        if not f.exists():
            warnings.append(f"{r.system}: per-image file missing ({f}) — excluded from paired tests")
            continue
        d = pd.read_csv(f)
        d.index = d["image"].map(isic_id)
        if d.index.duplicated().any():
            warnings.append(f"{r.system}: duplicated ISIC IDs in {f.name}")
        out[r.system] = d
    ids = [set(d.index) for d in out.values()]
    if ids and len(set.intersection(*ids)) != max(len(i) for i in ids):
        warnings.append(f"per-image files cover different images: common = {len(set.intersection(*ids))}, "
                        f"max = {max(len(i) for i in ids)} — paired tests use the common images only")
    return out


def holm(p: Iterable[float]) -> np.ndarray:
    """Holm–Bonferroni adjusted p-values (NaNs kept)."""
    p = np.asarray(list(p), dtype=float)
    adj = np.full_like(p, np.nan)
    ok = np.where(np.isfinite(p))[0]
    order = ok[np.argsort(p[ok])]
    m, running = len(order), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def rank_biserial(d: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation (zeros dropped): (W+ − W−) / (W+ + W−)."""
    from scipy.stats import rankdata

    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = rankdata(np.abs(d))
    wp, wm = r[d > 0].sum(), r[d < 0].sum()
    return float((wp - wm) / (wp + wm))


def paired_tests(per_image: dict[str, pd.DataFrame], metrics: Iterable[str] = PAIRED_METRICS,
                 n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """All pairs × metrics: mean difference A − B, bootstrap CI, Wilcoxon p, Holm p, effect size, wins."""
    from scipy.stats import wilcoxon

    names = list(per_image)
    common = sorted(set.intersection(*(set(d.index) for d in per_image.values()))) if names else []
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, len(common), size=(n_boot, len(common))) if common else None
    rows = []
    for k in metrics:
        if not all(k in d for d in per_image.values()):
            continue
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                x = per_image[a].loc[common, k].to_numpy(float)
                y = per_image[b].loc[common, k].to_numpy(float)
                ok = np.isfinite(x) & np.isfinite(y)
                d = x[ok] - y[ok]
                # The same resampled images for every pair (unless a metric has NaNs for this pair).
                idx = boot_idx if ok.all() else rng.integers(0, len(d), size=(n_boot, len(d)))
                boot = d[idx].mean(axis=1)
                p = float(wilcoxon(x[ok], y[ok], zero_method="wilcox").pvalue) if np.any(d != 0) else 1.0
                better = (d < 0) if k in LOWER_IS_BETTER else (d > 0)
                worse = (d > 0) if k in LOWER_IS_BETTER else (d < 0)
                rows.append({"metric": k, "A": a, "B": b, "n": int(ok.sum()), "mean_A": float(x[ok].mean()),
                             "mean_B": float(y[ok].mean()), "delta": float(d.mean()),
                             "ci95_low": float(np.quantile(boot, 0.025)), "ci95_high": float(np.quantile(boot, 0.975)),
                             "median_delta": float(np.median(d)), "wilcoxon_p": p,
                             "rank_biserial": rank_biserial(d), "A_better": int(better.sum()), "B_better": int(worse.sum()),
                             "ties": int((d == 0).sum())})
    res = pd.DataFrame(rows)
    if len(res):
        res["p_holm"] = res.groupby("metric")["wilcoxon_p"].transform(lambda s: holm(s))
        res["significant_0.05"] = res["p_holm"] < 0.05
    return res


def friedman_tests(per_image: dict[str, pd.DataFrame], metrics: Iterable[str] = PAIRED_METRICS) -> pd.DataFrame:
    """Friedman omnibus test across all systems per metric, Kendall's W and mean ranks (1 = best)."""
    from scipy.stats import friedmanchisquare, rankdata

    names = list(per_image)
    if len(names) < 3:
        return pd.DataFrame()
    common = sorted(set.intersection(*(set(d.index) for d in per_image.values())))
    rows = []
    for k in metrics:
        if not all(k in d for d in per_image.values()):
            continue
        m = np.column_stack([per_image[s].loc[common, k].to_numpy(float) for s in names])
        m = m[np.isfinite(m).all(axis=1)]
        stat, p = friedmanchisquare(*m.T)
        n, kk = m.shape
        ranks = rankdata(m if k in LOWER_IS_BETTER else -m, axis=1).mean(axis=0)
        rows.append({"metric": k, "n_images": n, "k_systems": kk, "chi2": float(stat), "p": float(p),
                     "kendall_w": float(stat / (n * (kk - 1))),
                     **{f"mean_rank[{s}]": float(r) for s, r in zip(names, ranks)}})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LaTeX (booktabs) without jinja2
# ---------------------------------------------------------------------------
_TEX_ESC = {"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_", "Δ": r"$\Delta$", "−": r"$-$", "±": r"$\pm$",
            "≤": r"$\leq$", "≥": r"$\geq$", "×": r"$\times$", "<": r"$<$", ">": r"$>$", "→": r"$\rightarrow$",
            "χ": r"$\chi$", "²": r"$^2$", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def tex_escape(s: Any) -> str:
    s = str(s)
    if s.startswith("$") and s.endswith("$"):
        return s
    out = []
    for part in re.split(r"(\$[^$]*\$)", s):       # keep inline math untouched
        if part.startswith("$") and part.endswith("$") and len(part) > 1:
            out.append(part)
        else:
            esc = "".join(_TEX_ESC.get(ch, ch) for ch in part)
            out.append(re.sub(r"(?<![\w.])-(?=\d)", r"$-$", esc))     # numeric minus, not hyphens
    return "".join(out)


def to_latex(df: pd.DataFrame, caption: str, label: str, bold: dict[str, list[int]] | None = None,
             notes: str | None = None, resize: bool = False) -> str:
    """booktabs table; ``bold`` = {column: [row positions]} to set in bold (best values)."""
    bold = bold or {}
    cols = list(df.columns)
    spec = "l" + "c" * (len(cols) - 1)
    lines = [r"\begin{table}[t]", r"\centering", r"\small", rf"\caption{{{tex_escape(caption)}}}", rf"\label{{{label}}}"]
    if resize:
        lines.append(r"\resizebox{\linewidth}{!}{%")
    lines += [rf"\begin{{tabular}}{{{spec}}}", r"\toprule", " & ".join(tex_escape(c) for c in cols) + r" \\", r"\midrule"]
    for i, (_, r) in enumerate(df.iterrows()):
        cells = []
        for c in cols:
            v = tex_escape(r[c])
            cells.append(rf"\textbf{{{v}}}" if i in bold.get(c, []) else v)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if resize:
        lines.append("}")
    if notes:
        lines.append(rf"\par\smallskip\footnotesize {tex_escape(notes)}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def best_rows(values: pd.Series, lower_is_better: bool) -> list[int]:
    v = values.to_numpy(float)
    if not np.isfinite(v).any():
        return []
    target = np.nanmin(v) if lower_is_better else np.nanmax(v)
    return [i for i, x in enumerate(v) if np.isfinite(x) and np.isclose(x, target)]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def setup_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelcolor": INK_2, "axes.edgecolor": INK_2, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
        "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 2, "lines.markersize": 7,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def log_axis(ax, axis: str = "x") -> None:
    """Log scale with plain-number ticks (1-2-5, 1-3 or decades, by the span of the data); call after plotting."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    (ax.set_xscale if axis == "x" else ax.set_yscale)("log")
    a = ax.xaxis if axis == "x" else ax.yaxis
    lo, hi = ax.get_xlim() if axis == "x" else ax.get_ylim()
    span = np.log10(hi / lo) if lo > 0 and hi > lo else 1.0
    a.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0) if span <= 1.6 else (1.0, 3.0) if span <= 2.6 else (1.0,)))
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}"))
    a.set_minor_formatter(NullFormatter())


def arch_legend(fig, archs: Iterable[str], extra: list | None = None, ncol: int | None = None, y: float = -0.01):
    from matplotlib.lines import Line2D

    h = [Line2D([], [], color=ARCH_BY_KEY[a].color, marker=ARCH_BY_KEY[a].marker, linestyle="", markersize=7,
                markeredgecolor=SURFACE, label=ARCH_BY_KEY[a].label) for a in archs] + (extra or [])
    fig.legend(handles=h, loc="lower center", ncol=ncol or len(h), bbox_to_anchor=(0.5, y))


def pareto_front(x: np.ndarray, y: np.ndarray, x_lower_better: bool) -> np.ndarray:
    """Indices of the points not dominated in (x, y), y higher is better."""
    xs = x if x_lower_better else -x
    idx = []
    for i in range(len(x)):
        if not (np.isfinite(x[i]) and np.isfinite(y[i])):
            continue
        dominated = any((xs[j] <= xs[i] and y[j] >= y[i]) and (xs[j] < xs[i] or y[j] > y[i])
                        for j in range(len(x)) if j != i and np.isfinite(x[j]) and np.isfinite(y[j]))
        if not dominated:
            idx.append(i)
    return np.array(sorted(idx, key=lambda i: x[i]), dtype=int)


def short_label(system: str) -> str:
    return system.replace("YOLO26-", "")


def fig_accuracy_vs_efficiency(sel: pd.DataFrame, metrics=("dsc", "jsi"), save=None):
    """Accuracy (rows) vs. latency, FPS and parameters (columns), one colour per architecture, 95 % CI."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    xcols = [("fwd_median_ms", "median forward latency (ms)", True), ("fwd_fps", "throughput (FPS)", False),
             ("params_M", "parameters (millions)", True)]
    metrics = [m for m in metrics if f"{m}_mean" in sel]
    fig, axes = plt.subplots(len(metrics), len(xcols), figsize=(7.2, 2.5 * len(metrics) + 0.4), squeeze=False,
                             sharey="row", sharex="col")
    for r, k in enumerate(metrics):
        y = sel[f"{k}_mean"].to_numpy(float)
        for c, (xc, xlab, lower) in enumerate(xcols):
            ax = axes[r, c]
            x = sel[xc].to_numpy(float)
            front = pareto_front(x, y, lower)
            if len(front) > 1:
                ax.plot(x[front], y[front], color=INK_2, linewidth=0.8, linestyle=":", zorder=1)
            for a in sel.arch.unique():
                d = sel[sel.arch == a]
                arch = ARCH_BY_KEY[a]
                if len(d) > 1:
                    ax.plot(d[xc], d[f"{k}_mean"], color=arch.color, linewidth=1, alpha=0.6, zorder=2)
                ax.errorbar(d[xc], d[f"{k}_mean"],
                            yerr=np.vstack([d[f"{k}_mean"] - d[f"{k}_ci95_low"], d[f"{k}_ci95_high"] - d[f"{k}_mean"]]),
                            fmt=arch.marker, color=arch.color, elinewidth=1, capsize=0, markersize=7,
                            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
                for i, (_, row) in enumerate(d.iterrows()):   # alternate below / above: neighbours never overprint
                    ax.annotate(short_label(row.system), (row[xc], row[f"{k}_mean"]), xytext=(4, -10 if i % 2 == 0 else 4),
                                textcoords="offset points", fontsize=7, color=INK_2)
            log_axis(ax, "x")
            ax.grid(True, axis="both")
            if r == len(metrics) - 1:
                ax.set_xlabel(xlab)
            if c == 0:
                ax.set_ylabel(f"{METRIC_LABEL[k]} (test)")
    arch_legend(fig, sel.arch.unique(), extra=[Line2D([], [], color=INK_2, linestyle=":", linewidth=0.8,
                                                        label="Pareto front")])
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    if save:
        save(fig, "article_fig1_accuracy_vs_efficiency")
    return fig


def fig_training_inference_time(sel: pd.DataFrame, train: pd.DataFrame, realtime_fps: float, save=None):
    """(a) training GPU hours (whole protocol vs. final model); (b) batch-1 latency median with P95 whisker."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    systems = list(sel.system)
    yy = np.arange(len(systems))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 0.42 * len(systems) + 1.6), sharey=True)
    ax = axes[0]
    if len(train):
        t = train.set_index("System").reindex(systems)
        for yi, (s, row) in zip(yy, t.iterrows()):
            col = ARCH_BY_KEY[sel.set_index("system").loc[s, "arch"]].color
            ax.plot([row["Ph 4 optimized (h)"], row["Total (h)"]], [yi, yi], color=col, linewidth=1, alpha=0.6)
            ax.plot(row["Total (h)"], yi, "o", color=col, markeredgecolor=SURFACE, markeredgewidth=1.5)
            ax.plot(row["Ph 4 optimized (h)"], yi, "o", markerfacecolor=SURFACE, markeredgecolor=col, markeredgewidth=1.5)
            ax.annotate(f"{row['Total (h)']:,.0f} h", (row["Total (h)"], yi), xytext=(6, -3),
                        textcoords="offset points", fontsize=7, color=INK_2)
        log_axis(ax, "x")
    ax.set_yticks(yy, systems)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.set_xlabel("training time (GPU hours, log)")
    ax.set_title("(a) Training cost")
    ax.legend(handles=[Line2D([], [], color=INK_2, marker="o", linestyle="", label="whole protocol (Ph 1–4)"),
                       Line2D([], [], color=INK_2, marker="o", linestyle="", markerfacecolor=SURFACE,
                              label="final model only (Ph 4)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1)
    ax = axes[1]
    offs = {("fwd", "fp32"): 0.24, ("e2e", "fp32"): 0.08, ("fwd", "fp16"): -0.08, ("e2e", "fp16"): -0.24}
    for (scope, prec), off in offs.items():
        for yi, (_, row) in zip(yy, sel.iterrows()):
            g = row["_by_prec"].get(prec) if "_by_prec" in row else None
            if g is None:
                continue
            med = g[f"{scope}_median_ms"]
            p95 = g["fwd_p95_ms"] if scope == "fwd" else g["e2e_p95_rt_ms"]
            col = ARCH_BY_KEY[row.arch].color
            ax.plot([med, p95], [yi + off] * 2, color=col, linewidth=1)
            filled = prec == "fp32"
            ax.plot(med, yi + off, "o" if scope == "fwd" else "s", color=col, markersize=5.5,
                    markerfacecolor=col if filled else SURFACE, markeredgecolor=col if not filled else SURFACE,
                    markeredgewidth=1.2)
    ax.axvline(1000 / realtime_fps, color=INK_2, linewidth=0.8, linestyle=":")
    log_axis(ax, "x")
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.set_xlabel("latency, batch = 1 (ms, log): median → P95")
    ax.set_title("(b) Inference time")
    mk = lambda m, filled, lab: Line2D([], [], color=INK_2, marker=m, linestyle="", markersize=5.5,  # noqa: E731
                                       markerfacecolor=INK_2 if filled else SURFACE, label=lab)
    ax.legend(handles=[mk("o", True, "forward, FP32"), mk("s", True, "end-to-end, FP32"),
                       mk("o", False, "forward, FP16"), mk("s", False, "end-to-end, FP16"),
                       Line2D([], [], color=INK_2, linestyle=":", linewidth=0.8,
                              label=f"{realtime_fps:g} FPS budget ({1000 / realtime_fps:.1f} ms)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    fig.legend(handles=[Line2D([], [], color=ARCH_BY_KEY[a].color, marker="o", linestyle="", label=ARCH_BY_KEY[a].label)
                        for a in sel.arch.unique()], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    if save:
        save(fig, "article_fig2_training_inference_time")
    return fig


def _horizontal_kw() -> dict[str, Any]:
    import matplotlib

    major, minor = (int(x) for x in matplotlib.__version__.split(".")[:2])
    return {"orientation": "horizontal"} if (major, minor) >= (3, 10) else {"vert": False}


def fig_per_image_distributions(sel: pd.DataFrame, per_image: dict[str, pd.DataFrame],
                                metrics=("dsc", "biou"), save=None):
    """Per-image test scores (box = IQR, whiskers = P5–P95, ◆ = mean) per system."""
    import matplotlib.pyplot as plt

    HORIZONTAL = _horizontal_kw()
    systems = [s for s in sel.system if s in per_image]
    metrics = [m for m in metrics if all(m in per_image[s] for s in systems)]
    yy = np.arange(len(systems))[::-1]
    fig, axes = plt.subplots(1, len(metrics), figsize=(7.2, 0.42 * len(systems) + 1.3), sharey=True, squeeze=False)
    for ax, k in zip(axes[0], metrics):
        for yi, s in zip(yy, systems):
            v = per_image[s][k].dropna().to_numpy(float)
            col = ARCH_BY_KEY[sel.set_index("system").loc[s, "arch"]].color
            ax.boxplot([v], positions=[yi], **HORIZONTAL, widths=0.55, whis=(5, 95), showfliers=False, patch_artist=True,
                       medianprops=dict(color=SURFACE, linewidth=1.5), boxprops=dict(facecolor=col, edgecolor=col),
                       whiskerprops=dict(color=col, linewidth=1.2), capprops=dict(color=col, linewidth=1.2))
            ax.plot(v.mean(), yi, marker="D", color=INK, markersize=4, zorder=4)
        ax.set_yticks(yy, systems)
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
        ax.set_xlabel(f"per-image {METRIC_LABEL[k]} (test)")
    axes[0, 0].set_title("Per-image distribution (◆ mean)")
    fig.tight_layout()
    if save:
        save(fig, "article_fig3_per_image_distributions")
    return fig


def fig_pairwise_matrix(tests: pd.DataFrame, systems: list[str], metric: str = "dsc", save=None):
    """Mean paired difference row − column (percentage points for overlap scores), Holm-adjusted significance."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, to_rgb

    t = tests[tests.metric == metric]
    n = len(systems)
    M = np.full((n, n), np.nan)
    P = np.full((n, n), np.nan)
    scale = 1.0 if metric in LOWER_IS_BETTER else 100.0
    for _, r in t.iterrows():
        i, j = systems.index(r.A), systems.index(r.B)
        M[i, j], M[j, i] = r.delta * scale, -r.delta * scale
        P[i, j] = P[j, i] = r.p_holm
    better = -M if metric in LOWER_IS_BETTER else M       # > 0: row better than column
    lim = np.nanmax(np.abs(better)) if np.isfinite(better).any() else 1.0
    cmap = LinearSegmentedColormap.from_list("div", DIVERGING)
    fig, ax = plt.subplots(figsize=(0.62 * n + 2.2, 0.55 * n + 1.4))
    im = ax.imshow(better, cmap=cmap, vmin=-lim, vmax=lim)
    for i in range(n):
        for j in range(n):
            if i == j or not np.isfinite(M[i, j]):
                continue
            stars = "***" if P[i, j] < 0.001 else "**" if P[i, j] < 0.01 else "*" if P[i, j] < 0.05 else ""
            rgb = cmap((better[i, j] + lim) / (2 * lim))[:3]
            lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
            val = 0.0 if abs(M[i, j]) < 0.05 else M[i, j]
            ax.text(j, i, f"{val:+.1f}{stars}" if val else f"0.0{stars}", ha="center", va="center", fontsize=7,
                    color=INK if lum > 0.55 else SURFACE)
    ax.set_xticks(range(n), systems, rotation=45, ha="right")
    ax.set_yticks(range(n), systems)
    ax.grid(False)
    unit = "px" if metric in LOWER_IS_BETTER else "percentage points"
    ax.set_title(f"Δ{METRIC_LABEL[metric]} row − column ({unit})")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("row better than column (blue) / worse (red)", color=INK_2)
    cb.outline.set_visible(False)
    fig.text(0.01, 0.005, "Wilcoxon signed-rank, Holm-adjusted: * p<.05  ** p<.01  *** p<.001", fontsize=7, color=INK_2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    if save:
        save(fig, f"article_fig4_pairwise_{metric}")
    return fig


def fig_boundary_metrics(sel: pd.DataFrame, save=None):
    """Boundary IoU, NSD and HD95 with bootstrap 95 % CI per system."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    keys = [(k, t) for k, t in (("biou", "Boundary IoU ↑"), ("nsd", "NSD ↑"), ("hd95", "HD95 (px) ↓"))
            if f"{k}_mean" in sel and sel[f"{k}_mean"].notna().any()]
    yy = np.arange(len(sel))[::-1]
    fig, axes = plt.subplots(1, len(keys), figsize=(7.2, 0.42 * len(sel) + 1.2), sharey=True, squeeze=False)
    for ax, (k, title) in zip(axes[0], keys):
        for yi, (_, r) in zip(yy, sel.iterrows()):
            col = ARCH_BY_KEY[r.arch].color
            ax.errorbar(r[f"{k}_mean"], yi, xerr=[[r[f"{k}_mean"] - r[f"{k}_ci95_low"]], [r[f"{k}_ci95_high"] - r[f"{k}_mean"]]],
                        fmt=ARCH_BY_KEY[r.arch].marker, color=col, elinewidth=1.2, capsize=0, markersize=6,
                        markeredgecolor=SURFACE, markeredgewidth=1.2)
        ax.set_yticks(yy, sel.system)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
        ax.set_title(title)
    axes[0, 0].set_xlabel("per-image mean (95 % CI)")
    fig.tight_layout()
    if save:
        save(fig, "article_fig5_boundary_metrics")
    return fig


def fig_memory(sel: pd.DataFrame, save=None):
    """Peak GPU memory of batch-1 inference: allocator peak (model) and process peak (device, CUDA context incl.)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    yy = np.arange(len(sel))[::-1]
    fig, ax = plt.subplots(figsize=(5.2, 0.42 * len(sel) + 1.3))
    for yi, (_, r) in zip(yy, sel.iterrows()):
        col = ARCH_BY_KEY[r.arch].color
        a, p = r.get("vram_peak_allocated_mb", np.nan), r.get("vram_process_peak_mb", np.nan)
        if np.isfinite(a) and np.isfinite(p):
            ax.plot([a, p], [yi, yi], color=col, linewidth=1, alpha=0.6)
        ax.plot(a, yi, "o", color=col, markeredgecolor=SURFACE, markeredgewidth=1.5)
        if np.isfinite(p):
            ax.plot(p, yi, "o", markerfacecolor=SURFACE, markeredgecolor=col, markeredgewidth=1.5)
    log_axis(ax, "x")
    ax.set_yticks(yy, sel.system)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.set_xlabel("peak VRAM, batch-1 inference (MB, log)")
    ax.legend(handles=[Line2D([], [], color=INK_2, marker="o", linestyle="", label="allocator peak (model)"),
                       Line2D([], [], color=INK_2, marker="o", linestyle="", markerfacecolor=SURFACE,
                              label="process peak (incl. CUDA context)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.tight_layout()
    if save:
        save(fig, "article_fig6_memory")
    return fig


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
@dataclass
class Article:
    root: Path
    out: Path
    variant: str
    precision: str
    realtime_fps: float
    pipes: dict[str, Pipeline]
    systems: pd.DataFrame
    sel: pd.DataFrame
    per_image: dict[str, pd.DataFrame]
    tests: pd.DataFrame
    friedman: pd.DataFrame
    tables: dict[str, pd.DataFrame]
    warnings: list[str]

    def save_fig(self, fig, name: str) -> None:
        d = self.out / "figures"
        d.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(d / f"{name}.{ext}", bbox_inches="tight", dpi=300)


def build(root: Path | None = None, pipeline_name: str = "pipeline_final_v1", variant: str = "optimized",
          precision: str = "fp32", realtime_fps: float = 30.0, out: Path | None = None, n_boot: int = 2000,
          seed: int = 0) -> Article:
    """Load the three pipelines and compute every table and statistic (no figures)."""
    warnings: list[str] = []
    root = discover_root(root)
    out = Path(out) if out else root / "article_outputs"
    pipes = find_pipelines(root, pipeline_name, warnings)
    systems = system_frame(pipes)
    if systems.empty:
        raise FileNotFoundError(f"no pipeline results found under {root} (pipeline '{pipeline_name}'); "
                                f"set $YOLO26_PIPELINE_DIR / $UNET_PIPELINE_DIR / $SAM3_PIPELINE_DIR")
    sel = select(systems, variant, precision)
    by_prec = {s: {p: g.iloc[0] for p, g in systems[(systems.variant == variant) & (systems.system == s)].groupby("precision")}
               for s in sel.system}
    sel["_by_prec"] = sel.system.map(by_prec)
    for _, r in sel.iterrows():
        if pd.notna(r.get("contended")) and bool(r.get("contended")):
            warnings.append(f"{r.system}: efficiency benchmark ran on a busy GPU (contended) — re-run Phase 5b")
    per_image = load_per_image(pipes, sel, variant, precision, warnings)
    tests = paired_tests(per_image, n_boot=n_boot, seed=seed)
    fried = friedman_tests(per_image)
    tables: dict[str, pd.DataFrame] = {}
    tables["accuracy"], tables["accuracy_numeric"] = accuracy_table(sel)
    tables["efficiency"], tables["efficiency_numeric"] = efficiency_table(sel, realtime_fps)
    tables["hpo_gain"] = hpo_table(pipes)
    tables["precision"] = precision_table(systems, variant)
    tables["training"] = training_table(pipes)
    tables["cv_vs_test"] = cv_vs_test_table(pipes, select(systems, "baseline", "fp32"))
    return Article(root, out, variant, precision, realtime_fps, pipes, systems, sel, per_image, tests, fried,
                   tables, warnings)


def pairwise_table(tests: pd.DataFrame, metric: str) -> pd.DataFrame:
    t = tests[tests.metric == metric]
    d = 1 if metric in LOWER_IS_BETTER else 4
    return pd.DataFrame({
        "A": t.A, "B": t.B, "n": t.n,
        f"Δ{METRIC_LABEL[metric]} (A−B) [95% CI]": [fmt_ci(m, lo, hi, d) for m, lo, hi in zip(t.delta, t.ci95_low, t.ci95_high)],
        "r (rank-biserial)": t.rank_biserial.map(lambda v: f"{v:+.2f}"),
        "A better / B better": [f"{a} / {b}" for a, b in zip(t.A_better, t.B_better)],
        "p (Wilcoxon)": t.wilcoxon_p.map(fmt_p), "p (Holm)": t.p_holm.map(fmt_p),
    }).reset_index(drop=True)


def write_tables(art: Article) -> list[Path]:
    """Write every table as CSV (numeric where available) and booktabs LaTeX."""
    tdir = art.out / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    written = []
    v, p = art.variant.capitalize(), art.precision.upper()
    acc, accn = art.tables["accuracy"], art.tables["accuracy_numeric"]
    bold = {}
    for k in ("dsc", "jsi", "jsi_thr", "biou", "nsd", "hd95"):
        lab = METRIC_LABEL.get(k)
        if lab in acc:
            bold[lab] = best_rows(accn[f"{k}_mean"], k in LOWER_IS_BETTER)
    specs = [
        ("article_table1_accuracy", acc, f"Test-set segmentation accuracy ({v} models, {p}, n = 1,000 ISIC 2018 Task 1 "
         "test images): per-image mean with seeded bootstrap 95 % CI. HD95 in pixels at dataset resolution "
         "(lower is better). Missed = images with an empty prediction (scored 0, never skipped).", bold, True),
    ]
    eff, effn = art.tables["efficiency"], art.tables["efficiency_numeric"]
    bold_e = {lab: best_rows(effn[c], c in LOWER_BETTER_EFF) for c, lab, _ in EFF_COLS if lab in eff and c in effn
              and c != "input_px"}
    specs.append(("article_table2_efficiency", eff, f"Batch-1 inference efficiency on one GPU ({v} models, {p}). "
                  "Fwd = network forward pass (CUDA events); E2E = image in host memory → binary mask at dataset "
                  "resolution in host memory; E2E P95 over 100 distinct test images. FPS = 1000 / mean latency. "
                  "VRAM alloc. = PyTorch allocator peak; VRAM process = device memory of the process incl. CUDA "
                  "context. GFLOPs at each model's native input (YOLO26/U-Net: thop; SAM 3: FlopCounterMode, "
                  f"attention products included). Real-time = E2E P95 ≤ {1000 / art.realtime_fps:.1f} ms.", bold_e, True))
    for name, df, cap in (("article_table3_hpo_gain", art.tables["hpo_gain"],
                           "Effect of hyperparameter optimisation on the test set (Optimized − Baseline, FP32): mean "
                           "paired difference with bootstrap 95 % CI and two-sided Wilcoxon signed-rank p."),
                          ("article_table4_fp16", art.tables["precision"].round(4),
                           "Half-precision deployment: accuracy cost and speed-up of FP16 vs. FP32 (batch 1)."),
                          ("article_table5_training_cost", art.tables["training"].drop(columns="arch", errors="ignore").round(2),
                           "Training cost per phase (wall-clock GPU hours, validation included)."),
                          ("article_table6_cv_vs_test", art.tables["cv_vs_test"],
                           "Phase 2 cross-validation (Baseline protocol, 5 folds, mean ± SD) vs. Baseline test score.")):
        if len(df):
            specs.append((name, df, cap, {}, True))
    pt = pairwise_table(art.tests, "dsc") if len(art.tests) else pd.DataFrame()
    if len(pt):
        specs.append(("article_table7_pairwise_dsc", pt, "Paired cross-architecture comparison of per-image DSC "
                      f"({v}, {p}): mean difference A − B with bootstrap 95 % CI, matched-pairs rank-biserial r, "
                      "two-sided Wilcoxon signed-rank p and Holm-adjusted p (all pairs).", {}, True))
    for name, df, cap, b, resize in specs:
        (tdir / f"{name}.tex").write_text(to_latex(df, cap, f"tab:{name}", bold=b, resize=resize))
        df.to_csv(tdir / f"{name}.csv", index=False)
        written.append(tdir / f"{name}.tex")
    num = art.out / "data"
    num.mkdir(parents=True, exist_ok=True)
    art.sel.drop(columns="_by_prec").to_csv(num / "systems_selected.csv", index=False)
    art.systems.to_csv(num / "systems_all_variants.csv", index=False)
    art.tests.to_csv(num / "paired_tests_all_metrics.csv", index=False)
    art.friedman.to_csv(num / "friedman_tests.csv", index=False)
    accn.to_csv(num / "accuracy_numeric.csv", index=False)
    effn.to_csv(num / "efficiency_numeric.csv", index=False)
    (num / "warnings.txt").write_text("\n".join(art.warnings) + "\n")
    return written


def make_figures(art: Article, show: bool = False) -> list[str]:
    import matplotlib.pyplot as plt

    setup_style()
    figs = [
        fig_accuracy_vs_efficiency(art.sel, save=art.save_fig),
        fig_training_inference_time(art.sel, art.tables["training"], art.realtime_fps, save=art.save_fig),
        fig_boundary_metrics(art.sel, save=art.save_fig),
        fig_memory(art.sel, save=art.save_fig),
    ]
    if art.per_image:
        figs.append(fig_per_image_distributions(art.sel, art.per_image, save=art.save_fig))
    if len(art.tests):
        systems = [s for s in art.sel.system if s in art.per_image]
        figs.append(fig_pairwise_matrix(art.tests, systems, "dsc", save=art.save_fig))
    if show:
        plt.show()
    for f in figs:
        plt.close(f)
    return sorted(p.name for p in (art.out / "figures").glob("*.pdf"))


def key_numbers(art: Article) -> list[str]:
    """Sentences with the headline numbers for the Results section."""
    s, lines = art.sel, []
    if s.empty:
        return lines
    best = s.loc[s.dsc_mean.idxmax()]
    lines.append(f"Highest test DSC: {best.system} with {fmt_ci(best.dsc_mean, best.dsc_ci95_low, best.dsc_ci95_high)} "
                 f"(JSI {fmt_ci(best.jsi_mean, best.jsi_ci95_low, best.jsi_ci95_high)}).")
    if "fwd_median_ms" in s and s.fwd_median_ms.notna().any():
        fast = s.loc[s.fwd_median_ms.idxmin()]
        lines.append(f"Fastest network: {fast.system}, forward median {fast.fwd_median_ms:.2f} ms "
                     f"(P95 {fast.fwd_p95_ms:.2f} ms, {fast.fwd_fps:.0f} FPS), end-to-end median "
                     f"{fast.e2e_median_ms:.2f} ms (P95 {fast.e2e_p95_rt_ms:.2f} ms).")
        rt = s[s.e2e_p95_rt_ms <= 1000 / art.realtime_fps].system.tolist()
        lines.append(f"Meeting the {art.realtime_fps:g}-FPS real-time criterion (E2E P95 ≤ {1000 / art.realtime_fps:.1f} ms, "
                     f"{art.precision.upper()}): {', '.join(rt) if rt else 'none'}.")
    if "sam3" in set(s.arch) and len(s[s.arch != "sam3"]):
        sam = s[s.arch == "sam3"].iloc[0]
        light = s[s.arch != "sam3"].loc[lambda d: d.dsc_mean.idxmax()]
        t = art.tests[(art.tests.metric == "dsc") & (((art.tests.A == sam.system) & (art.tests.B == light.system)) |
                                                       ((art.tests.B == sam.system) & (art.tests.A == light.system)))]
        if len(t):
            r = t.iloc[0]
            sign = 1 if r.A == sam.system else -1
            lines.append(f"SAM 3 vs. the most accurate lightweight model ({light.system}): ΔDSC = {sign * r.delta:+.4f} "
                         f"[{min(sign * r.ci95_low, sign * r.ci95_high):+.4f}, {max(sign * r.ci95_low, sign * r.ci95_high):+.4f}], "
                         f"Holm-adjusted {p_text(r.p_holm)}, rank-biserial r = {sign * r.rank_biserial:+.2f}.")
        ratio = lambda c: sam[c] / light[c] if np.isfinite(sam.get(c, np.nan)) and light.get(c, 0) else np.nan  # noqa: E731
        lines.append(f"Cost ratio SAM 3 / {light.system}: parameters ×{ratio('params_fused'):.0f}, GFLOPs ×"
                     f"{ratio('gflops'):.0f}, forward latency ×{ratio('fwd_median_ms'):.0f}, end-to-end latency ×"
                     f"{ratio('e2e_median_ms'):.0f}, peak VRAM (allocator) ×{ratio('vram_peak_allocated_mb'):.0f}.")
    for _, f in art.friedman.iterrows():
        if f.metric == "dsc":
            lines.append(f"Friedman test across the {f.k_systems} systems (DSC): χ² = {f.chi2:.1f}, {p_text(f.p)}, "
                         f"Kendall's W = {f.kendall_w:.3f}.")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Cross-architecture article tables and figures (YOLO26 / U-Net / SAM 3).")
    ap.add_argument("--root", type=Path, default=None, help="Folder containing the three sandbox_* repositories.")
    ap.add_argument("--pipeline-name", default="pipeline_final_v1")
    ap.add_argument("--variant", default="optimized", choices=("baseline", "optimized"))
    ap.add_argument("--precision", default="fp32", choices=("fp32", "fp16"))
    ap.add_argument("--realtime-fps", type=float, default=30.0)
    ap.add_argument("--out", type=Path, default=None, help="Output folder (default: <root>/article_outputs).")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args(argv)
    art = build(args.root, args.pipeline_name, args.variant, args.precision, args.realtime_fps, args.out, args.n_boot)
    print(f"root      : {art.root}")
    for k, p in art.pipes.items():
        print(f"pipeline  : {ARCH_BY_KEY[k].label:<11} {p.root}")
    tables = write_tables(art)
    figs = make_figures(art)
    print(f"tables    : {len(tables)} LaTeX + CSV in {art.out / 'tables'}")
    print(f"figures   : {len(figs)} (PDF + PNG) in {art.out / 'figures'}")
    print("\nKey numbers:")
    for line in key_numbers(art):
        print(f"  - {line}")
    if art.warnings:
        print(f"\n{len(art.warnings)} warning(s):")
        for w in art.warnings:
            print(f"  - {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
