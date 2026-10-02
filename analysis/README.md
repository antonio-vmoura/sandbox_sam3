# `analysis/` — cross-architecture results analysis

These files are **byte-identical in `sandbox_yolo26`, `sandbox_unet` and `sandbox_sam3`** (like
`segmentation_metrics.py`): whichever repository you pull, you get the same article tooling.

| File | Content |
|---|---|
| `results_analysis.ipynb` | Master notebook: unified LaTeX tables, paired cross-architecture statistics and figures (YOLO26-seg vs. U-Net vs. SAM 3) |
| `article_aggregator.py` | The logic behind the notebook; also runnable headless: `python article_aggregator.py` |
| `methodology_notes.md` | Unified methodology (5-phase design, fairness controls, early-stopping justification, model-specific quirks, Phase 0, metrics, efficiency protocol, limitations, Methods wording) |

**Where to run.** Put the notebook and `article_aggregator.py` in the folder that contains the three repositories
(e.g. `~/projects/` with `sandbox_yolo26/`, `sandbox_unet/`, `sandbox_sam3/` — any letter case), or run them from here:
the three repositories are discovered in the current folder, its parent or its grandparent. The pipelines are read
from `<repo>/logs/<PIPELINE_NAME>/` (default `pipeline_final_v1`); override with `$YOLO26_PIPELINE_DIR`,
`$UNET_PIPELINE_DIR`, `$SAM3_PIPELINE_DIR`. Outputs go to `<folder holding the repositories>/article_outputs/`
(`tables/`, `figures/`, `data/`), never inside a repository.

**Inputs.** Only Phase 5 outputs (`summary/*.csv`, `summary/final_results.json`, `phase5_test/per_image/*.csv`): no
training, no GPU. A pipeline that has not finished is skipped with a warning.

**Requirements.** `numpy`, `pandas`, `scipy`, `matplotlib` (+ `jupyterlab` for the notebook). LaTeX tables need
`\usepackage{booktabs,graphicx}`.
