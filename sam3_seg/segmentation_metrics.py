"""Pixel-level segmentation metrics (DSC, JSI, ...) shared by Phase 2, Phase 5 and the notebooks.

Ultralytics reports detection-style metrics (mAP, P, R) per *instance*. Lesion
segmentation papers (ISIC 2018 Task 1) report *pixel-level* overlap between
the ground-truth and the predicted lesion masks, which is computed here:

* **Ground truth** — every polygon of the YOLO label file, rasterised at the
  original image resolution (:func:`rasterize_yolo_label`) and merged (union).
* **Prediction** — the union of every predicted instance mask with
  confidence ``>= conf``, produced at the original resolution
  (``retina_masks=True``) (:func:`predicted_union_mask`).
* **Per-image scores** (:func:`pixel_scores`) from the confusion counts
  TP / FP / FN / TN:

  - ``DSC = 2TP / (2TP + FP + FN)`` (Dice / F1 of the pixels)
  - ``JSI = TP / (TP + FP + FN)`` (Jaccard / IoU)
  - ``JSI_thr = JSI if JSI >= 0.65 else 0`` (ISIC 2018 Task 1 official score)
  - sensitivity ``TP / (TP + FN)``, specificity ``TN / (TN + FP)``,
    pixel accuracy ``(TP + TN) / N``.

Empty masks are handled explicitly, never by dividing by zero:

* GT empty **and** prediction empty → DSC = JSI = 1 (perfect agreement); the
  row is flagged ``both_empty``. Sensitivity is undefined (NaN).
* Exactly one of them empty → DSC = JSI = 0 (flagged ``empty_pred`` or
  ``empty_gt``). A missed lesion therefore counts as a full failure in the mean.

Dataset-level aggregates (:func:`aggregate_scores`) report the per-image mean
(macro, the primary figure), sample std (ddof=1), median, IQR, a seeded
bootstrap 95 % CI of the mean, and the pooled (micro) DSC/JSI.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

#: ISIC 2018 Task 1 threshold for the thresholded Jaccard index.
ISIC_JSI_THRESHOLD: float = 0.65

#: Per-image score columns aggregated by :func:`aggregate_scores`.
SCORE_KEYS: tuple[str, ...] = (
    "dsc", "jsi", "jsi_thr", "sensitivity", "specificity", "accuracy",
)


# ----------------------------------------------------------------------------
# Masks
# ----------------------------------------------------------------------------
def label_path_for(image_path: Path) -> Path:
    """Return the YOLO label path of an image (``/images/`` → ``/labels/``, ``.txt``)."""
    return Path(str(image_path).replace("/images/", "/labels/")).with_suffix(".txt")


def rasterize_yolo_label(label_path: Path, height: int, width: int) -> np.ndarray:
    """Rasterise a YOLO label file into a binary union mask at full resolution.

    Segmentation lines (``cls x1 y1 x2 y2 ...``, normalised) are filled as
    polygons. Plain detection lines (``cls xc yc w h``) are filled as boxes.
    A missing or empty label file yields an empty mask (background image).

    Args:
        label_path: Path to the ``.txt`` label.
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        ``bool`` array of shape ``(height, width)``.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    if not Path(label_path).exists():
        return mask.astype(bool)
    scale = np.array([width, height], dtype=np.float64)
    for line in Path(label_path).read_text().splitlines():
        vals = line.split()
        if len(vals) < 5:
            continue
        coords = np.array(vals[1:], dtype=np.float64)
        if len(coords) == 4:  # bounding box
            xc, yc, bw, bh = coords * np.array([width, height, width, height])
            x0, y0 = int(round(xc - bw / 2)), int(round(yc - bh / 2))
            x1, y1 = int(round(xc + bw / 2)), int(round(yc + bh / 2))
            cv2.rectangle(mask, (x0, y0), (x1, y1), 1, thickness=-1)
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            pts = np.round(coords.reshape(-1, 2) * scale).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def predicted_union_mask(result: Any, height: int, width: int) -> np.ndarray:
    """Union of all predicted instance masks of one Ultralytics ``Results``.

    The prediction must have been made with ``retina_masks=True`` so masks are
    already at the original resolution. No detection → empty mask.

    Returns:
        ``bool`` array of shape ``(height, width)``.
    """
    if result.masks is None or len(result.masks) == 0:
        return np.zeros((height, width), dtype=bool)
    data = result.masks.data.cpu().numpy() > 0.5
    union = data.any(axis=0)
    if union.shape != (height, width):  # defensive: never expected with retina_masks
        union = cv2.resize(union.astype(np.uint8), (width, height),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
    return union


# ----------------------------------------------------------------------------
# Scores
# ----------------------------------------------------------------------------
def _ratio(num: float, den: float) -> float:
    """``num / den`` or NaN when the denominator is zero."""
    return num / den if den > 0 else math.nan


def pixel_scores(gt: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    """Compute the confusion counts and pixel scores of one image.

    Args:
        gt: Binary ground-truth mask.
        pred: Binary predicted mask (same shape).

    Returns:
        Dict with ``tp``, ``fp``, ``fn``, ``tn``, the :data:`SCORE_KEYS`,
        ``gt_px``, ``pred_px`` and the ``empty_gt`` / ``empty_pred`` /
        ``both_empty`` flags.
    """
    if gt.shape != pred.shape:
        raise ValueError(f"mask shapes differ: gt {gt.shape} vs pred {pred.shape}")
    gt, pred = gt.astype(bool), pred.astype(bool)
    tp = int(np.count_nonzero(gt & pred))
    fp = int(np.count_nonzero(~gt & pred))
    fn = int(np.count_nonzero(gt & ~pred))
    tn = int(gt.size - tp - fp - fn)
    empty_gt, empty_pred = tp + fn == 0, tp + fp == 0

    if empty_gt and empty_pred:
        dsc = jsi = 1.0
    else:  # denominators are > 0 whenever at least one mask is non-empty
        dsc = 2 * tp / (2 * tp + fp + fn)
        jsi = tp / (tp + fp + fn)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "gt_px": tp + fn, "pred_px": tp + fp,
        "dsc": dsc,
        "jsi": jsi,
        "jsi_thr": jsi if jsi >= ISIC_JSI_THRESHOLD else 0.0,
        "sensitivity": _ratio(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp),
        "accuracy": (tp + tn) / gt.size,
        "empty_gt": empty_gt,
        "empty_pred": empty_pred,
        "both_empty": empty_gt and empty_pred,
    }


def bootstrap_ci(values: np.ndarray, n_resamples: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Seeded percentile bootstrap CI of the mean (NaNs ignored)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_resamples, len(v)))].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def aggregate_scores(rows: list[dict[str, Any]], seed: int = 0) -> dict[str, Any]:
    """Aggregate per-image rows into dataset-level statistics.

    Returns:
        ``{"n_images", "n_empty_pred", "n_empty_gt", "n_both_empty",
        "pooled_dsc", "pooled_jsi", <score>: {mean, std, median, q1, q3,
        ci95_low, ci95_high, n}}``.
    """
    out: dict[str, Any] = {
        "n_images": len(rows),
        "n_empty_pred": sum(bool(r["empty_pred"]) for r in rows),
        "n_empty_gt": sum(bool(r["empty_gt"]) for r in rows),
        "n_both_empty": sum(bool(r["both_empty"]) for r in rows),
    }
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    out["pooled_dsc"] = _ratio(2 * tp, 2 * tp + fp + fn)
    out["pooled_jsi"] = _ratio(tp, tp + fp + fn)
    for key in SCORE_KEYS:
        v = np.array([r[key] for r in rows], dtype=float)
        v = v[~np.isnan(v)]
        if len(v) == 0:
            out[key] = {"n": 0}
            continue
        lo, hi = bootstrap_ci(v, seed=seed)
        out[key] = {
            "n": int(len(v)),
            "mean": float(v.mean()),
            "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "median": float(np.median(v)),
            "q1": float(np.quantile(v, 0.25)),
            "q3": float(np.quantile(v, 0.75)),
            "ci95_low": lo,
            "ci95_high": hi,
        }
    return out


# ----------------------------------------------------------------------------
# Inference loop
# ----------------------------------------------------------------------------
def evaluate_images(
    model: Any,
    images: Iterable[Path],
    *,
    conf: float,
    imgsz: int,
    half: bool,
    device: Any,
    mask_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Predict every image (batch=1) and score it against its YOLO label.

    Args:
        model: An Ultralytics ``YOLO`` segmentation model.
        images: Image paths; labels are found via :func:`label_path_for`.
        conf: Confidence threshold for the instances merged into the mask.
        imgsz: Inference size.
        half: FP16 inference.
        device: Ultralytics device argument.
        mask_dir: If given, the predicted union mask of each image is saved
            there as ``<stem>.png`` (0/255), for the visualisation notebook.

    Returns:
        One row per image: ``image``, ``height``, ``width``, ``n_pred``,
        ``max_conf``, ``infer_ms`` and every key of :func:`pixel_scores`.
    """
    if mask_dir is not None:
        Path(mask_dir).mkdir(parents=True, exist_ok=True)
    rows = []
    images = list(images)
    for i, img_path in enumerate(images, 1):
        t0 = time.perf_counter()
        result = model.predict(
            source=str(img_path), imgsz=imgsz, conf=conf, half=half, device=device,
            retina_masks=True, verbose=False,
        )[0]
        infer_ms = (time.perf_counter() - t0) * 1000
        h, w = result.orig_shape
        pred = predicted_union_mask(result, h, w)
        gt = rasterize_yolo_label(label_path_for(img_path), h, w)
        if mask_dir is not None:
            cv2.imwrite(str(Path(mask_dir) / f"{Path(img_path).stem}.png"),
                        pred.astype(np.uint8) * 255)
        boxes = result.boxes
        rows.append({
            "image": str(img_path), "height": h, "width": w,
            "n_pred": 0 if boxes is None else len(boxes),
            "max_conf": float(boxes.conf.max()) if boxes is not None and len(boxes) else 0.0,
            "infer_ms": infer_ms,
            **pixel_scores(gt, pred),
        })
        if i % 100 == 0 or i == len(images):
            print(f"    {i}/{len(images)} images", flush=True)
    return rows
