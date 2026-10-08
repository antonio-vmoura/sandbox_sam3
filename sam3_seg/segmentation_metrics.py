"""Pixel-level segmentation metrics (DSC, JSI, ...) shared by Phase 2, Phase 5 and the notebooks.

Ultralytics reports detection-style metrics (mAP, P, R) per *instance*. Lesion
segmentation papers (ISIC 2018 Task 1) report *pixel-level* overlap between
the ground-truth and the predicted lesion masks, which is computed here:

* **Ground truth** (:func:`ground_truth_mask`) — the dataset's mask image
  (``masks/<stem>.png``, the official ISIC mask at the working resolution)
  when the dataset has one; otherwise every polygon of the YOLO label file,
  rasterised at the image resolution (:func:`rasterize_yolo_label`) and
  merged (union).
* **Prediction** — the mask of the single highest-confidence instance among
  those with confidence ``>= conf`` (:data:`PIXEL_CONF` = 0.001), produced at
  the original resolution (``retina_masks=True``) (:func:`predicted_top1_mask`).
  ISIC 2018 Task 1 has exactly one lesion per image, so lower-ranked instances
  are background artefacts and are never merged into the lesion mask.
  :func:`predicted_union_mask` is kept for diagnostics only.
* **Per-image scores** (:func:`pixel_scores`) from the confusion counts
  TP / FP / FN / TN:

  - ``DSC = 2TP / (2TP + FP + FN)`` (Dice / F1 of the pixels)
  - ``JSI = TP / (TP + FP + FN)`` (Jaccard / IoU)
  - ``JSI_thr = JSI if JSI >= 0.65 else 0`` (ISIC 2018 Task 1 official score)
  - sensitivity ``TP / (TP + FN)``, specificity ``TN / (TN + FP)``,
    pixel accuracy ``(TP + TN) / N``.

* **Boundary scores** (contour adherence, as recommended by *Metrics
  Reloaded*, Maier-Hein et al., Nat. Methods 2024; overlap scores are
  insensitive to boundary errors on large lesions):

  - ``BIoU`` — Boundary IoU (Cheng et al., CVPR 2021): the IoU of the two
    *boundary bands*, i.e. the pixels of each mask within ``d`` pixels of its
    own contour, ``d`` = :data:`BOUNDARY_DILATION_RATIO` × image diagonal
    (2 %, the authors' setting; ≈ 26 px for a 1024 × 768 image).
  - ``NSD`` — Normalised Surface Distance / surface Dice (Nikolov et al.,
    2021): the fraction of contour pixels of both masks that lie within a
    tolerance ``tau`` of the other mask's contour, ``tau`` =
    :data:`NSD_TOLERANCE_RATIO` × image diagonal (1 %; ≈ 13 px for a 1024 × 768 image).
  - ``HD95`` — 95th-percentile symmetric Hausdorff distance, in **pixels at
    the dataset resolution** (lower is better): the larger of the two directed
    95th percentiles of the contour-to-contour distances (ground truth →
    prediction and prediction → ground truth), the convention of MONAI and of
    DeepMind's ``surface-distance`` package. It reports the size of the
    boundary errors, which BIoU and NSD (fractions within a tolerance) do not.

  Contours are the 1-pixel inner boundaries of the masks; the image border
  counts as background (zero padding), as in the reference BIoU code.

Empty masks are handled explicitly, never by dividing by zero:

* GT empty **and** prediction empty → DSC = JSI = 1 (perfect agreement); the
  row is flagged ``both_empty``. Sensitivity is undefined (NaN).
* Exactly one of them empty → DSC = JSI = 0 (flagged ``empty_pred`` or
  ``empty_gt``). A missed lesion therefore counts as a full failure in the mean.
  The boundary scores follow the same rule (1 if both empty, 0 if one is);
  HD95 is 0 if both are empty and the worst possible value, the image
  diagonal, if one is (Metrics Reloaded: penalise, never drop, missing
  predictions in distance-based metrics).

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

#: Confidence threshold of the pixel-level evaluation (Phase 2 pixels, Phase 5, end-to-end latency).
#: With :func:`predicted_top1_mask` a low threshold only guarantees that the primary lesion is never
#: dropped; it cannot add false-positive instances to the mask.
PIXEL_CONF: float = 0.001

#: ISIC 2018 Task 1 threshold for the thresholded Jaccard index.
ISIC_JSI_THRESHOLD: float = 0.65

#: ISIC 2018 Task 2 lesion attributes; the YOLO class id of an attribute is its index.
ISIC2018_ATTRIBUTES: tuple[str, ...] = (
    "pigment_network", "negative_network", "streaks", "milia_like_cyst", "globules",
)

#: Boundary-band width of the Boundary IoU, as a fraction of the image diagonal.
BOUNDARY_DILATION_RATIO: float = 0.02

#: Tolerance of the Normalised Surface Distance, as a fraction of the image diagonal.
NSD_TOLERANCE_RATIO: float = 0.01

#: Percentile of the robust Hausdorff distance (HD95).
HAUSDORFF_PERCENTILE: float = 95.0

#: Per-image score columns aggregated by :func:`aggregate_scores`.
SCORE_KEYS: tuple[str, ...] = (
    "dsc", "jsi", "jsi_thr", "sensitivity", "specificity", "accuracy", "biou", "nsd", "hd95",
)

#: Scores for which lower is better (distances, in pixels at dataset resolution).
LOWER_IS_BETTER: frozenset[str] = frozenset({"hd95"})


# ----------------------------------------------------------------------------
# Masks
# ----------------------------------------------------------------------------
def label_path_for(image_path: Path) -> Path:
    """Return the YOLO label path of an image (``/images/`` → ``/labels/``, ``.txt``)."""
    return Path(str(image_path).replace("/images/", "/labels/")).with_suffix(".txt")


def mask_path_for(image_path: Path) -> Path:
    """Return the ground-truth mask path of an image (``/images/`` → ``/masks/``, ``.png``)."""
    return Path(str(image_path).replace("/images/", "/masks/")).with_suffix(".png")


def ground_truth_mask(image_path: Path, height: int, width: int) -> np.ndarray:
    """Ground-truth lesion mask of a dataset image at ``height × width``.

    The official mask (``masks/<stem>.png``, written by Phase 0) is used when it exists — it is exact, whereas
    the YOLO label is a polygon approximation for training. Datasets without mask images fall back to the
    rasterised YOLO label.

    Raises:
        ValueError: If the mask image does not have the requested size.
    """
    mpath = mask_path_for(image_path)
    if mpath.exists():
        mask = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != (height, width):
            raise ValueError(f"{mpath}: mask {None if mask is None else mask.shape} != image {(height, width)}")
        return mask > 127
    return rasterize_yolo_label(label_path_for(image_path), height, width)


def class_mask_path_for(image_path: Path, class_name: str) -> Path:
    """Ground-truth mask of one class (Task 2 attribute): ``/images/`` → ``/masks/<class_name>/``, ``.png``."""
    return Path(str(image_path).replace("/images/", f"/masks/{class_name}/")).with_suffix(".png")


def ground_truth_class_mask(image_path: Path, class_name: str, class_id: int, height: int, width: int) -> np.ndarray:
    """Ground-truth mask of one class of a multi-label dataset (ISIC 2018 Task 2 attribute).

    The official attribute mask (``masks/<class_name>/<stem>.png``, written by Phase 0) when it exists,
    otherwise the polygons of class ``class_id`` of the YOLO label, rasterised. Classes may overlap.
    """
    mpath = class_mask_path_for(image_path, class_name)
    if mpath.exists():
        mask = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != (height, width):
            raise ValueError(f"{mpath}: mask {None if mask is None else mask.shape} != image {(height, width)}")
        return mask > 127
    return rasterize_yolo_label(label_path_for(image_path), height, width, class_id=class_id)


def rasterize_yolo_label(label_path: Path, height: int, width: int, class_id: int | None = None) -> np.ndarray:
    """Rasterise a YOLO label file into a binary union mask at full resolution.

    Segmentation lines (``cls x1 y1 x2 y2 ...``, normalised) are filled as
    polygons. Plain detection lines (``cls xc yc w h``) are filled as boxes.
    A missing or empty label file yields an empty mask (background image).

    Args:
        label_path: Path to the ``.txt`` label.
        height: Image height in pixels.
        width: Image width in pixels.
        class_id: Rasterise only the lines of this class (multi-label datasets); ``None`` = all lines.

    Returns:
        ``bool`` array of shape ``(height, width)``.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    if not Path(label_path).exists():
        return mask.astype(bool)
    scale = np.array([width, height], dtype=np.float64)
    for line in Path(label_path).read_text().splitlines():
        vals = line.split()
        if len(vals) < 5 or (class_id is not None and int(float(vals[0])) != class_id):
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


def predicted_top1_mask(result: Any, height: int, width: int) -> np.ndarray:
    """Mask of the highest-confidence predicted instance of one Ultralytics ``Results``.

    The pixel-evaluation rule for single-lesion data (ISIC 2018 Task 1): exactly one
    lesion per image, so only the top-ranked instance is the lesion and the others are
    discarded instead of merged. The argmax is explicit because YOLO26 is NMS-free and
    its output order is not guaranteed. The prediction must have been made with
    ``retina_masks=True``. No detection → empty mask.

    Returns:
        ``bool`` array of shape ``(height, width)``.
    """
    if result.masks is None or len(result.masks) == 0:
        return np.zeros((height, width), dtype=bool)
    top = int(result.boxes.conf.argmax())
    mask = result.masks.data[top].cpu().numpy() > 0.5
    if mask.shape != (height, width):  # defensive: never expected with retina_masks
        mask = cv2.resize(mask.astype(np.uint8), (width, height),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
    return mask


def predicted_union_mask(result: Any, height: int, width: int) -> np.ndarray:
    """Union of all predicted instance masks of one Ultralytics ``Results`` (diagnostics only).

    Not used for evaluation: on single-lesion data the union merges low-confidence
    background instances into the lesion mask (see :func:`predicted_top1_mask`).

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


def _distance_to_zero(mask: np.ndarray, border: int = 0) -> np.ndarray:
    """Euclidean distance of every pixel to the nearest zero pixel (outside the image = ``border``)."""
    padded = np.pad(mask.astype(np.uint8), 1, constant_values=border)
    return cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]


def _contour(mask: np.ndarray) -> np.ndarray:
    """1-pixel inner boundary of a binary mask (image border counts as background)."""
    eroded = cv2.erode(np.pad(mask.astype(np.uint8), 1), np.ones((3, 3), np.uint8))[1:-1, 1:-1]
    return mask & ~eroded.astype(bool)


def boundary_scores(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Boundary IoU, Normalised Surface Distance and HD95 of one image (see module docstring)."""
    diag = math.hypot(*gt.shape)
    if not gt.any() or not pred.any():
        both = not gt.any() and not pred.any()
        return {"biou": float(both), "nsd": float(both), "hd95": 0.0 if both else float(diag)}
    d = max(1.0, BOUNDARY_DILATION_RATIO * diag)
    tau = max(1.0, NSD_TOLERANCE_RATIO * diag)
    band_gt = gt & (_distance_to_zero(gt) <= d)
    band_pred = pred & (_distance_to_zero(pred) <= d)
    biou = np.count_nonzero(band_gt & band_pred) / np.count_nonzero(band_gt | band_pred)
    c_gt, c_pred = _contour(gt), _contour(pred)
    to_gt = _distance_to_zero(~c_gt, border=1)      # distance to the nearest ground-truth contour pixel
    to_pred = _distance_to_zero(~c_pred, border=1)
    gt_to_pred, pred_to_gt = to_pred[c_gt], to_gt[c_pred]
    hits = np.count_nonzero(gt_to_pred <= tau) + np.count_nonzero(pred_to_gt <= tau)
    nsd = hits / (len(gt_to_pred) + len(pred_to_gt))
    hd95 = max(np.percentile(gt_to_pred, HAUSDORFF_PERCENTILE), np.percentile(pred_to_gt, HAUSDORFF_PERCENTILE))
    return {"biou": float(biou), "nsd": float(nsd), "hd95": float(hd95)}


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
        **boundary_scores(gt, pred),
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
    """Predict every image (batch=1) and score its top-1 instance mask against the ground truth.

    Args:
        model: An Ultralytics ``YOLO`` segmentation model.
        images: Image paths; labels are found via :func:`label_path_for`.
        conf: Confidence threshold of the candidate instances (:data:`PIXEL_CONF`).
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
        pred = predicted_top1_mask(result, h, w)
        gt = ground_truth_mask(img_path, h, w)
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
