"""Phase 0 — Build the SAM 3 COCO dataset from the YOLO26 dataset (single source of truth).

SAM 3 is trained and evaluated on **exactly the same images, splits and
annotations** as YOLO26-seg and the U-Net. This script reads the YOLO-format
dataset (``data.yaml``: ``train`` / ``val`` / ``test`` + polygon labels) and
writes a COCO dataset in the format the official SAM 3 loaders consume
(compressed RLE masks, like the previous Roboflow COCO export)::

    <out>/images/<file>.jpg                 # one pool: hard links (identical bytes) or copies
    <out>/annotations/{train,val,test}.json # one COCO file per split
    <out>/folds/fold_<k>/{train,val}.json   # the 5 CV folds — identical to YOLO26's
    <out>/manifest_{train,val,test}.csv     # index, id, image, label, size, n_instances
    <out>/meta.json                         # provenance (source SHA-256, parameters, counts)

Per image (deterministic):

* Each YOLO polygon (one line of the label file) becomes one COCO instance:
  it is rasterised at the image's own resolution with **the convention of**
  :func:`segmentation_metrics.rasterize_yolo_label` (the function YOLO26's and
  the U-Net's evaluation use; their union is verified to equal it), encoded as
  compressed RLE with ``pycocotools``; ``bbox`` and ``area`` are derived from
  the mask.
* The image file is linked (or copied) unchanged — SAM 3 sees the same pixels
  as YOLO26 (the 640 × 640 export) and resizes them internally to 1008.
* The single category is named ``"skin lesion"`` (:data:`common.PROMPT`): SAM 3
  uses the category name as its text prompt.

Cross-validation folds: the pool is the train IDs followed by the val IDs in
YOLO26's order, split with YOLO26's algorithm (``RandomState(seed)`` shuffle +
contiguous folds), so the folds are identical to those of YOLO26 and the U-Net.

Checks: no ISIC ID in two splits; every instance mask non-empty; the cache is
rebuilt only when the source files (SHA-256) or the parameters change
(``--force`` rebuilds); the new dataset is written to a temporary folder and
swapped in atomically, a replaced one is kept as ``<out>.bak-<UTC>``.

Runs inside the ``sam3_ft`` image (needs ``pycocotools``).

Usage:
    python sam3_seg/prepare_dataset.py \\
        --yolo-data /workspace/datasets/isic_2018_task1_yolo26/data.yaml \\
        --out /workspace/datasets/isic_2018_task1_sam3
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from pycocotools import mask as mask_utils

from common import (
    DEFAULT_DATA_DIR,
    DEFAULT_YOLO_DATA_YAML,
    PROMPT,
    SEED,
    atomic_write_json,
    read_json,
    sha256_file,
    utc_now_iso,
    utc_stamp,
)
from segmentation_metrics import label_path_for

#: Version of the preprocessing method (part of the dataset fingerprint).
PREP_VERSION: int = 1

#: ``data.yaml`` split key → output split name.
SPLITS: dict[str, str] = {"train": "train", "val": "val", "test": "test"}

#: Image extensions (same as YOLO26's cross-validation).
IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

#: Number of CV folds (identical to YOLO26 / U-Net).
K_FOLDS: int = 5

#: The single COCO category (its name is SAM 3's text prompt).
CATEGORY: dict[str, Any] = {"id": 0, "name": PROMPT, "supercategory": "lesion"}


# ----------------------------------------------------------------------------
# Sources (same logic as the U-Net Phase 0)
# ----------------------------------------------------------------------------
def isic_id(image: Path) -> str:
    """``ISIC_0012169_jpg.rf.<hash>.jpg`` → ``ISIC_0012169``."""
    stem = image.name.split(".rf.")[0]
    for ext in ("_jpg", "_jpeg", "_png"):
        if stem.endswith(ext):
            return stem[: -len(ext)]
    return Path(stem).stem


def resolve_split(data_yaml: Path, key: str) -> tuple[Path, list[Path]]:
    """``(dataset_root, sorted image paths)`` of one ``data.yaml`` split."""
    data = yaml.safe_load(data_yaml.read_text()) or {}
    root = Path(data.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    if not root.is_dir():  # e.g. a /workspace/... path used outside the container
        root = data_yaml.parent
    value = data.get(key)
    if value is None:
        raise ValueError(f"{data_yaml}: no {key!r} split")
    dirs = [Path(v) if Path(v).is_absolute() else (root / v).resolve()
            for v in (value if isinstance(value, list) else [value])]
    images = [p for d in dirs for p in sorted(d.rglob("*")) if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not images:
        raise ValueError(f"{data_yaml}: split {key!r} has no images ({dirs})")
    return root, images


def source_fingerprint(images: list[Path], root: Path) -> str:
    """SHA-256 over (relative path, image SHA-256, label SHA-256) of a split."""
    h = hashlib.sha256()
    for img in images:
        lab = label_path_for(img)
        h.update(str(img.relative_to(root)).encode())
        h.update(sha256_file(img).encode())
        h.update((sha256_file(lab) if lab.exists() else "no-label").encode())
    return h.hexdigest()


def build_kfold_splits(ids: list[str], k: int, seed: int) -> list[tuple[list[str], list[str]]]:
    """K deterministic folds over ``ids`` — YOLO26's algorithm (identical to the U-Net's)."""
    if k < 2 or len(ids) < k:
        raise ValueError(f"invalid K-Fold: k={k}, n={len(ids)}")
    rng = np.random.RandomState(seed)
    indices = np.arange(len(ids))
    rng.shuffle(indices)
    sizes = np.full(k, len(ids) // k, dtype=int)
    sizes[: len(ids) % k] += 1
    splits, start = [], 0
    for size in sizes:
        stop = start + size
        val_idx, train_idx = indices[start:stop], np.concatenate([indices[:start], indices[stop:]])
        splits.append(([ids[i] for i in train_idx], [ids[i] for i in val_idx]))
        start = stop
    return splits


# ----------------------------------------------------------------------------
# Annotations
# ----------------------------------------------------------------------------
def instance_masks(label_path: Path, height: int, width: int) -> list[np.ndarray]:
    """One binary mask per YOLO label line, rasterised exactly like
    :func:`segmentation_metrics.rasterize_yolo_label` (whose result is their union)."""
    masks: list[np.ndarray] = []
    if not label_path.exists():
        return masks
    scale = np.array([width, height], dtype=np.float64)
    for line in label_path.read_text().splitlines():
        vals = line.split()
        if len(vals) < 5:
            continue
        coords = np.array(vals[1:], dtype=np.float64)
        m = np.zeros((height, width), dtype=np.uint8)
        if len(coords) == 4:  # bounding box
            xc, yc, bw, bh = coords * np.array([width, height, width, height])
            x0, y0 = int(round(xc - bw / 2)), int(round(yc - bh / 2))
            x1, y1 = int(round(xc + bw / 2)), int(round(yc + bh / 2))
            cv2.rectangle(m, (x0, y0), (x1, y1), 1, thickness=-1)
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            pts = np.round(coords.reshape(-1, 2) * scale).astype(np.int32)
            cv2.fillPoly(m, [pts], 1)
        else:
            continue
        masks.append(m)
    return masks


def encode(mask: np.ndarray) -> tuple[dict[str, Any], list[float], int]:
    """Compressed RLE (str counts, as the Roboflow export), COCO bbox [x, y, w, h], area."""
    rle = mask_utils.encode(np.asfortranarray(mask))
    bbox = [float(v) for v in mask_utils.toBbox(rle)]
    area = int(mask_utils.area(rle))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle, bbox, area


def describe_image(img: Path, root: Path) -> dict[str, Any]:
    """Size, ISIC ID and per-instance COCO annotations of one image (image IDs assigned later)."""
    bgr = cv2.imread(str(img), cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"cannot read {img}")
    h, w = bgr.shape[:2]
    anns = []
    for m in instance_masks(label_path_for(img), h, w):
        if not m.any():
            raise RuntimeError(f"{img}: degenerate (empty) instance polygon")
        rle, bbox, area = encode(m)
        anns.append({"segmentation": rle, "bbox": bbox, "area": area})
    return {"id": isic_id(img), "file_name": img.name, "width": w, "height": h,
            "image": str(img.relative_to(root)), "label": str(label_path_for(img).relative_to(root)),
            "anns": anns}


def coco_json(records: list[dict[str, Any]], description: str) -> dict[str, Any]:
    """COCO file for ``records`` (sequential image / annotation IDs, Roboflow-style structure)."""
    images, annotations = [], []
    for img_id, r in enumerate(records):
        images.append({"id": img_id, "width": r["width"], "height": r["height"], "file_name": r["file_name"],
                       "isic_id": r["id"], "license": 1, "date_captured": ""})
        for a in r["anns"]:
            annotations.append({"id": len(annotations), "image_id": img_id, "category_id": CATEGORY["id"],
                                "bbox": a["bbox"], "area": a["area"], "segmentation": a["segmentation"],
                                "iscrowd": 0})
    return {"info": {"description": description, "version": str(PREP_VERSION), "date_created": utc_now_iso()},
            "licenses": [{"id": 1, "name": "see ISIC 2018 / dataset README", "url": ""}],
            "categories": [CATEGORY], "images": images, "annotations": annotations}


def write_json(path: Path, payload: dict[str, Any]) -> str:
    """Write a compact JSON file; return its SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")))
    return sha256_file(path)


def link_or_copy(src: Path, dst: Path) -> str:
    """Hard-link ``src`` to ``dst`` (identical bytes, no extra space); copy if linking is impossible."""
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Phase 0 — SAM 3 COCO dataset from the YOLO26 dataset.")
    p.add_argument("--yolo-data", default=DEFAULT_YOLO_DATA_YAML, help="YOLO data.yaml (train/val/test).")
    p.add_argument("--out", default=DEFAULT_DATA_DIR, help=f"Output directory (default: {DEFAULT_DATA_DIR}).")
    p.add_argument("--seed", type=int, default=SEED, help="K-Fold seed (default: 0, as YOLO26).")
    p.add_argument("--force", action="store_true", help="Rebuild even if up to date.")
    return p.parse_args()


def main() -> int:
    """Build (or validate) the SAM 3 dataset.

    Returns:
        ``0`` on success / up to date, ``2`` on invalid input.
    """
    args = parse_args()
    data_yaml = Path(args.yolo_data).resolve()
    out = Path(args.out)
    try:
        resolved = {name: resolve_split(data_yaml, key) for key, name in SPLITS.items()}
    except (OSError, ValueError) as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2
    owner: dict[str, str] = {}
    for name, (_, images) in resolved.items():
        for img in images:
            if isic_id(img) in owner:
                print(f"[error] {isic_id(img)} appears in both {owner[isic_id(img)]!r} and {name!r}", file=sys.stderr)
                return 2
            owner[isic_id(img)] = name

    print(f"Phase 0 — SAM 3 COCO dataset from {data_yaml}")
    for name, (_, images) in resolved.items():
        print(f"  {name:<5}: {len(images)} images")
    params = {"prep_version": PREP_VERSION, "category": CATEGORY, "k_folds": K_FOLDS, "seed": args.seed,
              "mask_encoding": "compressed RLE (pycocotools)", "rasterisation": "segmentation_metrics convention"}
    print("  fingerprinting source files ...")
    sources = {name: source_fingerprint(images, root) for name, (root, images) in resolved.items()}
    meta = read_json(out / "meta.json")
    if meta and meta.get("sources") == sources and meta.get("params") == params and not args.force:
        print(f"  [skip] dataset up to date: {out}")
        return 0

    tmp = out.with_name(f"{out.name}.tmp-{utc_stamp()}")
    (tmp / "images").mkdir(parents=True)
    records: dict[str, list[dict[str, Any]]] = {}
    modes: dict[str, int] = {"link": 0, "copy": 0}
    for name, (root, images) in resolved.items():
        print(f"  {name}: rasterising + encoding ...")
        records[name] = []
        for i, img in enumerate(images):
            records[name].append(describe_image(img, root))
            modes[link_or_copy(img, tmp / "images" / img.name)] += 1
            if (i + 1) % 500 == 0:
                print(f"    {i + 1}/{len(images)}", flush=True)
        with (tmp / f"manifest_{name}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["index", "id", "file_name", "image", "label", "width", "height",
                                              "n_instances"])
            w.writeheader()
            for i, r in enumerate(records[name]):
                w.writerow({"index": i, "id": r["id"], "file_name": r["file_name"], "image": r["image"],
                            "label": r["label"], "width": r["width"], "height": r["height"],
                            "n_instances": len(r["anns"])})

    files: dict[str, str] = {}
    for name, recs in records.items():
        files[f"annotations/{name}.json"] = write_json(tmp / "annotations" / f"{name}.json",
                                                       coco_json(recs, f"ISIC 2018 Task 1 — {name} (from YOLO26)"))
    pool = records["train"] + records["val"]
    by_id = {r["id"]: r for r in pool}
    folds = build_kfold_splits([r["id"] for r in pool], K_FOLDS, args.seed)
    fold_meta = []
    for k, (tr, va) in enumerate(folds):
        for part, ids in (("train", tr), ("val", va)):
            files[f"folds/fold_{k}/{part}.json"] = write_json(
                tmp / "folds" / f"fold_{k}" / f"{part}.json",
                coco_json([by_id[i] for i in ids], f"ISIC 2018 Task 1 — CV fold {k} {part} (YOLO26 folds)"))
        fold_meta.append({"fold": k, "n_train": len(tr), "n_val": len(va),
                          "val_sha256": hashlib.sha256("\n".join(va).encode()).hexdigest()})

    atomic_write_json(tmp / "meta.json", {
        "created_at": utc_now_iso(), "source_data_yaml": str(data_yaml),
        "source_root": str(next(iter(resolved.values()))[0]), "sources": sources, "params": params,
        "splits": {n: {"n_images": len(r), "n_instances": sum(len(x["anns"]) for x in r)} for n, r in records.items()},
        "folds": fold_meta, "files_sha256": files, "image_pool": modes,
    })
    if out.exists():
        backup = out.with_name(f"{out.name}.bak-{utc_stamp()}")
        out.rename(backup)
        print(f"  previous dataset kept as {backup}")
    tmp.rename(out)

    print(f"\nDataset written to {out}  (images: {modes['link']} hard-linked, {modes['copy']} copied)")
    for name, r in records.items():
        print(f"  {name:<5}: {len(r)} images, {sum(len(x['anns']) for x in r)} instances")
    print(f"  folds : {[(m['n_train'], m['n_val']) for m in fold_meta]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
