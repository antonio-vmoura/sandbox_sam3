"""SAM 3 test-set inference and pixel scoring (Phase 5) — the validation path, reused.

Predictions are produced **exactly as during training-time validation** (the
quantity that selected every ``best.pt``), so test and validation scores are
the same function of the model:

1. the run's own ``config.yaml`` is loaded and the model, the validation
   dataset (official ``val_transforms``: resize to 1008 × 1008, pad, normalise),
   the collate function and the official postprocessor
   (``scratch.mask_postprocessor_thresholded``: score = sigmoid(logit) ×
   presence, mask logits bilinearly upsampled to the original image size,
   sigmoid > 0.5) are instantiated from it; the dataset reads the ``test``
   annotation file instead of ``val``;
2. per image (batch = 1) the predictions are converted to COCO records as the
   official ``PredictionDumper`` does (scores rounded to 5 decimals, top
   ``maxdets`` = 100 per image);
3. the predicted binary mask is the **union of the instance masks with score
   >= 0.5** (:data:`common.BASE_SETUP` ``score_threshold``), as in
   :func:`protocol_trainer.pixel_metrics_from_dump`, and is scored against the
   ground truth with :func:`segmentation_metrics.pixel_scores` — the code shared
   with YOLO26 and the U-Net. The ground truth is the union of the image's
   COCO RLE masks, rasterised in Phase 0 from the **same YOLO polygons** with
   the same convention, at the original dataset resolution.

FP16 is SAM 3's official mixed-precision path (``torch.autocast`` float16,
FP32 weights) — the model is not converted with ``.half()``.

Output rows use the YOLO26 / U-Net per-image CSV columns: ``image``,
``height``, ``width``, ``n_pred`` (instances with score >= threshold),
``max_conf`` (maximum instance score), ``infer_ms`` and every key of
:func:`pixel_scores`.
"""

from __future__ import annotations

import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

from common import BASE_SETUP
from protocol_trainer import union_masks
from segmentation_metrics import pixel_scores

#: Version of the evaluation method (part of the result cache keys).
EVAL_VERSION: int = 2   # 2: + boundary metrics (BIoU, NSD)

#: Instance score threshold of the merged binary mask (as in validation).
SCORE_THRESHOLD: float = float(BASE_SETUP["score_threshold"])

_RESOLVERS_REGISTERED = False


def load_run_config(run_dir: Path):
    """OmegaConf config of a training run (``<run_dir>/config.yaml``)."""
    global _RESOLVERS_REGISTERED
    from omegaconf import OmegaConf

    from sam3.train.utils.train_utils import register_omegaconf_resolvers

    if not _RESOLVERS_REGISTERED:
        register_omegaconf_resolvers()
        _RESOLVERS_REGISTERED = True
    return OmegaConf.load(Path(run_dir) / "config.yaml")


def build_model(cfg, weights: Path, device: torch.device) -> torch.nn.Module:
    """The run's SAM 3 architecture in eval mode with the ``best.pt`` weights."""
    from hydra.utils import instantiate

    model = instantiate(cfg.trainer.model, eval_mode=True, load_from_HF=False, device=device.type)
    state = torch.load(weights, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval()


def autocast(precision: str):
    """Context of the requested precision (FP16 = official autocast path)."""
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


class Predictor:
    """The validation pipeline of a run applied to any COCO annotation file."""

    def __init__(self, run_dir: Path, weights: Path, device: torch.device) -> None:
        from hydra.utils import instantiate

        from sam3.eval.coco_writer import PredictionDumper

        self.cfg = load_run_config(run_dir)
        self.device = device
        self.model = build_model(self.cfg, weights, device)
        self.postprocessor = instantiate(self.cfg.scratch.mask_postprocessor_thresholded)
        self.maxdets = int(self.cfg.trainer.meters.val.isic_val.detection.maxdets)
        self._dumper = PredictionDumper.__new__(PredictionDumper)   # only its COCO conversion is used

    def loader(self, image_dir: Path, ann_file: Path, workers: int = 4):
        """Batch-1 loader of ``ann_file`` with the run's validation transforms."""
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        vcfg = OmegaConf.to_container(self.cfg.trainer.data.val, resolve=True)
        vcfg["dataset"]["img_folder"] = str(image_dir) + "/"
        vcfg["dataset"]["ann_file"] = str(ann_file)
        vcfg.update(batch_size=1, num_workers=workers, enable_distributed_sampler=False)
        return instantiate(vcfg).get_loader(epoch=0)

    @torch.inference_mode()
    def predict_batch(self, batch, precision: str) -> tuple[list[int], dict[int, list[dict[str, Any]]]]:
        """Image ids of the batch and their COCO segmentation records (as dumped during validation)."""
        from sam3.model.utils.misc import copy_data_to_device

        _, b = batch.popitem()
        img_ids = sorted({int(i) for m in b.find_metadatas for i in m.original_image_id.reshape(-1).tolist()})
        b = copy_data_to_device(b, self.device, non_blocking=True)
        with autocast(precision):
            find_stages = self.model(b)
        preds = self.postprocessor.process_results(find_stages=find_stages, find_metadatas=b.find_metadatas)
        records = self._dumper.prepare_for_coco_segmentation(preds)
        by_image: dict[int, list[dict[str, Any]]] = {}
        for r in records:
            r["score"] = round(r["score"], 5)
            by_image.setdefault(int(r["image_id"]), []).append(r)
        for img_id, recs in by_image.items():
            recs.sort(key=lambda r: -r["score"])
            del recs[self.maxdets:]
        return img_ids, by_image


def evaluate_annotations(
    predictor: Predictor,
    image_dir: Path,
    ann_file: Path,
    *,
    precision: str = "fp32",
    mask_dir: Path | None = None,
    workers: int = 4,
) -> list[dict[str, Any]]:
    """Predict every image of ``ann_file`` (batch = 1) and score it against its ground truth.

    Args:
        predictor: Model + validation pipeline of a run.
        image_dir: Image pool of the Phase 0 dataset.
        ann_file: COCO annotation file (e.g. ``annotations/test.json``).
        precision: ``fp32`` or ``fp16`` (autocast).
        mask_dir: If given, each predicted mask is saved as ``<image stem>.png`` (0/255).
        workers: DataLoader workers.

    Returns:
        One row per image in the YOLO26 per-image schema.
    """
    import cv2

    gt = json.loads(Path(ann_file).read_text())
    images = {im["id"]: im for im in gt["images"]}
    gt_by: dict[int, list] = {}
    for a in gt["annotations"]:
        gt_by.setdefault(a["image_id"], []).append(a["segmentation"])
    if mask_dir is not None:
        Path(mask_dir).mkdir(parents=True, exist_ok=True)
    cuda = predictor.device.type == "cuda"

    rows, seen = [], set()
    for i, batch in enumerate(predictor.loader(image_dir, ann_file, workers)):
        if cuda:
            torch.cuda.synchronize(predictor.device)
        t0 = time.perf_counter()
        img_ids, by_image = predictor.predict_batch(batch, precision)
        if cuda:
            torch.cuda.synchronize(predictor.device)
        infer_ms = (time.perf_counter() - t0) * 1000
        for img_id in img_ids:
            if img_id in seen:
                raise RuntimeError(f"image {img_id} predicted twice")
            seen.add(img_id)
            im = images[img_id]
            h, w = im["height"], im["width"]
            recs = by_image.get(img_id, [])
            kept = [r["segmentation"] for r in recs if r["score"] >= SCORE_THRESHOLD]
            pred = union_masks(kept, h, w)
            if mask_dir is not None:
                cv2.imwrite(str(Path(mask_dir) / f"{Path(im['file_name']).stem}.png"), pred.astype(np.uint8) * 255)
            rows.append({"image": str(Path(image_dir) / im["file_name"]), "height": h, "width": w,
                         "n_pred": len(kept), "max_conf": max((r["score"] for r in recs), default=0.0),
                         "infer_ms": infer_ms, **pixel_scores(union_masks(gt_by.get(img_id, []), h, w), pred)})
        if (i + 1) % 100 == 0 or i + 1 == len(images):
            print(f"    {i + 1}/{len(images)} images", flush=True)
    if len(rows) != len(images):
        raise RuntimeError(f"scored {len(rows)} of {len(images)} images of {ann_file}")
    return rows
