"""Memory / throughput probe of SAM 3 full fine-tuning (official trainer and recipe).

Runs a short training epoch with the official ``sam3.train.trainer.Trainer`` and
the official Phase 1 recipe, in FP32, on the Phase 0 data, for several
configurations — each in a fresh subprocess so peak-memory counters and the
CUDA context never leak between them:

* ``P0`` — batch 2, activation checkpointing **off** (truly native);
* ``P1`` — batch 2, the official setting (activation checkpointing on);
* ``P1D`` — P1 in strict deterministic mode + deterministic ``grid_sample``
  (bit-exact; measured for the record, the study trains with P1 — see
  ``run_training.py``);
* ``P2`` — batch 2 as 2 chunks of 1 (gradient accumulation 2), checkpointing on.

For each configuration it reports whether the step fits in memory, the peak
VRAM allocated / reserved by PyTorch, the device-level total (incl. the CUDA
context, via NVML), the median / P95 seconds per optimiser step after warm-up,
and the projected time per training epoch (2,547 images).

Usage (inside the ``sam3_ft`` container, one GPU visible):
    python sam3_seg/probe_memory.py --data /workspace/datasets/isic_2018_task1_sam3
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

CONFIGS: dict[str, dict] = {
    "P0": {"batch": 2, "chunks": 1, "act_ckpt": False, "desc": "FP32, batch 2, act. ckpt OFF in ViT + text encoder (detector keeps it, required)"},
    "P1": {"batch": 2, "chunks": 1, "act_ckpt": True, "desc": "FP32, batch 2, official act. checkpointing ON"},
    "P1D": {"batch": 2, "chunks": 1, "act_ckpt": True, "deterministic": True, "desc": "P1 + strict deterministic algorithms (bit-exact; not used by the study: +33 % time)"},
    "P2": {"batch": 2, "chunks": 2, "act_ckpt": True, "desc": "FP32, 2 chunks of 1 (grad. accumulation 2), act. ckpt ON"},
}
BASE_YAML = "sam3/train/configs/custom/sam3_phase1_baseline.yaml"
N_TRAIN_FULL = 2547


def disable_act_ckpt(model) -> int:
    """Turn off activation checkpointing in the two large backbones; return how many switches.

    SAM 3's detector encoder and decoder *assert* activation checkpointing in
    training mode (``encoder.py`` / ``decoder.py``), so the small detector
    modules keep the official setting; the vision trunk (ViT, 454 M params) and
    the text encoder (354 M) — where nearly all the recompute cost is — run
    without it. Numerically identical to the official setting.
    """
    n = 0
    for root in (model.backbone.vision_backbone, model.backbone.language_backbone):
        for mod in root.modules():
            for attr in ("use_act_checkpoint", "grad_checkpointing"):
                if isinstance(getattr(mod, attr, None), bool) and getattr(mod, attr):
                    setattr(mod, attr, False)
                    n += 1
    return n


def worker(args) -> None:
    """Run one configuration and write a JSON result."""
    os.environ.update(MASTER_ADDR="localhost", MASTER_PORT=str(29500 + os.getpid() % 1000),
                      RANK="0", LOCAL_RANK="0", WORLD_SIZE="1")
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from sam3.train.trainer import Trainer
    from sam3.train.utils.train_utils import register_omegaconf_resolvers

    spec = CONFIGS[args.config]
    if spec.get("deterministic"):
        import determinism
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        determinism.install()
    register_omegaconf_resolvers()
    cfg = OmegaConf.load(BASE_YAML)
    data = Path(args.data)
    cfg.paths.experiment_log_dir = args.logdir
    cfg.roboflow_train.num_images = args.images
    cfg.scratch.train_batch_size = spec["batch"]
    cfg.scratch.gradient_accumulation_steps = spec["chunks"]
    if spec["chunks"] > 1:
        cfg.scratch.collate_fn = OmegaConf.create({
            "_target_": "sam3.train.data.collator.collate_fn_api_with_chunking", "_partial_": True,
            "num_chunks": spec["chunks"], "repeats": "${scratch.hybrid_repeats}", "dict_key": "all",
            "with_seg_masks": "${scratch.enable_segmentation}"})
    t = cfg.trainer
    t.mode, t.max_epochs, t.seed_value, t.skip_saving_ckpts = "train_only", 1, 0, True
    t.optim.amp.enabled = False
    t.data.train.dataset.img_folder = str(data / "images") + "/"
    t.data.train.dataset.ann_file = str(data / "annotations" / "train.json")
    t.data.train.num_workers = 4

    timings: list[float] = []
    state: dict = {"config": args.config, **spec, "images": args.images}

    class ProbeTrainer(Trainer):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            if not spec["act_ckpt"]:
                state["act_ckpt_switches_disabled"] = disable_act_ckpt(self.model)

        def _run_step(self, *a, **k):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = super()._run_step(*a, **k)
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - t0)
            return out

    # Build through Hydra with our subclass as target (resolved as __main__.ProbeTrainer).
    globals()["ProbeTrainer"] = ProbeTrainer
    cfg.trainer._target_ = "__main__.ProbeTrainer"
    torch.cuda.reset_peak_memory_stats()
    try:
        trainer = instantiate(cfg.trainer, _recursive_=False)
        state["params"] = sum(p.numel() for p in trainer.model.parameters())
        state["trainable"] = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        trainer.run()
        state["status"] = "ok"
    except torch.OutOfMemoryError as e:
        state["status"] = "OOM"
        state["error"] = str(e).splitlines()[0][:300]
    state["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    state["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
    state["step_s"] = timings
    Path(args.out).write_text(json.dumps(state))


def gpu_used_mib() -> float:
    """Device-level memory used (incl. CUDA contexts), via nvidia-smi."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.split(",")
    return float(out[0])


def main() -> int:
    p = argparse.ArgumentParser(description="SAM 3 FP32 fine-tuning memory / throughput probe.")
    p.add_argument("--data", default="/workspace/datasets/isic_2018_task1_sam3")
    p.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    p.add_argument("--images", type=int, default=40, help="Training images in the probe epoch (default: 40).")
    p.add_argument("--warmup", type=int, default=3, help="Optimiser steps excluded from timing.")
    p.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    p.add_argument("--config", default=None, help=argparse.SUPPRESS)
    p.add_argument("--logdir", default=None, help=argparse.SUPPRESS)
    p.add_argument("--out", default=None, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker:
        worker(args)
        return 0

    import numpy as np
    results = []
    for name in args.configs:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "r.json"
            print(f"\n=== {name}: {CONFIGS[name]['desc']}", flush=True)
            peak_dev = [0.0]
            # Worker output goes to a file: a pipe that is only read at the end fills up
            # (~64 KB) and blocks the worker's logging — a deadlock.
            log_path = Path(tmp) / "worker.log"
            with log_path.open("w") as log_f:
                proc = subprocess.Popen([sys.executable, __file__, "--worker", "1", "--config", name, "--data", args.data,
                                         "--images", str(args.images), "--logdir", str(Path(tmp) / "log"),
                                         "--out", str(out)], stdout=log_f, stderr=subprocess.STDOUT, text=True)
                while proc.poll() is None:
                    peak_dev[0] = max(peak_dev[0], gpu_used_mib())
                    time.sleep(0.5)
            log = log_path.read_text().splitlines()
            if not out.exists():
                print("\n".join(log[-25:]))
                results.append({"config": name, "status": f"crashed (exit {proc.returncode})"})
                continue
            r = json.loads(out.read_text())
            r["device_peak_used_gib"] = peak_dev[0] / 1024
            steps = r.pop("step_s")[args.warmup:]
            if steps:
                r["step_median_s"], r["step_p95_s"] = float(np.median(steps)), float(np.percentile(steps, 95))
                steps_per_epoch = N_TRAIN_FULL // CONFIGS[name]["batch"]
                r["epoch_min_projected"] = r["step_median_s"] * steps_per_epoch / 60
            results.append(r)
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
