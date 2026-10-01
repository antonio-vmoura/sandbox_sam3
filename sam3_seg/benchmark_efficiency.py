"""Phase 5b — Hardware-efficiency benchmark (batch=1) of the Baseline and Optimised SAM 3.

Same measurement method and output schema as YOLO26's
``yolo26_seg/benchmark_efficiency.py`` and the U-Net's (this file is derived
from them; only the model-specific parts differ). For every ``variant`` ×
``model`` × ``precision ∈ {fp32, fp16}`` it measures, on a **single GPU** with
**batch = 1**:

* **Latency** (ms) — mean, std, median, P90, P95, P99, min, max — of three
  scopes:

  - ``forward``: the whole network on a fixed, pre-processed real test image
    (1×3×1008×1008, SAM 3's native input) and the prompt ``"skin lesion"``:
    image encoder + text encoder + detector (``forward_grounding``), timed with
    ``torch.cuda.Event`` pairs and a ``synchronize`` after every iteration.
    Primary "model latency" figure (comparable to YOLO26 / U-Net ``forward``).
  - ``forward_cached_text``: the same without the text encoder — the text
    features of the fixed prompt are computed once (the deployment form when
    the prompt never changes).
  - ``end_to_end``: Meta's deployment API (``Sam3Processor``) on the decoded
    test image (640×640): host→device copy, resize to 1008, normalisation,
    image + text encoding, detection, score = sigmoid × presence, bilinear
    mask upsampling to 640×640, sigmoid > 0.5, union of the instances with
    score > 0.5 and device→host copy of the binary mask. Timed with
    ``time.perf_counter`` around a synchronised call (it includes CPU work).

* **FPS** = 1000 / mean latency (``fps``) and 1000 / median (``fps_median``).
* **Memory** — steady-state peak VRAM allocated / reserved by PyTorch during
  the timed iterations (peak counters reset **after** warm-up; the warm-up
  peak is kept as ``vram_peak_warmup_mb``), the VRAM of the weights alone, and
  host RAM (RSS after loading / benchmarking and peak RSS). The CUDA context is
  excluded.
* **Model size** — size of ``best.pt`` on disk, theoretical FP32/FP16 weight
  sizes, parameter count in total, per component and without the text
  encoder, and **GFLOPs measured natively with**
  ``torch.utils.flop_counter.FlopCounterMode`` (FLOPs = 2 × MACs of matrix
  products and convolutions, the YOLO26 / thop convention; element-wise ops,
  normalisation and interpolation are not counted by either tool). Unlike
  ``thop`` (module hooks, blind to ``F.scaled_dot_product_attention`` and
  functional matmuls), ``FlopCounterMode`` counts every ATen matrix product,
  including the attention products. Counting runs under ``torch.no_grad``
  (``inference_mode`` hides every op from ``FlopCounterMode``) and
  ``nn.MultiheadAttention``'s fused
  inference fast path is disabled during counting (its fused kernel has no
  FLOP formula) — the counted graph is otherwise the benchmarked one.
  Reported: ``gflops`` (whole network), ``gflops_by_component`` (image
  encoder / text encoder / detector), ``gflops_without_text`` and
  ``gflops_by_op`` (attention products QKᵀ and AV / linear layers / convolutions).
  SAM 3 resizes every input to 1008 × 1008, so ``gflops_640`` (YOLO26 /
  U-Net schema key) equals ``gflops``.

FP16 = SAM 3's official mixed-precision path (``torch.autocast`` float16, FP32
weights; ``vram_weights_mb`` is therefore the FP32 footprint in both rows).

Isolation & fairness (as YOLO26): every configuration runs in a **fresh
subprocess**; ``cudnn.benchmark=True`` (recorded); warm-up iterations are
discarded; GPU utilisation of other processes is recorded before/after each
run and a busy GPU is flagged ``contended``.

Outputs::

    <project>/phase5_test/efficiency/<variant>_<model>_<precision>.json

(including the raw per-iteration latencies). Results whose weights and
settings are unchanged are skipped.

Usage:
    python sam3_seg/benchmark_efficiency.py --project /workspace/logs/pipeline_final_v1 --device 0
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    DEFAULT_DATA_DIR,
    DEFAULT_ORDER,
    DEFAULT_PIPELINE_ROOT,
    PROMPT,
    RESOLUTION,
    SEED,
    PipelinePaths,
    atomic_write_json,
    config_hash,
    read_json,
    sha256_file,
    utc_now_iso,
)

VARIANTS: tuple[str, ...] = ("baseline", "optimized")
PRECISIONS: tuple[str, ...] = ("fp32", "fp16")

#: Version of the measurement method. Part of the cache key: bump it whenever
#: what or how this script measures changes, so stale results are recomputed.
BENCHMARK_VERSION: int = 1

#: GPU utilisation (%) above which a run is flagged as contended.
CONTENTION_UTIL_PCT: int = 5

MB: float = 1024 ** 2

#: Top-level SAM 3 components for the parameter split.
COMPONENTS: dict[str, str] = {
    "vision_backbone": "backbone.vision_backbone",
    "text_encoder": "backbone.language_backbone",
    "transformer": "transformer",
    "geometry_encoder": "geometry_encoder",
    "segmentation_head": "segmentation_head",
    "dot_prod_scoring": "dot_prod_scoring",
}


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def latency_stats(ms: list[float]) -> dict[str, float]:
    """Summarise per-iteration latencies (ms) and derive FPS."""
    a = np.asarray(ms, dtype=float)
    mean, median = float(a.mean()), float(np.median(a))
    return {
        "n": int(a.size),
        "mean_ms": mean,
        "std_ms": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "median_ms": median,
        "p90_ms": float(np.percentile(a, 90)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
        "fps": 1000.0 / mean,
        "fps_median": 1000.0 / median,
    }


def op_class(op_name: str) -> str:
    """FLOP class of an ATen op: attention products, linear (matmul) or convolution."""
    name = op_name.lower()
    if "attention" in name or "scaled_dot_product" in name:
        return "attention"
    if "conv" in name:
        return "convolution"
    if any(k in name for k in ("mm", "matmul", "linear", "einsum")):
        return "linear"
    return "other"


# ----------------------------------------------------------------------------
# Worker (runs in its own process)
# ----------------------------------------------------------------------------
def _rss_mb() -> float:
    import psutil

    return psutil.Process().memory_info().rss / MB


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KiB on Linux


def count_flops(fn) -> tuple[int, dict[str, int]]:
    """Total FLOPs of ``fn()`` and FLOPs per ATen op (``FlopCounterMode``, MHA fast path off)."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    fast = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with FlopCounterMode(display=False) as fc:
            fn()
    finally:
        torch.backends.mha.set_fastpath_enabled(fast)
    per_op = {str(op): int(n) for op, n in fc.get_flop_counts().get("Global", {}).items()}
    return int(fc.get_total_flops()), per_op


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    """Benchmark one weights file at one precision; return the measurement dict."""
    import PIL.Image
    import torch
    from torchvision.transforms import v2

    from inference import SCORE_THRESHOLD, autocast, build_model, load_run_config
    from sam3.model.sam3_image_processor import Sam3Processor

    torch.manual_seed(SEED)
    dev = torch.device(f"cuda:{args.device}")
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    rss_start = _rss_mb()
    torch.cuda.set_device(dev)
    torch.zeros(1, device=dev)  # create the CUDA context before measuring
    torch.cuda.synchronize(dev)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)
    rss_ctx = _rss_mb()
    alloc0 = torch.cuda.memory_allocated(dev)

    # ---- Model ----------------------------------------------------------------
    weights = Path(args.weights)
    model = build_model(load_run_config(weights.parent.parent), weights, dev)
    for p in model.parameters():
        p.requires_grad_(False)
    weights_vram = (torch.cuda.memory_allocated(dev) - alloc0) / MB
    rss_model = _rss_mb()
    params = sum(p.numel() for p in model.parameters())
    params_by_component = {k: sum(p.numel() for p in model.get_submodule(path).parameters())
                           for k, path in COMPONENTS.items()}
    params_by_component["other"] = params - sum(params_by_component.values())

    image = PIL.Image.open(args.sample_image).convert("RGB")
    processor = Sam3Processor(model, resolution=RESOLUTION, device="cuda", confidence_threshold=SCORE_THRESHOLD)
    x = processor.transform(v2.functional.to_image(image).to(dev)).unsqueeze(0)   # 1×3×1008×1008
    find_stage = processor.find_stage

    def encode_image():
        return model.backbone.forward_image(x)

    def encode_text():
        return model.backbone.forward_text([PROMPT], device=dev)

    def detect(backbone_out):
        return model.forward_grounding(backbone_out=backbone_out, find_input=find_stage,
                                       geometric_prompt=model._get_dummy_prompt(), find_target=None)

    def forward(text_out=None):
        bo = encode_image()
        bo.update(text_out if text_out is not None else encode_text())
        return detect(bo)

    # ---- GFLOPs (FP32, eager, FlopCounterMode) ------------------------------
    # no_grad, NOT inference_mode: under inference_mode the dispatcher bypasses
    # Python dispatch modes and FlopCounterMode counts nothing.
    with torch.no_grad():
        f_total, per_op = count_flops(forward)
        f_image, _ = count_flops(encode_image)
        f_text, _ = count_flops(encode_text)
        text_cached = encode_text()
        bo_cached = encode_image()
        bo_cached.update(text_cached)
        f_detect, _ = count_flops(lambda: detect(dict(bo_cached)))
    by_op: dict[str, float] = {}
    for op, n in per_op.items():
        by_op[op_class(op)] = by_op.get(op_class(op), 0.0) + n / 1e9
    del bo_cached
    torch.cuda.empty_cache()

    # ---- Forward-pass benchmarks (with / without the text encoder) ----------
    def time_forward(fn) -> tuple[list[float], float, float, float]:
        ms: list[float] = []
        with torch.inference_mode(), autocast(args.precision):
            for _ in range(args.warmup):
                fn()
            torch.cuda.synchronize(dev)
            warm_peak = (torch.cuda.max_memory_allocated(dev) - alloc0) / MB
            torch.cuda.reset_peak_memory_stats(dev)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            for _ in range(args.iters):
                start.record()
                fn()
                end.record()
                torch.cuda.synchronize(dev)
                ms.append(start.elapsed_time(end))
        return (ms, warm_peak, (torch.cuda.max_memory_allocated(dev) - alloc0) / MB,
                torch.cuda.max_memory_reserved(dev) / MB)

    fwd_ms, warmup_peak, fwd_peak_alloc, fwd_peak_reserved = time_forward(forward)
    with torch.inference_mode(), autocast(args.precision):
        text_cached = encode_text()
    cached_ms, _, cached_peak_alloc, _ = time_forward(lambda: forward(text_cached))
    del text_cached

    # ---- End-to-end prediction pipeline (Sam3Processor) ---------------------
    def predict(img):
        with torch.inference_mode(), autocast(args.precision):
            state = processor.set_image(img)
            state = processor.set_text_prompt(PROMPT, state)
            mask = state["masks"].any(dim=0)[0] if len(state["masks"]) else torch.zeros(
                img.size[1], img.size[0], dtype=torch.bool, device=dev)
            return mask.cpu().numpy()

    torch.cuda.empty_cache()
    for _ in range(args.e2e_warmup):
        predict(image)
    torch.cuda.synchronize(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    e2e_ms: list[float] = []
    for _ in range(args.e2e_iters):
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        predict(image)
        torch.cuda.synchronize(dev)
        e2e_ms.append((time.perf_counter() - t0) * 1000)
    e2e_peak_alloc = (torch.cuda.max_memory_allocated(dev) - alloc0) / MB
    e2e_peak_reserved = torch.cuda.max_memory_reserved(dev) / MB

    size_disk = os.path.getsize(args.weights) / MB
    gflops = f_total / 1e9
    return {
        "forward": {**latency_stats(fwd_ms), "timer": "torch.cuda.Event",
                    "input_shape": [1, 3, RESOLUTION, RESOLUTION], "prompt": PROMPT,
                    "scope": "image encoder + text encoder + detector", "raw_ms": fwd_ms},
        "forward_cached_text": {**latency_stats(cached_ms), "timer": "torch.cuda.Event",
                                "input_shape": [1, 3, RESOLUTION, RESOLUTION],
                                "scope": "image encoder + detector (text features precomputed)",
                                "raw_ms": cached_ms},
        "end_to_end": {**latency_stats(e2e_ms), "timer": "perf_counter+synchronize",
                       "pipeline": "sam3.model.sam3_image_processor.Sam3Processor",
                       "sample_image": args.sample_image, "raw_ms": e2e_ms},
        "memory": {
            "vram_weights_mb": weights_vram,
            "vram_peak_allocated_mb": fwd_peak_alloc,
            "vram_peak_reserved_mb": fwd_peak_reserved,
            "vram_peak_allocated_cached_text_mb": cached_peak_alloc,
            "vram_peak_allocated_e2e_mb": e2e_peak_alloc,
            "vram_peak_reserved_e2e_mb": e2e_peak_reserved,
            "vram_peak_warmup_mb": warmup_peak,
            "ram_rss_start_mb": rss_start,
            "ram_rss_after_cuda_init_mb": rss_ctx,
            "ram_rss_model_loaded_mb": rss_model,
            "ram_rss_end_mb": _rss_mb(),
            "ram_peak_rss_mb": _peak_rss_mb(),
            "ram_model_delta_mb": rss_model - rss_ctx,
            "note": "VRAM from the PyTorch allocator (CUDA context excluded), steady state "
                    "after warm-up; vram_peak_warmup_mb includes cuDNN autotune workspaces; "
                    "*_e2e_* covers the deployed pipeline (Sam3Processor). FP16 = autocast, "
                    "FP32 weights.",
        },
        "model": {
            "params": int(params),
            "params_fused": int(params),
            "params_by_component": params_by_component,
            "params_without_text": int(params - params_by_component["text_encoder"]),
            "gflops": gflops,
            "gflops_640": gflops,
            "gflops_input": f"{RESOLUTION}x{RESOLUTION} (every input is resized to it) + prompt '{PROMPT}'",
            "gflops_by_component": {"image_encoder": f_image / 1e9, "text_encoder": f_text / 1e9,
                                    "detector": f_detect / 1e9},
            "gflops_without_text": (f_total - f_text) / 1e9,
            "gflops_by_op": by_op,
            "attention_flop_share": by_op.get("attention", 0.0) / gflops if gflops else None,
            "flops_per_op": per_op,
            "flop_counter": "torch.utils.flop_counter.FlopCounterMode (2 x MACs; MHA fast path off)",
            "size_mb_disk": size_disk,
            "size_disk_dtype": "fp32 (model state_dict + metadata)",
            "size_mb_fp32_theoretical": params * 4 / MB,
            "size_mb_fp16_theoretical": params * 2 / MB,
        },
        "env": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cudnn_benchmark": args.cudnn_benchmark,
            "gpu": torch.cuda.get_device_name(dev),
            "cpu": platform.processor() or platform.machine(),
            "fp16_mode": "torch.autocast(float16), FP32 weights" if args.precision == "fp16" else None,
        },
    }


# ----------------------------------------------------------------------------
# Parent orchestration
# ----------------------------------------------------------------------------
def gpu_snapshot(device: str) -> dict[str, Any]:
    """Utilisation / memory of the benchmark GPU right before a run (via nvidia-smi)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", str(device), "--format=csv,noheader,nounits",
             "--query-gpu=name,driver_version,utilization.gpu,memory.used,memory.total,"
             "clocks.sm,clocks.max.sm,temperature.gpu"],
            check=True, capture_output=True, text=True, timeout=30,
        ).stdout.strip().split(", ")
    except (OSError, subprocess.SubprocessError) as e:
        return {"error": str(e)}
    keys = ["name", "driver", "util_pct", "mem_used_mb", "mem_total_mb", "sm_clock_mhz",
            "sm_clock_max_mhz", "temp_c"]
    snap: dict[str, Any] = dict(zip(keys, out))
    for k in keys[2:]:
        try:
            snap[k] = float(snap[k])
        except (KeyError, ValueError):
            pass
    snap["contended"] = isinstance(snap.get("util_pct"), float) and snap["util_pct"] > CONTENTION_UTIL_PCT
    return snap


def benchmark_one(
    variant: str, model_name: str, precision: str, args: argparse.Namespace,
    paths: PipelinePaths, sample_image: str,
) -> dict[str, Any]:
    """Run (or skip) one configuration in a fresh worker process and save its JSON."""
    weights = paths.best_pt(variant, model_name)
    if not weights.exists():
        raise RuntimeError(f"weights not found: {weights} (run Phase 1/4 first)")
    out_json = paths.phase5_efficiency_json(variant, model_name, precision)
    settings = {
        "weights_sha256": sha256_file(weights), "precision": precision, "imgsz": RESOLUTION,
        "prompt": PROMPT, "warmup": args.warmup, "iters": args.iters, "e2e_warmup": args.e2e_warmup,
        "e2e_iters": args.e2e_iters, "cudnn_benchmark": args.cudnn_benchmark,
        "device": args.device, "sample_image": sample_image,
        "run_config_sha256": sha256_file(weights.parent.parent / "config.yaml"),
        "benchmark_version": BENCHMARK_VERSION,
    }
    previous = read_json(out_json)
    if previous and previous.get("settings_hash") == config_hash(settings) and not args.force:
        return {"tag": out_json.stem, "skipped": True, "payload": previous}

    before = gpu_snapshot(args.device)
    if before.get("contended"):
        print(f"  [warn] GPU {args.device} is busy ({before['util_pct']:.0f}% util, "
              f"{before['mem_used_mb']:.0f} MB used) — latencies will be flagged as contended")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_json = Path(tmp) / "result.json"
        cmd = [
            sys.executable, str(Path(__file__).resolve()), "--worker",
            "--weights", str(weights), "--precision", precision, "--device", args.device,
            "--sample-image", sample_image, "--out", str(tmp_json),
            "--warmup", str(args.warmup), "--iters", str(args.iters),
            "--e2e-warmup", str(args.e2e_warmup), "--e2e-iters", str(args.e2e_iters),
        ]
        if not args.cudnn_benchmark:
            cmd.append("--no-cudnn-benchmark")
        subprocess.run(cmd, check=True)
        result = json.loads(tmp_json.read_text())
    after = gpu_snapshot(args.device)

    payload = {
        "variant": variant, "model": model_name, "precision": precision, "batch": 1,
        "weights": str(weights), **result,
        "gpu_before": before, "gpu_after": after,
        "contended": bool(before.get("contended") or after.get("contended")),
        "settings": settings, "settings_hash": config_hash(settings), "created_at": utc_now_iso(),
    }
    atomic_write_json(out_json, payload)
    return {"tag": out_json.stem, "skipped": False, "payload": payload}


def _first_test_image(data_dir: str) -> str:
    """Deterministic sample image for the end-to-end benchmark (first test image, 640×640)."""
    from data import CocoData

    data = CocoData(data_dir)
    first = json.loads(data.split_json("test").read_text())["images"][0]
    return str(data.image_dir / first["file_name"])


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments (parent and ``--worker`` modes)."""
    p = argparse.ArgumentParser(description="Phase 5b — batch=1 efficiency benchmark of SAM 3 (FP32 / FP16).")
    p.add_argument("--models", nargs="+", default=DEFAULT_ORDER, choices=DEFAULT_ORDER)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    p.add_argument("--precisions", nargs="+", default=list(PRECISIONS), choices=PRECISIONS)
    p.add_argument("--data", default=DEFAULT_DATA_DIR, help="Phase 0 dataset (first test image = e2e sample).")
    p.add_argument("--device", default="0", help="Single GPU id (default: 0).")
    p.add_argument("--project", default=DEFAULT_PIPELINE_ROOT, help="Pipeline root.")
    p.add_argument("--warmup", type=int, default=50, help="Discarded forward iterations (default: 50).")
    p.add_argument("--iters", type=int, default=500, help="Timed forward iterations (default: 500).")
    p.add_argument("--e2e-warmup", type=int, default=20, help="Discarded predict() calls (default: 20).")
    p.add_argument("--e2e-iters", type=int, default=200, help="Timed predict() calls (default: 200).")
    p.add_argument("--no-cudnn-benchmark", dest="cudnn_benchmark", action="store_false",
                   help="Disable cuDNN autotuning (default: enabled, as in deployment).")
    p.add_argument("--force", action="store_true", help="Re-run even if results are up to date.")
    # Worker mode (internal).
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--weights", help=argparse.SUPPRESS)
    p.add_argument("--precision", choices=PRECISIONS, help=argparse.SUPPRESS)
    p.add_argument("--sample-image", help=argparse.SUPPRESS)
    p.add_argument("--out", help=argparse.SUPPRESS)
    return p.parse_args()


def main() -> int:
    """Benchmark every requested configuration (or run one worker).

    Returns:
        ``0`` on success, ``1`` if any configuration failed, ``2`` on bad usage.
    """
    args = parse_args()
    if args.worker:
        Path(args.out).write_text(json.dumps(run_worker(args)))
        return 0
    if "," in args.device or args.device == "cpu":
        print("[error] Phase 5b must run on a single GPU (e.g. --device 0).", file=sys.stderr)
        return 2

    paths = PipelinePaths(Path(args.project))
    sample_image = _first_test_image(args.data)
    print(f"Phase 5b — efficiency (batch=1) on device {args.device}")
    print(f"  variants={args.variants} models={args.models} precisions={args.precisions}")
    print(f"  forward: {args.warmup} warm-up + {args.iters} timed | "
          f"predict(): {args.e2e_warmup} warm-up + {args.e2e_iters} timed")

    failures = 0
    for variant in args.variants:
        for m in args.models:
            for precision in args.precisions:
                try:
                    r = benchmark_one(variant, m, precision, args, paths, sample_image)
                    pl = r["payload"]
                    status = "skip (up to date)" if r["skipped"] else "ok"
                    print(
                        f"  [{status}] {r['tag']:<26} fwd median={pl['forward']['median_ms']:.1f} ms "
                        f"P95={pl['forward']['p95_ms']:.1f} ms  FPS={pl['forward']['fps']:.2f} | "
                        f"cached-text median={pl['forward_cached_text']['median_ms']:.1f} ms | "
                        f"e2e median={pl['end_to_end']['median_ms']:.1f} ms | "
                        f"VRAM peak={pl['memory']['vram_peak_allocated_mb']:.0f} MB | "
                        f"GFLOPs={pl['model']['gflops']:.1f}"
                        f"{'  [CONTENDED]' if pl['contended'] else ''}",
                    )
                except Exception:
                    failures += 1
                    print(f"  [fail] {variant}/{m}/{precision}:\n{traceback.format_exc()}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
