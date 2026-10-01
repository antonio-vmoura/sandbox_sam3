"""Run one SAM 3 training job from a generated config (one process, one GPU).

Equivalent to the official ``sam3/train/train.py`` single-node path
(``single_proc_run``) without Hydra's config search path, so generated configs
can live in the run directory (``logs/``) instead of the source tree. The
official trainer auto-resumes from ``<run>/checkpoints/checkpoint.pt``;
:class:`protocol_trainer.ProtocolTrainer` restores the RNG / early-stopping
state on top of it.

Determinism (protocol keys ``deterministic`` / ``strict_determinism``)
---------------------------------------------------------------------
``deterministic``: seeds, cuDNN deterministic / no benchmark,
``torch.use_deterministic_algorithms(True, warn_only=True)`` and the
deterministic ``F.grid_sample`` of :mod:`determinism` (process-local).

SAM 3 training then has exactly one nondeterministic CUDA op left: the
backward of the ViT's memory-efficient attention (``F.scaled_dot_product_attention``
on the V100). PyTorch switches it to a deterministic algorithm only in strict
mode (``warn_only=False``), which costs +33 % step time on the V100S (6.67 vs
5.01 s / step, ``probe_memory.py`` P1D vs P1). The study does **not** use strict
mode (``strict_determinism = False``); measured consequence: two identical
3-epoch runs differ by <= 6.8e-5 in the weights and in the 5th digit of the
validation JSI — ordinary GPU run-to-run noise. Repeated or resumed runs are
therefore statistically equivalent, not bit-identical.

With ``strict_determinism = True`` training was verified bit-exact (identical
weights across two runs and across a kill / resume) and any other
nondeterministic op would raise.

Exit codes: 0 completed (``protocol_final.json`` written); 1 error; 3 CUDA out
of memory.

Usage (called by :func:`training.train_or_resume`):
    python sam3_seg/run_training.py <run_dir>/config.yaml
"""

from __future__ import annotations

import os
import sys


def main(config_path: str) -> int:
    """Instantiate the trainer from ``config_path`` and run it."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.update(MASTER_ADDR="localhost", RANK="0", LOCAL_RANK="0", WORLD_SIZE="1")
    os.environ.setdefault("MASTER_PORT", str(20000 + os.getpid() % 20000))
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from sam3.train.utils.train_utils import register_omegaconf_resolvers

    register_omegaconf_resolvers()
    cfg = OmegaConf.load(config_path)
    study = cfg.get("study", {})
    if study.get("deterministic", False):
        import determinism

        torch.use_deterministic_algorithms(True, warn_only=not study.get("strict_determinism", False))
        determinism.install()   # grid_sample backward has no deterministic CUDA kernel (see determinism.py)
    os.makedirs(cfg.launcher.experiment_log_dir, exist_ok=True)
    try:
        trainer = instantiate(cfg.trainer, _recursive_=False)
        trainer.run()
    except torch.OutOfMemoryError as e:
        print(f"[run_training] CUDA out of memory: {e}", file=sys.stderr, flush=True)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
