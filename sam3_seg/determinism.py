"""Deterministic replacement for ``F.grid_sample`` in SAM 3 training.

SAM 3 training has two nondeterministic CUDA operations: the memory-efficient
attention backward of the ViT (deterministic only in PyTorch's strict mode, not
used by the study for speed — see ``run_training.py``) and
``grid_sampler_2d_backward_cuda`` (gradient accumulation with atomic adds; no
deterministic CUDA kernel exists, strict mode would raise),
called by the mask loss's point sampling (``sam3/train/loss/mask_sampling.py``)
and by the geometry encoder. It made two identical training runs differ (max
weight difference 2.5e-5 after 3 epochs).

:func:`grid_sample_deterministic` computes the same bilinear interpolation
(zero padding, ``align_corners`` honoured) as an explicit weighted sum of the
four gathered neighbours. Its backward pass is a ``scatter_add``, which has a
deterministic CUDA implementation in PyTorch's deterministic mode. The
forward result equals ``F.grid_sample`` up to floating-point rounding.

:func:`install` replaces ``torch.nn.functional.grid_sample`` in the training
process only (the vendored ``sam3`` code is not modified); calls with other
modes / padding fall back to the original implementation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_ORIGINAL_GRID_SAMPLE = F.grid_sample


def grid_sample_deterministic(input: torch.Tensor, grid: torch.Tensor, mode: str = "bilinear",
                              padding_mode: str = "zeros", align_corners: bool | None = None) -> torch.Tensor:
    """Bilinear, zero-padded ``grid_sample`` for 4-D inputs with a deterministic backward."""
    if mode != "bilinear" or padding_mode != "zeros" or input.dim() != 4:
        return _ORIGINAL_GRID_SAMPLE(input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    align_corners = bool(align_corners)
    n, c, h, w = input.shape
    _, hg, wg, _ = grid.shape
    gx, gy = grid[..., 0], grid[..., 1]
    if align_corners:
        x, y = (gx + 1) * (w - 1) / 2, (gy + 1) * (h - 1) / 2
    else:
        x, y = ((gx + 1) * w - 1) / 2, ((gy + 1) * h - 1) / 2
    x0, y0 = torch.floor(x), torch.floor(y)
    wx1, wy1 = x - x0, y - y0
    wx0, wy0 = 1 - wx1, 1 - wy1
    flat = input.reshape(n, c, h * w)
    out = input.new_zeros(n, c, hg * wg)
    for xi, yi, wgt in ((x0, y0, wx0 * wy0), (x0 + 1, y0, wx1 * wy0), (x0, y0 + 1, wx0 * wy1), (x0 + 1, y0 + 1, wx1 * wy1)):
        valid = (xi >= 0) & (xi <= w - 1) & (yi >= 0) & (yi <= h - 1)
        idx = (yi.clamp(0, h - 1) * w + xi.clamp(0, w - 1)).long().reshape(n, 1, hg * wg).expand(n, c, hg * wg)
        vals = torch.gather(flat, 2, idx)
        out = out + vals * (wgt * valid).reshape(n, 1, hg * wg).to(vals.dtype)
    return out.reshape(n, c, hg, wg)


def install() -> None:
    """Route every ``F.grid_sample`` call of this process through the deterministic version."""
    F.grid_sample = grid_sample_deterministic
    torch.nn.functional.grid_sample = grid_sample_deterministic
