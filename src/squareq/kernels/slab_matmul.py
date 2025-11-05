# SPDX-License-Identifier: Apache-2.0
"""
Python wrapper around the fused Triton slab matmul kernel.

The kernel expects per-row scale and zero-point vectors and supports an
optional LoRA residual.  It is primarily used by the Squared-Q slab runtime
to provide a faster path than the reference PyTorch-based matmul.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import triton
import torch

from .slab_matmul_triton import slab_matmul_kernel

DEFAULT_BLOCK_M = 64
DEFAULT_BLOCK_N = 64
DEFAULT_BLOCK_K = 32


@dataclass(slots=True)
class SlabKernelConfig:
    """Convenience container for kernel launch parameters."""

    block_m: int = DEFAULT_BLOCK_M
    block_n: int = DEFAULT_BLOCK_N
    block_k: int = DEFAULT_BLOCK_K
    apply_relu: bool = True


def run_slab_matmul(
    qweight: torch.Tensor,
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    lora_a: Optional[torch.Tensor] = None,
    lora_b: Optional[torch.Tensor] = None,
    *,
    alpha: float = 1.0,
    config: Optional[SlabKernelConfig] = None,
) -> torch.Tensor:
    """
    Execute the fused slab matmul kernel.

    Args:
        qweight: Int8 weight matrix of shape ``[M, K]``.
        x: Activation matrix of shape ``[K, N]`` (FP16/BF16).
        scale: Per-output scale vector of shape ``[M]``.
        zero_point: Per-output zero-point vector of shape ``[M]``.
        lora_a: Optional LoRA A matrix of shape ``[M, R]``.
        lora_b: Optional LoRA B matrix of shape ``[R, N]``.
        alpha: Scaling factor applied to the LoRA product.
        config: Kernel launch configuration overrides.

    Returns:
        Output matrix of shape ``[M, N]`` in ``float32``.
    """

    if not torch.cuda.is_available():  # pragma: no cover - runtime safeguard
        raise RuntimeError("Squared-Q fused kernels require CUDA")

    cfg = config or SlabKernelConfig()
    if qweight.dtype != torch.int8:
        raise TypeError(f"qweight must be int8; received {qweight.dtype}")
    if scale.dim() != 1 or zero_point.dim() != 1:
        raise ValueError("scale and zero_point must be 1-D tensors")
    if scale.numel() != qweight.shape[0] or zero_point.numel() != qweight.shape[0]:
        raise ValueError(
            "scale and zero_point must have the same length as the qweight rows"
        )

    use_lora = lora_a is not None and lora_b is not None
    if use_lora:
        if lora_a.shape[0] != qweight.shape[0]:
            raise ValueError("LoRA A must align with qweight rows")
        if lora_b.shape[0] != lora_a.shape[1]:
            raise ValueError("LoRA A/B inner dimensions must match")

    if use_lora:
        rank = lora_a.shape[1]
    else:
        rank = 0

    M, K = qweight.shape
    if x.shape[0] != K:
        raise ValueError(f"Activation K mismatch: expected {K}, got {x.shape[0]}")
    N = x.shape[1]

    qweight_c = qweight.contiguous()
    x_c = x.contiguous()
    scale_c = scale.contiguous()
    zp_c = zero_point.contiguous()
    out = torch.empty((M, N), device="cuda", dtype=torch.float32)

    if use_lora:
        lora_a_c = lora_a.contiguous()
        lora_b_c = lora_b.contiguous()
    else:
        lora_a_c = torch.empty(1, 1, device="cuda")
        lora_b_c = torch.empty(1, 1, device="cuda")

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    slab_matmul_kernel[grid](
        qweight_c,
        x_c,
        scale_c,
        zp_c,
        lora_a_c,
        lora_b_c,
        alpha,
        use_lora,
        out,
        M,
        N,
        K,
        rank,
        qweight_c.stride(0),
        qweight_c.stride(1),
        x_c.stride(0),
        x_c.stride(1),
        out.stride(0),
        out.stride(1),
        lora_a_c.stride(0),
        lora_a_c.stride(1),
        lora_b_c.stride(0),
        lora_b_c.stride(1),
        apply_relu=cfg.apply_relu,
        BLOCK_M=cfg.block_m,
        BLOCK_N=cfg.block_n,
        BLOCK_K=cfg.block_k,
    )

    return out
