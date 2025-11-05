from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional

import torch
from torch import nn

from squareq.kernels import slab_matmul


@lru_cache(None)
def _get_kernel():
    """Load and cache the Triton slab matmul kernel module."""
    return slab_matmul()


class QuantLinearLoRA(nn.Module):
    """
    Quantized linear layer with LoRA adapters.

    The base weights remain in INT8 (qweight + scale + zero_point), while the trainable
    LoRA matrices stay in FP32 to preserve gradient fidelity.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: int = 8,
        alpha: float = 1.0,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.compute_dtype = compute_dtype

        self.register_buffer("qweight", torch.empty(self.out_features, self.in_features, dtype=torch.int8))
        self.register_buffer("scale", torch.empty(self.out_features, dtype=torch.float32))
        self.register_buffer("zero_point", torch.empty(self.out_features, dtype=torch.float32))
        self.register_buffer("bias", torch.zeros(self.out_features, dtype=torch.float32), persistent=False)

        # LoRA trainables use FP32 for numerical stability.
        self.lora_A = nn.Parameter(torch.empty(self.in_features, self.rank, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.empty(self.rank, self.out_features, dtype=torch.float32))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def set_quant_state(
        self,
        *,
        qweight: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> None:
        if qweight.dtype != torch.int8:
            raise TypeError(f"qweight must be int8, received {qweight.dtype}")
        if scale.dtype != torch.float32 or zero_point.dtype != torch.float32:
            raise TypeError("scale/zero_point must be float32")
        if qweight.shape != self.qweight.shape:
            raise ValueError(f"qweight shape mismatch: expected {tuple(self.qweight.shape)}, got {tuple(qweight.shape)}")

        device = qweight.device
        self.qweight.copy_(qweight.to(device=device))
        self.scale.copy_(scale.to(device=device))
        self.zero_point.copy_(zero_point.to(device=device))
        if bias is not None:
            self.bias.copy_(bias.to(device=device, dtype=torch.float32))
        else:
            self.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.qweight.numel() == 0:
            raise RuntimeError("QuantLinearLoRA called before quant state was set.")
        if not x.is_cuda:
            raise RuntimeError("QuantLinearLoRA expects CUDA activations.")

        leading_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features)
        activations = x_flat.to(dtype=torch.bfloat16).transpose(0, 1).contiguous()

        kernel = _get_kernel()
        base = kernel.run_slab_matmul(
            self.qweight,
            activations,
            self.scale,
            self.zero_point,
            alpha=1.0,
        ).transpose(0, 1)

        if self.bias is not None:
            base += self.bias.to(base.dtype)

        lora_out = (x_flat @ self.lora_A) @ self.lora_B
        out = base + lora_out * self.alpha
        out = out.to(self.compute_dtype).contiguous()
        out = out.view(*leading_shape, self.out_features)
        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, compute_dtype={self.compute_dtype}"
        )
