from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import torch
from torch import nn

from squareq.kernels import slab_matmul


@lru_cache(None)
def _get_slab_kernel():
    """Load and cache the Triton slab matmul kernel module."""
    return slab_matmul()


class QuantLinear(nn.Module):
    """
    Quantized linear layer backed by the SquareQ fused Triton kernel.

    The layer keeps row-wise INT8 weights alongside FP32 scale and zero-point
    vectors. During the forward pass the fused kernel performs dequantisation
    and GEMM on the fly, writing FP32 accumulators that are converted to the
    requested compute dtype (BF16 by default).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.compute_dtype = compute_dtype
        self.padded_in_features = self.in_features

        self.register_buffer(
            "qweight",
            torch.empty((0, 0), dtype=torch.int8),
            persistent=False,
        )
        self.register_buffer(
            "scale",
            torch.empty(0, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "zero_point",
            torch.empty(0, dtype=torch.float32),
            persistent=False,
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_features, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

        if os.environ.get("SQUAREQ_USE_TRITON", "1") == "0":
            raise RuntimeError("SquareQ QuantLinear requires Triton fused kernels.")

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
            raise TypeError("scale and zero_point must be float32 tensors")
        if qweight.shape[0] != self.out_features:
            raise ValueError(
                f"qweight rows ({qweight.shape[0]}) do not match out_features ({self.out_features})"
            )

        device = qweight.device
        self.qweight = qweight.contiguous().to(device)
        self.scale = scale.contiguous().to(device)
        self.zero_point = zero_point.contiguous().to(device)
        self.padded_in_features = self.qweight.shape[1]
        if bias is not None:
            if self.bias is None:
                raise ValueError("Bias tensor provided but layer was initialised without bias.")
            self.bias.data.copy_(bias.to(device=device, dtype=torch.float32))
        elif self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.qweight.numel() == 0:
            raise RuntimeError("QuantLinear called before quantised state was set.")

        if not x.is_cuda:
            raise RuntimeError("QuantLinear expects CUDA activations.")

        batch_shape = x.shape[:-1]
        x_flat = x.view(-1, self.in_features)
        if self.padded_in_features != self.in_features:
            pad = self.padded_in_features - self.in_features
            x_flat = torch.nn.functional.pad(x_flat, (0, pad))

        activations = x_flat.to(dtype=torch.bfloat16).transpose(0, 1).contiguous()

        kernel = _get_slab_kernel()
        out = kernel.run_slab_matmul(
            self.qweight,
            activations,
            self.scale,
            self.zero_point,
            alpha=1.0,
        ).transpose(0, 1)

        if self.bias is not None:
            out += self.bias.to(out.dtype)

        out = out.to(self.compute_dtype).contiguous()
        out = out.view(*batch_shape, self.out_features)
        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"compute_dtype={self.compute_dtype}, bias={self.bias is not None}"
        )

    def to(self, *args, **kwargs):  # type: ignore[override]
        kwargs.pop("dtype", None)
        return super().to(*args, **kwargs)
