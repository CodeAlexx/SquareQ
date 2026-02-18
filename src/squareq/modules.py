"""QuantLinear and QuantLinearLoRA modules for INT8 fused compute.

Adapted from SquareQ (https://github.com/CodeAlexx/SquareQ).
Row-wise INT8 weights with FP32 scale/zero_point, computed via fused
Triton kernel.  QuantLinear.to() strips dtype kwargs to prevent
accidental BF16 promotion of INT8 buffers.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

__all__ = ["QuantLinear", "QuantLinearLoRA"]


class QuantLinear(nn.Module):
    """Quantized linear layer backed by INT8 weights + fused kernel.

    Buffers: qweight (int8), scale (fp32), zero_point (fp32).
    No float ``weight`` Parameter — buffers only.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = False,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.compute_dtype = compute_dtype
        self.padded_in_features = self.in_features

        # Empty buffers — populated by set_quant_state().
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
            self.bias = nn.Parameter(
                torch.zeros(self.out_features, dtype=torch.float32),
            )
        else:
            self.register_parameter("bias", None)

    def set_quant_state(
        self,
        *,
        qweight: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> None:
        """Load externally-provided quantized tensors into this module."""
        if qweight.dtype != torch.int8:
            raise TypeError(f"qweight must be int8, received {qweight.dtype}")
        if scale.dtype != torch.float32 or zero_point.dtype != torch.float32:
            raise TypeError("scale and zero_point must be float32 tensors")
        if qweight.shape[0] != self.out_features:
            raise ValueError(
                f"qweight rows ({qweight.shape[0]}) != out_features ({self.out_features})"
            )

        device = qweight.device
        self.qweight = qweight.contiguous().to(device)
        self.scale = scale.contiguous().to(device)
        self.zero_point = zero_point.contiguous().to(device)
        self.padded_in_features = self.qweight.shape[1]

        if bias is not None:
            if self.bias is None:
                raise ValueError("Bias provided but layer has no bias slot.")
            self.bias.data.copy_(bias.to(device=device, dtype=torch.float32))
        elif self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """CPU reference forward: dequant INT8→float, matmul, cast to compute_dtype."""
        if self.qweight.numel() == 0:
            raise RuntimeError(
                "QuantLinear called before quantized state was set."
            )
        # Dequantize: float_weight = qweight * scale (per-row).
        dequant = self.qweight.float() * self.scale.unsqueeze(1)
        padded_in = dequant.shape[1]

        # Pad input if qweight was K-padded.
        x_f = x.float()
        if padded_in > self.in_features:
            pad_size = padded_in - self.in_features
            x_f = torch.nn.functional.pad(x_f, (0, pad_size))

        out = x_f @ dequant.T

        if self.bias is not None:
            out = out + self.bias.float()

        return out.to(self.compute_dtype)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"compute_dtype={self.compute_dtype}, bias={self.bias is not None}"
        )

    def to(self, *args, **kwargs):  # type: ignore[override]
        """Strip dtype kwarg to prevent INT8→float promotion."""
        kwargs.pop("dtype", None)
        return super().to(*args, **kwargs)

    def _apply(self, fn):  # type: ignore[override]
        """Protect INT8/FP32 buffers from dtype promotion via model.to(dtype=...).

        PyTorch's ``Module.to(dtype=...)`` recurses through ``_apply(fn)``
        which bypasses our ``.to()`` override.  We snapshot the protected
        buffers, let ``_apply`` run normally (handling device moves and
        parameter dtype changes), then restore buffer dtypes while
        preserving any device change.
        """
        # Snapshot references before _apply overwrites them.
        qw_snap, sc_snap, zp_snap = self.qweight, self.scale, self.zero_point

        super()._apply(fn)

        # Restore dtype-protected buffers, keeping device changes.
        for attr, orig, expected_dtype in [
            ("qweight", qw_snap, torch.int8),
            ("scale", sc_snap, torch.float32),
            ("zero_point", zp_snap, torch.float32),
        ]:
            current = getattr(self, attr)
            if current.dtype != expected_dtype and orig.numel() > 0:
                setattr(self, attr, orig.to(device=current.device))

        return self


class QuantLinearLoRA(nn.Module):
    """Quantized linear with LoRA adapters (FP32 A/B on frozen INT8 base).

    Forward: ``base(x) + x @ lora_A @ lora_B * (alpha / rank)``.
    Base weights are INT8 buffers (frozen); lora_A and lora_B are trainable
    FP32 Parameters.  lora_B is initialized to zero so the initial LoRA
    contribution is zero.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: int = 8,
        alpha: float = 1.0,
        bias: bool = False,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.compute_dtype = compute_dtype
        self.padded_in_features = self.in_features

        # INT8 base buffers — populated by set_quant_state().
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

        # Trainable LoRA parameters.
        self.lora_A = nn.Parameter(torch.empty(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        nn.init.kaiming_uniform_(self.lora_A)

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(self.out_features, dtype=torch.float32),
            )
        else:
            self.register_parameter("bias", None)

    def set_quant_state(
        self,
        *,
        qweight: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> None:
        """Load externally-provided quantized tensors into this module."""
        if qweight.dtype != torch.int8:
            raise TypeError(f"qweight must be int8, received {qweight.dtype}")
        if scale.dtype != torch.float32 or zero_point.dtype != torch.float32:
            raise TypeError("scale and zero_point must be float32 tensors")
        if qweight.shape[0] != self.out_features:
            raise ValueError(
                f"qweight rows ({qweight.shape[0]}) != out_features ({self.out_features})"
            )

        device = qweight.device
        self.qweight = qweight.contiguous().to(device)
        self.scale = scale.contiguous().to(device)
        self.zero_point = zero_point.contiguous().to(device)
        self.padded_in_features = self.qweight.shape[1]

        if bias is not None:
            if self.bias is None:
                raise ValueError("Bias provided but layer has no bias slot.")
            self.bias.data.copy_(bias.to(device=device, dtype=torch.float32))
        elif self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: base(x) + lora(x) * scaling."""
        if self.qweight.numel() == 0:
            raise RuntimeError(
                "QuantLinearLoRA called before quantized state was set."
            )
        # Base path: dequant INT8→float, matmul.
        dequant = self.qweight.float() * self.scale.unsqueeze(1)
        padded_in = dequant.shape[1]

        x_f = x.float()
        if padded_in > self.in_features:
            pad_size = padded_in - self.in_features
            x_f = torch.nn.functional.pad(x_f, (0, pad_size))

        base_out = x_f @ dequant.T

        # LoRA path: x @ A @ B * scaling.
        lora_out = x.float() @ self.lora_A.float() @ self.lora_B.float() * self.scaling

        out = base_out + lora_out

        if self.bias is not None:
            out = out + self.bias.float()

        return out.to(self.compute_dtype)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, "
            f"compute_dtype={self.compute_dtype}, bias={self.bias is not None}"
        )

    def to(self, *args, **kwargs):  # type: ignore[override]
        """Strip dtype kwarg to prevent INT8→float promotion."""
        kwargs.pop("dtype", None)
        return super().to(*args, **kwargs)

    def _apply(self, fn):  # type: ignore[override]
        """Protect INT8/FP32 buffers from dtype promotion."""
        qw_snap, sc_snap, zp_snap = self.qweight, self.scale, self.zero_point

        super()._apply(fn)

        for attr, orig, expected_dtype in [
            ("qweight", qw_snap, torch.int8),
            ("scale", sc_snap, torch.float32),
            ("zero_point", zp_snap, torch.float32),
        ]:
            current = getattr(self, attr)
            if current.dtype != expected_dtype and orig.numel() > 0:
                setattr(self, attr, orig.to(device=current.device))

        return self
