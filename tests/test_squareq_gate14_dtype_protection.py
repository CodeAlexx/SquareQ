"""Gate 14: dtype Protection & Model Portability — SquareQ Tests.

Tests that QuantLinear and QuantLinearLoRA INT8 buffers survive
model.to(dtype=...), model.half(), model.bfloat16(), and _apply()
chains.  Without this protection, training frameworks that call
model.to() liberally would silently corrupt INT8 weights.

Run::

    python -m pytest serenity/tests/test_squareq_gate14_dtype_protection.py -x -v
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from squareq.modules import QuantLinear, QuantLinearLoRA

__all__: list[str] = []


# ── helpers ────────────────────────────────────────────────────────────────


def _make_quant_linear(
    in_f: int = 32, out_f: int = 32, *, bias: bool = True,
) -> QuantLinear:
    """Create a QuantLinear with populated INT8 quant state."""
    mod = QuantLinear(in_f, out_f, bias=bias, compute_dtype=torch.bfloat16)
    w = torch.randn(out_f, in_f)
    flat = w.view(out_f, -1)
    pad = (-flat.shape[1]) % 64
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    amax = flat.abs().amax(dim=1)
    scale = (amax / 127.0).clamp(min=1e-8)
    inv = 1.0 / scale
    q = torch.round(flat * inv.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
    zp = torch.zeros(out_f, dtype=torch.float32)
    bias_t = torch.randn(out_f) if bias else None
    mod.set_quant_state(qweight=q, scale=scale, zero_point=zp, bias=bias_t)
    return mod


def _make_quant_lora(
    in_f: int = 32, out_f: int = 32, rank: int = 4,
) -> QuantLinearLoRA:
    """Create a QuantLinearLoRA with populated INT8 base + FP32 LoRA."""
    mod = QuantLinearLoRA(
        in_f, out_f, rank=rank, alpha=float(rank),
        bias=True, compute_dtype=torch.bfloat16,
    )
    w = torch.randn(out_f, in_f)
    flat = w.view(out_f, -1)
    pad = (-flat.shape[1]) % 64
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    amax = flat.abs().amax(dim=1)
    scale = (amax / 127.0).clamp(min=1e-8)
    inv = 1.0 / scale
    q = torch.round(flat * inv.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
    zp = torch.zeros(out_f, dtype=torch.float32)
    mod.set_quant_state(
        qweight=q, scale=scale, zero_point=zp,
        bias=torch.randn(out_f),
    )
    return mod


class _QuantModel(nn.Module):
    """Wrapper to test parent .to() propagation to QuantLinear child."""

    def __init__(self) -> None:
        super().__init__()
        self.quant = _make_quant_linear()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quant(x)


class _LoRAModel(nn.Module):
    """Wrapper to test parent .to() propagation to QuantLinearLoRA child."""

    def __init__(self) -> None:
        super().__init__()
        self.lora = _make_quant_lora()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora(x)


# ── SQG-14.1: Direct .to(dtype=...) on QuantLinear ──────────────────────


class TestGate141DirectTo:
    """SQG-14.1: Direct .to() strips dtype kwarg — INT8 preserved."""

    def test_to_dtype_bfloat16_keyword(self) -> None:
        """quant.to(dtype=bfloat16) should be stripped — INT8 preserved."""
        mod = _make_quant_linear()
        mod.to(dtype=torch.bfloat16)
        assert mod.qweight.dtype == torch.int8, (
            "SQG-14.1: qweight should remain int8 after .to(dtype=bf16)."
        )
        assert mod.scale.dtype == torch.float32, (
            "SQG-14.1: scale should remain fp32 after .to(dtype=bf16)."
        )
        assert mod.zero_point.dtype == torch.float32, (
            "SQG-14.1: zero_point should remain fp32 after .to(dtype=bf16)."
        )

    def test_to_dtype_float16_keyword(self) -> None:
        """quant.to(dtype=float16) should be stripped — INT8 preserved."""
        mod = _make_quant_linear()
        mod.to(dtype=torch.float16)
        assert mod.qweight.dtype == torch.int8
        assert mod.scale.dtype == torch.float32

    def test_half_call(self) -> None:
        """.half() goes through _apply — INT8 still protected."""
        mod = _make_quant_linear()
        mod.half()
        assert mod.qweight.dtype == torch.int8, (
            "SQG-14.1: qweight should remain int8 after .half()."
        )
        assert mod.scale.dtype == torch.float32, (
            "SQG-14.1: scale should remain fp32 after .half()."
        )

    def test_bfloat16_call(self) -> None:
        """.bfloat16() goes through _apply — INT8 still protected."""
        mod = _make_quant_linear()
        mod.bfloat16()
        assert mod.qweight.dtype == torch.int8
        assert mod.scale.dtype == torch.float32


# ── SQG-14.2: Parent model .to() propagation ────────────────────────────


class TestGate142ParentApply:
    """SQG-14.2: Parent .to(dtype) propagates via _apply — INT8 protected."""

    def test_parent_to_bf16_preserves_int8(self) -> None:
        """Parent model.to(dtype=bf16) should not corrupt child INT8."""
        model = _QuantModel()
        model.to(dtype=torch.bfloat16)
        assert model.quant.qweight.dtype == torch.int8, (
            "SQG-14.2: qweight should remain int8 after parent .to(bf16)."
        )

    def test_parent_to_bf16_preserves_fp32_scale(self) -> None:
        """scale and zero_point should stay fp32 after parent .to()."""
        model = _QuantModel()
        model.to(dtype=torch.bfloat16)
        assert model.quant.scale.dtype == torch.float32, (
            "SQG-14.2: scale should remain fp32."
        )
        assert model.quant.zero_point.dtype == torch.float32, (
            "SQG-14.2: zero_point should remain fp32."
        )

    def test_forward_unchanged_after_parent_to(self) -> None:
        """Forward output approximately unchanged after parent .to(dtype=bf16).

        Bias is a regular parameter (not INT8-protected) so it converts
        fp32→bf16, causing small numerical differences.
        """
        model = _QuantModel()
        x = torch.randn(4, 32)
        out_before = model(x).clone()

        model.to(dtype=torch.bfloat16)
        out_after = model(x)

        torch.testing.assert_close(
            out_before, out_after, atol=1e-2, rtol=1e-2,
            msg="SQG-14.2: Forward should be approximately unchanged after .to(bf16).",
        )


# ── SQG-14.3: QuantLinearLoRA dtype protection ──────────────────────────


class TestGate143LoRADtype:
    """SQG-14.3: QuantLinearLoRA protects INT8 base, allows LoRA change."""

    def test_lora_base_int8_preserved(self) -> None:
        """Base INT8 qweight preserved after parent .to(bf16)."""
        model = _LoRAModel()
        model.to(dtype=torch.bfloat16)
        assert model.lora.qweight.dtype == torch.int8, (
            "SQG-14.3: LoRA base qweight should remain int8."
        )
        assert model.lora.scale.dtype == torch.float32, (
            "SQG-14.3: LoRA scale should remain fp32."
        )

    def test_lora_params_converted_via_parent(self) -> None:
        """LoRA params (lora_A, lora_B) should change dtype via parent _apply."""
        model = _LoRAModel()
        # Before: LoRA params are float32 (default).
        assert model.lora.lora_A.dtype == torch.float32

        model.to(dtype=torch.bfloat16)

        # After: LoRA params should be bf16 (not protected by _apply override).
        assert model.lora.lora_A.dtype == torch.bfloat16, (
            "SQG-14.3: lora_A should be bf16 after parent .to(bf16)."
        )
        assert model.lora.lora_B.dtype == torch.bfloat16, (
            "SQG-14.3: lora_B should be bf16 after parent .to(bf16)."
        )

    def test_lora_forward_unchanged_after_parent_to(self) -> None:
        """Forward output unchanged after parent .to(bf16) — base is same."""
        model = _LoRAModel()
        x = torch.randn(4, 32)
        out_before = model(x).clone()

        model.to(dtype=torch.bfloat16)
        out_after = model(x)

        # Tolerance needed because LoRA params changed precision (fp32→bf16).
        torch.testing.assert_close(
            out_before, out_after, atol=1e-2, rtol=1e-2,
            msg="SQG-14.3: Forward should be approximately unchanged.",
        )


# ── Gate 14 integration ─────────────────────────────────────────────────


class TestGate14Integration:
    """Integration: edge cases for dtype protection."""

    def test_sequential_to_calls(self) -> None:
        """Multiple .to() calls in sequence preserve INT8."""
        mod = _make_quant_linear()
        mod.to(dtype=torch.bfloat16)
        mod.to(dtype=torch.float16)
        mod.to(dtype=torch.float32)
        mod.bfloat16()
        mod.half()
        assert mod.qweight.dtype == torch.int8, (
            "SQG-14-INT: INT8 should survive sequential .to() calls."
        )
        assert mod.scale.dtype == torch.float32, (
            "SQG-14-INT: scale should survive sequential .to() calls."
        )

    def test_qweight_values_unchanged_after_to(self) -> None:
        """INT8 qweight values are bit-exact after dtype roundtrip."""
        mod = _make_quant_linear()
        qw_before = mod.qweight.clone()
        sc_before = mod.scale.clone()
        zp_before = mod.zero_point.clone()

        # Parent model to + back.
        model = nn.Sequential(mod)
        model.to(dtype=torch.bfloat16)
        model.to(dtype=torch.float32)

        assert torch.equal(mod.qweight, qw_before), (
            "SQG-14-INT: qweight values must be bit-exact after .to() roundtrip."
        )
        assert torch.equal(mod.scale, sc_before), (
            "SQG-14-INT: scale values must be bit-exact."
        )
        assert torch.equal(mod.zero_point, zp_before), (
            "SQG-14-INT: zero_point values must be bit-exact."
        )

    def test_no_bias_module_to_bf16(self) -> None:
        """QuantLinear without bias also survives .to(bf16)."""
        mod = _make_quant_linear(bias=False)
        assert mod.bias is None
        mod.bfloat16()
        assert mod.qweight.dtype == torch.int8
        assert mod.scale.dtype == torch.float32
