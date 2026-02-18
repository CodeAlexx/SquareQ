"""Gate 4: LoRA on Quantized Base — SquareQ ↔ Stagehand Integration Tests.

TDD tests for QuantLinearLoRA: FP32 lora_A/lora_B adapters on a frozen
INT8 base.  Gradients flow through LoRA only; base buffers stay frozen.

All gate codes use the ``SQG-4.x`` prefix for CI triage.

Run::

    python -m pytest serenity/tests/test_squareq_gate4_lora.py -x -v
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from torch import nn

from squareq.modules import QuantLinear, QuantLinearLoRA

__all__: list[str] = []


# ── helpers ────────────────────────────────────────────────────────────────


def _make_quant_state(
    out_features: int,
    in_features: int,
    *,
    pack_k: int = 64,
    bias: bool = False,
) -> dict[str, torch.Tensor]:
    """Create synthetic INT8 quant state for testing."""
    padded_in = in_features + ((-in_features) % pack_k)
    weight = torch.randn(out_features, in_features)
    flat = weight.view(out_features, -1).float()
    if padded_in > in_features:
        flat = torch.nn.functional.pad(flat, (0, padded_in - in_features))
    max_vals = flat.abs().amax(dim=1)
    scale = (max_vals / 127.0).clamp(min=1e-8)
    inv_scale = 1.0 / scale
    qweight = torch.round(flat * inv_scale.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
    zero_point = torch.zeros_like(scale)

    result = {
        "qweight": qweight,
        "scale": scale,
        "zero_point": zero_point,
    }
    if bias:
        result["bias"] = torch.randn(out_features)
    return result


# ── Gate 4.1: Construction ────────────────────────────────────────────────


class TestGate41Construction:
    """SQG-4.1: QuantLinearLoRA initializes with correct structure."""

    def test_stores_rank_and_alpha(self) -> None:
        module = QuantLinearLoRA(64, 128, rank=16, alpha=8.0)
        assert module.rank == 16, "SQG-4.1: rank not stored."
        assert module.alpha == 8.0, "SQG-4.1: alpha not stored."

    def test_lora_A_shape(self) -> None:
        module = QuantLinearLoRA(64, 128, rank=16)
        assert module.lora_A.shape == (64, 16), (
            f"SQG-4.1: lora_A shape {module.lora_A.shape} != expected (64, 16)."
        )

    def test_lora_B_shape(self) -> None:
        module = QuantLinearLoRA(64, 128, rank=16)
        assert module.lora_B.shape == (16, 128), (
            f"SQG-4.1: lora_B shape {module.lora_B.shape} != expected (16, 128)."
        )

    def test_lora_B_initialized_to_zero(self) -> None:
        """Standard LoRA init: B starts at zero so initial LoRA contribution is 0."""
        module = QuantLinearLoRA(64, 128, rank=8)
        assert torch.all(module.lora_B == 0), (
            "SQG-4.1-INIT: lora_B should be initialized to zeros."
        )

    def test_base_buffers_exist(self) -> None:
        module = QuantLinearLoRA(64, 128, rank=8)
        assert hasattr(module, "qweight"), "SQG-4.1: Missing qweight buffer."
        assert hasattr(module, "scale"), "SQG-4.1: Missing scale buffer."
        assert hasattr(module, "zero_point"), "SQG-4.1: Missing zero_point buffer."

    def test_bias_parameter_when_requested(self) -> None:
        module = QuantLinearLoRA(64, 128, rank=8, bias=True)
        assert module.bias is not None, "SQG-4.1: bias should exist when bias=True."

    def test_no_bias_when_not_requested(self) -> None:
        module = QuantLinearLoRA(64, 128, rank=8, bias=False)
        assert module.bias is None, "SQG-4.1: bias should be None when bias=False."

    def test_scaling_factor(self) -> None:
        """Scaling should follow Serenity convention: alpha / rank."""
        module = QuantLinearLoRA(64, 128, rank=8, alpha=16.0)
        expected = 16.0 / 8
        assert module.scaling == expected, (
            f"SQG-4.1: scaling {module.scaling} != expected {expected} (alpha/rank)."
        )


# ── Gate 4.2: Trainability ────────────────────────────────────────────────


class TestGate42Trainability:
    """SQG-4.2: LoRA params are trainable, base buffers are frozen."""

    def test_lora_params_require_grad(self) -> None:
        module = QuantLinearLoRA(64, 128, rank=8)
        assert module.lora_A.requires_grad, (
            "SQG-4.2-FROZEN_LORA: lora_A should require grad."
        )
        assert module.lora_B.requires_grad, (
            "SQG-4.2-FROZEN_LORA: lora_B should require grad."
        )

    def test_base_buffers_no_grad(self) -> None:
        """Buffers (qweight, scale, zero_point) don't participate in autograd."""
        module = QuantLinearLoRA(64, 128, rank=8)
        state = _make_quant_state(128, 64)
        module.set_quant_state(**state)

        # Buffers should not require grad.
        assert not module.qweight.requires_grad, (
            "SQG-4.2-TRAINABLE_BASE: qweight should not require grad."
        )
        assert not module.scale.requires_grad, (
            "SQG-4.2-TRAINABLE_BASE: scale should not require grad."
        )

    def test_named_parameters_contents(self) -> None:
        """named_parameters should yield only lora_A, lora_B, and optionally bias."""
        module = QuantLinearLoRA(64, 128, rank=8, bias=True)
        param_names = {n for n, _ in module.named_parameters()}
        expected = {"lora_A", "lora_B", "bias"}
        assert param_names == expected, (
            f"SQG-4.2: named_parameters has {param_names}, expected {expected}."
        )

    def test_named_parameters_no_bias(self) -> None:
        module = QuantLinearLoRA(64, 128, rank=8, bias=False)
        param_names = {n for n, _ in module.named_parameters()}
        expected = {"lora_A", "lora_B"}
        assert param_names == expected, (
            f"SQG-4.2: named_parameters has {param_names}, expected {expected}."
        )


# ── Gate 4.3: set_quant_state ─────────────────────────────────────────────


class TestGate43SetQuantState:
    """SQG-4.3: set_quant_state populates base INT8 buffers."""

    def test_populates_base(self) -> None:
        module = QuantLinearLoRA(64, 128, rank=8)
        state = _make_quant_state(128, 64)
        module.set_quant_state(**state)

        assert module.qweight.numel() > 0, "SQG-4.3: qweight empty after set_quant_state."
        assert module.qweight.dtype == torch.int8, "SQG-4.3: qweight not int8."
        assert module.scale.dtype == torch.float32, "SQG-4.3: scale not float32."

    def test_wrong_dtype_raises(self) -> None:
        module = QuantLinearLoRA(64, 128, rank=8)
        with pytest.raises(TypeError):
            module.set_quant_state(
                qweight=torch.randn(128, 64),  # float, not int8
                scale=torch.ones(128),
                zero_point=torch.zeros(128),
            )

    def test_with_bias(self) -> None:
        module = QuantLinearLoRA(64, 128, rank=8, bias=True)
        state = _make_quant_state(128, 64, bias=True)
        module.set_quant_state(**state)
        assert module.bias is not None
        assert module.bias.numel() > 0


# ── Gate 4.4: Forward computation ─────────────────────────────────────────


class TestGate44Forward:
    """SQG-4.4: Forward = base(x) + lora(x) * scaling."""

    def test_forward_output_shape(self) -> None:
        module = QuantLinearLoRA(64, 128, rank=8)
        state = _make_quant_state(128, 64)
        module.set_quant_state(**state)

        x = torch.randn(4, 64)
        out = module(x)
        assert out.shape == (4, 128), (
            f"SQG-4.4: Output shape {out.shape} != expected (4, 128)."
        )

    def test_forward_3d_input(self) -> None:
        module = QuantLinearLoRA(64, 128, rank=8)
        state = _make_quant_state(128, 64)
        module.set_quant_state(**state)

        x = torch.randn(2, 5, 64)
        out = module(x)
        assert out.shape == (2, 5, 128), (
            f"SQG-4.4: 3D output shape {out.shape} != expected (2, 5, 128)."
        )

    def test_zero_lora_equals_base(self) -> None:
        """When lora_B=0 (init state), output should equal base-only output."""
        module = QuantLinearLoRA(64, 128, rank=8)
        state = _make_quant_state(128, 64)
        module.set_quant_state(**state)

        # Also create a plain QuantLinear with same base.
        base = QuantLinear(64, 128, bias=False)
        base.set_quant_state(**state)

        x = torch.randn(4, 64)
        lora_out = module(x)
        base_out = base(x)

        torch.testing.assert_close(
            lora_out, base_out,
            atol=1e-4, rtol=1e-4,
            msg="SQG-4.4: With zero lora_B, QuantLinearLoRA output should match QuantLinear.",
        )

    def test_lora_contribution_proportional_to_alpha(self) -> None:
        """Doubling alpha should double the LoRA contribution (by norm)."""
        mod_a1 = QuantLinearLoRA(64, 128, rank=8, alpha=1.0)
        mod_a2 = QuantLinearLoRA(64, 128, rank=8, alpha=2.0)
        state = _make_quant_state(128, 64)
        mod_a1.set_quant_state(**state)
        mod_a2.set_quant_state(**state)

        # Set identical non-zero LoRA weights.
        torch.manual_seed(42)
        lora_A = torch.randn(64, 8)
        lora_B = torch.randn(8, 128)
        with torch.no_grad():
            mod_a1.lora_A.copy_(lora_A)
            mod_a1.lora_B.copy_(lora_B)
            mod_a2.lora_A.copy_(lora_A)
            mod_a2.lora_B.copy_(lora_B)

        x = torch.randn(4, 64)
        out_a1 = mod_a1(x)
        out_a2 = mod_a2(x)

        # Compute base-only output to isolate LoRA contribution.
        base = QuantLinear(64, 128, bias=False)
        base.set_quant_state(**state)
        base_out = base(x)

        lora_contrib_a1 = out_a1.float() - base_out.float()
        lora_contrib_a2 = out_a2.float() - base_out.float()

        # Use L2 norms to compare — avoids element-wise division instability.
        norm_a1 = lora_contrib_a1.norm()
        norm_a2 = lora_contrib_a2.norm()
        ratio = (norm_a2 / norm_a1).item()
        assert abs(ratio - 2.0) < 0.15, (
            f"SQG-4.4: Alpha scaling ratio {ratio:.4f} != expected ~2.0."
        )

    def test_forward_with_bias(self) -> None:
        module = QuantLinearLoRA(64, 128, rank=8, bias=True)
        state = _make_quant_state(128, 64, bias=True)
        module.set_quant_state(**state)

        x = torch.randn(4, 64)
        out = module(x)
        assert out.shape == (4, 128)
        # Verify bias has visible effect.
        module_no_bias = QuantLinearLoRA(64, 128, rank=8, bias=False)
        state_nb = {k: v for k, v in state.items() if k != "bias"}
        module_no_bias.set_quant_state(**state_nb)
        with torch.no_grad():
            module_no_bias.lora_A.copy_(module.lora_A)
            module_no_bias.lora_B.copy_(module.lora_B)
        out_nb = module_no_bias(x)
        diff = (out.float() - out_nb.float()).abs().mean()
        assert diff > 1e-6, "SQG-4.4: Bias should change the output."

    def test_forward_before_quant_state_raises(self) -> None:
        module = QuantLinearLoRA(64, 128, rank=8)
        x = torch.randn(4, 64)
        with pytest.raises(RuntimeError, match="[Bb]efore.*state|[Bb]efore.*quant"):
            module(x)


# ── Gate 4.5: Gradient flow ───────────────────────────────────────────────


class TestGate45GradientFlow:
    """SQG-4.5: Gradients flow through LoRA params only, not base."""

    def test_lora_A_gets_grad(self) -> None:
        module = QuantLinearLoRA(64, 128, rank=8)
        state = _make_quant_state(128, 64)
        module.set_quant_state(**state)

        # Set non-zero lora_B so gradients can flow.
        with torch.no_grad():
            module.lora_B.fill_(0.1)

        x = torch.randn(4, 64)
        out = module(x)
        loss = out.sum()
        loss.backward()

        assert module.lora_A.grad is not None, (
            "SQG-4.5-NO_GRAD: lora_A.grad is None after backward."
        )
        assert module.lora_A.grad.abs().sum() > 0, (
            "SQG-4.5-NO_GRAD: lora_A.grad is all zeros."
        )

    def test_lora_B_gets_grad(self) -> None:
        module = QuantLinearLoRA(64, 128, rank=8)
        state = _make_quant_state(128, 64)
        module.set_quant_state(**state)

        x = torch.randn(4, 64)
        out = module(x)
        loss = out.sum()
        loss.backward()

        assert module.lora_B.grad is not None, (
            "SQG-4.5-NO_GRAD: lora_B.grad is None after backward."
        )

    def test_base_buffers_no_grad_after_backward(self) -> None:
        module = QuantLinearLoRA(64, 128, rank=8)
        state = _make_quant_state(128, 64)
        module.set_quant_state(**state)

        x = torch.randn(4, 64)
        out = module(x)
        loss = out.sum()
        loss.backward()

        # Buffers should never accumulate grad.
        assert not module.qweight.requires_grad, (
            "SQG-4.5-BASE_GRAD: qweight should not require grad."
        )
        assert not module.scale.requires_grad, (
            "SQG-4.5-BASE_GRAD: scale should not require grad."
        )

    def test_multiple_backward_accumulates(self) -> None:
        """Two backward passes should accumulate gradients (standard PyTorch)."""
        module = QuantLinearLoRA(64, 128, rank=8)
        state = _make_quant_state(128, 64)
        module.set_quant_state(**state)
        with torch.no_grad():
            module.lora_B.fill_(0.1)

        x = torch.randn(4, 64)
        out1 = module(x)
        out1.sum().backward()
        grad_after_1 = module.lora_A.grad.clone()

        out2 = module(x)
        out2.sum().backward()
        grad_after_2 = module.lora_A.grad.clone()

        assert (grad_after_2.abs() >= grad_after_1.abs() - 1e-6).all(), (
            "SQG-4.5: Gradients should accumulate over multiple backward passes."
        )


# ── Gate 4.6: LoRA state management ──────────────────────────────────────


class TestGate46StateManagement:
    """SQG-4.6: LoRA weights can be extracted and reset."""

    def test_extract_lora_state(self) -> None:
        module = QuantLinearLoRA(64, 128, rank=8)
        state = _make_quant_state(128, 64)
        module.set_quant_state(**state)
        with torch.no_grad():
            module.lora_A.fill_(1.0)
            module.lora_B.fill_(2.0)

        # Should be able to extract just the LoRA parameters.
        lora_state = {
            n: p.clone() for n, p in module.named_parameters()
            if "lora" in n
        }
        assert "lora_A" in lora_state, "SQG-4.6: lora_A not in parameters."
        assert "lora_B" in lora_state, "SQG-4.6: lora_B not in parameters."
        assert torch.all(lora_state["lora_A"] == 1.0)
        assert torch.all(lora_state["lora_B"] == 2.0)

    def test_reset_lora(self) -> None:
        """Zeroing lora_B resets LoRA contribution to zero."""
        module = QuantLinearLoRA(64, 128, rank=8)
        state = _make_quant_state(128, 64)
        module.set_quant_state(**state)

        # Set non-zero LoRA.
        with torch.no_grad():
            module.lora_A.fill_(1.0)
            module.lora_B.fill_(1.0)

        x = torch.randn(4, 64)
        out_with_lora = module(x).clone()

        # Reset LoRA by zeroing B.
        with torch.no_grad():
            module.lora_B.zero_()
        out_after_reset = module(x)

        # After reset, output should match base-only.
        base = QuantLinear(64, 128, bias=False)
        base.set_quant_state(**state)
        base_out = base(x)

        torch.testing.assert_close(
            out_after_reset, base_out,
            atol=1e-4, rtol=1e-4,
            msg="SQG-4.6: After lora_B.zero_(), output should match base.",
        )

    def test_dtype_protection(self) -> None:
        """model.to(bfloat16) should not corrupt INT8 base."""
        module = QuantLinearLoRA(64, 128, rank=8)
        state = _make_quant_state(128, 64)
        module.set_quant_state(**state)

        module.to(dtype=torch.bfloat16)
        assert module.qweight.dtype == torch.int8, (
            "SQG-4.6: qweight promoted from int8 after .to(bfloat16)."
        )
