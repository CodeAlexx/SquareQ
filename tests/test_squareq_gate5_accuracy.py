"""Gate 5: Numerical Accuracy — SquareQ Forward Path Tests.

TDD tests verifying that the QuantLinear CPU reference forward path
(dequant + matmul) produces numerically correct results compared to
float reference computations.

All gate codes use the ``SQG-5.x`` prefix for CI triage.

Run::

    python -m pytest serenity/tests/test_squareq_gate5_accuracy.py -x -v
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from squareq.modules import QuantLinear, QuantLinearLoRA

__all__: list[str] = []

_HAS_CUDA = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="CUDA not available")


# ── helpers ────────────────────────────────────────────────────────────────


def _quantize_reference(
    weight: torch.Tensor,
    pack_k: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a float weight matrix, return (qweight, scale, zero_point, padded_weight).

    Returns the padded float weight for computing expected results.
    """
    out_f, in_f = weight.shape
    padded_in = in_f + ((-in_f) % pack_k)
    flat = weight.float()
    if padded_in > in_f:
        flat = torch.nn.functional.pad(flat, (0, padded_in - in_f))

    max_vals = flat.abs().amax(dim=1)
    scale = (max_vals / 127.0).clamp(min=1e-8)
    inv_scale = 1.0 / scale
    qweight = torch.round(flat * inv_scale.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
    zero_point = torch.zeros_like(scale)

    return qweight, scale, zero_point, flat


def _make_loaded_quantlinear(
    in_features: int,
    out_features: int,
    *,
    bias: bool = False,
    pack_k: int = 64,
) -> tuple[QuantLinear, torch.Tensor]:
    """Create and load a QuantLinear, return (module, original_float_weight)."""
    weight = torch.randn(out_features, in_features)
    qweight, scale, zero_point, _ = _quantize_reference(weight, pack_k)
    module = QuantLinear(in_features, out_features, bias=bias)
    kwargs = {"qweight": qweight, "scale": scale, "zero_point": zero_point}
    if bias:
        kwargs["bias"] = torch.randn(out_features)
    module.set_quant_state(**kwargs)
    return module, weight


# ── Gate 5.1: Forward output shape ───────────────────────────────────────


class TestGate51OutputShape:
    """SQG-5.1: QuantLinear.forward produces correct output shapes."""

    def test_2d_input(self) -> None:
        module, _ = _make_loaded_quantlinear(64, 128)
        x = torch.randn(4, 64)
        out = module(x)
        assert out.shape == (4, 128), (
            f"SQG-5.1: Output shape {out.shape} != expected (4, 128)."
        )

    def test_3d_input(self) -> None:
        module, _ = _make_loaded_quantlinear(64, 128)
        x = torch.randn(2, 5, 64)
        out = module(x)
        assert out.shape == (2, 5, 128), (
            f"SQG-5.1: 3D output shape {out.shape} != expected (2, 5, 128)."
        )

    def test_1_sample(self) -> None:
        module, _ = _make_loaded_quantlinear(32, 48)
        x = torch.randn(1, 32)
        out = module(x)
        assert out.shape == (1, 48)

    def test_large_batch(self) -> None:
        module, _ = _make_loaded_quantlinear(64, 128)
        x = torch.randn(256, 64)
        out = module(x)
        assert out.shape == (256, 128)


# ── Gate 5.2: Accuracy vs float reference ────────────────────────────────


class TestGate52Accuracy:
    """SQG-5.2: Quantized forward is close to float reference."""

    def test_matches_dequant_matmul(self) -> None:
        """Output should match manual dequant + matmul within quant error."""
        in_f, out_f = 64, 128
        weight = torch.randn(out_f, in_f)
        qweight, scale, zero_point, padded_weight = _quantize_reference(weight)

        module = QuantLinear(in_f, out_f)
        module.set_quant_state(
            qweight=qweight, scale=scale, zero_point=zero_point,
        )

        x = torch.randn(4, in_f)

        # Module output (returns compute_dtype=bfloat16).
        out = module(x)

        # Manual dequant reference — also cast through compute_dtype for fair comparison.
        dequant = qweight.float() * scale.unsqueeze(1)
        padded_in = dequant.shape[1]
        x_padded = x.float()
        if padded_in > in_f:
            x_padded = torch.nn.functional.pad(x_padded, (0, padded_in - in_f))
        expected = (x_padded @ dequant.T).to(module.compute_dtype)

        torch.testing.assert_close(
            out, expected,
            atol=1e-3, rtol=1e-3,
            msg="SQG-5.2: QuantLinear output doesn't match dequant+matmul reference.",
        )

    def test_close_to_float_original(self) -> None:
        """Quantized output should be reasonably close to float original."""
        in_f, out_f = 128, 256
        weight = torch.randn(out_f, in_f)
        qweight, scale, zero_point, _ = _quantize_reference(weight)

        module = QuantLinear(in_f, out_f)
        module.set_quant_state(
            qweight=qweight, scale=scale, zero_point=zero_point,
        )

        x = torch.randn(8, in_f)
        out = module(x).float()

        # Float reference (using original non-quantized weight).
        expected = (x @ weight.T).float()

        # INT8 introduces ~1% relative error for normal distributions.
        # bf16 cast adds additional error. Allow generous tolerance.
        rel_error = (out - expected).abs() / (expected.abs() + 1e-6)
        mean_rel_error = rel_error.mean()
        assert mean_rel_error < 0.15, (
            f"SQG-5.2: Mean relative error {mean_rel_error:.4f} too high vs float reference."
        )

    def test_small_weight_values(self) -> None:
        """Accuracy with small weight values (tests scale precision)."""
        in_f, out_f = 32, 64
        weight = torch.randn(out_f, in_f) * 0.01  # Small weights.
        qweight, scale, zero_point, _ = _quantize_reference(weight)

        module = QuantLinear(in_f, out_f)
        module.set_quant_state(
            qweight=qweight, scale=scale, zero_point=zero_point,
        )

        x = torch.randn(4, in_f)
        out = module(x)

        # Dequant reference — cast through compute_dtype.
        dequant = qweight.float() * scale.unsqueeze(1)
        padded_in = dequant.shape[1]
        x_padded = x.float()
        if padded_in > in_f:
            x_padded = torch.nn.functional.pad(x_padded, (0, padded_in - in_f))
        expected = (x_padded @ dequant.T).to(module.compute_dtype)

        torch.testing.assert_close(
            out, expected,
            atol=1e-4, rtol=1e-3,
            msg="SQG-5.2: Small weights: output doesn't match reference.",
        )

    def test_large_input_values(self) -> None:
        """Accuracy with large input values."""
        in_f, out_f = 64, 128
        weight = torch.randn(out_f, in_f)
        qweight, scale, zero_point, _ = _quantize_reference(weight)

        module = QuantLinear(in_f, out_f)
        module.set_quant_state(
            qweight=qweight, scale=scale, zero_point=zero_point,
        )

        x = torch.randn(4, in_f) * 100.0  # Large inputs.
        out = module(x)

        dequant = qweight.float() * scale.unsqueeze(1)
        padded_in = dequant.shape[1]
        x_padded = x.float()
        if padded_in > in_f:
            x_padded = torch.nn.functional.pad(x_padded, (0, padded_in - in_f))
        expected = (x_padded @ dequant.T).to(module.compute_dtype)

        torch.testing.assert_close(
            out, expected,
            atol=1e-1, rtol=1e-3,
            msg="SQG-5.2: Large inputs: output doesn't match reference.",
        )


# ── Gate 5.3: Bias handling ──────────────────────────────────────────────


class TestGate53Bias:
    """SQG-5.3: Bias is correctly added in forward pass."""

    def test_bias_adds_to_output(self) -> None:
        # Use float32 compute_dtype so bf16 rounding doesn't affect bias comparison.
        in_f, out_f = 64, 128
        weight = torch.randn(out_f, in_f)
        qweight, scale, zero_point, _ = _quantize_reference(weight)
        bias_val = torch.randn(out_f)

        mod_bias = QuantLinear(in_f, out_f, bias=True, compute_dtype=torch.float32)
        mod_bias.set_quant_state(
            qweight=qweight, scale=scale, zero_point=zero_point,
            bias=bias_val,
        )

        mod_nobias = QuantLinear(in_f, out_f, bias=False, compute_dtype=torch.float32)
        mod_nobias.set_quant_state(
            qweight=qweight, scale=scale, zero_point=zero_point,
        )

        x = torch.randn(4, in_f)
        out_bias = mod_bias(x)
        out_nobias = mod_nobias(x)

        diff = out_bias - out_nobias
        expected_diff = bias_val.unsqueeze(0).expand_as(diff)
        torch.testing.assert_close(
            diff, expected_diff,
            atol=1e-4, rtol=1e-4,
            msg="SQG-5.3: Bias not correctly added to output.",
        )

    def test_zero_bias_no_effect(self) -> None:
        in_f, out_f = 64, 128
        weight = torch.randn(out_f, in_f)
        qweight, scale, zero_point, _ = _quantize_reference(weight)

        mod_bias = QuantLinear(in_f, out_f, bias=True)
        mod_bias.set_quant_state(
            qweight=qweight, scale=scale, zero_point=zero_point,
            # bias defaults to zero
        )

        mod_nobias = QuantLinear(in_f, out_f, bias=False)
        mod_nobias.set_quant_state(
            qweight=qweight, scale=scale, zero_point=zero_point,
        )

        x = torch.randn(4, in_f)
        torch.testing.assert_close(
            mod_bias(x), mod_nobias(x),
            atol=0.0, rtol=0.0,
            msg="SQG-5.3: Zero bias should produce same output as no bias.",
        )


# ── Gate 5.4: Compute dtype ─────────────────────────────────────────────


class TestGate54ComputeDtype:
    """SQG-5.4: Output dtype matches compute_dtype when possible."""

    def test_output_dtype_bfloat16(self) -> None:
        module, _ = _make_loaded_quantlinear(64, 128)
        x = torch.randn(4, 64, dtype=torch.bfloat16)
        out = module(x)
        # Output should be in compute_dtype (bfloat16 by default).
        assert out.dtype == torch.bfloat16, (
            f"SQG-5.4: Output dtype {out.dtype} != expected bfloat16."
        )

    def test_output_dtype_float32_input(self) -> None:
        module, _ = _make_loaded_quantlinear(64, 128)
        x = torch.randn(4, 64, dtype=torch.float32)
        out = module(x)
        # With float32 input, output should still be compute_dtype (bfloat16).
        assert out.dtype == module.compute_dtype, (
            f"SQG-5.4: Output dtype {out.dtype} != compute_dtype {module.compute_dtype}."
        )

    def test_custom_compute_dtype(self) -> None:
        in_f, out_f = 64, 128
        weight = torch.randn(out_f, in_f)
        qweight, scale, zero_point, _ = _quantize_reference(weight)

        module = QuantLinear(in_f, out_f, compute_dtype=torch.float32)
        module.set_quant_state(
            qweight=qweight, scale=scale, zero_point=zero_point,
        )

        x = torch.randn(4, in_f)
        out = module(x)
        assert out.dtype == torch.float32, (
            f"SQG-5.4: Output dtype {out.dtype} != expected float32."
        )


# ── Gate 5.5: Determinism ───────────────────────────────────────────────


class TestGate55Determinism:
    """SQG-5.5: Forward pass is deterministic."""

    def test_same_input_same_output(self) -> None:
        module, _ = _make_loaded_quantlinear(64, 128)
        x = torch.randn(4, 64)
        out1 = module(x)
        out2 = module(x)
        torch.testing.assert_close(
            out1, out2,
            atol=0.0, rtol=0.0,
            msg="SQG-5.5: Same input should produce identical output.",
        )

    def test_deterministic_across_calls(self) -> None:
        """Multiple calls with same input should be bitwise identical."""
        module, _ = _make_loaded_quantlinear(128, 256)
        x = torch.randn(8, 128)
        results = [module(x) for _ in range(5)]
        for i, r in enumerate(results[1:], 1):
            assert torch.equal(results[0], r), (
                f"SQG-5.5: Call {i} differs from call 0."
            )


# ── Gate 5.6: Padding correctness ───────────────────────────────────────


class TestGate56Padding:
    """SQG-5.6: K-padding does not affect output correctness."""

    def test_padded_vs_unpadded_in_features(self) -> None:
        """in_features not aligned to pack_k should still produce correct output."""
        # 100 is not aligned to pack_k=64, so padded_in=128.
        in_f, out_f = 100, 64
        weight = torch.randn(out_f, in_f)
        qweight, scale, zero_point, _ = _quantize_reference(weight, pack_k=64)

        module = QuantLinear(in_f, out_f)
        module.set_quant_state(
            qweight=qweight, scale=scale, zero_point=zero_point,
        )

        x = torch.randn(4, in_f)
        out = module(x)

        # Manual reference — cast through compute_dtype.
        dequant = qweight.float() * scale.unsqueeze(1)
        padded_in = dequant.shape[1]
        x_padded = x.float()
        if padded_in > in_f:
            x_padded = torch.nn.functional.pad(x_padded, (0, padded_in - in_f))
        expected = (x_padded @ dequant.T).to(module.compute_dtype)

        torch.testing.assert_close(
            out, expected,
            atol=1e-3, rtol=1e-3,
            msg="SQG-5.6: Padded in_features produces incorrect output.",
        )

    def test_aligned_in_features(self) -> None:
        """in_features already aligned to pack_k should work without padding."""
        in_f, out_f = 128, 64  # 128 is aligned to 64.
        weight = torch.randn(out_f, in_f)
        qweight, scale, zero_point, _ = _quantize_reference(weight, pack_k=64)

        module = QuantLinear(in_f, out_f)
        module.set_quant_state(
            qweight=qweight, scale=scale, zero_point=zero_point,
        )

        x = torch.randn(4, in_f)
        out = module(x)

        dequant = qweight.float() * scale.unsqueeze(1)
        expected = (x.float() @ dequant.T).to(module.compute_dtype)

        torch.testing.assert_close(
            out, expected,
            atol=1e-3, rtol=1e-3,
            msg="SQG-5.6: Aligned in_features produces incorrect output.",
        )


# ── Gate 5 Integration ──────────────────────────────────────────────────


class TestGate5Integration:
    """Integration: full pipeline forward pass correctness."""

    def test_scaffold_load_forward(self) -> None:
        """Full pipeline: build slab → scaffold → load → forward."""
        from pathlib import Path
        import tempfile

        from squareq.builder import build_safetensors_slab
        from squareq.loader import load_quant_state_from_slab
        from squareq.manifest import load_manifest
        from squareq.scaffold import prepare_model_for_quantized_streaming

        class _TinyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc1 = nn.Linear(32, 64, bias=False)
                self.fc2 = nn.Linear(64, 32, bias=True)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc2(self.fc1(x))

        with tempfile.TemporaryDirectory() as tmp:
            source = _TinyModel()
            build_safetensors_slab(
                model=source,
                output_dir=tmp,
                slab_name="test_acc",
                architecture_id="test",
                pack_k=64,
            )
            manifest = load_manifest(Path(tmp) / "test_acc.manifest.json")
            model = _TinyModel()
            prepare_model_for_quantized_streaming(model, manifest)
            load_quant_state_from_slab(
                model, manifest, Path(tmp) / "test_acc.safetensors",
            )

            # Both fc1 and fc2 should now be QuantLinear and support forward.
            x = torch.randn(4, 32)
            out = model(x)
            assert out.shape == (4, 32), (
                f"SQG-5-INT: Output shape {out.shape} != expected (4, 32)."
            )
            # Output should be finite.
            assert torch.isfinite(out).all(), (
                "SQG-5-INT: Output contains non-finite values."
            )

    def test_lora_forward_accuracy(self) -> None:
        """QuantLinearLoRA forward = base_forward + lora * scaling."""
        in_f, out_f, rank = 64, 128, 8
        weight = torch.randn(out_f, in_f)
        qweight, scale, zero_point, _ = _quantize_reference(weight)

        # Use float32 compute_dtype to avoid bf16 rounding in comparison.
        base = QuantLinear(in_f, out_f, compute_dtype=torch.float32)
        base.set_quant_state(
            qweight=qweight, scale=scale, zero_point=zero_point,
        )

        lora_mod = QuantLinearLoRA(
            in_f, out_f, rank=rank, alpha=4.0, compute_dtype=torch.float32,
        )
        lora_mod.set_quant_state(
            qweight=qweight, scale=scale, zero_point=zero_point,
        )

        # Set known LoRA weights.
        torch.manual_seed(123)
        lora_A_val = torch.randn(in_f, rank)
        lora_B_val = torch.randn(rank, out_f)
        with torch.no_grad():
            lora_mod.lora_A.copy_(lora_A_val)
            lora_mod.lora_B.copy_(lora_B_val)

        x = torch.randn(4, in_f)
        lora_out = lora_mod(x)
        base_out = base(x)

        # Expected LoRA contribution: x @ A @ B * scaling.
        scaling = 4.0 / rank
        lora_expected = x.float() @ lora_A_val.float() @ lora_B_val.float() * scaling
        expected_total = base_out + lora_expected

        torch.testing.assert_close(
            lora_out, expected_total,
            atol=1e-4, rtol=1e-4,
            msg="SQG-5-INT: LoRA forward doesn't match base + lora*scaling.",
        )
