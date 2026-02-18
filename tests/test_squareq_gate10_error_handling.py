"""Gate 10: Error Handling & Recovery — SquareQ ↔ Stagehand Tests.

Maps to original SQG-6.1, 6.2, 6.3 from the merge gate checklist.
Tests that missing tensors, shape mismatches, and corrupted data produce
deterministic hard fails or explicit diagnostics — never silent degradation.

Run::

    python -m pytest serenity/tests/test_squareq_gate10_error_handling.py -x -v
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from squareq.builder import build_safetensors_slab
from squareq.manifest import (
    CURRENT_KERNEL_ABI_VERSION,
    load_and_validate_manifest,
    load_manifest,
    validate_manifest_against_safetensors,
)
from squareq.modules import QuantLinear

__all__: list[str] = []

# ── helpers ───────────────────────────────────────────────────────────────


def _make_model(hidden: int = 32) -> nn.Module:
    """Small 2-layer model for error-handling tests."""
    m = nn.Sequential()
    m.add_module("linear_a", nn.Linear(hidden, hidden, bias=True))
    m.add_module("linear_b", nn.Linear(hidden, hidden, bias=False))
    return m


def _build_slab(model: nn.Module, tmp_path: Path) -> tuple[Path, Path]:
    """Build a valid V2 slab and return (safetensors_path, manifest_path)."""
    build_safetensors_slab(
        model=model,
        output_dir=str(tmp_path),
        slab_name="test_slab",
        architecture_id="test",
        pack_k=1,
    )
    st = tmp_path / "test_slab.safetensors"
    mf = tmp_path / "test_slab.manifest.json"
    return st, mf


def _remove_tensor_from_safetensors(
    st_path: Path, key_to_remove: str,
) -> None:
    """Rebuild safetensors file with one tensor removed."""
    from safetensors.torch import load_file, save_file

    tensors = load_file(str(st_path))
    del tensors[key_to_remove]
    save_file(tensors, str(st_path))


def _replace_tensor_in_safetensors(
    st_path: Path, key: str, new_tensor: torch.Tensor,
) -> None:
    """Replace a single tensor in the safetensors file."""
    from safetensors.torch import load_file, save_file

    tensors = load_file(str(st_path))
    tensors[key] = new_tensor
    save_file(tensors, str(st_path))


def _tamper_manifest(mf_path: Path, fn) -> None:
    """Load manifest JSON, apply tampering function, write back."""
    raw = json.loads(mf_path.read_text())
    fn(raw)
    mf_path.write_text(json.dumps(raw, indent=2))


# ── SQG-6.1: Missing slab entry → deterministic fail ─────────────────────


class TestGate101MissingSlab:
    """Missing tensors must be detected, never silently skipped."""

    def test_validate_detects_missing_qweight(self, tmp_path: Path) -> None:
        model = _make_model()
        st, mf = _build_slab(model, tmp_path)
        _remove_tensor_from_safetensors(st, "linear_a.qweight")

        errors = validate_manifest_against_safetensors(mf, st)
        qweight_errors = [e for e in errors if "linear_a.qweight" in e]
        assert len(qweight_errors) > 0, (
            "SQG-6.1-SILENT_MISS: Missing qweight tensor not detected by validator."
        )

    def test_validate_detects_missing_scale(self, tmp_path: Path) -> None:
        model = _make_model()
        st, mf = _build_slab(model, tmp_path)
        _remove_tensor_from_safetensors(st, "linear_a.scale")

        errors = validate_manifest_against_safetensors(mf, st)
        scale_errors = [e for e in errors if "linear_a.scale" in e]
        assert len(scale_errors) > 0, (
            "SQG-6.1-SILENT_MISS: Missing scale tensor not detected by validator."
        )

    def test_validate_detects_missing_zero_point(self, tmp_path: Path) -> None:
        model = _make_model()
        st, mf = _build_slab(model, tmp_path)
        _remove_tensor_from_safetensors(st, "linear_a.zero_point")

        errors = validate_manifest_against_safetensors(mf, st)
        zp_errors = [e for e in errors if "linear_a.zero_point" in e]
        assert len(zp_errors) > 0, (
            "SQG-6.1-SILENT_MISS: Missing zero_point tensor not detected."
        )

    def test_validate_detects_missing_bias(self, tmp_path: Path) -> None:
        model = _make_model()
        st, mf = _build_slab(model, tmp_path)
        _remove_tensor_from_safetensors(st, "linear_a.bias")

        errors = validate_manifest_against_safetensors(mf, st)
        bias_errors = [e for e in errors if "linear_a.bias" in e]
        assert len(bias_errors) > 0, (
            "SQG-6.1-SILENT_MISS: Missing bias tensor not detected."
        )

    def test_validate_returns_all_errors(self, tmp_path: Path) -> None:
        """Validator should report ALL missing tensors, not stop at first."""
        model = _make_model()
        st, mf = _build_slab(model, tmp_path)
        _remove_tensor_from_safetensors(st, "linear_a.qweight")
        _remove_tensor_from_safetensors(st, "linear_b.scale")

        errors = validate_manifest_against_safetensors(mf, st)
        assert len(errors) >= 2, (
            f"SQG-6.1-INCOMPLETE: Expected ≥2 errors, got {len(errors)}. "
            "Validator should report all issues, not short-circuit."
        )

    def test_v2_loader_raises_on_missing_qweight(self, tmp_path: Path) -> None:
        """get_squareq_v2_layers must raise on missing required tensor."""
        from squareq.bridge import get_squareq_v2_layers

        model = _make_model()
        st, mf = _build_slab(model, tmp_path)
        _remove_tensor_from_safetensors(st, "linear_a.qweight")

        with pytest.raises(RuntimeError, match="missing.*tensor|Missing.*tensor"):
            get_squareq_v2_layers(str(st), str(mf))

    def test_v2_loader_raises_on_missing_scale(self, tmp_path: Path) -> None:
        """get_squareq_v2_layers must raise on missing scale."""
        from squareq.bridge import get_squareq_v2_layers

        model = _make_model()
        st, mf = _build_slab(model, tmp_path)
        _remove_tensor_from_safetensors(st, "linear_a.scale")

        with pytest.raises(RuntimeError, match="missing.*tensor|Missing.*tensor"):
            get_squareq_v2_layers(str(st), str(mf))


# ── SQG-6.2: Shape mismatch → hard fail, no coercion ─────────────────────


class TestGate102ShapeMismatch:
    """Shape and dtype mismatches must fail hard, never silently coerce."""

    def test_validate_detects_qweight_shape_mismatch(self, tmp_path: Path) -> None:
        model = _make_model(hidden=32)
        st, mf = _build_slab(model, tmp_path)
        # Replace qweight with wrong-shaped tensor.
        _replace_tensor_in_safetensors(
            st, "linear_a.qweight", torch.zeros(16, 16, dtype=torch.int8),
        )

        errors = validate_manifest_against_safetensors(mf, st)
        shape_errors = [e for e in errors if "shape" in e.lower()]
        assert len(shape_errors) > 0, (
            "SQG-6.2-SILENT_COERCE: Shape mismatch not detected by validator."
        )

    def test_validate_detects_dtype_mismatch(self, tmp_path: Path) -> None:
        model = _make_model(hidden=32)
        st, mf = _build_slab(model, tmp_path)
        # Replace int8 qweight with float32 tensor of same shape.
        from safetensors.torch import load_file

        tensors = load_file(str(st))
        orig_shape = tensors["linear_a.qweight"].shape
        _replace_tensor_in_safetensors(
            st, "linear_a.qweight",
            torch.zeros(orig_shape, dtype=torch.float32),
        )

        errors = validate_manifest_against_safetensors(mf, st)
        dtype_errors = [e for e in errors if "dtype" in e.lower()]
        assert len(dtype_errors) > 0, (
            "SQG-6.2-SILENT_COERCE: Dtype mismatch not detected."
        )

    def test_set_quant_state_rejects_wrong_dtype(self) -> None:
        """QuantLinear.set_quant_state must reject non-int8 qweight."""
        ql = QuantLinear(32, 32)
        with pytest.raises(TypeError, match="int8"):
            ql.set_quant_state(
                qweight=torch.zeros(32, 32, dtype=torch.float32),
                scale=torch.ones(32, dtype=torch.float32),
                zero_point=torch.zeros(32, dtype=torch.float32),
            )

    def test_set_quant_state_rejects_wrong_rows(self) -> None:
        """QuantLinear.set_quant_state must reject mismatched out_features."""
        ql = QuantLinear(32, 32)
        with pytest.raises(ValueError, match="out_features"):
            ql.set_quant_state(
                qweight=torch.zeros(16, 32, dtype=torch.int8),
                scale=torch.ones(16, dtype=torch.float32),
                zero_point=torch.zeros(16, dtype=torch.float32),
            )

    def test_set_quant_state_rejects_wrong_scale_dtype(self) -> None:
        """Scale and zero_point must be float32."""
        ql = QuantLinear(32, 32)
        with pytest.raises(TypeError, match="float32"):
            ql.set_quant_state(
                qweight=torch.zeros(32, 32, dtype=torch.int8),
                scale=torch.ones(32, dtype=torch.float16),
                zero_point=torch.zeros(32, dtype=torch.float32),
            )

    def test_no_silent_reshape(self, tmp_path: Path) -> None:
        """Validator catches shape mismatch — never reshape silently."""
        model = _make_model(hidden=32)
        st, mf = _build_slab(model, tmp_path)
        # Tamper manifest to claim different packed_shape.
        def tamper(raw):
            raw["layers"][0]["packed_shape"] = [64, 64]
        _tamper_manifest(mf, tamper)

        errors = validate_manifest_against_safetensors(mf, st)
        assert len(errors) > 0, (
            "SQG-6.2-SILENT_COERCE: Manifest shape tamper not detected."
        )


# ── SQG-6.3: Corrupted tensor → clear diagnostic ─────────────────────────


class TestGate103CorruptedTensor:
    """Corrupted tensors (NaN, Inf, degenerate) must be detected."""

    def test_validate_detects_nan_in_scale(self, tmp_path: Path) -> None:
        model = _make_model()
        st, mf = _build_slab(model, tmp_path)
        from safetensors.torch import load_file

        tensors = load_file(str(st))
        bad_scale = tensors["linear_a.scale"].clone()
        bad_scale[0] = float("nan")
        _replace_tensor_in_safetensors(st, "linear_a.scale", bad_scale)

        errors = validate_manifest_against_safetensors(mf, st)
        nan_errors = [e for e in errors if "nan" in e.lower() or "NaN" in e]
        assert len(nan_errors) > 0, (
            "SQG-6.3-SILENT_CORRUPT: NaN in scale tensor not detected."
        )

    def test_validate_detects_inf_in_scale(self, tmp_path: Path) -> None:
        model = _make_model()
        st, mf = _build_slab(model, tmp_path)
        from safetensors.torch import load_file

        tensors = load_file(str(st))
        bad_scale = tensors["linear_a.scale"].clone()
        bad_scale[0] = float("inf")
        _replace_tensor_in_safetensors(st, "linear_a.scale", bad_scale)

        errors = validate_manifest_against_safetensors(mf, st)
        inf_errors = [e for e in errors if "inf" in e.lower() or "Inf" in e]
        assert len(inf_errors) > 0, (
            "SQG-6.3-SILENT_CORRUPT: Inf in scale tensor not detected."
        )

    def test_validate_detects_all_zero_scale(self, tmp_path: Path) -> None:
        """All-zero scale means all dequantized weights are zero — degenerate."""
        model = _make_model()
        st, mf = _build_slab(model, tmp_path)
        from safetensors.torch import load_file

        tensors = load_file(str(st))
        zero_scale = torch.zeros_like(tensors["linear_a.scale"])
        _replace_tensor_in_safetensors(st, "linear_a.scale", zero_scale)

        errors = validate_manifest_against_safetensors(mf, st)
        zero_errors = [e for e in errors if "zero" in e.lower() and "scale" in e.lower()]
        assert len(zero_errors) > 0, (
            "SQG-6.3-SILENT_CORRUPT: All-zero scale not detected. "
            "Dequantization would produce all zeros."
        )

    def test_validate_detects_nan_in_zero_point(self, tmp_path: Path) -> None:
        model = _make_model()
        st, mf = _build_slab(model, tmp_path)
        from safetensors.torch import load_file

        tensors = load_file(str(st))
        bad_zp = tensors["linear_a.zero_point"].clone()
        bad_zp[0] = float("nan")
        _replace_tensor_in_safetensors(st, "linear_a.zero_point", bad_zp)

        errors = validate_manifest_against_safetensors(mf, st)
        nan_errors = [e for e in errors if "nan" in e.lower() or "NaN" in e]
        assert len(nan_errors) > 0, (
            "SQG-6.3-SILENT_CORRUPT: NaN in zero_point not detected."
        )


# ── SQG-6 Integration ────────────────────────────────────────────────────


class TestGate104SignatureAndABI:
    """Signature, ABI, and quant-bits guards from Gates 0.5-0.7."""

    def test_signature_mismatch_raises(self, tmp_path: Path) -> None:
        model = _make_model()
        _st, mf = _build_slab(model, tmp_path)
        _tamper_manifest(mf, lambda r: r.update(model_signature="deadbeef"))

        different_model = nn.Sequential(
            nn.Linear(64, 64),
        )
        with pytest.raises(RuntimeError, match="[Ss]ignature"):
            load_and_validate_manifest(mf, model=different_model)

    def test_abi_version_too_high_raises(self, tmp_path: Path) -> None:
        model = _make_model()
        _st, mf = _build_slab(model, tmp_path)
        _tamper_manifest(mf, lambda r: r.update(kernel_abi_version=999))

        with pytest.raises(RuntimeError, match="ABI"):
            load_and_validate_manifest(mf)

    def test_unsupported_quant_bits_raises(self, tmp_path: Path) -> None:
        model = _make_model()
        _st, mf = _build_slab(model, tmp_path)

        def tamper(raw):
            raw["layers"][0]["quant_bits"] = 4

        _tamper_manifest(mf, tamper)

        with pytest.raises(RuntimeError, match="quant.*bits|Unsupported"):
            load_and_validate_manifest(mf)

    def test_valid_manifest_passes(self, tmp_path: Path) -> None:
        model = _make_model()
        _st, mf = _build_slab(model, tmp_path)
        manifest = load_and_validate_manifest(mf, model=model)
        assert manifest.layer_count == 2


class TestGate10Integration:
    """End-to-end error handling integration."""

    def test_build_validate_roundtrip_clean(self, tmp_path: Path) -> None:
        """A freshly built slab passes all validation with zero errors."""
        model = _make_model()
        st, mf = _build_slab(model, tmp_path)
        errors = validate_manifest_against_safetensors(mf, st)
        assert errors == [], (
            f"SQG-6-INT: Fresh slab has validation errors: {errors}"
        )

    def test_tampered_signature_blocks_load(self, tmp_path: Path) -> None:
        """Tampered model signature must block loading."""
        model = _make_model()
        _st, mf = _build_slab(model, tmp_path)
        _tamper_manifest(mf, lambda r: r.update(model_signature="tampered"))

        with pytest.raises(RuntimeError, match="[Ss]ignature"):
            load_and_validate_manifest(mf, model=model)

    def test_multiple_corruption_types_all_detected(self, tmp_path: Path) -> None:
        """Multiple corruptions in a single slab are all reported."""
        model = _make_model()
        st, mf = _build_slab(model, tmp_path)
        from safetensors.torch import load_file

        tensors = load_file(str(st))
        # Corrupt scale with NaN.
        bad_scale = tensors["linear_a.scale"].clone()
        bad_scale[0] = float("nan")
        _replace_tensor_in_safetensors(st, "linear_a.scale", bad_scale)
        # Remove a tensor.
        _remove_tensor_from_safetensors(st, "linear_b.zero_point")

        errors = validate_manifest_against_safetensors(mf, st)
        assert len(errors) >= 2, (
            f"SQG-6-INT: Expected ≥2 errors for multi-corruption, got {len(errors)}."
        )
