"""Gate 2: Residency Adapter — SquareQ ↔ Stagehand Integration Tests.

TDD tests for loading INT8 quant state from a V2 safetensors slab
into scaffolded QuantLinear modules.  This is the bridge between
Stagehand's block lifecycle and QuantLinear.set_quant_state().

All gate codes use the ``SQG-2.x`` prefix for CI triage.

Run::

    python -m pytest serenity/tests/test_squareq_gate2_residency.py -x -v
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from squareq.builder import build_safetensors_slab
from squareq.loader import load_quant_state_from_slab
from squareq.manifest import SlabManifestV2, load_manifest
from squareq.modules import QuantLinear
from squareq.scaffold import prepare_model_for_quantized_streaming

__all__: list[str] = []

_HAS_CUDA = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="CUDA not available")


# ── helpers ────────────────────────────────────────────────────────────────


class _SmallFluxModel(nn.Module):
    """Minimal model: 5 blocks × 3 Linear = 15 layers + proj_out = 16 total."""

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.embed = nn.Embedding(100, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.transformer_blocks = nn.ModuleList()
        for _ in range(5):
            block = nn.Module()
            block.attn = nn.Module()
            block.attn.to_q = nn.Linear(hidden, hidden, bias=False)
            block.attn.to_k = nn.Linear(hidden, hidden, bias=False)
            block.attn.to_v = nn.Linear(hidden, hidden, bias=True)
            self.transformer_blocks.append(block)
        self.proj_out = nn.Linear(hidden, hidden, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.transformer_blocks:
            x = block.attn.to_q(x) + block.attn.to_k(x) + block.attn.to_v(x)
        return self.proj_out(x)


def _build_and_scaffold(tmp_path: Path) -> tuple[nn.Module, SlabManifestV2, Path]:
    """Build slab → scaffold model → return (model, manifest, st_path)."""
    # Build the slab from a source model (with float weights).
    source = _SmallFluxModel()
    output_dir = tmp_path / "slab_output"
    output_dir.mkdir()
    slab_name = "test_bp8"

    build_safetensors_slab(
        model=source,
        output_dir=str(output_dir),
        slab_name=slab_name,
        architecture_id="test_flux",
        pack_k=64,
    )

    st_path = output_dir / f"{slab_name}.safetensors"
    manifest_path = output_dir / f"{slab_name}.manifest.json"
    manifest = load_manifest(manifest_path)

    # Scaffold a fresh model (same architecture, different random weights).
    model = _SmallFluxModel()
    prepare_model_for_quantized_streaming(model, manifest)

    return model, manifest, st_path


def _collect_quantlinear(model: nn.Module) -> dict[str, QuantLinear]:
    """Collect all QuantLinear modules by name."""
    return {
        name: m for name, m in model.named_modules()
        if isinstance(m, QuantLinear)
    }


# ── Gate 2.1: Quant state loading from slab ───────────────────────────────


class TestGate21QuanthStateLoading:
    """SQG-2.1: load_quant_state_from_slab populates all QuantLinear buffers."""

    def test_load_populates_all_quantlinear_buffers(self, tmp_path: Path) -> None:
        model, manifest, st_path = _build_and_scaffold(tmp_path)
        load_quant_state_from_slab(model, manifest, st_path)

        for name, module in _collect_quantlinear(model).items():
            assert module.qweight.numel() > 0, (
                f"SQG-2.1-EMPTY_BUFFER: Module \"{name}\" qweight is empty "
                "after load_quant_state_from_slab."
            )
            assert module.scale.numel() > 0, (
                f"SQG-2.1-EMPTY_BUFFER: Module \"{name}\" scale is empty "
                "after load_quant_state_from_slab."
            )
            assert module.zero_point.numel() > 0, (
                f"SQG-2.1-EMPTY_BUFFER: Module \"{name}\" zero_point is empty "
                "after load_quant_state_from_slab."
            )

    def test_load_returns_count(self, tmp_path: Path) -> None:
        model, manifest, st_path = _build_and_scaffold(tmp_path)
        count = load_quant_state_from_slab(model, manifest, st_path)

        assert isinstance(count, int), (
            "SQG-2.1: load_quant_state_from_slab must return an int."
        )
        assert count > 0, (
            "SQG-2.1: load_quant_state_from_slab returned 0 — nothing loaded."
        )

    def test_load_count_matches_manifest(self, tmp_path: Path) -> None:
        model, manifest, st_path = _build_and_scaffold(tmp_path)
        count = load_quant_state_from_slab(model, manifest, st_path)

        assert count == manifest.layer_count, (
            f"SQG-2.1-COUNT_MISMATCH: Loaded {count} modules but manifest "
            f"has {manifest.layer_count} layers."
        )

    def test_load_idempotent(self, tmp_path: Path) -> None:
        """Loading twice should overwrite cleanly without error."""
        model, manifest, st_path = _build_and_scaffold(tmp_path)
        count1 = load_quant_state_from_slab(model, manifest, st_path)
        count2 = load_quant_state_from_slab(model, manifest, st_path)

        assert count1 == count2, (
            f"SQG-2.1: Idempotent load returned {count1} then {count2}."
        )


# ── Gate 2.2: Shape and dtype correctness ─────────────────────────────────


class TestGate22ShapeDtype:
    """SQG-2.2: Loaded buffers have correct shapes and dtypes."""

    def test_qweight_is_int8(self, tmp_path: Path) -> None:
        model, manifest, st_path = _build_and_scaffold(tmp_path)
        load_quant_state_from_slab(model, manifest, st_path)

        for name, module in _collect_quantlinear(model).items():
            assert module.qweight.dtype == torch.int8, (
                f"SQG-2.2-WRONG_DTYPE: Module \"{name}\" qweight dtype "
                f"{module.qweight.dtype} != expected int8."
            )

    def test_scale_is_float32(self, tmp_path: Path) -> None:
        model, manifest, st_path = _build_and_scaffold(tmp_path)
        load_quant_state_from_slab(model, manifest, st_path)

        for name, module in _collect_quantlinear(model).items():
            assert module.scale.dtype == torch.float32, (
                f"SQG-2.2-WRONG_DTYPE: Module \"{name}\" scale dtype "
                f"{module.scale.dtype} != expected float32."
            )

    def test_zero_point_is_float32(self, tmp_path: Path) -> None:
        model, manifest, st_path = _build_and_scaffold(tmp_path)
        load_quant_state_from_slab(model, manifest, st_path)

        for name, module in _collect_quantlinear(model).items():
            assert module.zero_point.dtype == torch.float32, (
                f"SQG-2.2-WRONG_DTYPE: Module \"{name}\" zero_point dtype "
                f"{module.zero_point.dtype} != expected float32."
            )

    def test_qweight_shape_matches_manifest(self, tmp_path: Path) -> None:
        model, manifest, st_path = _build_and_scaffold(tmp_path)
        load_quant_state_from_slab(model, manifest, st_path)

        layer_lookup = {e["canonical_name"]: e for e in manifest.layers}
        for name, module in _collect_quantlinear(model).items():
            entry = layer_lookup.get(name)
            if entry is None:
                continue
            expected = tuple(entry["packed_shape"])
            actual = tuple(module.qweight.shape)
            assert actual == expected, (
                f"SQG-2.2-SHAPE_MISMATCH: Module \"{name}\" qweight shape "
                f"{actual} != manifest packed_shape {expected}."
            )

    def test_scale_shape_matches_out_features(self, tmp_path: Path) -> None:
        model, manifest, st_path = _build_and_scaffold(tmp_path)
        load_quant_state_from_slab(model, manifest, st_path)

        for name, module in _collect_quantlinear(model).items():
            assert module.scale.shape == (module.out_features,), (
                f"SQG-2.2-SHAPE_MISMATCH: Module \"{name}\" scale shape "
                f"{module.scale.shape} != expected ({module.out_features},)."
            )

    def test_zero_point_shape_matches_out_features(self, tmp_path: Path) -> None:
        model, manifest, st_path = _build_and_scaffold(tmp_path)
        load_quant_state_from_slab(model, manifest, st_path)

        for name, module in _collect_quantlinear(model).items():
            assert module.zero_point.shape == (module.out_features,), (
                f"SQG-2.2-SHAPE_MISMATCH: Module \"{name}\" zero_point shape "
                f"{module.zero_point.shape} != expected ({module.out_features},)."
            )

    def test_padded_in_features_updated(self, tmp_path: Path) -> None:
        """padded_in_features should reflect the K-padded dimension."""
        model, manifest, st_path = _build_and_scaffold(tmp_path)
        load_quant_state_from_slab(model, manifest, st_path)

        layer_lookup = {e["canonical_name"]: e for e in manifest.layers}
        for name, module in _collect_quantlinear(model).items():
            entry = layer_lookup.get(name)
            if entry is None:
                continue
            expected_padded = entry["packed_shape"][1]
            assert module.padded_in_features == expected_padded, (
                f"SQG-2.2: Module \"{name}\" padded_in_features "
                f"{module.padded_in_features} != expected {expected_padded}."
            )


# ── Gate 2.3: Bias handling ───────────────────────────────────────────────


class TestGate23BiasHandling:
    """SQG-2.3: Bias is correctly loaded for layers that have it."""

    def test_bias_populated_for_bias_layers(self, tmp_path: Path) -> None:
        model, manifest, st_path = _build_and_scaffold(tmp_path)
        load_quant_state_from_slab(model, manifest, st_path)

        layer_lookup = {e["canonical_name"]: e for e in manifest.layers}
        for name, module in _collect_quantlinear(model).items():
            entry = layer_lookup.get(name)
            if entry is None:
                continue
            if entry.get("bias_key") is not None:
                assert module.bias is not None, (
                    f"SQG-2.3-BIAS_MISSING: Module \"{name}\" should have "
                    "bias but bias is None."
                )
                assert module.bias.numel() > 0, (
                    f"SQG-2.3-BIAS_MISSING: Module \"{name}\" bias is empty."
                )

    def test_no_bias_for_non_bias_layers(self, tmp_path: Path) -> None:
        model, manifest, st_path = _build_and_scaffold(tmp_path)
        load_quant_state_from_slab(model, manifest, st_path)

        layer_lookup = {e["canonical_name"]: e for e in manifest.layers}
        for name, module in _collect_quantlinear(model).items():
            entry = layer_lookup.get(name)
            if entry is None:
                continue
            if entry.get("bias_key") is None:
                assert module.bias is None, (
                    f"SQG-2.3-UNEXPECTED_BIAS: Module \"{name}\" should not "
                    f"have bias but bias is {module.bias}."
                )

    def test_bias_values_match_original(self, tmp_path: Path) -> None:
        """Bias should be exactly preserved (not quantized)."""
        # Build slab from a known model.
        source = _SmallFluxModel()
        output_dir = tmp_path / "slab_output"
        output_dir.mkdir()
        build_safetensors_slab(
            model=source,
            output_dir=str(output_dir),
            slab_name="test_bp8",
            architecture_id="test_flux",
            pack_k=64,
        )
        st_path = output_dir / "test_bp8.safetensors"
        manifest_path = output_dir / "test_bp8.manifest.json"
        manifest = load_manifest(manifest_path)

        # Record original biases.
        orig_biases: dict[str, torch.Tensor] = {}
        for name, module in source.named_modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                orig_biases[name] = module.bias.data.clone().float()

        # Scaffold + load a fresh model.
        model = _SmallFluxModel()
        prepare_model_for_quantized_streaming(model, manifest)
        load_quant_state_from_slab(model, manifest, st_path)

        # Verify bias values are exactly preserved.
        for name, module in _collect_quantlinear(model).items():
            if name in orig_biases and module.bias is not None:
                torch.testing.assert_close(
                    module.bias.data.float(),
                    orig_biases[name],
                    atol=1e-6,
                    rtol=1e-6,
                    msg=f"SQG-2.3: Bias mismatch for \"{name}\"",
                )


# ── Gate 2.4: Round-trip quantization accuracy ────────────────────────────


class TestGate24RoundTripAccuracy:
    """SQG-2.4: Dequantized weights approximate original within INT8 error."""

    def test_dequantized_weight_close_to_original(self, tmp_path: Path) -> None:
        """(qweight * scale) should approximate original weight."""
        source = _SmallFluxModel()
        output_dir = tmp_path / "slab_output"
        output_dir.mkdir()
        build_safetensors_slab(
            model=source,
            output_dir=str(output_dir),
            slab_name="test_bp8",
            architecture_id="test_flux",
            pack_k=64,
        )
        st_path = output_dir / "test_bp8.safetensors"
        manifest_path = output_dir / "test_bp8.manifest.json"
        manifest = load_manifest(manifest_path)

        # Record original weights.
        orig_weights: dict[str, torch.Tensor] = {}
        for name, module in source.named_modules():
            if isinstance(module, nn.Linear):
                orig_weights[name] = module.weight.data.clone().float()

        # Scaffold + load.
        model = _SmallFluxModel()
        prepare_model_for_quantized_streaming(model, manifest)
        load_quant_state_from_slab(model, manifest, st_path)

        for name, module in _collect_quantlinear(model).items():
            if name not in orig_weights:
                continue
            orig = orig_weights[name]
            out_features, in_features = orig.shape

            # Dequantize: qweight * scale (symmetric, zero_point=0).
            qw = module.qweight[:, :in_features].float()
            dequant = qw * module.scale.unsqueeze(1)

            # Max error per row should be bounded by scale / 2 (rounding).
            row_max_err = (orig - dequant).abs().amax(dim=1)
            row_scale = module.scale
            # Allow up to scale (generous bound for edge cases).
            assert (row_max_err <= row_scale + 1e-6).all(), (
                f"SQG-2.4-ACCURACY: Module \"{name}\" has row errors "
                f"exceeding scale. max_err={row_max_err.max():.6f}, "
                f"max_scale={row_scale.max():.6f}."
            )

    def test_relative_error_reasonable(self, tmp_path: Path) -> None:
        """Overall relative error should be small for random weights."""
        source = _SmallFluxModel()
        output_dir = tmp_path / "slab_output"
        output_dir.mkdir()
        build_safetensors_slab(
            model=source,
            output_dir=str(output_dir),
            slab_name="test_bp8",
            architecture_id="test_flux",
            pack_k=64,
        )
        st_path = output_dir / "test_bp8.safetensors"
        manifest_path = output_dir / "test_bp8.manifest.json"
        manifest = load_manifest(manifest_path)

        orig_weights: dict[str, torch.Tensor] = {}
        for name, module in source.named_modules():
            if isinstance(module, nn.Linear):
                orig_weights[name] = module.weight.data.clone().float()

        model = _SmallFluxModel()
        prepare_model_for_quantized_streaming(model, manifest)
        load_quant_state_from_slab(model, manifest, st_path)

        for name, module in _collect_quantlinear(model).items():
            if name not in orig_weights:
                continue
            orig = orig_weights[name]
            out_features, in_features = orig.shape
            qw = module.qweight[:, :in_features].float()
            dequant = qw * module.scale.unsqueeze(1)

            # RMSE / mean(|orig|) should be < 5% for INT8.
            rmse = (orig - dequant).pow(2).mean().sqrt()
            mean_abs = orig.abs().mean()
            if mean_abs > 1e-6:
                rel_err = rmse / mean_abs
                assert rel_err < 0.05, (
                    f"SQG-2.4-ACCURACY: Module \"{name}\" relative RMSE "
                    f"{rel_err:.4f} > 5%."
                )


# ── Gate 2.5: Device placement ────────────────────────────────────────────


class TestGate25DevicePlacement:
    """SQG-2.5: Loaded tensors are on the requested device."""

    def test_load_to_cpu(self, tmp_path: Path) -> None:
        model, manifest, st_path = _build_and_scaffold(tmp_path)
        load_quant_state_from_slab(model, manifest, st_path, device="cpu")

        for name, module in _collect_quantlinear(model).items():
            assert module.qweight.device.type == "cpu", (
                f"SQG-2.5-DEVICE: Module \"{name}\" qweight on "
                f"{module.qweight.device}, expected cpu."
            )
            assert module.scale.device.type == "cpu", (
                f"SQG-2.5-DEVICE: Module \"{name}\" scale on "
                f"{module.scale.device}, expected cpu."
            )

    @requires_cuda
    def test_load_to_cuda(self, tmp_path: Path) -> None:
        model, manifest, st_path = _build_and_scaffold(tmp_path)
        load_quant_state_from_slab(model, manifest, st_path, device="cuda")

        for name, module in _collect_quantlinear(model).items():
            assert module.qweight.device.type == "cuda", (
                f"SQG-2.5-DEVICE: Module \"{name}\" qweight on "
                f"{module.qweight.device}, expected cuda."
            )
            assert module.scale.device.type == "cuda", (
                f"SQG-2.5-DEVICE: Module \"{name}\" scale on "
                f"{module.scale.device}, expected cuda."
            )


# ── Gate 2 integration ────────────────────────────────────────────────────


class TestGate2Integration:
    """Integration: build → scaffold → load → verify full pipeline."""

    def test_full_pipeline(self, tmp_path: Path) -> None:
        """Build slab → scaffold → load → all QuantLinear modules populated."""
        model, manifest, st_path = _build_and_scaffold(tmp_path)
        count = load_quant_state_from_slab(model, manifest, st_path)

        # Every QuantLinear should have non-empty qweight.
        ql_modules = _collect_quantlinear(model)
        assert count == len(ql_modules), (
            f"SQG-2-INTEGRATION: Loaded {count} but model has "
            f"{len(ql_modules)} QuantLinear modules."
        )
        for name, module in ql_modules.items():
            assert module.qweight.numel() > 0
            assert module.qweight.dtype == torch.int8
            assert module.scale.dtype == torch.float32

    def test_non_target_modules_untouched(self, tmp_path: Path) -> None:
        """Embedding and LayerNorm should be unaffected by the load."""
        model, manifest, st_path = _build_and_scaffold(tmp_path)
        load_quant_state_from_slab(model, manifest, st_path)

        assert isinstance(model.embed, nn.Embedding), (
            "SQG-2-INTEGRATION: model.embed was modified."
        )
        assert isinstance(model.norm, nn.LayerNorm), (
            "SQG-2-INTEGRATION: model.norm was modified."
        )
