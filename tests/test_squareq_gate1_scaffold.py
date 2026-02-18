"""Gate 1: Scaffold Replacement — SquareQ ↔ Stagehand Integration Tests.

TDD tests for the model scaffold replacement step: replacing ``nn.Linear``
modules with empty ``QuantLinear`` instances that await external quant data
from Stagehand at runtime.

All gate codes use the ``SQG-1.x`` prefix for CI triage.

Run::

    python -m pytest serenity/tests/test_squareq_gate1_scaffold.py -x -v
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

# ── imports from the new integration module (TDD — does not exist yet) ──

from squareq.builder import build_safetensors_slab
from squareq.manifest import SlabManifestV2, load_manifest
from squareq.modules import QuantLinear
from squareq.scaffold import prepare_model_for_quantized_streaming

__all__: list[str] = []

_HAS_CUDA = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="CUDA not available")


# ── helpers ────────────────────────────────────────────────────────────────


class _SmallFluxModel(nn.Module):
    """Minimal model mimicking Flux layer naming.

    5 blocks with 3 Linear layers each = 15 quantizable layers.
    Plus a LayerNorm and Embedding that should NOT be replaced.
    """

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


def _build_test_slab(tmp_path: Path) -> tuple[Path, Path]:
    """Build a slab and return (safetensors_path, manifest_path)."""
    model = _SmallFluxModel()
    output_dir = tmp_path / "slab_output"
    output_dir.mkdir()
    slab_name = "test_model_bp8"
    build_safetensors_slab(
        model=model,
        output_dir=str(output_dir),
        slab_name=slab_name,
        architecture_id="test_flux",
        pack_k=64,
    )
    return (
        output_dir / f"{slab_name}.safetensors",
        output_dir / f"{slab_name}.manifest.json",
    )


def _count_modules_of_type(model: nn.Module, target_type: type) -> int:
    """Count modules of a specific type in the model tree."""
    return sum(1 for _, m in model.named_modules() if isinstance(m, target_type))


def _collect_modules_of_type(
    model: nn.Module, target_type: type,
) -> dict[str, nn.Module]:
    """Collect all modules of a specific type with their full names."""
    return {
        name: m for name, m in model.named_modules()
        if isinstance(m, target_type)
    }


# ── Gate 1.1: Scaffold produces negligible GPU allocation ──────────────────


class TestGate11ScaffoldAllocation:
    """SQG-1.1: prepare_model_for_quantized_streaming() uses < 1MB GPU."""

    @requires_cuda
    def test_negligible_gpu_delta(self, tmp_path: Path) -> None:
        _, manifest_path = _build_test_slab(tmp_path)
        manifest = load_manifest(manifest_path)
        model = _SmallFluxModel()

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()

        mapping = prepare_model_for_quantized_streaming(model, manifest)

        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        delta = peak - baseline

        threshold = 1 * 1024 * 1024  # 1 MB
        assert delta <= threshold, (
            f"SQG-1.1-SCAFFOLD_ALLOC: {delta} bytes GPU memory delta during "
            f"scaffold; threshold=1MB ({threshold} bytes)."
        )

    def test_no_float_weight_parameters_after_swap(self, tmp_path: Path) -> None:
        """No module should hold a float weight Parameter after scaffold."""
        _, manifest_path = _build_test_slab(tmp_path)
        manifest = load_manifest(manifest_path)
        model = _SmallFluxModel()
        mapping = prepare_model_for_quantized_streaming(model, manifest)

        for name, module in model.named_modules():
            if isinstance(module, QuantLinear):
                # QuantLinear should NOT have a .weight Parameter.
                assert not hasattr(module, "weight") or module.weight is None or (
                    hasattr(module.weight, "dtype") and module.weight.dtype == torch.int8
                ), (
                    f"SQG-1.1-SCAFFOLD_ALLOC: Module \"{name}\" has a float "
                    f"weight parameter after scaffold swap."
                )

    def test_mapping_returned(self, tmp_path: Path) -> None:
        """The function should return a mapping of replaced module names."""
        _, manifest_path = _build_test_slab(tmp_path)
        manifest = load_manifest(manifest_path)
        model = _SmallFluxModel()
        mapping = prepare_model_for_quantized_streaming(model, manifest)

        assert isinstance(mapping, dict), (
            "SQG-1.1: prepare_model_for_quantized_streaming must return a dict mapping."
        )
        assert len(mapping) > 0, (
            "SQG-1.1: Mapping is empty — no modules were replaced."
        )


# ── Gate 1.2: Module type correctness ──────────────────────────────────────


class TestGate12ModuleTypeCorrectness:
    """SQG-1.2: All target nn.Linear modules become QuantLinear."""

    def test_all_quantized_scope_is_quantlinear(self, tmp_path: Path) -> None:
        _, manifest_path = _build_test_slab(tmp_path)
        manifest = load_manifest(manifest_path)
        raw = json.loads(manifest_path.read_text())
        manifest_names = {e["canonical_name"] for e in raw.get("layers", [])}

        model = _SmallFluxModel()
        prepare_model_for_quantized_streaming(model, manifest)

        for name, module in model.named_modules():
            if name in manifest_names:
                assert isinstance(module, QuantLinear), (
                    f"SQG-1.2-WRONG_TYPE: Module \"{name}\" is "
                    f"{type(module).__name__}, expected QuantLinear."
                )

    def test_no_nn_linear_in_quantized_scope(self, tmp_path: Path) -> None:
        _, manifest_path = _build_test_slab(tmp_path)
        manifest = load_manifest(manifest_path)
        raw = json.loads(manifest_path.read_text())
        manifest_names = {e["canonical_name"] for e in raw.get("layers", [])}

        model = _SmallFluxModel()
        prepare_model_for_quantized_streaming(model, manifest)

        for name, module in model.named_modules():
            if name in manifest_names:
                assert not isinstance(module, nn.Linear) or isinstance(module, QuantLinear), (
                    f"SQG-1.2-WRONG_TYPE: Module \"{name}\" is still nn.Linear. "
                    "Should have been replaced with QuantLinear."
                )

    def test_non_target_modules_untouched(self, tmp_path: Path) -> None:
        """LayerNorm, Embedding, and non-quantized modules should be unchanged."""
        _, manifest_path = _build_test_slab(tmp_path)
        manifest = load_manifest(manifest_path)
        raw = json.loads(manifest_path.read_text())
        manifest_names = {e["canonical_name"] for e in raw.get("layers", [])}

        model = _SmallFluxModel()
        # Record non-target types before.
        before_types: dict[str, type] = {}
        for name, module in model.named_modules():
            if name not in manifest_names and name:
                before_types[name] = type(module)

        prepare_model_for_quantized_streaming(model, manifest)

        # Verify non-target types are unchanged.
        for name, module in model.named_modules():
            if name in before_types:
                assert type(module) == before_types[name], (
                    f"SQG-1.2-COLLATERAL: Non-target module \"{name}\" changed "
                    f"from {before_types[name].__name__} to {type(module).__name__}."
                )

    def test_embedding_still_embedding(self, tmp_path: Path) -> None:
        _, manifest_path = _build_test_slab(tmp_path)
        manifest = load_manifest(manifest_path)
        model = _SmallFluxModel()
        prepare_model_for_quantized_streaming(model, manifest)

        assert isinstance(model.embed, nn.Embedding), (
            f"SQG-1.2-COLLATERAL: model.embed is {type(model.embed).__name__}, "
            "expected nn.Embedding."
        )

    def test_layernorm_still_layernorm(self, tmp_path: Path) -> None:
        _, manifest_path = _build_test_slab(tmp_path)
        manifest = load_manifest(manifest_path)
        model = _SmallFluxModel()
        prepare_model_for_quantized_streaming(model, manifest)

        assert isinstance(model.norm, nn.LayerNorm), (
            f"SQG-1.2-COLLATERAL: model.norm is {type(model.norm).__name__}, "
            "expected nn.LayerNorm."
        )


# ── Gate 1.3: Empty QuantLinear rejects forward ────────────────────────────


class TestGate13EmptyQuantLinearForward:
    """SQG-1.3: Forward on empty QuantLinear raises RuntimeError immediately."""

    def test_empty_forward_raises(self) -> None:
        module = QuantLinear(in_features=64, out_features=64, bias=False)
        x = torch.randn(1, 64)

        with pytest.raises(RuntimeError, match="[Bb]efore.*state.*set|[Bb]efore.*quant"):
            module.forward(x)

    def test_empty_forward_does_not_produce_output(self) -> None:
        """Verify it raises, not silently returns garbage."""
        module = QuantLinear(in_features=128, out_features=64, bias=True)
        x = torch.randn(2, 128)

        raised = False
        try:
            _ = module(x)
        except RuntimeError:
            raised = True

        assert raised, (
            "SQG-1.3-SILENT_EMPTY: Empty QuantLinear did not raise on forward. "
            "Silent corruption possible."
        )

    @requires_cuda
    def test_empty_forward_on_cuda_raises(self) -> None:
        """Even on CUDA, empty QuantLinear should reject forward."""
        module = QuantLinear(in_features=64, out_features=64, bias=False).cuda()
        x = torch.randn(1, 64, device="cuda")

        with pytest.raises(RuntimeError):
            module(x)

    def test_after_scaffold_forward_raises(self, tmp_path: Path) -> None:
        """Scaffolded model's QuantLinear modules should reject forward."""
        _, manifest_path = _build_test_slab(tmp_path)
        manifest = load_manifest(manifest_path)
        model = _SmallFluxModel()
        prepare_model_for_quantized_streaming(model, manifest)

        # Find a QuantLinear and try to forward through it.
        for name, module in model.named_modules():
            if isinstance(module, QuantLinear):
                x = torch.randn(1, module.in_features)
                with pytest.raises(RuntimeError):
                    module(x)
                break  # One is enough.


# ── Gate 1.4: Canonical-name mapping bijection ─────────────────────────────


class TestGate14CanonicalNameBijection:
    """SQG-1.4: Manifest names ↔ model QuantLinear names is a 1:1 mapping."""

    def test_bijection(self, tmp_path: Path) -> None:
        _, manifest_path = _build_test_slab(tmp_path)
        manifest = load_manifest(manifest_path)
        raw = json.loads(manifest_path.read_text())
        manifest_names = {e["canonical_name"] for e in raw.get("layers", [])}

        model = _SmallFluxModel()
        prepare_model_for_quantized_streaming(model, manifest)

        model_names = {
            name for name, m in model.named_modules()
            if isinstance(m, QuantLinear)
        }

        in_manifest_only = manifest_names - model_names
        in_model_only = model_names - manifest_names

        assert manifest_names == model_names, (
            f"SQG-1.4-NAME_COLLISION: Canonical mapping mismatch. "
            f"In manifest only: {sorted(in_manifest_only)}. "
            f"In model only: {sorted(in_model_only)}."
        )

    def test_no_duplicate_canonical_names(self, tmp_path: Path) -> None:
        _, manifest_path = _build_test_slab(tmp_path)
        raw = json.loads(manifest_path.read_text())
        names = [e["canonical_name"] for e in raw.get("layers", [])]
        assert len(names) == len(set(names)), (
            f"SQG-1.4-NAME_COLLISION: Duplicate canonical names in manifest: "
            f"{[n for n in names if names.count(n) > 1]}"
        )

    def test_correct_dimensions_preserved(self, tmp_path: Path) -> None:
        """QuantLinear dimensions should match the original nn.Linear."""
        _, manifest_path = _build_test_slab(tmp_path)
        manifest = load_manifest(manifest_path)
        raw = json.loads(manifest_path.read_text())

        # Record original dimensions.
        model_before = _SmallFluxModel()
        orig_dims: dict[str, tuple[int, int, bool]] = {}
        for name, module in model_before.named_modules():
            if isinstance(module, nn.Linear):
                orig_dims[name] = (
                    module.in_features,
                    module.out_features,
                    module.bias is not None,
                )

        # Do scaffold swap.
        model = _SmallFluxModel()
        prepare_model_for_quantized_streaming(model, manifest)

        # Verify dimensions.
        for name, module in model.named_modules():
            if isinstance(module, QuantLinear) and name in orig_dims:
                orig_in, orig_out, orig_bias = orig_dims[name]
                assert module.in_features == orig_in, (
                    f"SQG-1.4: \"{name}\" in_features {module.in_features} != "
                    f"original {orig_in}"
                )
                assert module.out_features == orig_out, (
                    f"SQG-1.4: \"{name}\" out_features {module.out_features} != "
                    f"original {orig_out}"
                )
                has_bias = module.bias is not None
                assert has_bias == orig_bias, (
                    f"SQG-1.4: \"{name}\" bias={has_bias} != original bias={orig_bias}"
                )


# ── Gate 1 integration: scaffold → QuantLinear count ───────────────────────


class TestGate1Integration:
    """Integration: scaffold replaces exactly the right number of modules."""

    def test_replaced_count_matches_manifest(self, tmp_path: Path) -> None:
        _, manifest_path = _build_test_slab(tmp_path)
        manifest = load_manifest(manifest_path)
        raw = json.loads(manifest_path.read_text())
        expected_count = len(raw["layers"])

        model = _SmallFluxModel()
        mapping = prepare_model_for_quantized_streaming(model, manifest)

        actual_count = _count_modules_of_type(model, QuantLinear)
        assert actual_count == expected_count, (
            f"SQG-1-INTEGRATION: Expected {expected_count} QuantLinear modules, "
            f"found {actual_count}."
        )

    def test_idempotent_scaffold(self, tmp_path: Path) -> None:
        """Running scaffold twice should not change module count."""
        _, manifest_path = _build_test_slab(tmp_path)
        manifest = load_manifest(manifest_path)

        model = _SmallFluxModel()
        prepare_model_for_quantized_streaming(model, manifest)
        count_first = _count_modules_of_type(model, QuantLinear)

        # Second call should be a no-op (modules already replaced).
        prepare_model_for_quantized_streaming(model, manifest)
        count_second = _count_modules_of_type(model, QuantLinear)

        assert count_first == count_second, (
            f"SQG-1-INTEGRATION: Scaffold not idempotent. "
            f"First pass: {count_first}, second pass: {count_second}."
        )

    def test_scaffold_preserves_module_hierarchy(self, tmp_path: Path) -> None:
        """The model's module tree structure should be preserved."""
        _, manifest_path = _build_test_slab(tmp_path)
        manifest = load_manifest(manifest_path)

        model = _SmallFluxModel()

        # Record hierarchy before.
        children_before = {
            name: list(m.named_children())
            for name, m in model.named_modules()
        }

        prepare_model_for_quantized_streaming(model, manifest)

        # Verify parent-child relationships are preserved (same child names).
        for name, module in model.named_modules():
            if name in children_before:
                child_names_now = [c[0] for c in module.named_children()]
                child_names_before = [c[0] for c in children_before[name]]
                assert child_names_now == child_names_before, (
                    f"SQG-1-INTEGRATION: Module \"{name}\" children changed. "
                    f"Before: {child_names_before}, after: {child_names_now}."
                )
