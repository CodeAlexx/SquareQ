"""Gate 3: No-Materialization — SquareQ ↔ Stagehand Integration Tests.

TDD tests verifying that no full-precision float weight tensor ever
materializes during the quant streaming lifecycle.  INT8 data must
flow from safetensors → QuantLinear buffers without float intermediates.

All gate codes use the ``SQG-3.x`` prefix for CI triage.

Run::

    python -m pytest serenity/tests/test_squareq_gate3_no_materialization.py -x -v
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from squareq.builder import build_safetensors_slab
from squareq.loader import load_quant_state_from_slab
from squareq.manifest import load_manifest
from squareq.modules import QuantLinear
from squareq.scaffold import prepare_model_for_quantized_streaming

__all__: list[str] = []

_HAS_CUDA = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="CUDA not available")


# ── helpers ────────────────────────────────────────────────────────────────


class _SmallFluxModel(nn.Module):
    """Minimal model: 5 blocks × 3 Linear = 15 layers + proj_out = 16."""

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


def _build_scaffold_load(tmp_path: Path) -> nn.Module:
    """Build slab → scaffold → load quant state → return model."""
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

    model = _SmallFluxModel()
    prepare_model_for_quantized_streaming(model, manifest)
    load_quant_state_from_slab(model, manifest, st_path)
    return model


def _collect_quantlinear(model: nn.Module) -> dict[str, QuantLinear]:
    return {
        name: m for name, m in model.named_modules()
        if isinstance(m, QuantLinear)
    }


# ── Gate 3.1: No float weight Parameter ───────────────────────────────────


class TestGate31NoFloatWeightParam:
    """SQG-3.1: QuantLinear holds no float .weight Parameter."""

    def test_no_weight_attribute(self, tmp_path: Path) -> None:
        """QuantLinear should NOT have a .weight attribute at all."""
        model = _build_scaffold_load(tmp_path)

        for name, module in _collect_quantlinear(model).items():
            has_weight = (
                hasattr(module, "weight")
                and module.weight is not None
                and isinstance(module.weight, (nn.Parameter, torch.Tensor))
                and module.weight.dtype in (torch.float32, torch.float16, torch.bfloat16)
            )
            assert not has_weight, (
                f"SQG-3.1-FLOAT_WEIGHT: Module \"{name}\" has a float .weight "
                f"attribute of dtype {module.weight.dtype}. "
                "QuantLinear must not hold float weights."
            )

    def test_named_parameters_no_float_weight(self, tmp_path: Path) -> None:
        """named_parameters() should not yield any float 'weight' tensor."""
        model = _build_scaffold_load(tmp_path)

        ql_names = set(_collect_quantlinear(model).keys())
        for param_name, param in model.named_parameters():
            # Check if this param belongs to a QuantLinear and is a weight.
            for ql_name in ql_names:
                if param_name.startswith(ql_name) and param_name.endswith(".weight"):
                    assert param.dtype not in (torch.float32, torch.float16, torch.bfloat16), (
                        f"SQG-3.1-FLOAT_WEIGHT: Parameter \"{param_name}\" is "
                        f"float dtype {param.dtype} inside a QuantLinear."
                    )

    def test_quantlinear_parameters_are_bias_only(self, tmp_path: Path) -> None:
        """QuantLinear.parameters() should yield only bias (if any)."""
        model = _build_scaffold_load(tmp_path)

        for name, module in _collect_quantlinear(model).items():
            param_names = [n for n, _ in module.named_parameters()]
            for pn in param_names:
                assert pn == "bias", (
                    f"SQG-3.1-UNEXPECTED_PARAM: Module \"{name}\" has "
                    f"parameter \"{pn}\" — only 'bias' expected."
                )


# ── Gate 3.2: Memory footprint is INT8, not float ─────────────────────────


class TestGate32MemoryFootprint:
    """SQG-3.2: Total buffer memory reflects INT8 storage, not float."""

    def test_buffer_memory_is_int8_based(self, tmp_path: Path) -> None:
        """Per-module buffer bytes should be INT8 + scale/zero overhead."""
        model = _build_scaffold_load(tmp_path)

        for name, module in _collect_quantlinear(model).items():
            # qweight: out × padded_in × 1 byte (int8).
            qw_bytes = module.qweight.numel() * module.qweight.element_size()
            assert module.qweight.element_size() == 1, (
                f"SQG-3.2-FLOAT_STORAGE: Module \"{name}\" qweight element_size "
                f"is {module.qweight.element_size()}, expected 1 (int8)."
            )

            # scale + zero_point: out × 4 bytes each (float32).
            scale_bytes = module.scale.numel() * module.scale.element_size()
            zp_bytes = module.zero_point.numel() * module.zero_point.element_size()
            total = qw_bytes + scale_bytes + zp_bytes

            # Float16 equivalent would be: out × padded_in × 2 bytes.
            float16_bytes = module.qweight.shape[0] * module.qweight.shape[1] * 2

            assert total < float16_bytes, (
                f"SQG-3.2-FLOAT_STORAGE: Module \"{name}\" buffer memory "
                f"{total} bytes >= float16 equivalent {float16_bytes} bytes. "
                "INT8 storage should be smaller."
            )

    def test_total_model_memory_smaller_than_float(self, tmp_path: Path) -> None:
        """Total QuantLinear buffer memory should be ~4x smaller than float32."""
        model = _build_scaffold_load(tmp_path)

        total_quant_bytes = 0
        total_float32_bytes = 0

        for name, module in _collect_quantlinear(model).items():
            # Actual quant storage.
            total_quant_bytes += module.qweight.numel() * 1  # int8
            total_quant_bytes += module.scale.numel() * 4     # float32
            total_quant_bytes += module.zero_point.numel() * 4  # float32

            # Float32 equivalent.
            total_float32_bytes += (
                module.out_features * module.padded_in_features * 4
            )

        ratio = total_float32_bytes / total_quant_bytes if total_quant_bytes > 0 else 0
        assert ratio > 2.0, (
            f"SQG-3.2-FLOAT_STORAGE: Compression ratio {ratio:.2f}x is too "
            "low. INT8 + overhead should be at least 2x smaller than float32."
        )


# ── Gate 3.3: .to() dtype stripping ───────────────────────────────────────


class TestGate33DtypeStripping:
    """SQG-3.3: model.to(dtype=...) does NOT promote INT8 buffers."""

    def test_to_bfloat16_preserves_int8(self, tmp_path: Path) -> None:
        model = _build_scaffold_load(tmp_path)
        model.to(dtype=torch.bfloat16)

        for name, module in _collect_quantlinear(model).items():
            assert module.qweight.dtype == torch.int8, (
                f"SQG-3.3-DTYPE_PROMOTION: Module \"{name}\" qweight became "
                f"{module.qweight.dtype} after model.to(dtype=bfloat16). "
                "INT8 must be preserved."
            )

    def test_to_float16_preserves_int8(self, tmp_path: Path) -> None:
        model = _build_scaffold_load(tmp_path)
        model.to(dtype=torch.float16)

        for name, module in _collect_quantlinear(model).items():
            assert module.qweight.dtype == torch.int8, (
                f"SQG-3.3-DTYPE_PROMOTION: Module \"{name}\" qweight became "
                f"{module.qweight.dtype} after model.to(dtype=float16). "
                "INT8 must be preserved."
            )

    def test_to_float32_preserves_int8(self, tmp_path: Path) -> None:
        model = _build_scaffold_load(tmp_path)
        model.to(dtype=torch.float32)

        for name, module in _collect_quantlinear(model).items():
            assert module.qweight.dtype == torch.int8, (
                f"SQG-3.3-DTYPE_PROMOTION: Module \"{name}\" qweight became "
                f"{module.qweight.dtype} after model.to(dtype=float32). "
                "INT8 must be preserved."
            )

    def test_scale_preserved_through_dtype_cast(self, tmp_path: Path) -> None:
        """scale/zero_point should stay float32 even after .to(bfloat16)."""
        model = _build_scaffold_load(tmp_path)
        model.to(dtype=torch.bfloat16)

        for name, module in _collect_quantlinear(model).items():
            assert module.scale.dtype == torch.float32, (
                f"SQG-3.3-DTYPE_PROMOTION: Module \"{name}\" scale became "
                f"{module.scale.dtype} after model.to(dtype=bfloat16)."
            )
            assert module.zero_point.dtype == torch.float32, (
                f"SQG-3.3-DTYPE_PROMOTION: Module \"{name}\" zero_point became "
                f"{module.zero_point.dtype} after model.to(dtype=bfloat16)."
            )

    @requires_cuda
    def test_to_cuda_preserves_dtypes(self, tmp_path: Path) -> None:
        """model.to('cuda') moves but preserves all dtypes."""
        model = _build_scaffold_load(tmp_path)
        model.to("cuda")

        for name, module in _collect_quantlinear(model).items():
            assert module.qweight.dtype == torch.int8, (
                f"SQG-3.3-DTYPE_PROMOTION: Module \"{name}\" qweight dtype "
                f"changed to {module.qweight.dtype} on .to('cuda')."
            )
            assert module.qweight.device.type == "cuda", (
                f"SQG-3.3-DEVICE: Module \"{name}\" qweight not on cuda."
            )
            assert module.scale.dtype == torch.float32, (
                f"SQG-3.3-DTYPE_PROMOTION: Module \"{name}\" scale dtype "
                f"changed to {module.scale.dtype} on .to('cuda')."
            )


# ── Gate 3.4: No float intermediates in load path ─────────────────────────


class TestGate34NoFloatIntermediates:
    """SQG-3.4: The load path never creates a float weight tensor."""

    def test_buffers_only_expected_dtypes(self, tmp_path: Path) -> None:
        """QuantLinear buffers should be exactly: int8, float32, float32."""
        model = _build_scaffold_load(tmp_path)

        expected_dtypes = {
            "qweight": torch.int8,
            "scale": torch.float32,
            "zero_point": torch.float32,
        }

        for name, module in _collect_quantlinear(model).items():
            for buf_name, buf in module.named_buffers():
                if buf_name in expected_dtypes:
                    assert buf.dtype == expected_dtypes[buf_name], (
                        f"SQG-3.4-FLOAT_INTERMEDIATE: Module \"{name}\" "
                        f"buffer \"{buf_name}\" dtype {buf.dtype} != "
                        f"expected {expected_dtypes[buf_name]}."
                    )

    def test_no_dangling_float_tensors(self, tmp_path: Path) -> None:
        """After scaffold + load, no QuantLinear should hold a float tensor
        that looks like a full weight matrix.
        """
        model = _build_scaffold_load(tmp_path)

        for name, module in _collect_quantlinear(model).items():
            # Check all attributes on the module.
            for attr_name in dir(module):
                if attr_name.startswith("_"):
                    continue
                try:
                    attr = getattr(module, attr_name)
                except Exception:
                    continue
                if not isinstance(attr, torch.Tensor):
                    continue
                # A float tensor with shape (out, in) would be a leaked weight.
                if (
                    attr.dtype in (torch.float32, torch.float16, torch.bfloat16)
                    and attr.ndim == 2
                    and attr.shape[0] == module.out_features
                    and attr.shape[1] >= module.in_features
                ):
                    assert False, (
                        f"SQG-3.4-FLOAT_INTERMEDIATE: Module \"{name}\" has "
                        f"a 2D float tensor \"{attr_name}\" with shape "
                        f"{tuple(attr.shape)} that looks like a leaked weight."
                    )


# ── Gate 3 integration ────────────────────────────────────────────────────


class TestGate3Integration:
    """Integration: full pipeline preserves INT8 invariant."""

    def test_scaffold_only_no_float_weights(self, tmp_path: Path) -> None:
        """After scaffold (before load), no float weight should exist."""
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
        manifest = load_manifest(output_dir / "test_bp8.manifest.json")

        model = _SmallFluxModel()
        prepare_model_for_quantized_streaming(model, manifest)

        for name, module in _collect_quantlinear(model).items():
            # After scaffold, qweight should be empty (size 0).
            assert module.qweight.numel() == 0, (
                f"SQG-3-INTEGRATION: Module \"{name}\" qweight is non-empty "
                "after scaffold (before load). Should be empty."
            )
            # No float weight attribute.
            assert not hasattr(module, "weight") or module.weight is None, (
                f"SQG-3-INTEGRATION: Module \"{name}\" has .weight after scaffold."
            )

    def test_after_load_all_invariants_hold(self, tmp_path: Path) -> None:
        """After full pipeline: INT8 buffers, no float weights, correct dtypes."""
        model = _build_scaffold_load(tmp_path)
        ql_modules = _collect_quantlinear(model)

        assert len(ql_modules) > 0, "No QuantLinear modules found."

        for name, module in ql_modules.items():
            # qweight is int8 and non-empty.
            assert module.qweight.dtype == torch.int8
            assert module.qweight.numel() > 0
            # scale is float32.
            assert module.scale.dtype == torch.float32
            # zero_point is float32.
            assert module.zero_point.dtype == torch.float32
            # No float .weight parameter.
            for pn, _ in module.named_parameters():
                assert pn != "weight", (
                    f"SQG-3-INTEGRATION: Module \"{name}\" has a 'weight' parameter."
                )

    @requires_cuda
    def test_gpu_round_trip_preserves_int8(self, tmp_path: Path) -> None:
        """CPU → GPU → CPU round-trip preserves INT8 dtypes."""
        model = _build_scaffold_load(tmp_path)
        model.to("cuda")

        for name, module in _collect_quantlinear(model).items():
            assert module.qweight.dtype == torch.int8
            assert module.qweight.device.type == "cuda"

        model.to("cpu")

        for name, module in _collect_quantlinear(model).items():
            assert module.qweight.dtype == torch.int8
            assert module.qweight.device.type == "cpu"
