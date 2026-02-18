"""Gate 0: Format Unification — SquareQ ↔ Stagehand Integration Tests.

TDD tests for the new safetensors-only slab format.  These define the
contract that the SquareQ slab builder must satisfy before any further
integration work can proceed.

All gate codes use the ``SQG-0.x`` prefix for CI triage.

Run::

    python -m pytest serenity/tests/test_squareq_gate0_format.py -x -v
"""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

# ── imports from the new integration module (TDD — does not exist yet) ──
# These will fail until the implementation is written.

from squareq.builder import build_safetensors_slab
from squareq.manifest import (
    CURRENT_KERNEL_ABI_VERSION,
    SUPPORTED_QUANT_BITS,
    SlabManifestV2,
    load_manifest,
    validate_manifest_against_safetensors,
)
from squareq.modules import QuantLinear

__all__: list[str] = []

# ── constants ──────────────────────────────────────────────────────────────

# Required fields per layer entry in the manifest.
REQUIRED_LAYER_FIELDS = {
    "canonical_name": str,
    "qweight_key": str,
    "scale_key": str,
    "zero_point_key": str,
    "bias_key": (str, type(None)),
    "orig_shape": list,
    "packed_shape": list,
    "pad_k": int,
    "quant_bits": int,
    "quant_scheme": str,
    "quant_axis": int,
    "dtype_qweight": str,
    "dtype_scale": str,
    "dtype_zero_point": str,
    "block_id": (str, type(None)),
}

# Required top-level fields in the manifest.
REQUIRED_TOP_LEVEL_FIELDS = {
    "model_signature": str,
    "architecture_id": str,
    "quant_version": str,
    "kernel_abi_version": int,
    "min_runtime_version": str,
    "layer_count": int,
    "total_qweight_bytes": int,
}


# ── helpers ────────────────────────────────────────────────────────────────


class _SmallFluxModel(nn.Module):
    """Minimal model mimicking Flux layer naming for test purposes.

    5 blocks with 3 Linear layers each = 15 quantizable layers.
    """

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList()
        for _ in range(5):
            block = nn.Module()
            block.attn = nn.Module()
            block.attn.to_q = nn.Linear(hidden, hidden, bias=False)
            block.attn.to_k = nn.Linear(hidden, hidden, bias=False)
            block.attn.to_v = nn.Linear(hidden, hidden, bias=True)
            self.transformer_blocks.append(block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.transformer_blocks:
            x = block.attn.to_q(x) + block.attn.to_k(x) + block.attn.to_v(x)
        return x


def _build_test_slab(tmp_path: Path) -> tuple[Path, Path]:
    """Build a slab from _SmallFluxModel and return (safetensors_path, manifest_path)."""
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

    safetensors_path = output_dir / f"{slab_name}.safetensors"
    manifest_path = output_dir / f"{slab_name}.manifest.json"
    return safetensors_path, manifest_path


# ── Gate 0.1: Safetensors-only slab output ─────────────────────────────────


class TestGate01SafetensorsOnly:
    """SQG-0.1: Builder produces .safetensors + .manifest.json, no .fpk."""

    def test_output_files_exist(self, tmp_path: Path) -> None:
        st_path, manifest_path = _build_test_slab(tmp_path)
        assert st_path.exists(), (
            f"SQG-0.1-MISSING_SAFETENSORS: Expected safetensors at {st_path}"
        )
        assert manifest_path.exists(), (
            f"SQG-0.1-MISSING_MANIFEST: Expected manifest at {manifest_path}"
        )

    def test_no_fpk_artifacts(self, tmp_path: Path) -> None:
        st_path, _ = _build_test_slab(tmp_path)
        output_dir = st_path.parent
        fpk_files = list(output_dir.glob("*.fpk"))
        assert len(fpk_files) == 0, (
            f"SQG-0.1-LEGACY_FORMAT: Found .fpk output at {fpk_files[0]}. "
            "Builder must emit safetensors only."
        )

    def test_no_torch_save_artifacts(self, tmp_path: Path) -> None:
        st_path, _ = _build_test_slab(tmp_path)
        output_dir = st_path.parent
        pt_files = list(output_dir.glob("*.pt")) + list(output_dir.glob("*.pth"))
        assert len(pt_files) == 0, (
            f"SQG-0.1-LEGACY_FORMAT: Found torch.save output at {pt_files[0]}. "
            "Builder must emit safetensors only."
        )

    def test_exactly_two_output_files(self, tmp_path: Path) -> None:
        """Output dir should contain exactly the safetensors + manifest."""
        st_path, _ = _build_test_slab(tmp_path)
        output_dir = st_path.parent
        output_files = sorted(f.name for f in output_dir.iterdir() if f.is_file())
        expected = sorted(["test_model_bp8.safetensors", "test_model_bp8.manifest.json"])
        assert output_files == expected, (
            f"SQG-0.1-EXTRA_FILES: Expected {expected}, got {output_files}"
        )


# ── Gate 0.2: Manifest schema completeness ─────────────────────────────────


class TestGate02ManifestSchema:
    """SQG-0.2: Every manifest entry has all required fields with correct types."""

    def test_top_level_fields_present(self, tmp_path: Path) -> None:
        _, manifest_path = _build_test_slab(tmp_path)
        manifest = load_manifest(manifest_path)
        raw = json.loads(manifest_path.read_text())

        for field_name, expected_type in REQUIRED_TOP_LEVEL_FIELDS.items():
            assert field_name in raw, (
                f"SQG-0.2-MISSING_FIELD: Top-level missing field \"{field_name}\". "
                "Manifest schema incomplete."
            )
            assert isinstance(raw[field_name], expected_type), (
                f"SQG-0.2-WRONG_TYPE: Top-level field \"{field_name}\" "
                f"is {type(raw[field_name]).__name__}, expected {expected_type.__name__}"
            )

    def test_layer_entries_have_all_fields(self, tmp_path: Path) -> None:
        _, manifest_path = _build_test_slab(tmp_path)
        raw = json.loads(manifest_path.read_text())
        layers = raw.get("layers", [])
        assert len(layers) > 0, "SQG-0.2-NO_LAYERS: Manifest contains no layer entries."

        for entry in layers:
            name = entry.get("canonical_name", "<unknown>")
            for field_name, expected_type in REQUIRED_LAYER_FIELDS.items():
                assert field_name in entry, (
                    f"SQG-0.2-MISSING_FIELD: Layer \"{name}\" missing field "
                    f"\"{field_name}\". Manifest schema incomplete."
                )
                if isinstance(expected_type, tuple):
                    assert isinstance(entry[field_name], expected_type), (
                        f"SQG-0.2-WRONG_TYPE: Layer \"{name}\" field \"{field_name}\" "
                        f"is {type(entry[field_name]).__name__}, expected one of "
                        f"{[t.__name__ for t in expected_type]}"
                    )
                else:
                    assert isinstance(entry[field_name], expected_type), (
                        f"SQG-0.2-WRONG_TYPE: Layer \"{name}\" field \"{field_name}\" "
                        f"is {type(entry[field_name]).__name__}, expected {expected_type.__name__}"
                    )

    def test_layer_count_matches(self, tmp_path: Path) -> None:
        _, manifest_path = _build_test_slab(tmp_path)
        raw = json.loads(manifest_path.read_text())
        layers = raw.get("layers", [])
        layer_count = raw.get("layer_count", -1)
        assert layer_count == len(layers), (
            f"SQG-0.2-COUNT_MISMATCH: layer_count={layer_count} but "
            f"{len(layers)} layer entries in manifest."
        )

    def test_quant_scheme_is_per_row_symmetric(self, tmp_path: Path) -> None:
        """All BP8 layers should use per_row_symmetric scheme."""
        _, manifest_path = _build_test_slab(tmp_path)
        raw = json.loads(manifest_path.read_text())
        for entry in raw.get("layers", []):
            name = entry.get("canonical_name", "<unknown>")
            assert entry.get("quant_scheme") == "per_row_symmetric", (
                f"SQG-0.2-WRONG_SCHEME: Layer \"{name}\" quant_scheme="
                f"\"{entry.get('quant_scheme')}\", expected \"per_row_symmetric\""
            )

    def test_dtype_fields_are_correct(self, tmp_path: Path) -> None:
        _, manifest_path = _build_test_slab(tmp_path)
        raw = json.loads(manifest_path.read_text())
        for entry in raw.get("layers", []):
            name = entry.get("canonical_name", "<unknown>")
            assert entry.get("dtype_qweight") == "int8", (
                f"SQG-0.2-WRONG_DTYPE: Layer \"{name}\" dtype_qweight="
                f"\"{entry.get('dtype_qweight')}\", expected \"int8\""
            )
            assert entry.get("dtype_scale") == "float32", (
                f"SQG-0.2-WRONG_DTYPE: Layer \"{name}\" dtype_scale="
                f"\"{entry.get('dtype_scale')}\", expected \"float32\""
            )
            assert entry.get("dtype_zero_point") == "float32", (
                f"SQG-0.2-WRONG_DTYPE: Layer \"{name}\" dtype_zero_point="
                f"\"{entry.get('dtype_zero_point')}\", expected \"float32\""
            )


# ── Gate 0.3: Manifest-tensor agreement ─────────────────────────────────────


class TestGate03ManifestTensorAgreement:
    """SQG-0.3: Every key in manifest exists in safetensors with declared dtype/shape."""

    def test_all_qweight_keys_match(self, tmp_path: Path) -> None:
        st_path, manifest_path = _build_test_slab(tmp_path)
        errors = validate_manifest_against_safetensors(manifest_path, st_path)
        assert len(errors) == 0, (
            f"SQG-0.3-TENSOR_MISMATCH: {len(errors)} mismatches:\n"
            + "\n".join(errors)
        )

    def test_qweight_tensors_are_int8(self, tmp_path: Path) -> None:
        st_path, manifest_path = _build_test_slab(tmp_path)
        from safetensors.torch import load_file
        raw = json.loads(manifest_path.read_text())
        tensors = load_file(str(st_path))

        for entry in raw.get("layers", []):
            name = entry["canonical_name"]
            qw_key = entry["qweight_key"]
            assert qw_key in tensors, (
                f"SQG-0.3-TENSOR_MISMATCH: Tensor \"{qw_key}\" for layer "
                f"\"{name}\" not found in safetensors."
            )
            t = tensors[qw_key]
            assert t.dtype == torch.int8, (
                f"SQG-0.3-TENSOR_MISMATCH: Tensor \"{qw_key}\" dtype "
                f"{t.dtype} != expected int8."
            )

    def test_scale_tensors_are_float32(self, tmp_path: Path) -> None:
        st_path, manifest_path = _build_test_slab(tmp_path)
        from safetensors.torch import load_file
        raw = json.loads(manifest_path.read_text())
        tensors = load_file(str(st_path))

        for entry in raw.get("layers", []):
            name = entry["canonical_name"]
            sc_key = entry["scale_key"]
            assert sc_key in tensors, (
                f"SQG-0.3-TENSOR_MISMATCH: Tensor \"{sc_key}\" for layer "
                f"\"{name}\" not found in safetensors."
            )
            t = tensors[sc_key]
            assert t.dtype == torch.float32, (
                f"SQG-0.3-TENSOR_MISMATCH: Tensor \"{sc_key}\" dtype "
                f"{t.dtype} != expected float32."
            )

    def test_packed_shapes_match(self, tmp_path: Path) -> None:
        st_path, manifest_path = _build_test_slab(tmp_path)
        from safetensors.torch import load_file
        raw = json.loads(manifest_path.read_text())
        tensors = load_file(str(st_path))

        for entry in raw.get("layers", []):
            name = entry["canonical_name"]
            qw_key = entry["qweight_key"]
            packed_shape = tuple(entry["packed_shape"])
            t = tensors[qw_key]
            assert tuple(t.shape) == packed_shape, (
                f"SQG-0.3-TENSOR_MISMATCH: Tensor \"{qw_key}\" shape "
                f"{tuple(t.shape)} != manifest packed_shape {packed_shape}."
            )


# ── Gate 0.4: Stagehand FileParamSpec compatibility ──────────────────────────


class TestGate04FileParamSpecCompat:
    """SQG-0.4: Stagehand can build FileParamSpec from manifest + safetensors."""

    def test_build_param_specs_from_manifest(self, tmp_path: Path) -> None:
        from squareq.manifest import build_stagehand_param_specs

        st_path, manifest_path = _build_test_slab(tmp_path)
        specs = build_stagehand_param_specs(manifest_path, st_path)

        assert len(specs) > 0, (
            "SQG-0.4-FILESPEC_INCOMPAT: No param specs generated from manifest."
        )
        for spec in specs:
            assert hasattr(spec, "param_name"), (
                "SQG-0.4-FILESPEC_INCOMPAT: Spec missing param_name attribute."
            )
            assert hasattr(spec, "layer_name"), (
                "SQG-0.4-FILESPEC_INCOMPAT: Spec missing layer_name attribute."
            )

    def test_param_specs_cover_all_layers(self, tmp_path: Path) -> None:
        from squareq.manifest import build_stagehand_param_specs

        st_path, manifest_path = _build_test_slab(tmp_path)
        raw = json.loads(manifest_path.read_text())
        manifest_names = {e["canonical_name"] for e in raw.get("layers", [])}

        specs = build_stagehand_param_specs(manifest_path, st_path)
        spec_layer_names = {s.layer_name for s in specs}

        missing = manifest_names - spec_layer_names
        assert len(missing) == 0, (
            f"SQG-0.4-FILESPEC_INCOMPAT: Layers without param specs: {missing}"
        )


# ── Gate 0.5: Model-signature mismatch hard fail ────────────────────────────


class TestGate05SignatureMismatch:
    """SQG-0.5: Wrong model_signature causes immediate hard failure."""

    def test_wrong_signature_raises(self, tmp_path: Path) -> None:
        from squareq.manifest import load_and_validate_manifest

        _, manifest_path = _build_test_slab(tmp_path)

        # Corrupt the model_signature.
        raw = json.loads(manifest_path.read_text())
        raw["model_signature"] = "WRONG_SIGNATURE_0000"
        manifest_path.write_text(json.dumps(raw))

        # Build a fresh model to get its real signature.
        model = _SmallFluxModel()

        with pytest.raises(RuntimeError, match="[Mm]odel signature mismatch"):
            load_and_validate_manifest(manifest_path, model=model)

    def test_correct_signature_passes(self, tmp_path: Path) -> None:
        from squareq.manifest import load_and_validate_manifest

        _, manifest_path = _build_test_slab(tmp_path)
        model = _SmallFluxModel()
        # Should not raise.
        manifest = load_and_validate_manifest(manifest_path, model=model)
        assert manifest is not None


# ── Gate 0.6: Kernel ABI version check ──────────────────────────────────────


class TestGate06KernelABIVersion:
    """SQG-0.6: Slab with future ABI version is rejected."""

    def test_future_abi_raises(self, tmp_path: Path) -> None:
        from squareq.manifest import load_and_validate_manifest

        _, manifest_path = _build_test_slab(tmp_path)
        model = _SmallFluxModel()

        raw = json.loads(manifest_path.read_text())
        raw["kernel_abi_version"] = 999
        manifest_path.write_text(json.dumps(raw))

        with pytest.raises(RuntimeError, match="kernel ABI"):
            load_and_validate_manifest(manifest_path, model=model)

    def test_current_abi_passes(self, tmp_path: Path) -> None:
        from squareq.manifest import load_and_validate_manifest

        _, manifest_path = _build_test_slab(tmp_path)
        model = _SmallFluxModel()

        raw = json.loads(manifest_path.read_text())
        assert raw["kernel_abi_version"] <= CURRENT_KERNEL_ABI_VERSION, (
            f"SQG-0.6-ABI_BYPASS: Builder emitted ABI v{raw['kernel_abi_version']} "
            f"but runtime provides v{CURRENT_KERNEL_ABI_VERSION}."
        )
        # Should not raise.
        load_and_validate_manifest(manifest_path, model=model)


# ── Gate 0.7: Quant-bit compatibility gate ──────────────────────────────────


class TestGate07QuantBitCompat:
    """SQG-0.7: Unsupported quant_bits values are rejected."""

    def test_unsupported_bits_raises(self, tmp_path: Path) -> None:
        from squareq.manifest import load_and_validate_manifest

        _, manifest_path = _build_test_slab(tmp_path)
        model = _SmallFluxModel()

        raw = json.loads(manifest_path.read_text())
        # Set first layer to unsupported 4-bit.
        raw["layers"][0]["quant_bits"] = 4
        manifest_path.write_text(json.dumps(raw))

        with pytest.raises(RuntimeError, match="quant"):
            load_and_validate_manifest(manifest_path, model=model)

    def test_supported_bits_passes(self, tmp_path: Path) -> None:
        from squareq.manifest import load_and_validate_manifest

        _, manifest_path = _build_test_slab(tmp_path)
        model = _SmallFluxModel()

        raw = json.loads(manifest_path.read_text())
        for entry in raw["layers"]:
            assert entry["quant_bits"] in SUPPORTED_QUANT_BITS, (
                f"SQG-0.7-QBIT_BYPASS: Layer \"{entry['canonical_name']}\" has "
                f"quant_bits={entry['quant_bits']} which is not in {SUPPORTED_QUANT_BITS}."
            )
        # Should not raise.
        load_and_validate_manifest(manifest_path, model=model)


# ── Gate 0 integration: round-trip ──────────────────────────────────────────


class TestGate0RoundTrip:
    """Integration: build slab → load manifest → validate tensors → all pass."""

    def test_full_round_trip(self, tmp_path: Path) -> None:
        st_path, manifest_path = _build_test_slab(tmp_path)

        # Load and validate manifest.
        raw = json.loads(manifest_path.read_text())

        # Verify basic structure.
        assert "layers" in raw
        assert "model_signature" in raw
        assert raw["layer_count"] == len(raw["layers"])

        # Verify tensor agreement.
        errors = validate_manifest_against_safetensors(manifest_path, st_path)
        assert len(errors) == 0, f"Round-trip validation failed:\n" + "\n".join(errors)

        # Verify all qweights are INT8, all scales/zeros are FP32.
        from safetensors.torch import load_file
        tensors = load_file(str(st_path))
        for entry in raw["layers"]:
            qw = tensors[entry["qweight_key"]]
            sc = tensors[entry["scale_key"]]
            zp = tensors[entry["zero_point_key"]]
            assert qw.dtype == torch.int8
            assert sc.dtype == torch.float32
            assert zp.dtype == torch.float32
            # Scale and zero_point should be 1D with length == out_features.
            assert sc.shape == (entry["orig_shape"][0],)
            assert zp.shape == (entry["orig_shape"][0],)

    def test_total_qweight_bytes_accurate(self, tmp_path: Path) -> None:
        st_path, manifest_path = _build_test_slab(tmp_path)
        from safetensors.torch import load_file
        raw = json.loads(manifest_path.read_text())
        tensors = load_file(str(st_path))

        actual_bytes = sum(
            tensors[entry["qweight_key"]].numel() * tensors[entry["qweight_key"]].element_size()
            for entry in raw["layers"]
        )
        assert raw["total_qweight_bytes"] == actual_bytes, (
            f"total_qweight_bytes={raw['total_qweight_bytes']} != actual {actual_bytes}"
        )
