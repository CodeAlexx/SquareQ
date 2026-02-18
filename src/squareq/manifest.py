"""SquareQ slab manifest — V2 safetensors format.

Defines the manifest schema, validation logic, and Stagehand param-spec
generation for the new safetensors-only slab format.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

__all__ = [
    "CURRENT_KERNEL_ABI_VERSION",
    "SUPPORTED_QUANT_BITS",
    "SlabManifestV2",
    "SquareQParamSpec",
    "load_manifest",
    "load_and_validate_manifest",
    "validate_manifest_against_safetensors",
    "build_stagehand_param_specs",
]

# Current kernel ABI version supported by this runtime.
CURRENT_KERNEL_ABI_VERSION: int = 1

# Quant bit-widths the current kernel can handle.
SUPPORTED_QUANT_BITS: frozenset[int] = frozenset({8})

# dtype string mapping for validation.
_DTYPE_STR_MAP: dict[str, torch.dtype] = {
    "int8": torch.int8,
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


# ── manifest dataclass ────────────────────────────────────────────────────


@dataclass
class SlabManifestV2:
    """Top-level manifest for the V2 safetensors slab format.

    Loaded from ``{slab_name}.manifest.json``.
    """

    model_signature: str = ""
    architecture_id: str = ""
    quant_version: str = "squareq_bp8_v2"
    kernel_abi_version: int = CURRENT_KERNEL_ABI_VERSION
    min_runtime_version: str = "0.1.0"
    layer_count: int = 0
    total_qweight_bytes: int = 0
    layers: list[dict[str, Any]] = field(default_factory=list)


# ── signature computation ──────────────────────────────────────────────────


def compute_model_signature(model: nn.Module) -> str:
    """Compute a deterministic signature from Linear layer topology.

    Hashes the sorted set of (name, in_features, out_features, has_bias)
    for every ``nn.Linear`` in the model tree.
    """
    entries: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            entries.append(
                f"{name}:{module.in_features}:{module.out_features}"
                f":{module.bias is not None}"
            )
    entries.sort()
    digest = hashlib.sha256("|".join(entries).encode()).hexdigest()
    return digest[:16]


# ── loading ────────────────────────────────────────────────────────────────


def load_manifest(path: str | Path) -> SlabManifestV2:
    """Load a manifest JSON file and return the parsed dataclass."""
    raw = json.loads(Path(path).read_text())
    return SlabManifestV2(
        model_signature=raw.get("model_signature", ""),
        architecture_id=raw.get("architecture_id", ""),
        quant_version=raw.get("quant_version", "squareq_bp8_v2"),
        kernel_abi_version=raw.get("kernel_abi_version", CURRENT_KERNEL_ABI_VERSION),
        min_runtime_version=raw.get("min_runtime_version", "0.1.0"),
        layer_count=raw.get("layer_count", 0),
        total_qweight_bytes=raw.get("total_qweight_bytes", 0),
        layers=raw.get("layers", []),
    )


def load_and_validate_manifest(
    path: str | Path,
    *,
    model: nn.Module | None = None,
) -> SlabManifestV2:
    """Load manifest, validate signature/ABI/quant-bits against model and runtime.

    Raises ``RuntimeError`` with clear messages on mismatch (Gates 0.5-0.7).
    """
    manifest = load_manifest(path)

    # Gate 0.6: kernel ABI version check.
    if manifest.kernel_abi_version > CURRENT_KERNEL_ABI_VERSION:
        raise RuntimeError(
            f"Unsupported kernel ABI version {manifest.kernel_abi_version}. "
            f"Runtime supports up to v{CURRENT_KERNEL_ABI_VERSION}."
        )

    # Gate 0.7: quant-bit compatibility.
    for entry in manifest.layers:
        bits = entry.get("quant_bits")
        if bits not in SUPPORTED_QUANT_BITS:
            raise RuntimeError(
                f"Unsupported quant bits {bits} for layer "
                f"'{entry.get('canonical_name', '<unknown>')}'. "
                f"Supported: {sorted(SUPPORTED_QUANT_BITS)}."
            )

    # Gate 0.5: model signature check.
    if model is not None:
        expected = compute_model_signature(model)
        if manifest.model_signature != expected:
            raise RuntimeError(
                f"Model signature mismatch: manifest has "
                f"'{manifest.model_signature}', model has '{expected}'."
            )

    return manifest


# ── validation ─────────────────────────────────────────────────────────────


def validate_manifest_against_safetensors(
    manifest_path: str | Path,
    safetensors_path: str | Path,
) -> list[str]:
    """Check every manifest key exists in safetensors with correct dtype/shape.

    Returns a list of error strings (empty = all pass).
    """
    from safetensors.torch import load_file

    raw = json.loads(Path(manifest_path).read_text())
    tensors = load_file(str(safetensors_path))

    errors: list[str] = []
    for entry in raw.get("layers", []):
        name = entry.get("canonical_name", "<unknown>")

        # Check qweight.
        qw_key = entry.get("qweight_key")
        if qw_key not in tensors:
            errors.append(f"Missing tensor '{qw_key}' for layer '{name}'")
        else:
            t = tensors[qw_key]
            expected_dtype = _DTYPE_STR_MAP.get(entry.get("dtype_qweight", ""))
            if expected_dtype is not None and t.dtype != expected_dtype:
                errors.append(
                    f"Tensor '{qw_key}' dtype {t.dtype} != expected {expected_dtype}"
                )
            expected_shape = tuple(entry.get("packed_shape", []))
            if tuple(t.shape) != expected_shape:
                errors.append(
                    f"Tensor '{qw_key}' shape {tuple(t.shape)} != "
                    f"expected {expected_shape}"
                )

        # Check scale.
        sc_key = entry.get("scale_key")
        if sc_key not in tensors:
            errors.append(f"Missing tensor '{sc_key}' for layer '{name}'")
        else:
            t = tensors[sc_key]
            if t.dtype != torch.float32:
                errors.append(
                    f"Tensor '{sc_key}' dtype {t.dtype} != expected float32"
                )
            if t.dtype.is_floating_point:
                if torch.isnan(t).any():
                    errors.append(
                        f"Tensor '{sc_key}' for layer '{name}' contains NaN values"
                    )
                if torch.isinf(t).any():
                    errors.append(
                        f"Tensor '{sc_key}' for layer '{name}' contains Inf values"
                    )
                if t.numel() > 0 and (t == 0).all():
                    errors.append(
                        f"Tensor '{sc_key}' for layer '{name}' is all-zero scale "
                        "(dequantization would produce all zeros)"
                    )

        # Check zero_point.
        zp_key = entry.get("zero_point_key")
        if zp_key not in tensors:
            errors.append(f"Missing tensor '{zp_key}' for layer '{name}'")
        else:
            t = tensors[zp_key]
            if t.dtype != torch.float32:
                errors.append(
                    f"Tensor '{zp_key}' dtype {t.dtype} != expected float32"
                )
            if t.dtype.is_floating_point:
                if torch.isnan(t).any():
                    errors.append(
                        f"Tensor '{zp_key}' for layer '{name}' contains NaN values"
                    )
                if torch.isinf(t).any():
                    errors.append(
                        f"Tensor '{zp_key}' for layer '{name}' contains Inf values"
                    )

        # Check bias if declared.
        bias_key = entry.get("bias_key")
        if bias_key is not None:
            if bias_key not in tensors:
                errors.append(f"Missing tensor '{bias_key}' for layer '{name}'")

    return errors


# ── param spec dataclass ──────────────────────────────────────────────────


@dataclass
class SquareQParamSpec:
    """Descriptor for a parameter sourced from a SquareQ BP8 slab."""

    param_name: str
    layer_name: str
    kind: str  # "weight" or "bias"
    out_features: int
    in_features: int
    padded_in_features: int


# ── Stagehand integration ─────────────────────────────────────────────────


def build_stagehand_param_specs(
    manifest_path: str | Path,
    safetensors_path: str | Path,
) -> list[SquareQParamSpec]:
    """Build SquareQParamSpec entries from manifest for Stagehand registry.

    Returns a list of ``SquareQParamSpec`` instances.
    """
    raw = json.loads(Path(manifest_path).read_text())
    specs: list[SquareQParamSpec] = []

    for entry in raw.get("layers", []):
        name = entry["canonical_name"]
        out_features = entry["orig_shape"][0]
        in_features = entry["orig_shape"][1]
        padded_in = entry["packed_shape"][1]

        specs.append(
            SquareQParamSpec(
                param_name=f"{name}.weight",
                layer_name=name,
                kind="weight",
                out_features=out_features,
                in_features=in_features,
                padded_in_features=padded_in,
            )
        )

        if entry.get("bias_key") is not None:
            specs.append(
                SquareQParamSpec(
                    param_name=f"{name}.bias",
                    layer_name=name,
                    kind="bias",
                    out_features=out_features,
                    in_features=in_features,
                    padded_in_features=padded_in,
                )
            )

    return specs
