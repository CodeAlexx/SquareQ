"""SquareQ slab builder — safetensors output format.

Quantizes model weights to per-row symmetric INT8 and writes a
safetensors file + JSON manifest.  Replaces the old .fpk torch.save path.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from squareq.manifest import (
    CURRENT_KERNEL_ABI_VERSION,
    compute_model_signature,
)

__all__ = ["build_safetensors_slab"]


def _quantize_rowwise_int8(
    weight: torch.Tensor,
    pack_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Per-row symmetric INT8 quantization with optional K-padding.

    Returns (qweight, scale, zero_point, padded_in_features).
    """
    out_features = weight.shape[0]
    flat = weight.view(out_features, -1).to(torch.float32)

    # Pad K dimension to pack_k multiple for kernel alignment.
    if pack_k > 1:
        pad = (-flat.shape[1]) % pack_k
        if pad:
            flat = torch.nn.functional.pad(flat, (0, pad))

    padded_in = flat.shape[1]

    # Per-row absmax scaling.
    max_vals = flat.abs().amax(dim=1)
    scale = (max_vals / 127.0).clamp(min=1e-8)
    inv_scale = torch.where(scale > 0, 1.0 / scale, torch.zeros_like(scale))
    qweight = (
        torch.round(flat * inv_scale.unsqueeze(1))
        .clamp(-127, 127)
        .to(torch.int8)
    )
    zero_point = torch.zeros_like(scale, dtype=torch.float32)

    return qweight, scale.float(), zero_point, padded_in


def build_safetensors_slab(
    *,
    model: nn.Module,
    output_dir: str,
    slab_name: str,
    architecture_id: str = "unknown",
    pack_k: int = 64,
) -> None:
    """Quantize all eligible Linear layers and write safetensors + manifest.

    Output files:
        ``{output_dir}/{slab_name}.safetensors``
        ``{output_dir}/{slab_name}.manifest.json``
    """
    from safetensors.torch import save_file

    output_path = Path(output_dir)
    st_path = output_path / f"{slab_name}.safetensors"
    manifest_path = output_path / f"{slab_name}.manifest.json"

    tensors: dict[str, torch.Tensor] = {}
    layers: list[dict] = []
    total_qweight_bytes = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        weight = module.weight.data.float()
        out_features, in_features = weight.shape

        qweight, scale, zero_point, padded_in = _quantize_rowwise_int8(
            weight, pack_k,
        )

        qw_key = f"{name}.qweight"
        sc_key = f"{name}.scale"
        zp_key = f"{name}.zero_point"

        tensors[qw_key] = qweight
        tensors[sc_key] = scale
        tensors[zp_key] = zero_point

        bias_key = None
        if module.bias is not None:
            bias_key = f"{name}.bias"
            tensors[bias_key] = module.bias.data.float()

        total_qweight_bytes += qweight.numel() * qweight.element_size()

        layers.append({
            "canonical_name": name,
            "qweight_key": qw_key,
            "scale_key": sc_key,
            "zero_point_key": zp_key,
            "bias_key": bias_key,
            "orig_shape": [out_features, in_features],
            "packed_shape": [out_features, padded_in],
            "pad_k": pack_k,
            "quant_bits": 8,
            "quant_scheme": "per_row_symmetric",
            "quant_axis": 0,
            "dtype_qweight": "int8",
            "dtype_scale": "float32",
            "dtype_zero_point": "float32",
            "block_id": None,
        })

    layers.sort(key=lambda x: x["canonical_name"])

    manifest = {
        "model_signature": compute_model_signature(model),
        "architecture_id": architecture_id,
        "quant_version": "squareq_bp8_v2",
        "kernel_abi_version": CURRENT_KERNEL_ABI_VERSION,
        "min_runtime_version": "0.1.0",
        "layer_count": len(layers),
        "total_qweight_bytes": total_qweight_bytes,
        "layers": layers,
    }

    save_file(tensors, str(st_path))
    manifest_path.write_text(json.dumps(manifest, indent=2))
