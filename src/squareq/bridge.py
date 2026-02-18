"""SquareQ V2 ↔ Stagehand layer bridge.

Loads INT8 quantized layers from V2 safetensors slabs using the JSON
manifest sidecar.  Returns layer dicts compatible with
``_copy_squareq_backed_params_into_buffer``.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

__all__ = ["get_squareq_v2_layers"]


def get_squareq_v2_layers(
    safetensors_path: str,
    manifest_path: str,
) -> dict[str, dict[str, torch.Tensor | None]]:
    """Load V2 SquareQ layers from safetensors using manifest metadata.

    Returns a dict mapping canonical layer name to a sub-dict with keys
    ``qweight`` (int8), ``scale`` (fp32), ``zero_point`` (fp32), and
    optionally ``bias`` (fp32).  This format is compatible with
    ``_copy_squareq_backed_params_into_buffer`` in the stagehand
    scheduler.
    """
    from safetensors.torch import load_file

    raw = json.loads(Path(manifest_path).read_text())
    tensors = load_file(safetensors_path)

    layers: dict[str, dict[str, torch.Tensor | None]] = {}
    for entry in raw.get("layers", []):
        name = entry.get("canonical_name", "")
        if not name:
            continue

        layer_data: dict[str, torch.Tensor | None] = {}

        # Required tensors — raise on missing.
        for field, dict_key in [
            ("qweight_key", "qweight"),
            ("scale_key", "scale"),
            ("zero_point_key", "zero_point"),
        ]:
            tensor_key = entry.get(field)
            if tensor_key and tensor_key not in tensors:
                raise RuntimeError(
                    f"Missing tensor '{tensor_key}' for layer '{name}'. "
                    f"Cannot stage — slab file is incomplete."
                )
            if tensor_key and tensor_key in tensors:
                layer_data[dict_key] = tensors[tensor_key]

        bias_key = entry.get("bias_key")
        if bias_key and bias_key in tensors:
            layer_data["bias"] = tensors[bias_key]
        else:
            layer_data["bias"] = None

        layers[name] = layer_data

    return layers
