"""SquareQ slab loader — load INT8 quant state into scaffolded QuantLinear.

Bridges the V2 safetensors slab format and QuantLinear.set_quant_state().
Used by Stagehand's residency adapter to populate QuantLinear modules
at block-load time without materializing float weights.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from squareq.manifest import SlabManifestV2
from squareq.modules import QuantLinear

__all__ = ["load_quant_state_from_slab"]


def load_quant_state_from_slab(
    model: nn.Module,
    manifest: SlabManifestV2,
    safetensors_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> int:
    """Load INT8 quant state from V2 slab into scaffolded QuantLinear modules.

    Reads quantized tensors from the safetensors file and calls
    ``set_quant_state()`` on each matching QuantLinear in the model.
    No float weight is ever materialized — INT8 data flows directly
    from disk into the QuantLinear buffers.

    Parameters
    ----------
    model:
        Model previously scaffolded with ``prepare_model_for_quantized_streaming``.
    manifest:
        Loaded slab manifest (from ``load_manifest``).
    safetensors_path:
        Path to the ``.safetensors`` file containing INT8 tensors.
    device:
        Target device for the loaded tensors.

    Returns
    -------
    int:
        Number of QuantLinear modules whose state was loaded.
    """
    from safetensors.torch import load_file

    tensors = load_file(str(safetensors_path), device=str(device))

    # Build module lookup.
    module_map: dict[str, nn.Module] = dict(model.named_modules())

    count = 0
    for entry in manifest.layers:
        name = entry["canonical_name"]
        module = module_map.get(name)
        if module is None or not isinstance(module, QuantLinear):
            continue

        qweight = tensors[entry["qweight_key"]]
        scale = tensors[entry["scale_key"]]
        zero_point = tensors[entry["zero_point_key"]]

        bias = None
        if entry.get("bias_key") is not None:
            bias = tensors[entry["bias_key"]]

        module.set_quant_state(
            qweight=qweight,
            scale=scale,
            zero_point=zero_point,
            bias=bias,
        )
        count += 1

    return count
