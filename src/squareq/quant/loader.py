from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch import nn

from squareq.quant.modules import QuantLinear
from squareq.slab.schema import LayerRecord, SlabStorage, load_slab


def _split_module_path(name: str) -> Tuple[str, str]:
    if "." not in name:
        return "", name
    parts = name.split(".")
    module_path = ".".join(parts[:-1])
    return module_path, parts[-1]


def _locate_parent(model: nn.Module, module_path: str) -> nn.Module:
    """
    Traverse ``module_path`` relative to ``model`` and return the parent module
    that owns the attribute (or ModuleList index) we want to replace.
    """

    if not module_path:
        return model

    parent = model
    for token in module_path.split("."):
        parent = parent[int(token)] if token.isdigit() else getattr(parent, token)
    return parent


def _quant_layer_to_device(
    layer: LayerRecord,
    data: SlabStorage,
    device: torch.device,
    compute_dtype: torch.dtype,
) -> QuantLinear:
    layer_data = data.layers[layer.name]
    qlinear = QuantLinear(
        in_features=layer.in_features,
        out_features=layer.out_features,
        bias=layer.has_bias,
        compute_dtype=compute_dtype,
    ).to(device)
    bias = layer_data.bias.to(device) if layer_data.bias is not None else None
    qlinear.set_quant_state(
        qweight=layer_data.qweight.to(device),
        scale=layer_data.scale.to(device),
        zero_point=layer_data.zero_point.to(device),
        bias=bias,
    )
    return qlinear


def attach_bp8_slab(
    model: nn.Module,
    slab_path: str | Path,
    *,
    device: str | torch.device = "cuda",
    compute_dtype: torch.dtype = torch.bfloat16,
) -> None:
    """
    Replace all Linear modules in ``model`` with INT8 quantised QuantLinear modules
    loaded from ``slab_path``. The function assumes the slab contains entries for
    every layer (all-layer slab).
    """

    slab = load_slab(slab_path)
    device = torch.device(device)

    # Perform replacements while the scaffold resides on CPU to keep VRAM low.
    for layer in slab.manifest.layers:
        module_path, attr = _split_module_path(layer.name)
        parent = _locate_parent(model, module_path)
        target_attr = attr if module_path else attr or layer.name

        module = parent[int(target_attr)] if target_attr.isdigit() else getattr(parent, target_attr)
        if not isinstance(module, nn.Linear):
            raise TypeError(f"Expected nn.Linear at {layer.name}, found {type(module)}")

        quant_module = _quant_layer_to_device(layer, slab, device, compute_dtype)

        if target_attr.isdigit():
            parent[int(target_attr)] = quant_module  # type: ignore[index]
        else:
            setattr(parent, target_attr, quant_module)

        # Drop references to the original module to allow GC.
        del module

    # Move remaining scaffold to the requested device (without dtype cast).
    model.to(device)
