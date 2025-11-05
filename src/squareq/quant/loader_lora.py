from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from squareq.quant.loader import attach_bp8_slab
from squareq.quant.modules import QuantLinear
from squareq.quant.modules_lora import QuantLinearLoRA


def _resolve_parent(model: nn.Module, module_path: str) -> tuple[nn.Module, str]:
    if not module_path:
        return model, ""
    parts = module_path.split(".")
    parent = model
    for token in parts[:-1]:
        parent = parent[int(token)] if token.isdigit() else getattr(parent, token)
    return parent, parts[-1]


def _replace_module(root: nn.Module, module_path: str, new_module: nn.Module) -> None:
    if not module_path:
        raise ValueError("Cannot replace the root module.")
    parent, name = _resolve_parent(root, module_path)
    if name.isdigit():
        parent[int(name)] = new_module  # type: ignore[index]
    else:
        setattr(parent, name, new_module)


def _iter_quant_linear_modules(model: nn.Module) -> Iterable[tuple[str, QuantLinear]]:
    for name, module in list(model.named_modules()):
        if isinstance(module, QuantLinear):
            yield name, module


def prepare_flux_for_lora_training(
    model: nn.Module,
    slab_path: str,
    *,
    rank: int = 8,
    alpha: float = 1.0,
    device: str | torch.device = "cuda",
) -> nn.Module:
    """
    Attach a BP8 slab, wrap each quantized linear block with a LoRA-enabled module,
    and freeze all non-LoRA parameters.
    """

    attach_bp8_slab(model, slab_path, device=device, compute_dtype=torch.bfloat16)

    replaced = 0
    for name, module in _iter_quant_linear_modules(model):
        ql = QuantLinearLoRA(
            module.in_features,
            module.out_features,
            rank=rank,
            alpha=alpha,
            compute_dtype=module.compute_dtype,
        ).to(device)
        ql.set_quant_state(
            qweight=module.qweight,
            scale=module.scale,
            zero_point=module.zero_point,
            bias=module.bias.detach() if module.bias is not None else None,
        )
        _replace_module(model, name, ql)
        replaced += 1

    if replaced == 0:
        raise RuntimeError("No QuantLinear modules were found to wrap with LoRA.")

    for param in model.parameters():
        param.requires_grad_(False)

    for module in model.modules():
        if isinstance(module, QuantLinearLoRA):
            module.lora_A.requires_grad_(True)
            module.lora_B.requires_grad_(True)

    model.to(device)

    return model
