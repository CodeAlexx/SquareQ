"""SquareQ scaffold — replace nn.Linear with empty QuantLinear.

Prepares a model for quantized streaming by replacing target modules
with empty QuantLinear instances.  No data is loaded — Stagehand
provides the INT8 weights at runtime via set_quant_state().
"""
from __future__ import annotations

from torch import nn

from squareq.manifest import SlabManifestV2
from squareq.modules import QuantLinear

__all__ = ["prepare_model_for_quantized_streaming"]


def prepare_model_for_quantized_streaming(
    model: nn.Module,
    manifest: SlabManifestV2,
) -> dict[str, dict]:
    """Replace target nn.Linear modules with empty QuantLinear.

    Walks the model tree and replaces every module whose name appears
    in the manifest with an empty ``QuantLinear``.  No GPU memory is
    allocated — only the module structure changes.

    Parameters
    ----------
    model:
        The model to modify in-place.
    manifest:
        Loaded slab manifest describing which layers to replace.

    Returns
    -------
    dict:
        Mapping of ``{module_name: {in_features, out_features, bias}}``
        for every replaced module.
    """
    # Build lookup from manifest layer entries.
    layer_info: dict[str, dict] = {}
    for entry in manifest.layers:
        layer_info[entry["canonical_name"]] = entry

    # Build module lookup once for parent resolution.
    module_map: dict[str, nn.Module] = dict(model.named_modules())

    replaced: dict[str, dict] = {}

    for name, entry in layer_info.items():
        # Skip if already a QuantLinear (idempotency).
        current = module_map.get(name)
        if current is None:
            continue
        if isinstance(current, QuantLinear):
            continue

        out_features = entry["orig_shape"][0]
        in_features = entry["orig_shape"][1]
        has_bias = entry.get("bias_key") is not None

        new_module = QuantLinear(
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
        )

        # Resolve parent module and attribute name.
        if "." in name:
            parent_path, attr_name = name.rsplit(".", 1)
            parent = module_map[parent_path]
        else:
            parent = model
            attr_name = name

        setattr(parent, attr_name, new_module)

        # Update module_map so subsequent lookups see the new module.
        module_map[name] = new_module

        replaced[name] = {
            "in_features": in_features,
            "out_features": out_features,
            "bias": has_bias,
        }

    return replaced
