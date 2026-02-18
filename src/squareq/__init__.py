"""SquareQ — INT8 quantized linear layers with safetensors slab format."""

from squareq.bridge import get_squareq_v2_layers
from squareq.builder import build_safetensors_slab
from squareq.loader import load_quant_state_from_slab
from squareq.manifest import (
    CURRENT_KERNEL_ABI_VERSION,
    SUPPORTED_QUANT_BITS,
    SlabManifestV2,
    SquareQParamSpec,
    build_stagehand_param_specs,
    load_and_validate_manifest,
    load_manifest,
    validate_manifest_against_safetensors,
)
from squareq.modules import QuantLinear, QuantLinearLoRA
from squareq.scaffold import prepare_model_for_quantized_streaming

__all__ = [
    "CURRENT_KERNEL_ABI_VERSION",
    "SUPPORTED_QUANT_BITS",
    "QuantLinear",
    "QuantLinearLoRA",
    "SlabManifestV2",
    "SquareQParamSpec",
    "build_safetensors_slab",
    "build_stagehand_param_specs",
    "get_squareq_v2_layers",
    "load_and_validate_manifest",
    "load_manifest",
    "load_quant_state_from_slab",
    "prepare_model_for_quantized_streaming",
    "validate_manifest_against_safetensors",
]
