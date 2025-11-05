#!/usr/bin/env python
"""
Build an all-layers BP8 slab for SQUARE-Q.

This script quantises every weight-bearing layer of the Flux transformer into
row-wise INT8 tensors and stores them alongside scale/zero-point metadata for
fast device staging.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from safetensors import safe_open
from tqdm import tqdm

from squareq.slab.schema import LayerData, LayerRecord, SlabManifest, SlabStorage, save_slab

INCLUDE_PREFIXES: Tuple[str, ...] = (
    # Core transformer stacks
    "transformer_blocks.",
    "single_transformer_blocks.",
    # Embedders and projections
    "time_text_embed.",
    "context_embedder",
    "x_embedder",
    "pos_embed",
    "norm_out.",
    "proj_out",
    "final_layer.",
)


def discover_safetensors(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    files = sorted(path.glob("**/*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors files found under {path}")
    return files


def build_key_index(files: Iterable[Path]) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as f:
            for key in f.keys():
                index[key] = file
    return index


def load_tensor(index: Dict[str, Path], key: str) -> torch.Tensor:
    file = index.get(key)
    if file is None:
        raise KeyError(f"Key '{key}' not present in safetensors index")
    with safe_open(file, framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def quantize_rowwise_int8(weight: torch.Tensor, pack_k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if weight.dim() < 2:
        raise ValueError("Expected 2D weight tensor for quantisation")
    out_features = weight.shape[0]
    flat = weight.view(out_features, -1).to(torch.float32)
    original_in = flat.shape[1]

    if pack_k > 1:
        pad = (-original_in) % pack_k
        if pad:
            flat = torch.nn.functional.pad(flat, (0, pad))
    else:
        pad = 0

    max_vals = flat.abs().amax(dim=1)
    scale = (max_vals / 127.0).clamp(min=1e-8)
    inv_scale = torch.where(scale > 0, 1.0 / scale, torch.zeros_like(scale))
    q = torch.round(flat * inv_scale.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
    q = q.view(out_features, flat.shape[1])
    zero_point = torch.zeros_like(scale, dtype=torch.float32)

    return q, scale.float(), zero_point, flat.shape[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an all-layer BP8 slab for Flux.")
    parser.add_argument("--model-path", required=True, help="Path to Flux safetensors directory or file")
    parser.add_argument("--output", required=True, help="Destination .fpk file")
    parser.add_argument("--metadata-name", default="FLUX.1-dev", help="Model name for metadata")
    parser.add_argument("--quant-version", default="squareq_bp8_v1", help="Quant version string")
    parser.add_argument("--layout", default="rowwise_sym_int8", help="Layout descriptor stored in manifest")
    parser.add_argument("--pack-k", type=int, default=64, help="Pad input dimension to this multiple")
    args = parser.parse_args()

    model_path = Path(args.model_path).expanduser()
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    safetensor_files = discover_safetensors(model_path)
    key_index = build_key_index(safetensor_files)

    seen_modules: set[str] = set()
    layer_records: List[LayerRecord] = []
    layer_data: Dict[str, LayerData] = {}

    total_params = 0
    quant_params = 0
    bytes_raw = 0
    bytes_quant = 0

    for key, file in tqdm(key_index.items(), desc="Processing", unit="param"):
        if not key.endswith(".weight"):
            continue
        if not key.startswith(INCLUDE_PREFIXES):
            continue
        module_name = key[:-7]  # strip ".weight"
        if module_name in seen_modules:
            continue

        weight = load_tensor(key_index, key)
        if weight.dim() < 2:
            continue

        bias_key = f"{module_name}.bias"
        bias_tensor = load_tensor(key_index, bias_key).to(torch.float32) if bias_key in key_index else None

        qweight, scale, zero_point, padded_in = quantize_rowwise_int8(weight, args.pack_k)

        out_features = weight.shape[0]
        in_features = weight.view(out_features, -1).shape[1]

        record = LayerRecord(
            name=module_name,
            out_features=out_features,
            in_features=in_features,
            padded_in_features=padded_in,
            has_bias=bias_tensor is not None,
        )
        data = LayerData(
            qweight=qweight.contiguous(),
            scale=scale.contiguous(),
            zero_point=zero_point.contiguous(),
            bias=bias_tensor.contiguous() if bias_tensor is not None else None,
        )

        layer_records.append(record)
        layer_data[module_name] = data
        seen_modules.add(module_name)

        total_params += weight.numel()
        quant_params += qweight.numel()
        bytes_raw += weight.element_size() * weight.numel()
        bytes_quant += (
            qweight.element_size() * qweight.numel()
            + scale.element_size() * scale.numel()
            + zero_point.element_size() * zero_point.numel()
            + (bias_tensor.element_size() * bias_tensor.numel() if bias_tensor is not None else 0)
        )

    layer_records.sort(key=lambda rec: rec.name)

    manifest = SlabManifest(
        model_name=args.metadata_name,
        quant_version=args.quant_version,
        layout=args.layout,
        pack_k=args.pack_k,
        layers=layer_records,
    )
    storage = SlabStorage(manifest=manifest, layers=layer_data)
    save_slab(output_path, storage)

    compression_ratio = bytes_raw / bytes_quant if bytes_quant else 1.0
    print(f"Saved slab to {output_path}")
    print(f"Total params: {total_params / 1e9:.2f}B | Quantised params: {quant_params / 1e9:.2f}B")
    print(f"Compression ratio: {compression_ratio:.2f}x (raw {bytes_raw/1e9:.2f} GB -> quant {bytes_quant/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
