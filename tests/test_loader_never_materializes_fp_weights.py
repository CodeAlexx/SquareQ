import os
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from squareq.quant.loader import attach_bp8_slab
from squareq.quant.modules import QuantLinear
from squareq.slab.schema import LayerData, LayerRecord, SlabManifest, SlabStorage, save_slab
from squareq.tools.squareq_build_slab import quantize_rowwise_int8


class TinyFlux(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = torch.nn.Module()

        double_block = torch.nn.Module()
        double_block.attn = torch.nn.Module()
        double_block.attn.to_q = torch.nn.Linear(4, 4, bias=True)
        double_block.attn.to_k = torch.nn.Linear(4, 4, bias=True)
        double_block.attn.to_v = torch.nn.Linear(4, 4, bias=True)
        double_block.attn.to_out = torch.nn.ModuleList([torch.nn.Linear(4, 4, bias=True)])
        self.transformer.transformer_blocks = torch.nn.ModuleList([double_block])

        single_block = torch.nn.Module()
        single_block.linear1 = torch.nn.Linear(12, 4, bias=True)
        single_block.linear2 = torch.nn.Linear(4, 16, bias=True)
        single_block.modulation = torch.nn.Module()
        single_block.modulation.lin = torch.nn.Linear(4, 12, bias=True)
        self.transformer.single_transformer_blocks = torch.nn.ModuleList([single_block])

        self.final_layer = torch.nn.Module()
        self.final_layer.adaLN_modulation = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Linear(4, 8, bias=True)])
        self.final_layer.linear = torch.nn.Linear(8, 4, bias=False)


def build_tiny_slab(model: TinyFlux, tmp_path: Path) -> Path:
    layers = {}
    records = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            weight = module.weight.detach().clone()
            qweight, scale, zero, padded_in = quantize_rowwise_int8(weight, pack_k=1)
            bias = module.bias.detach().clone().to(torch.float32) if module.bias is not None else None
            record = LayerRecord(
                name=name,
                out_features=weight.shape[0],
                in_features=weight.view(weight.shape[0], -1).shape[1],
                padded_in_features=padded_in,
                has_bias=bias is not None,
            )
            layers[name] = LayerData(qweight=qweight, scale=scale, zero_point=zero, bias=bias)
            records.append(record)

    manifest = SlabManifest(
        model_name="tiny",
        quant_version="test",
        layout="rowwise_sym_int8",
        pack_k=1,
        layers=sorted(records, key=lambda r: r.name),
    )
    storage = SlabStorage(manifest=manifest, layers=layers)
    slab_path = tmp_path / "tiny.slab"
    save_slab(slab_path, storage)
    return slab_path


@pytest.mark.cuda
def test_loader_never_casts_dtype(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    model = TinyFlux()
    slab_path = build_tiny_slab(model, tmp_path)

    dtype_calls = []
    original_to = torch.nn.Module.to

    def patched_to(self, *args, **kwargs):
        if "dtype" in kwargs:
            dtype_calls.append(kwargs["dtype"])
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.nn.Module, "to", patched_to, raising=False)

    attach_bp8_slab(model, slab_path, device="cuda")

    assert dtype_calls == []
    q_module = model.transformer.transformer_blocks[0].attn.to_q
    assert isinstance(q_module, QuantLinear)
    assert not hasattr(q_module, "weight")
