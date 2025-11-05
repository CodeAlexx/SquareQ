#!/usr/bin/env python
"""
Measure Flux transformer VRAM during a LoRA training step with BP8 slabs.

This script loads the Flux transformer on CPU, attaches the SquareQ BP8 slab,
wraps linear modules with LoRA adapters, and runs a dummy forward/backward step
while printing VRAM usage after key phases.
"""

from __future__ import annotations

import argparse
import os

import torch

from squareq.quant.loader_lora import prepare_flux_for_lora_training
from scripts.test_flux_bp8_pipeline import build_dummy_inputs, load_transformer


def log_vram(tag: str) -> None:
    alloc = torch.cuda.memory_allocated() / (1024**3)
    peak = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"[VRAM:{tag}] alloc={alloc:.2f}GB peak={peak:.2f}GB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Flux BP8 LoRA VRAM smoke test")
    parser.add_argument("--model-path", default="/home/alex/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev")
    parser.add_argument("--slab-path", default="output/flux1_dev_bp8_all_layers.fpk")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for VRAM measurement")

    device = torch.device("cuda")
    transformer = load_transformer(args.model_path, torch.bfloat16)

    transformer = prepare_flux_for_lora_training(
        transformer,
        args.slab_path,
        rank=args.rank,
        alpha=args.alpha,
        device=device,
    )

    lora_params = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0)

    inputs = build_dummy_inputs(
        transformer,
        height=args.height,
        width=args.width,
        dtype=torch.bfloat16,
        device=device,
    )

    torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        def _forward() -> torch.Tensor:
            return transformer(
                hidden_states=inputs.hidden_states,
                encoder_hidden_states=inputs.encoder_hidden_states,
                pooled_projections=inputs.pooled_projections,
                timestep=inputs.timestep / 1000.0,
                guidance=torch.tensor([3.5], device=device, dtype=torch.bfloat16),
                img_ids=inputs.img_ids,
                txt_ids=inputs.text_ids,
            )

        outputs = _forward()
        sample = outputs.sample if hasattr(outputs, "sample") else outputs
        loss = sample.pow(2).mean()

    log_vram("after_forward")
    loss.backward()
    log_vram("after_backward")
    optimizer.step()
    log_vram("after_optimizer_step")


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
