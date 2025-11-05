#!/usr/bin/env python
"""
Minimal Flux transformer VRAM smoke test.

Loads only the Flux transformer, applies an optional BP8 slab, then runs a single
forward pass with synthetic latents / text embeddings to measure peak memory.
No text encoder or VAE weights are materialised.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch

from diffusers import FluxTransformer2DModel

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in os.sys.path:
    os.sys.path.insert(0, str(repo_root))

from squareq.quant.loader import attach_bp8_slab

# Avoid spawning inductor worker pools – they require OS semaphores that might
# be unavailable in sandboxed environments.
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")


@dataclass
class FluxInputs:
    hidden_states: torch.Tensor
    encoder_hidden_states: torch.Tensor
    pooled_projections: torch.Tensor
    text_ids: torch.Tensor
    img_ids: torch.Tensor
    timestep: torch.Tensor
    guidance: torch.Tensor


def log_vram(tag: str) -> None:
    if not torch.cuda.is_available():
        print(f"[{tag}] CUDA unavailable.")
        return
    alloc = torch.cuda.memory_allocated() / (1024**3)
    peak = torch.cuda.max_memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    print(f"[{tag}] alloc={alloc:.2f} GB | peak={peak:.2f} GB | reserved={reserved:.2f} GB")


def pack_latents(latents: torch.Tensor, batch: int, channels: int, height: int, width: int) -> torch.Tensor:
    latents = latents.view(batch, channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(batch, (height // 2) * (width // 2), channels * 4)


def prepare_latent_image_ids(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    latent_image_ids = torch.zeros(height, width, 3, device=device, dtype=dtype)
    latent_image_ids[..., 1] += torch.arange(height, device=device, dtype=dtype)[:, None]
    latent_image_ids[..., 2] += torch.arange(width, device=device, dtype=dtype)[None, :]
    latent_image_ids = latent_image_ids.view(height * width, 3)
    return latent_image_ids


def build_dummy_inputs(
    transformer: FluxTransformer2DModel,
    *,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
) -> FluxInputs:
    vae_scale_factor = 8
    latent_height = 2 * (height // (vae_scale_factor * 2))
    latent_width = 2 * (width // (vae_scale_factor * 2))

    num_channels_latents = transformer.config.in_channels // 4
    latents = torch.randn(
        1,
        num_channels_latents,
        latent_height,
        latent_width,
        device=device,
        dtype=dtype,
    )
    hidden_states = pack_latents(latents, 1, num_channels_latents, latent_height, latent_width)

    text_seq_len = 512
    joint_dim = transformer.config.joint_attention_dim
    encoder_hidden_states = torch.randn(1, text_seq_len, joint_dim, device=device, dtype=dtype)

    pooled_dim = transformer.config.pooled_projection_dim or joint_dim
    pooled_projections = torch.randn(1, pooled_dim, device=device, dtype=dtype)

    text_ids = torch.zeros(text_seq_len, 3, device=device, dtype=dtype)
    img_ids = prepare_latent_image_ids(latent_height // 2, latent_width // 2, device, dtype)

    timestep = torch.tensor([50.0], device=device, dtype=dtype)
    guidance = torch.tensor([3.5], device=device, dtype=dtype)

    return FluxInputs(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        pooled_projections=pooled_projections,
        text_ids=text_ids,
        img_ids=img_ids,
        timestep=timestep,
        guidance=guidance,
    )


def load_transformer(model_path: str, dtype: torch.dtype) -> FluxTransformer2DModel:
    # Load on CPU to avoid allocating baseline weights on GPU.
    transformer = FluxTransformer2DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=torch.float32,
    )
    transformer.to(torch.device("cpu"))
    transformer.eval()
    for param in transformer.parameters():
        param.requires_grad_(False)
    return transformer


def run_pass(
    mode: str,
    model_path: str,
    slab_path: str | None,
    height: int,
    width: int,
) -> Tuple[float, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for this benchmark.")

    device = torch.device("cuda")
    dtype = torch.bfloat16

    transformer = load_transformer(model_path, dtype)
    transformer.text_encoder = None
    transformer.text_encoder_2 = None
    transformer.vae = None

    torch.cuda.reset_peak_memory_stats(device)
    if mode == "bp8":
        if slab_path is None:
            raise ValueError("BP8 mode requires --slab-path.")
        print(f"Applying BP8 slab from {slab_path}")
        attach_bp8_slab(transformer, slab_path, device=device, compute_dtype=dtype)
    else:
        transformer.to(device, dtype=dtype)
    log_vram("After to(device)")

    inputs = build_dummy_inputs(transformer, height=height, width=width, dtype=dtype, device=device)

    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        _ = transformer(
            hidden_states=inputs.hidden_states,
            encoder_hidden_states=inputs.encoder_hidden_states,
            pooled_projections=inputs.pooled_projections,
            timestep=inputs.timestep / 1000.0,
            guidance=inputs.guidance,
            img_ids=inputs.img_ids,
            txt_ids=inputs.text_ids,
        )
    torch.cuda.synchronize(device)
    duration = time.perf_counter() - start
    log_vram("After forward")

    peak = torch.cuda.max_memory_allocated(device) / (1024**3)
    return peak, duration


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flux BP8 vs BF16 VRAM micro-benchmark")
    parser.add_argument(
        "--mode",
        choices=("bf16", "bp8"),
        default="bp8",
        help="Run in BF16 baseline mode or apply the BP8 slab.",
    )
    parser.add_argument(
        "--model-path",
        default="/home/alex/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev",
        help="Path to the local FLUX.1-Dev model directory.",
    )
    parser.add_argument(
        "--slab-path",
        default="output/flux1_dev_bp8_all_layers.fpk",
        help="Path to the BP8 slab (required when mode=bp8).",
    )
    parser.add_argument("--height", type=int, default=1024, help="Generation height in pixels.")
    parser.add_argument("--width", type=int, default=1024, help="Generation width in pixels.")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    peak, duration = run_pass(
        mode=args.mode,
        model_path=args.model_path,
        slab_path=args.slab_path if args.mode == "bp8" else None,
        height=args.height,
        width=args.width,
    )
    print(f"[RESULT] mode={args.mode} peak_vram={peak:.2f} GB | forward_time={duration:.3f} s")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
