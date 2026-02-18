#!/usr/bin/env python3
"""SDXL INT8 LoRA training validation — real UNet forward/backward/optimizer.

Validates end-to-end LoRA training through the INT8-quantized SDXL UNet:
  1. Load UNet, scaffold with QuantLinearLoRA, load INT8 slab
  2. Freeze everything except LoRA params
  3. Run forward pass with synthetic latents + text embeddings + timestep
  4. Compute MSE loss, backward, optimizer step
  5. Verify: finite loss, LoRA params changed, INT8 frozen

Usage:
    python scripts/test_sdxl_int8_training.py [--slab-dir /tmp/sdxl_int8_slab]
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import torch
from torch import nn

# ── arg parsing ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="SDXL INT8 LoRA training validation")
parser.add_argument(
    "--slab-dir", type=str, default="/tmp/sdxl_int8_slab",
    help="Directory containing the pre-built INT8 slab",
)
parser.add_argument(
    "--model-id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0",
    help="HuggingFace model ID",
)
parser.add_argument("--rank", type=int, default=8, help="LoRA rank")
parser.add_argument("--alpha", type=float, default=1.0, help="LoRA alpha")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--steps", type=int, default=5, help="Training steps")
args = parser.parse_args()

SLAB_DIR = Path(args.slab_dir)
MODEL_ID = args.model_id
SLAB_NAME = "sdxl_unet_int8"

if not torch.cuda.is_available():
    print("ERROR: CUDA required for training validation.")
    sys.exit(1)

st_path = SLAB_DIR / f"{SLAB_NAME}.safetensors"
manifest_path = SLAB_DIR / f"{SLAB_NAME}.manifest.json"

if not st_path.exists() or not manifest_path.exists():
    print(f"ERROR: Slab files not found in {SLAB_DIR}")
    print("  Run test_sdxl_int8_e2e.py first to build the slab.")
    sys.exit(1)


def banner(msg: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {msg}")
    print(f"{'=' * 70}")


def ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════
# STEP 1: Load UNet + scaffold with QuantLinearLoRA
# ══════════════════════════════════════════════════════════════════════════
banner("STEP 1: Load UNet + scaffold with QuantLinearLoRA")

from diffusers import UNet2DConditionModel

t0 = time.time()
unet = UNet2DConditionModel.from_pretrained(
    MODEL_ID, subfolder="unet",
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)
ok(f"UNet loaded in {time.time() - t0:.1f}s")

from squareq.manifest import load_manifest
from squareq.scaffold import prepare_model_for_quantized_lora_training
from squareq.loader import load_quant_state_from_slab
from squareq.modules import QuantLinearLoRA

manifest = load_manifest(manifest_path)

t0 = time.time()
replaced = prepare_model_for_quantized_lora_training(
    unet, manifest, rank=args.rank, alpha=args.alpha,
)
scaffold_time = time.time() - t0
ok(f"Scaffolded {len(replaced)} layers -> QuantLinearLoRA (rank={args.rank}) in {scaffold_time:.2f}s")

# Load INT8 state
t0 = time.time()
loaded = load_quant_state_from_slab(unet, manifest, st_path, device="cpu")
load_time = time.time() - t0
ok(f"Loaded INT8 state into {loaded} modules in {load_time:.1f}s")

# Verify all QuantLinearLoRA have populated buffers
for name, mod in unet.named_modules():
    if isinstance(mod, QuantLinearLoRA):
        if mod.qweight.numel() == 0:
            fail(f"{name}: qweight is empty after loading")
ok("All QuantLinearLoRA modules have populated INT8 buffers")


# ══════════════════════════════════════════════════════════════════════════
# STEP 2: Freeze base, unfreeze LoRA params
# ══════════════════════════════════════════════════════════════════════════
banner("STEP 2: Freeze base weights, unfreeze LoRA params")

# Freeze everything first
for p in unet.parameters():
    p.requires_grad_(False)

# Unfreeze only LoRA A/B parameters
lora_params = []
for name, mod in unet.named_modules():
    if isinstance(mod, QuantLinearLoRA):
        mod.lora_A.requires_grad_(True)
        mod.lora_B.requires_grad_(True)
        lora_params.extend([mod.lora_A, mod.lora_B])

total_params = sum(p.numel() for p in unet.parameters())
trainable_params = sum(p.numel() for p in lora_params)
frozen_params = total_params - trainable_params

ok(f"Total params:     {total_params:,}")
ok(f"Trainable (LoRA): {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
ok(f"Frozen:           {frozen_params:,}")

# Snapshot INT8 buffers for later comparison
int8_snapshots: dict[str, torch.Tensor] = {}
for name, mod in unet.named_modules():
    if isinstance(mod, QuantLinearLoRA):
        int8_snapshots[name] = mod.qweight.clone()

# Snapshot LoRA params for checking they changed
lora_a_snapshots: dict[str, torch.Tensor] = {}
lora_b_snapshots: dict[str, torch.Tensor] = {}
for name, mod in unet.named_modules():
    if isinstance(mod, QuantLinearLoRA):
        lora_a_snapshots[name] = mod.lora_A.data.clone()
        lora_b_snapshots[name] = mod.lora_B.data.clone()


# ══════════════════════════════════════════════════════════════════════════
# STEP 3: Set up optimizer + synthetic training data
# ══════════════════════════════════════════════════════════════════════════
banner("STEP 3: Set up optimizer + synthetic data")

optimizer = torch.optim.AdamW(lora_params, lr=args.lr)
ok(f"AdamW optimizer with lr={args.lr}, {len(lora_params)} param groups")

# Move UNet to GPU
unet = unet.to("cuda")
torch.cuda.reset_peak_memory_stats()

# SDXL UNet expects:
#   sample: (B, 4, H/8, W/8) — latent noise
#   timestep: scalar or (B,) — diffusion timestep
#   encoder_hidden_states: (B, seq_len, 2048) — CLIP text embeddings
#   added_cond_kwargs: {"text_embeds": (B, 1280), "time_ids": (B, 6)}
B = 1
H, W = 64, 64  # 512x512 latent space
SEQ_LEN = 77

# Create synthetic batch (deterministic)
torch.manual_seed(42)
sample = torch.randn(B, 4, H, W, dtype=torch.bfloat16, device="cuda")
timestep = torch.tensor([500], device="cuda")
encoder_hidden_states = torch.randn(B, SEQ_LEN, 2048, dtype=torch.bfloat16, device="cuda")
text_embeds = torch.randn(B, 1280, dtype=torch.bfloat16, device="cuda")
time_ids = torch.zeros(B, 6, dtype=torch.bfloat16, device="cuda")
# Standard SDXL time_ids: [orig_h, orig_w, crop_top, crop_left, target_h, target_w]
time_ids[0] = torch.tensor([512, 512, 0, 0, 512, 512], dtype=torch.bfloat16)

# Target for MSE loss (random noise prediction target)
target = torch.randn_like(sample)

ok(f"Synthetic batch: sample={list(sample.shape)}, "
   f"hidden_states={list(encoder_hidden_states.shape)}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 4: Training loop
# ══════════════════════════════════════════════════════════════════════════
banner(f"STEP 4: Training loop ({args.steps} steps)")

unet.train()
losses = []

for step in range(args.steps):
    optimizer.zero_grad()

    # Forward pass
    t0 = time.time()
    output = unet(
        sample=sample,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
    )
    pred = output.sample
    fwd_time = time.time() - t0

    # MSE loss
    loss = torch.nn.functional.mse_loss(pred, target)
    loss_val = loss.item()
    losses.append(loss_val)

    # Backward
    t0 = time.time()
    loss.backward()
    bwd_time = time.time() - t0

    # Check for NaN/Inf before stepping
    if not torch.isfinite(loss):
        fail(f"Step {step}: loss is {loss_val} (not finite)")

    has_nan_grad = False
    for p in lora_params:
        if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
            has_nan_grad = True
            break
    if has_nan_grad:
        fail(f"Step {step}: NaN/Inf in LoRA gradients")

    # Optimizer step
    optimizer.step()

    print(f"  Step {step}: loss={loss_val:.6f}  fwd={fwd_time:.2f}s  bwd={bwd_time:.2f}s")

ok(f"All {args.steps} steps completed with finite loss")
ok(f"Loss trend: {losses[0]:.6f} -> {losses[-1]:.6f}")

peak_vram = torch.cuda.max_memory_allocated()
ok(f"Peak VRAM: {peak_vram / 1e9:.2f} GB")


# ══════════════════════════════════════════════════════════════════════════
# STEP 5: Verification
# ══════════════════════════════════════════════════════════════════════════
banner("STEP 5: Post-training verification")

# 5.1: LoRA params changed
lora_a_changed = 0
lora_b_changed = 0
for name, mod in unet.named_modules():
    if isinstance(mod, QuantLinearLoRA):
        if not torch.equal(mod.lora_A.data.cpu(), lora_a_snapshots[name]):
            lora_a_changed += 1
        if not torch.equal(mod.lora_B.data.cpu(), lora_b_snapshots[name]):
            lora_b_changed += 1

ok(f"LoRA A changed: {lora_a_changed}/{len(replaced)} layers")
ok(f"LoRA B changed: {lora_b_changed}/{len(replaced)} layers")

if lora_a_changed == 0:
    fail("No LoRA A parameters changed — optimizer did nothing")

# 5.2: INT8 base weights unchanged
int8_changed = 0
for name, mod in unet.named_modules():
    if isinstance(mod, QuantLinearLoRA):
        if not torch.equal(mod.qweight.cpu(), int8_snapshots[name]):
            int8_changed += 1

if int8_changed > 0:
    fail(f"{int8_changed} INT8 base weights were modified during training!")
ok(f"All {len(int8_snapshots)} INT8 base weight buffers are unchanged")

# 5.3: INT8 buffers still correct dtype
for name, mod in unet.named_modules():
    if isinstance(mod, QuantLinearLoRA):
        if mod.qweight.dtype != torch.int8:
            fail(f"{name}.qweight dtype = {mod.qweight.dtype}, expected int8")
        if mod.scale.dtype != torch.float32:
            fail(f"{name}.scale dtype = {mod.scale.dtype}, expected float32")
ok("All INT8 buffers preserved correct dtypes (int8/float32)")


# ══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════
banner("SUMMARY")

print(f"""
  Model:               {MODEL_ID} (UNet only)
  LoRA rank:           {args.rank}
  LoRA alpha:          {args.alpha}
  Learning rate:       {args.lr}
  Training steps:      {args.steps}

  Quantized layers:    {len(replaced)}
  Total params:        {total_params:,}
  Trainable (LoRA):    {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)

  Loss trend:          {' -> '.join(f'{l:.4f}' for l in losses)}
  Peak VRAM:           {peak_vram / 1e9:.2f} GB

  LoRA A changed:      {lora_a_changed}/{len(replaced)} layers
  LoRA B changed:      {lora_b_changed}/{len(replaced)} layers
  INT8 base modified:  {int8_changed}/{len(int8_snapshots)} layers (expected: 0)

  Scaffold time:       {scaffold_time:.2f}s
  Slab load time:      {load_time:.1f}s

  VERDICT:             {"PASSED" if int8_changed == 0 and lora_a_changed > 0 else "FAILED"}
""")

# Cleanup
del unet, optimizer, sample, target
gc.collect()
torch.cuda.empty_cache()

print("Done.")
