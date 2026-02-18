#!/usr/bin/env python3
"""End-to-end INT8 slab test on SDXL base 1.0 UNet.

Builds a real INT8 slab from SDXL UNet weights, then validates:
  - Gate 0: Manifest schema, tensor shapes, dtypes, signature
  - Gate 1: Scaffold replacement + quant state loading on CPU
  - GPU forward pass: cosine similarity vs BF16 reference per block
  - LoRA backward pass: gradients land on A/B only, no NaN/Inf
  - Summary: slab size, compression ratio, accuracy, VRAM

Usage:
    python scripts/test_sdxl_int8_e2e.py [--output-dir /tmp/sdxl_slab]
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch import nn

# ── arg parsing ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="SDXL INT8 slab E2E test")
parser.add_argument(
    "--output-dir", type=str, default="/tmp/sdxl_int8_slab",
    help="Directory to write slab files",
)
parser.add_argument(
    "--model-id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0",
    help="HuggingFace model ID",
)
parser.add_argument(
    "--skip-gpu", action="store_true",
    help="Skip GPU forward/backward tests even if CUDA available",
)
args = parser.parse_args()

OUTPUT_DIR = args.output_dir
MODEL_ID = args.model_id
SLAB_NAME = "sdxl_unet_int8"
HAS_CUDA = torch.cuda.is_available() and not args.skip_gpu

# SDXL UNet prefixes for Linear layers
SDXL_UNET_PREFIXES = (
    "down_blocks.",
    "mid_block.",
    "up_blocks.",
    "time_embedding.",
    "add_embedding.",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

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
# STEP 1: Load SDXL UNet from HuggingFace
# ══════════════════════════════════════════════════════════════════════════
banner("STEP 1: Loading SDXL UNet from HuggingFace")

from diffusers import UNet2DConditionModel

t0 = time.time()
unet = UNet2DConditionModel.from_pretrained(
    MODEL_ID, subfolder="unet",
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)
unet.eval()
load_time = time.time() - t0

linear_count = sum(
    1 for _, m in unet.named_modules() if isinstance(m, nn.Linear)
)
total_bf16_bytes = sum(
    p.numel() * p.element_size() for p in unet.parameters()
)
ok(f"Loaded UNet in {load_time:.1f}s — {linear_count} Linear layers, "
   f"{total_bf16_bytes / 1e9:.2f} GB BF16")

# Count only the layers that match our prefixes
filtered_linear_count = sum(
    1 for n, m in unet.named_modules()
    if isinstance(m, nn.Linear)
    and any(n.startswith(p) for p in SDXL_UNET_PREFIXES)
)
print(f"  Layers matching SDXL_UNET_PREFIXES: {filtered_linear_count}/{linear_count}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 2: Build INT8 slab (safetensors format)
# ══════════════════════════════════════════════════════════════════════════
banner("STEP 2: Building INT8 slab")

from squareq.builder import build_safetensors_slab

t0 = time.time()
build_safetensors_slab(
    model=unet,
    output_dir=OUTPUT_DIR,
    slab_name=SLAB_NAME,
    architecture_id="sdxl_unet_base_1.0",
    pack_k=64,
    include_prefixes=SDXL_UNET_PREFIXES,
)
build_time = time.time() - t0

st_path = Path(OUTPUT_DIR) / f"{SLAB_NAME}.safetensors"
manifest_path = Path(OUTPUT_DIR) / f"{SLAB_NAME}.manifest.json"

slab_size = st_path.stat().st_size
manifest_size = manifest_path.stat().st_size

ok(f"Slab built in {build_time:.1f}s")
ok(f"Slab file: {slab_size / 1e9:.3f} GB ({st_path})")
ok(f"Manifest: {manifest_size / 1e3:.1f} KB ({manifest_path})")


# ══════════════════════════════════════════════════════════════════════════
# GATE 0: Manifest schema validation
# ══════════════════════════════════════════════════════════════════════════
banner("GATE 0: Manifest schema + tensor validation")

from squareq.manifest import (
    load_manifest,
    load_and_validate_manifest,
    validate_manifest_against_safetensors,
)

# 0.0: Load manifest, check all required fields
manifest = load_manifest(manifest_path)
REQUIRED_FIELDS = [
    "model_signature", "architecture_id", "quant_version",
    "kernel_abi_version", "min_runtime_version", "layer_count",
    "total_qweight_bytes",
]
raw = json.loads(manifest_path.read_text())
for field in REQUIRED_FIELDS:
    if field not in raw:
        fail(f"Missing required manifest field: {field}")
ok(f"All {len(REQUIRED_FIELDS)} required top-level fields present")

# 0.1: Check layer count matches filtered Linears
if manifest.layer_count != filtered_linear_count:
    fail(f"layer_count {manifest.layer_count} != expected {filtered_linear_count}")
ok(f"layer_count = {manifest.layer_count} (matches filtered Linear count)")

# 0.2: Per-layer required fields
LAYER_FIELDS = [
    "canonical_name", "qweight_key", "scale_key", "zero_point_key",
    "orig_shape", "packed_shape", "pad_k", "quant_bits",
    "quant_scheme", "quant_axis", "dtype_qweight", "dtype_scale",
    "dtype_zero_point",
]
for entry in manifest.layers:
    for lf in LAYER_FIELDS:
        if lf not in entry:
            fail(f"Layer '{entry.get('canonical_name', '?')}' missing field: {lf}")
ok(f"All {len(LAYER_FIELDS)} per-layer fields present in all {len(manifest.layers)} layers")

# 0.3: Tensor dtype/shape agreement
errors = validate_manifest_against_safetensors(manifest_path, st_path)
if errors:
    for e in errors:
        print(f"    ERROR: {e}")
    fail(f"{len(errors)} tensor validation errors")
ok("All tensor shapes and dtypes match manifest")

# 0.4: Signature and ABI validation
validated = load_and_validate_manifest(manifest_path, model=unet)
ok(f"Signature check passed: {validated.model_signature}")
ok(f"ABI version: {validated.kernel_abi_version}")

# 0.5: Quant bits
for entry in manifest.layers:
    if entry["quant_bits"] != 8:
        fail(f"Layer '{entry['canonical_name']}' has quant_bits={entry['quant_bits']}")
ok("All layers have quant_bits=8")

# 0.6: Verify SDXL prefixes are present
found_prefixes = set()
for entry in manifest.layers:
    for p in SDXL_UNET_PREFIXES:
        if entry["canonical_name"].startswith(p):
            found_prefixes.add(p)
ok(f"Found layers with prefixes: {sorted(found_prefixes)}")

print(f"\n  GATE 0 PASSED — all checks pass")


# ══════════════════════════════════════════════════════════════════════════
# GATE 1: Scaffold + load on CPU
# ══════════════════════════════════════════════════════════════════════════
banner("GATE 1: Scaffold replacement + CPU loading")

from squareq.scaffold import prepare_model_for_quantized_streaming
from squareq.loader import load_quant_state_from_slab
from squareq.modules import QuantLinear

# 1.0: Create a fresh UNet for scaffolding (to avoid mutating the original)
unet_scaffold = UNet2DConditionModel.from_pretrained(
    MODEL_ID, subfolder="unet",
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)
unet_scaffold.eval()

# 1.1: Replace nn.Linear -> QuantLinear
replaced = prepare_model_for_quantized_streaming(unet_scaffold, manifest)
ok(f"Scaffolded {len(replaced)} modules -> QuantLinear")

# 1.2: Verify replacement
quant_linear_count = sum(
    1 for _, m in unet_scaffold.named_modules() if isinstance(m, QuantLinear)
)
if quant_linear_count != manifest.layer_count:
    fail(f"QuantLinear count {quant_linear_count} != manifest {manifest.layer_count}")
ok(f"All {quant_linear_count} layers are now QuantLinear")

# 1.3: Load INT8 state from slab
loaded = load_quant_state_from_slab(unet_scaffold, manifest, st_path, device="cpu")
if loaded != manifest.layer_count:
    fail(f"Loaded {loaded} layers, expected {manifest.layer_count}")
ok(f"Loaded quant state into {loaded} modules on CPU")

# 1.4: Verify INT8 buffers
for name, mod in unet_scaffold.named_modules():
    if not isinstance(mod, QuantLinear):
        continue
    if mod.qweight.dtype != torch.int8:
        fail(f"{name}.qweight dtype = {mod.qweight.dtype}, expected int8")
    if mod.scale.dtype != torch.float32:
        fail(f"{name}.scale dtype = {mod.scale.dtype}, expected float32")
    if mod.qweight.numel() == 0:
        fail(f"{name}.qweight is empty after loading")
ok("All QuantLinear modules have valid INT8 qweight + FP32 scale buffers")

# 1.5: Verify no float .weight Parameter exists
for name, mod in unet_scaffold.named_modules():
    if isinstance(mod, QuantLinear):
        if hasattr(mod, 'weight') and isinstance(getattr(mod, 'weight', None), nn.Parameter):
            fail(f"{name} has a float .weight Parameter (should not exist)")
ok("No float .weight Parameters on any QuantLinear")

print(f"\n  GATE 1 PASSED — scaffold + load verified on CPU")


# ══════════════════════════════════════════════════════════════════════════
# GPU forward pass: compare quantized vs BF16 reference
# ══════════════════════════════════════════════════════════════════════════
if HAS_CUDA:
    banner("GPU TEST: Forward pass accuracy (INT8 vs BF16)")

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    vram_before = torch.cuda.memory_allocated()

    # Pick representative layers from each major block group
    TARGET_LAYERS = [
        "time_embedding.linear_1",
        "add_embedding.linear_1",
        "down_blocks.0.resnets.0.time_emb_proj",
        "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q",
        "down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_q",
        "down_blocks.2.attentions.1.transformer_blocks.9.ff.net.0.proj",
        "mid_block.attentions.0.transformer_blocks.0.attn1.to_q",
        "mid_block.attentions.0.transformer_blocks.5.ff.net.2",
        "up_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q",
        "up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k",
        "up_blocks.2.attentions.2.transformer_blocks.0.attn1.to_q",
    ]
    manifest_names = {e["canonical_name"] for e in manifest.layers}
    test_layers = [n for n in TARGET_LAYERS if n in manifest_names]

    if len(test_layers) < 5:
        # Fallback: first layer from each prefix
        seen = set()
        for entry in manifest.layers:
            name = entry["canonical_name"]
            for p in SDXL_UNET_PREFIXES:
                if name.startswith(p) and p not in seen:
                    test_layers.append(name)
                    seen.add(p)
                    break

    # Build module maps for both models
    orig_map = dict(unet.named_modules())
    quant_map = dict(unet_scaffold.named_modules())

    cos_results = []
    for layer_name in test_layers:
        orig_mod = orig_map.get(layer_name)
        quant_mod = quant_map.get(layer_name)
        if orig_mod is None or quant_mod is None:
            continue

        in_f = orig_mod.in_features if hasattr(orig_mod, 'in_features') else quant_mod.in_features
        # Create synthetic input
        torch.manual_seed(42)
        x = torch.randn(1, 4, in_f, dtype=torch.bfloat16, device=device)

        # BF16 reference
        orig_mod_gpu = orig_mod.to(device)
        with torch.no_grad():
            ref_out = orig_mod_gpu(x)
        orig_mod.cpu()

        # INT8 quantized
        quant_mod_gpu = quant_mod.to(device)
        with torch.no_grad():
            q_out = quant_mod_gpu(x)
        quant_mod.cpu()

        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            ref_out.float().flatten(),
            q_out.float().flatten(),
            dim=0,
        ).item()

        cos_results.append((layer_name, cos_sim))
        status = "PASS" if cos_sim > 0.98 else "WARN" if cos_sim > 0.95 else "FAIL"
        print(f"  [{status}] {layer_name}: cosine_sim = {cos_sim:.6f}")

        # Free GPU memory
        del ref_out, q_out, x
        torch.cuda.empty_cache()

    avg_cos = sum(c for _, c in cos_results) / len(cos_results) if cos_results else 0
    min_cos = min(c for _, c in cos_results) if cos_results else 0

    if min_cos < 0.98:
        print(f"\n  WARNING: min cosine similarity {min_cos:.6f} < 0.98")
    else:
        ok(f"All {len(cos_results)} layers pass cosine > 0.98")

    ok(f"Average cosine similarity: {avg_cos:.6f}")
    ok(f"Min cosine similarity: {min_cos:.6f}")

    peak_vram_forward = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    print(f"  Peak VRAM (forward): {peak_vram_forward / 1e9:.2f} GB")

    # ══════════════════════════════════════════════════════════════════════
    # LoRA backward pass through quantized path
    # ══════════════════════════════════════════════════════════════════════
    banner("GPU TEST: LoRA backward pass on quantized layer")

    from squareq.modules import QuantLinearLoRA

    # Pick a representative layer for LoRA test
    test_entry = manifest.layers[0]
    lora_name = test_entry["canonical_name"]
    out_f = test_entry["orig_shape"][0]
    in_f = test_entry["orig_shape"][1]
    has_bias = test_entry.get("bias_key") is not None

    print(f"  Testing LoRA on: {lora_name} [{out_f}x{in_f}]")

    # Create QuantLinearLoRA module
    lora_mod = QuantLinearLoRA(
        in_features=in_f,
        out_features=out_f,
        rank=8,
        alpha=1.0,
        bias=has_bias,
        compute_dtype=torch.bfloat16,
    )

    # Load INT8 state from slab for this specific layer
    from safetensors.torch import load_file as st_load_file
    slab_tensors = st_load_file(str(st_path), device="cpu")

    lora_mod.set_quant_state(
        qweight=slab_tensors[test_entry["qweight_key"]],
        scale=slab_tensors[test_entry["scale_key"]],
        zero_point=slab_tensors[test_entry["zero_point_key"]],
        bias=slab_tensors[test_entry["bias_key"]] if has_bias else None,
    )

    lora_mod = lora_mod.to(device)
    torch.cuda.reset_peak_memory_stats()

    # Forward + backward
    torch.manual_seed(42)
    x = torch.randn(2, 16, in_f, dtype=torch.bfloat16, device=device)
    x.requires_grad_(True)

    out = lora_mod(x)
    loss = out.sum()
    loss.backward()

    # Check gradients
    grad_a = lora_mod.lora_A.grad
    grad_b = lora_mod.lora_B.grad
    if grad_a is None:
        fail("lora_A has no gradient!")
    if grad_b is None:
        fail("lora_B has no gradient!")

    # Verify no NaN/Inf in gradients
    if torch.isnan(grad_a).any() or torch.isinf(grad_a).any():
        fail("lora_A gradient contains NaN or Inf!")
    if torch.isnan(grad_b).any() or torch.isinf(grad_b).any():
        fail("lora_B gradient contains NaN or Inf!")

    ok(f"lora_A grad shape: {list(grad_a.shape)}, norm: {grad_a.float().norm():.6f}")
    ok(f"lora_B grad shape: {list(grad_b.shape)}, norm: {grad_b.float().norm():.6f}")

    # Verify INT8 buffers have NO gradient
    if lora_mod.qweight.requires_grad:
        fail("qweight should not require gradients!")
    if lora_mod.scale.requires_grad:
        fail("scale should not require gradients!")
    ok("INT8 buffers (qweight, scale, zero_point) are frozen — no gradients")

    # Verify x got gradients too (for training)
    if x.grad is None:
        fail("Input x has no gradient — backward not flowing through")
    ok(f"Input gradient flows through quantized path")

    peak_vram_backward = torch.cuda.max_memory_allocated()
    print(f"  Peak VRAM (LoRA backward): {peak_vram_backward / 1e9:.2f} GB")

    # Cleanup
    del lora_mod, x, out, loss, grad_a, grad_b, slab_tensors
    torch.cuda.empty_cache()
    gc.collect()

else:
    banner("GPU TESTS SKIPPED — no CUDA available or --skip-gpu")
    peak_vram_forward = 0
    peak_vram_backward = 0
    cos_results = []
    avg_cos = 0
    min_cos = 0


# ══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════
banner("SUMMARY")

# Compute BF16 size of only the quantized layers
bf16_linear_bytes = 0
for name, mod in unet.named_modules():
    if isinstance(mod, nn.Linear) and any(name.startswith(p) for p in SDXL_UNET_PREFIXES):
        bf16_linear_bytes += mod.weight.numel() * 2  # BF16 = 2 bytes
        if mod.bias is not None:
            bf16_linear_bytes += mod.bias.numel() * 2

compression_ratio = bf16_linear_bytes / slab_size if slab_size > 0 else 0

print(f"""
  Model:               {MODEL_ID} (UNet only)
  Architecture ID:     sdxl_unet_base_1.0
  Linear layers:       {manifest.layer_count} (filtered by SDXL prefixes)
  Total Linear layers: {linear_count} (in full UNet)

  Original BF16 size (Linear weights): {bf16_linear_bytes / 1e9:.3f} GB
  INT8 slab size on disk:              {slab_size / 1e9:.3f} GB
  Manifest size:                       {manifest_size / 1e3:.1f} KB
  Compression ratio:                   {compression_ratio:.2f}x

  Build time:         {build_time:.1f}s
  Model load time:    {load_time:.1f}s
""")

if HAS_CUDA and cos_results:
    print(f"  Per-block accuracy (cosine similarity):")
    for name, cos in cos_results:
        short = name[:60] + "..." if len(name) > 60 else name
        print(f"    {short:<63s} {cos:.6f}")
    print(f"\n  Average cosine sim:  {avg_cos:.6f}")
    print(f"  Min cosine sim:      {min_cos:.6f}")
    print(f"  Peak VRAM (forward): {peak_vram_forward / 1e9:.2f} GB")
    print(f"  Peak VRAM (backward):{peak_vram_backward / 1e9:.2f} GB")

print(f"""
  Gate 0 (manifest schema):   PASSED
  Gate 1 (scaffold + load):   PASSED
  GPU forward accuracy:        {'PASSED' if HAS_CUDA and min_cos > 0.98 else 'SKIPPED' if not HAS_CUDA else 'FAILED'}
  LoRA backward pass:          {'PASSED' if HAS_CUDA else 'SKIPPED'}
""")

# Cleanup
del unet, unet_scaffold
gc.collect()
if HAS_CUDA:
    torch.cuda.empty_cache()

print("Done.")
