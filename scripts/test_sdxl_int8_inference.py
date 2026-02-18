#!/usr/bin/env python3
"""SDXL INT8 inference validation — full pipeline image generation.

Generates an image using the full StableDiffusionXLPipeline with:
  1. BF16 reference UNet (original)
  2. INT8 quantized UNet (scaffolded + slab-loaded)

Compares SSIM / pixel difference and saves side-by-side output.

Usage:
    python scripts/test_sdxl_int8_inference.py [--slab-dir /tmp/sdxl_int8_slab]
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import torch

# ── arg parsing ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="SDXL INT8 inference validation")
parser.add_argument(
    "--slab-dir", type=str, default="/tmp/sdxl_int8_slab",
    help="Directory containing the pre-built INT8 slab",
)
parser.add_argument(
    "--model-id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0",
    help="HuggingFace model ID",
)
parser.add_argument(
    "--output-dir", type=str, default="/tmp/sdxl_int8_inference",
    help="Directory to save output images",
)
parser.add_argument("--steps", type=int, default=20, help="Diffusion steps")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
args = parser.parse_args()

SLAB_DIR = Path(args.slab_dir)
MODEL_ID = args.model_id
OUTPUT_DIR = Path(args.output_dir)
SLAB_NAME = "sdxl_unet_int8"
PROMPT = "a photo of a cat sitting on a red couch, high quality, detailed"
NEGATIVE_PROMPT = "blurry, low quality"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not torch.cuda.is_available():
    print("ERROR: CUDA required for pipeline inference.")
    sys.exit(1)

st_path = SLAB_DIR / f"{SLAB_NAME}.safetensors"
manifest_path = SLAB_DIR / f"{SLAB_NAME}.manifest.json"

if not st_path.exists() or not manifest_path.exists():
    print(f"ERROR: Slab files not found in {SLAB_DIR}")
    print(f"  Expected: {st_path}")
    print(f"  Expected: {manifest_path}")
    print("  Run test_sdxl_int8_e2e.py first to build the slab.")
    sys.exit(1)


def banner(msg: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {msg}")
    print(f"{'=' * 70}")


def ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


# ══════════════════════════════════════════════════════════════════════════
# STEP 1: Generate reference image with BF16 UNet
# ══════════════════════════════════════════════════════════════════════════
banner("STEP 1: Generate reference image (BF16 UNet)")

from diffusers import StableDiffusionXLPipeline

t0 = time.time()
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)
pipe = pipe.to("cuda")
load_time = time.time() - t0
ok(f"Pipeline loaded in {load_time:.1f}s")

generator = torch.Generator(device="cuda").manual_seed(args.seed)
torch.cuda.reset_peak_memory_stats()

t0 = time.time()
ref_image = pipe(
    prompt=PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    num_inference_steps=args.steps,
    generator=generator,
    guidance_scale=7.5,
    width=1024,
    height=1024,
).images[0]
gen_time = time.time() - t0
peak_vram_ref = torch.cuda.max_memory_allocated()

ref_path = OUTPUT_DIR / "reference_bf16.png"
ref_image.save(ref_path)
ok(f"Reference image saved: {ref_path}")
ok(f"Generation time: {gen_time:.1f}s, Peak VRAM: {peak_vram_ref / 1e9:.2f} GB")

# Free BF16 pipeline
del pipe
gc.collect()
torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════════════════
# STEP 2: Load pipeline, scaffold UNet with INT8, generate same image
# ══════════════════════════════════════════════════════════════════════════
banner("STEP 2: Generate quantized image (INT8 UNet)")

from squareq.manifest import load_manifest
from squareq.scaffold import prepare_model_for_quantized_streaming
from squareq.loader import load_quant_state_from_slab

# Load pipeline again (need fresh UNet for scaffolding)
t0 = time.time()
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)
ok(f"Pipeline reloaded in {time.time() - t0:.1f}s")

# Scaffold UNet: replace nn.Linear -> QuantLinear
manifest = load_manifest(manifest_path)
replaced = prepare_model_for_quantized_streaming(pipe.unet, manifest)
ok(f"Scaffolded {len(replaced)} layers -> QuantLinear")

# Load INT8 state from slab
loaded = load_quant_state_from_slab(pipe.unet, manifest, st_path, device="cpu")
ok(f"Loaded INT8 state into {loaded} modules")

# Move pipeline to GPU — QuantLinear.to() strips dtype, protecting INT8
pipe = pipe.to("cuda")

# Verify UNet dtype detection still works (pipeline uses this internally)
from diffusers.utils import get_parameter_dtype  # type: ignore[attr-defined]
try:
    unet_dtype = get_parameter_dtype(pipe.unet)
    ok(f"UNet detected dtype: {unet_dtype} (pipeline encode_prompt uses this)")
except Exception:
    # Some diffusers versions use a different function
    unet_dtype = next(pipe.unet.parameters()).dtype
    ok(f"UNet first param dtype: {unet_dtype}")

# Generate with same seed
generator = torch.Generator(device="cuda").manual_seed(args.seed)
torch.cuda.reset_peak_memory_stats()

t0 = time.time()
quant_image = pipe(
    prompt=PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    num_inference_steps=args.steps,
    generator=generator,
    guidance_scale=7.5,
    width=1024,
    height=1024,
).images[0]
quant_time = time.time() - t0
peak_vram_quant = torch.cuda.max_memory_allocated()

quant_path = OUTPUT_DIR / "quantized_int8.png"
quant_image.save(quant_path)
ok(f"Quantized image saved: {quant_path}")
ok(f"Generation time: {quant_time:.1f}s, Peak VRAM: {peak_vram_quant / 1e9:.2f} GB")


# ══════════════════════════════════════════════════════════════════════════
# STEP 3: Compare images
# ══════════════════════════════════════════════════════════════════════════
banner("STEP 3: Image comparison")

import numpy as np
from PIL import Image

ref_np = np.array(ref_image).astype(np.float64)
quant_np = np.array(quant_image).astype(np.float64)

# Mean Absolute Error (pixel-level)
mae = np.mean(np.abs(ref_np - quant_np))
ok(f"Mean Absolute Error: {mae:.2f} (out of 255)")

# Peak Signal-to-Noise Ratio
mse = np.mean((ref_np - quant_np) ** 2)
if mse == 0:
    psnr = float("inf")
else:
    psnr = 10 * np.log10(255.0 ** 2 / mse)
ok(f"PSNR: {psnr:.2f} dB")

# SSIM (per-channel, averaged)
try:
    from skimage.metrics import structural_similarity as ssim
    ssim_val = ssim(ref_np, quant_np, channel_axis=2, data_range=255)
    ok(f"SSIM: {ssim_val:.6f}")
except ImportError:
    # Manual SSIM approximation
    print("  [INFO] scikit-image not available, computing basic correlation")
    correlation = np.corrcoef(ref_np.flatten(), quant_np.flatten())[0, 1]
    ssim_val = correlation
    ok(f"Pearson correlation: {ssim_val:.6f}")

# Create side-by-side comparison
side_by_side = Image.new("RGB", (2048, 1024))
side_by_side.paste(ref_image, (0, 0))
side_by_side.paste(quant_image, (1024, 0))
comparison_path = OUTPUT_DIR / "comparison_bf16_vs_int8.png"
side_by_side.save(comparison_path)
ok(f"Side-by-side saved: {comparison_path}")

# Difference map (amplified for visibility)
diff = np.abs(ref_np - quant_np).astype(np.uint8)
diff_amplified = np.clip(diff * 10, 0, 255).astype(np.uint8)
diff_image = Image.fromarray(diff_amplified)
diff_path = OUTPUT_DIR / "difference_map_10x.png"
diff_image.save(diff_path)
ok(f"Difference map (10x amplified) saved: {diff_path}")


# ══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════
banner("SUMMARY")

print(f"""
  Model:               {MODEL_ID}
  Prompt:              "{PROMPT}"
  Steps:               {args.steps}
  Seed:                {args.seed}
  Resolution:          1024x1024

  BF16 generation:     {gen_time:.1f}s  (Peak VRAM: {peak_vram_ref / 1e9:.2f} GB)
  INT8 generation:     {quant_time:.1f}s  (Peak VRAM: {peak_vram_quant / 1e9:.2f} GB)

  SSIM:                {ssim_val:.6f}
  PSNR:                {psnr:.2f} dB
  MAE:                 {mae:.2f} / 255

  Output files:
    {ref_path}
    {quant_path}
    {comparison_path}
    {diff_path}
""")

quality = "EXCELLENT" if ssim_val > 0.95 else "GOOD" if ssim_val > 0.90 else "FAIR" if ssim_val > 0.80 else "POOR"
print(f"  Quality assessment: {quality} (SSIM {ssim_val:.4f})")

# Cleanup
del pipe
gc.collect()
torch.cuda.empty_cache()

print("\nDone.")
