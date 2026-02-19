# SQUARE-Q

INT8 quantized block-swapping for diffusion model training on consumer GPUs.

SquareQ stores model weights as an INT8 slab on disk/CPU and streams one block at a time into VRAM. It computes directly on INT8 weights with a fused kernel then evicts the block, so the full model never lives in GPU memory. Base weights stay frozen in INT8; only small LoRA adapters receive FP32 gradients, so backprop fits on 24 GB.

> **Honest scope:** This is not the fastest INT8 library nor a new quantization algorithm. It's a practical, working stack to fine-tune diffusion models on 24 GB by combining quantization **and** block-level streaming.

---

## Supported Models

| Model | Layers | BF16 Size | INT8 Slab | Ratio | Cosine Sim | Status |
|-------|--------|-----------|-----------|-------|------------|--------|
| **SDXL Base 1.0** (UNet) | 743 | 4.47 GB | 2.25 GB | 1.99x | 0.9999 avg | Verified |
| **Flux 1 Dev** (57 blocks) | all | ~22 GB | ~11 GB | ~2x | - | Verified |

More models coming (SD 1.5, SD3, Flux 2 Klein, Chroma, etc).

---

## Key Features

- **V2 safetensors slab format**: `{name}.safetensors` + `{name}.manifest.json`. No more `.fpk` torch.save.
- **Per-row symmetric INT8**: `qweight:int8` + `scale:fp32` + `zero_point:fp32` (+ optional `bias:fp32`) for every targeted layer. No float weights in the slab.
- **`include_prefixes` filtering**: quantize only layers matching specific prefixes. Pass `None` to quantize all `nn.Linear` modules (default), or filter by architecture — e.g. SDXL uses `("down_blocks.", "mid_block.", "up_blocks.", "time_embedding.", "add_embedding.")`.
- **CPU-stage loader**: replaces `nn.Linear` with `QuantLinear` wrappers before any GPU cast; only INT8/FP32 buffers land on GPU.
- **Triton fused matmul**: INT8 x BF16 -> FP32 accumulate with optional fused epilogue (bias/GeLU); autotune tiles.
- **LoRA training path**: `QuantLinearLoRA` adds FP32 adapters (A/B) while base INT8 is frozen. Gradients land on A/B only.
- **dtype protection**: `QuantLinear.to(dtype=...)` and `_apply()` preserve INT8 buffers — no silent promotion to BF16.
- **291 gate tests** (Gates 0-15) covering format, scaffold, residency, materialization, LoRA, accuracy, registry, staging, scheduler, E2E, error handling, cleanup, forward accuracy, eviction stability, dtype protection, and telemetry.

---

## Install

```bash
git clone https://github.com/CodeAlexx/SquareQ.git
cd SquareQ
pip install -e .
```

---

## Quickstart

### 1) Build an INT8 slab (Python API)

```python
from diffusers import UNet2DConditionModel
from squareq.builder import build_safetensors_slab
import torch

# Load model
unet = UNet2DConditionModel.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    subfolder="unet", torch_dtype=torch.bfloat16,
)

# Build slab — only quantize UNet linear layers
build_safetensors_slab(
    model=unet,
    output_dir="./output",
    slab_name="sdxl_unet_int8",
    architecture_id="sdxl_unet_base_1.0",
    pack_k=64,
    include_prefixes=("down_blocks.", "mid_block.", "up_blocks.",
                      "time_embedding.", "add_embedding."),
)
# Output: output/sdxl_unet_int8.safetensors + output/sdxl_unet_int8.manifest.json
```

### 2) Load and use quantized model

```python
from squareq.manifest import load_manifest
from squareq.scaffold import prepare_model_for_quantized_streaming
from squareq.loader import load_quant_state_from_slab

manifest = load_manifest("output/sdxl_unet_int8.manifest.json")
prepare_model_for_quantized_streaming(unet, manifest)   # nn.Linear -> QuantLinear
load_quant_state_from_slab(unet, manifest, "output/sdxl_unet_int8.safetensors")
# All 743 layers now backed by INT8 buffers — no float weights
```

### 3) Scaffold for LoRA training

```python
from squareq.scaffold import prepare_model_for_quantized_lora_training

manifest = load_manifest("output/sdxl_unet_int8.manifest.json")
prepare_model_for_quantized_lora_training(unet, manifest, rank=8, alpha=1.0)
load_quant_state_from_slab(unet, manifest, "output/sdxl_unet_int8.safetensors")
# All 743 layers now QuantLinearLoRA: frozen INT8 base + trainable LoRA A/B
```

### 4) Run validation scripts

```bash
# Build slab + gate checks + per-layer cosine sim + LoRA backward
python scripts/test_sdxl_int8_e2e.py --output-dir /tmp/sdxl_slab

# Full pipeline inference: BF16 vs INT8 side-by-side image + SSIM
python scripts/test_sdxl_int8_inference.py --slab-dir /tmp/sdxl_int8_slab

# LoRA training: forward/backward/optimizer steps on quantized UNet
python scripts/test_sdxl_int8_training.py --slab-dir /tmp/sdxl_int8_slab
```

---

## SDXL Validation Results (2026-02-18)

```
Model:               stabilityai/stable-diffusion-xl-base-1.0 (UNet)
Linear layers:       743
Original BF16 size:  4.467 GB
INT8 slab on disk:   2.248 GB
Compression ratio:   1.99x
Build time:          8.3s

Per-block cosine similarity (INT8 vs BF16):
  time_embedding.linear_1                          0.999980
  add_embedding.linear_1                           0.999655
  down_blocks.0.resnets.0.time_emb_proj            0.999880
  down_blocks.1.*.attn1.to_q                       0.999967
  down_blocks.2.*.attn1.to_q                       0.999968
  down_blocks.2.*.ff.net.0.proj                    0.999968
  mid_block.*.attn1.to_q                           0.999963
  mid_block.*.ff.net.2                             0.999955
  up_blocks.0.*.attn1.to_q                         0.999965
  up_blocks.1.*.attn2.to_k                         0.999950
  Average:  0.999925    Min:  0.999655

Gate 0 (manifest schema):  PASSED
Gate 1 (scaffold + load):  PASSED
GPU forward accuracy:      PASSED (all > 0.98)
LoRA backward pass:        PASSED (grads on A/B only, no NaN/Inf)
```

---

## SDXL INT8 LoRA Training + Inference

SDXL is the first model to complete the full SquareQ pipeline: quantize to INT8 slab, train LoRA on the frozen INT8 base, and run inference with the trained LoRA applied.

<p align="center">
  <img src="assets/sdxl_int8_lora_inference.png" width="512" />
</p>

<p align="center"><em>1024x1024 generation from SquareQ INT8 UNet + trained LoRA (Euler, Karras schedule, CFG 4.0)</em></p>

**Training config:**

| Parameter | Value |
|-----------|-------|
| Base model | `stabilityai/stable-diffusion-xl-base-1.0` |
| INT8 slab | 743 layers, 2.25 GB |
| Training method | LoRA (rank 16, alpha 16) |
| Resolution | 1024x1024 |
| Learning rate | 1e-4 |
| Precision | bfloat16 |
| Memory strategy | Stagehand block-swapping |
| LoRA output | 560 target modules, 46.6 MB |

**Inference stats (RTX 3090 Ti):**

| | INT8 UNet | INT8 + LoRA |
|---|-----------|-------------|
| Peak VRAM | 7.8 GB | 7.8 GB |
| Denoise time (20 steps) | 14.2s | 18.3s |

LoRA is applied at inference via forward hooks on each `QuantLinear` module -- no weight merging or dequantization needed.

---

## Architecture

### Slab Format (V2)

Output: `{slab_name}.safetensors` + `{slab_name}.manifest.json`

Tensors per layer:
- `{name}.qweight` — INT8, shape `[out_features, padded_in_features]`
- `{name}.scale` — FP32, shape `[out_features]`
- `{name}.zero_point` — FP32, shape `[out_features]`
- `{name}.bias` (optional) — FP32, shape `[out_features]`

Manifest tracks: model signature, architecture ID, ABI version, per-layer shapes/dtypes/quant config.

### Modules

- **`QuantLinear`**: INT8 buffers only (no float `.weight` Parameter). Forward: dequant -> matmul -> cast to compute_dtype.
- **`QuantLinearLoRA`**: Frozen INT8 base + trainable FP32 `lora_A`/`lora_B`. Forward: `base(x) + x @ A @ B * scaling`.

Both modules protect INT8 buffers from `.to(dtype=...)` and `_apply()` chains.

### Pipeline

```
build_safetensors_slab()                    -> .safetensors + .manifest.json
prepare_model_for_quantized_streaming()     -> nn.Linear -> QuantLinear (inference)
prepare_model_for_quantized_lora_training() -> nn.Linear -> QuantLinearLoRA (training)
load_quant_state_from_slab()                -> populate INT8 buffers from slab
```

---

## Tests

```bash
# Package tests (Gates 0-5)
pytest tests/ -q

# Serenity integration tests (Gates 0-15, 291 tests)
pytest serenity/tests/test_squareq_gate*.py -q
```

### Gate Structure

| Gate | What it checks |
|------|----------------|
| 0 | Safetensors format, manifest schema, tensor agreement, signature, ABI |
| 1 | Scaffold replacement (nn.Linear -> QuantLinear), bijection |
| 2 | Quant state loading, shapes, bias, round-trip accuracy |
| 3 | No float weight materialization, dtype stripping |
| 4 | QuantLinearLoRA construction, trainability, grad flow |
| 5 | Forward accuracy vs float reference, determinism, padding |
| 6 | Registry V2 integration, BlockEntry, SquareQParamSpec |
| 7 | Scheduler staging, buffer dequant, mixed trainable/frozen |
| 8 | Scheduler V2 dispatch routing |
| 9 | Full E2E: quantize -> scaffold -> load -> forward -> backward -> LoRA update |
| 10 | Error handling: missing tensors, corrupted data -> hard fail |
| 11 | Cleanup: no legacy code, terminology consistency |
| 12 | Multi-block forward accuracy via scheduler |
| 13 | Eviction and re-staging stability |
| 14 | INT8 dtype protection through .to() and _apply() chains |
| 15 | Telemetry recording, NumericGuard NaN/Inf detection |

---

## Validation Scripts

| Script | What it does |
|--------|-------------|
| `test_sdxl_int8_e2e.py` | Build slab, Gate 0/1 checks, per-layer cosine sim, LoRA backward |
| `test_sdxl_int8_inference.py` | Full SDXL pipeline: BF16 vs INT8 image gen, SSIM/PSNR comparison |
| `test_sdxl_int8_training.py` | LoRA training loop: scaffold, freeze, forward/backward, verify INT8 frozen |

---

## Roadmap

- Test more model architectures (SD 1.5, SD3, Flux 2 Klein 4B/9B, Chroma, etc.)
- Threshold-based mixed precision: per-layer MSE/FID budget, auto-escalate W4->W6->W8
- SmoothQuant + learned rounding (AdaRound) for better compression without quality loss
- Streaming-aware block scheduling (lean on frequent blocks, allow W6/W8 on sensitive heads)
- Optional per-group INT4/INT8 for MLP layers when quality thresholds allow

---

## License

TBD by repository owner.
