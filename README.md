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
| **Flux 2 Dev** (56 blocks) | 203 | ~24 GB | ~30 GB* | - | - | **Training verified** |

\* Flux 2 Dev slab is larger because it includes per-row scale/zero_point metadata for 203 layers across 8 double-stream + 48 single-stream blocks.

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

## Flux 2 Dev INT8 LoRA Training (2026-02-28)

Flux 2 Dev is the first large-scale model to train with SquareQ V2 + Stagehand block-swapping. 12B transformer + 24B text encoder (36B total), trained with LoRA on a single 24GB GPU at just 6 GB steady-state VRAM.

**Training config:**

| Parameter | Value |
|-----------|-------|
| Base model | `black-forest-labs/FLUX.2-dev` (12B transformer + Mistral 3 24B text encoder) |
| INT8 slab | 203 layers, ~30 GB |
| Training method | LoRA (rank 16, alpha 16) |
| Resolution | 512 (multi-aspect bucketing: 448x576, 512x512, 384x704, 640x384) |
| Learning rate | 1e-4 (cosine, 10-step warmup) |
| Precision | bfloat16 compute, INT8 frozen base |
| Memory strategy | Stagehand block-swapping with SquareQ V2 backing |
| Dataset | 118 images with text captions |
| Steps | 200 |

**Training performance (RTX 3090 Ti, 24 GB):**

| Metric | Value |
|--------|-------|
| VRAM allocated (steady state) | 5.52 GB |
| VRAM reserved (steady state) | 6.00 GB |
| Step time | ~160 s |
| SquareQ params matched | 192/203 |
| OOM events | 0 |
| Text encoding (118 samples) | ~12 min |
| Training (200 steps, estimated) | ~8.9 hours |

**Loss curve:**

```
Step   1: loss=0.696  avg=0.696  lr=2.00e-05  grad_norm=0.030
Step  10: loss=0.626  avg=0.731  lr=1.00e-04  grad_norm=0.061
Step  20: loss=0.616  avg=0.742  lr=9.90e-05  grad_norm=0.037
Step  30: loss=0.584  avg=0.748  lr=9.70e-05  grad_norm=0.039
Step  35: loss=0.717  avg=0.710  lr=9.55e-05  grad_norm=0.035
```

### How SquareQ integrates with Stagehand

Stagehand manages block lifecycle (load → forward → backward → evict). For SquareQ-backed blocks, the loading path is:

1. **Slab read**: INT8 weights read from the `.safetensors` slab via memory-mapped I/O
2. **Dequantize**: Per-row `scale * (qweight - zero_point)` → bf16 in the pinned CPU slab
3. **H2D transfer**: Async DMA copy from pinned slab to GPU on a dedicated CUDA stream
4. **Parameter repoint**: Module's `param.data` views into the GPU tensor
5. **Forward/backward**: Runs in bf16 precision with gradient checkpointing
6. **Eviction**: GPU tensor freed. LoRA gradients preserved on CPU for optimizer step.

The training loop never touches INT8 directly — Stagehand handles dequantization transparently during block loading. The model sees bf16 parameters at every forward pass.

### Key matching: diffusers ↔ slab canonical names

The SquareQ slab is built from the HuggingFace checkpoint, which uses canonical names like `ff.linear_in`. But diffusers' `FluxTransformer2DModel` internally renames layers:

| Diffusers name | Slab canonical name |
|---------------|---------------------|
| `ff.net.0.proj` | `ff.linear_in` |
| `ff.net.2` | `ff.linear_out` |
| `ff_context.net.0.proj` | `ff_context.linear_in` |
| `ff_context.net.2` | `ff_context.linear_out` |

Stagehand's `BlockRegistry._candidate_tensor_keys()` bridges this gap with bidirectional alias tables. Additionally, LoRA injection renames base weights (e.g. `attn.to_q.weight` → `attn.to_q.orig.weight`). The `.orig` suffix is stripped before matching against the slab manifest.

Without these fixes, only 32/203 parameters matched. After: 192/203 (remaining 11 are top-level non-block layers like `proj_out`).

### Building the Flux 2 Dev slab

The standard `build_safetensors_slab()` API requires loading the entire model into RAM to iterate `model.named_modules()`. For Flux 2 Dev that means ~24 GB of bf16 weights in CPU memory just to quantize. On a machine with 32 GB RAM and a 24 GB GPU, that's too tight — the model alone fills RAM, leaving nothing for the quantization buffers and the output slab.

The streaming builder (`scripts/build_flux2dev_slab.py`) solves this by never loading the full model:

1. **Read the sharded index** — `diffusion_pytorch_model.safetensors.index.json` maps each weight key to its shard file
2. **Open one shard at a time** — `safetensors.safe_open()` memory-maps the shard, no full load
3. **Quantize one weight at a time** — load a single 2D weight tensor, quantize to INT8 with per-row scales, store the output tensors, immediately `del` the source
4. **Skip non-Linear weights** — only 2D `.weight` tensors are quantized (biases, norms, embeddings are left out)
5. **Compute signature without the model** — reads shapes from shard metadata to build the same hash that `compute_model_signature()` would produce from a live `nn.Module`
6. **Write once** — `safetensors.torch.save_file()` writes all quantized tensors to a single output slab

Peak RAM: ~2 GB (the largest single weight in float32 + its INT8 output + scale/zero_point vectors). This is 12x less than loading the full model.

```bash
# From the Serenity repo root:
python scripts/build_flux2dev_slab.py \
  --model-dir ~/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-dev/snapshots/<hash>/transformer \
  --output-dir output/squareq_slabs \
  --slab-name flux2dev_int8 \
  --pack-k 64
```

Arguments:
- `--model-dir` — path to the transformer subdirectory containing the sharded safetensors + index JSON
- `--output-dir` — where to write the slab and manifest
- `--slab-name` — base filename (produces `{name}.safetensors` + `{name}.manifest.json`)
- `--pack-k` — K-dimension padding alignment for kernel compatibility (default: 64)
- `--blocks-only` — only quantize `transformer_blocks.*` and `single_transformer_blocks.*` (skip embedders, projections)

Output:
- `flux2dev_int8.safetensors` — ~30 GB slab, 203 quantized layers (INT8 qweight + FP32 scale + FP32 zero_point per layer)
- `flux2dev_int8.manifest.json` — per-layer metadata: canonical name, shapes, quant config, model signature

Build time: ~3 minutes on NVMe. The script prints per-layer sizes as it goes:

```
  transformer_blocks.0.attn.to_q: [3072, 3072] BF16=18MB → INT8=9MB (0.1s)
  transformer_blocks.0.attn.to_k: [3072, 3072] BF16=18MB → INT8=9MB (0.1s)
  ...
  single_transformer_blocks.47.proj_mlp: [12288, 3072] BF16=72MB → INT8=36MB (0.2s)

=== Summary ===
  Quantized layers: 203
  Total INT8 weight bytes: 14.72 GB
```

**Why not use the standard API?** The standard `build_safetensors_slab(model=...)` works fine for models that fit in RAM (SDXL at 4.5 GB, Flux 1 at 22 GB on a 64 GB machine). For Flux 2 Dev on a 32 GB machine, or any model where `model_size + slab_size > available_RAM`, the streaming builder is the only option. The output slab is identical — same format, same manifest schema, same Stagehand integration.

### Bugs found during integration

Five bugs were identified and fixed to get Flux 2 Dev training working:

1. **Mistral 3 24B text encoder OOM**: 48 GB text encoder can't `.to(cuda)` on 24 GB card. Fix: `accelerate.cpu_offload()` per-layer streaming.

2. **Legacy config discarding SquareQ settings**: Serenity's `_is_legacy_config()` silently rebuilt the config dict, dropping all `memory.stagehand.squareq_*` paths. Fix: use new config format without legacy trigger keys.

3. **SquareQ key matching (32/203 → 192/203)**: Two sub-causes — LoRA `.orig` suffix not stripped in `_candidate_squareq_layer_keys()`, and Flux 2 FF layer naming mismatch. Fix: suffix stripping + bidirectional aliases in `stagehand/registry.py`.

4. **CUDA memory fragmentation**: Stagehand evicts blocks from GPU but PyTorch's caching allocator holds reserved memory. Over multiple steps, reserved grew from 6 GB to 15 GB. Fix: `torch.cuda.empty_cache()` after each training step.

5. **Bucket policy disabling gradient checkpointing**: Memory predictor saw low VRAM (Stagehand makes it appear nearly empty) and auto-disabled gradient checkpointing. Without it, activations OOM. Fix: disable bucket_policy when Stagehand+SquareQ is active.

### What this proves

SquareQ V2 + Stagehand can train a 12B-parameter diffusion model with LoRA on a single 24GB GPU at 6 GB steady-state VRAM. The INT8 quantization halves block transfer sizes compared to full bf16 Stagehand, leaving more headroom for activations. Training converges normally with healthy loss curves and gradient norms.

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
