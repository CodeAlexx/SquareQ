# SquareQ

Quantized-frozen-base training for diffusion models on consumer GPUs.
Newest first: **v3** (W4/NVFP4, Mojo runtime + torch reference) is the
current line; **v2** (INT8 block-swapping, PyTorch) remains supported below.

---

# SquareQ v3 — W4 (Hadamard-256 + low-rank + int4 / NVFP4)

v3 moves from INT8 block-swapping to a **4-bit quantized-frozen-base training
format** at ~0.29x bf16 bytes, with the error-recovery branch inside the format:

```
W = lora_up @ lora_down^T  +  inverse_H256( dequant( residual ) )
```

- **Rotation**: block-diagonal normalized Hadamard-256 over the input dim
  (any K % 256 == 0 — covers FLUX.2 Klein, Krea-2, and friends).
- **Residual codecs**: `int4` group-32/64 scales with MSE-swept fractions
  (dequant-first, any GPU), or `nvfp4` (e2m1 + ue4m3 block-16 scales in the
  cuBLASLt tiled layout + per-tensor global scale) for native FP4 tensor-core
  GEMMs on Blackwell (measured 6.1–7.4x vs bf16 at model shapes on an RTX 5080).
- **Low-rank branch**: rank-32 SVD of the weight, always on, bf16.
- **One artifact**: the same slab trains (frozen base + LoRA) and infers;
  the builder is streaming + restartable (kill -9 mid-build resumes to
  byte-identical output).

### Measured results — pure-Mojo runtime, RTX 5080 16 GB (same seeds/data, 2026-07-28)

**FLUX.2 Klein-4B LoRA, 512px, 300 steps** (rank-16, AdamW, inline 1024px samples):

| arm | bytes | mean loss vs bf16 | s/step | peak VRAM |
|---|---|---|---|---|
| bf16 streamed | 1.0x | — | 1.162 | 3.61 GB |
| fp8_e4m3 resident | 0.50x | +0.0005 | 1.089 | 7.04 GB |
| **SquareQ int4 g32+MSE** | **0.294x** | **+0.0048** | 1.297 | **5.52 GB** |

Loss curves are parallel (constant level-offset, no divergence); step-300
1024x1024 samples are clean.

**Krea-2 Raw (22B-class) LoRA, 1024px, 200 steps** — the memory-bound case:

| arm | residency on 16 GB | mean loss | s/step | peak VRAM |
|---|---|---|---|---|
| fp8_e4m3 | 28-resident: **OOM**. 22: **OOM**. max 14 + disk-stream | 0.1667 | 6.84 | 8.16 GB |
| **SquareQ int4 g32+MSE** | **all 28 blocks resident, zero disk reads** | 0.1669 (**+0.18%**) | **6.02** | 9.13 GB |

The larger the model, the smaller the W4 penalty (+0.64% rel on 4B ->
+0.18% on 22B-class) — and the residency win turns into a speed win.

**Native NVFP4 GEMM (Blackwell sm_120, cuBLASLt block-scaled)** vs the
production bf16 GEMM at model shapes:

| M x N x K | bf16 | fp4 | speedup |
|---|---|---|---|
| 1536 x 3072 x 3072 | 271 us | 45 us | 6.1x |
| 1536 x 9216 x 3072 | 837 us | 125 us | 6.7x |
| 4480 x 6144 x 6144 | 2973 us | 418 us | 7.1x |
| 4480 x 16384 x 6144 | 7930 us | 1074 us | 7.4x |
| 4480 x 6144 x 16384 | 7522 us | 1023 us | 7.4x |

Op-level parity of the full FP4 forward (fused H256-rotate + block-quant +
tiled scales + GEMM + epilogue): cos 0.99987 vs the bit-level oracle; the
bwd-side bf16 reconstruct from the same payload: cos 0.999994.

End-to-end: the Klein-4B trainer runs with `quantized_resident=squareq_nvfp4`
(native FP4 forward, bf16-reconstruct backward from the same payload) — loss
matches the int4 arm within 2% over the smoke, and an nsys trace shows the
cutlass sm120 block-scaled ue4m3xe2m1 GEMM executing (no silent bf16
fallback). Forward-only visits now skip the bf16 reconstruct entirely
(the FP4 payload drives the GEMM; only the backward rebuilds bf16):
measured forward -43%, step time 0.864 -> 0.785 s/step vs the int4 arm,
loss curve unchanged. Sensitive layers (AdaLN/modulation/norms) are kept
bf16 by an explicit, reported builder policy — and per-layer weight
fidelity in squareq-plan.json is a proxy, not a verdict: judge damage by
training curves and samples, which is how every number here was gated.

**Builder** (streaming, restartable): Klein-4B 7.3 GB -> 2.3 GB slab in 92 s
at 5.3 GB peak RSS; `kill -9` mid-build resumes to byte-identical output.
Krea-2 29 GB-class in 6 min at 8.1 GB RSS.

Weight-fidelity vs a public ConvRot INT8 checkpoint of the same base
(`lilcheaty/Krea2-INT8-ConvRot`, 0.49x bytes): int8 wins per-layer cosine
(0.99996 vs our 0.9964–0.9990) at 1.7x our footprint — 8 bits vs 4 bits.
SquareQ's class is the 0.29x row: full-model residency where int8 cannot fit.

### Layout: `src/squareq/v3/`
`core.py` (codecs + Hadamard + SVD init + NVFP4 tiled scale packer),
`build_slab.py` (streaming restartable builder CLI), `stwriter.py`
(streaming safetensors writer), `plan.py` (manifest/build-state),
`reference.py` (pure-torch runtime: `SquareQLinear`, `SquareQLoRALinear`,
`lora_train_demo` — verify every claim without the Mojo stack),
`selftest.py` (gates: pack/unpack bit-exact, rotation self-inverse,
determinism, NVFP4 round-trip; `--klein <ckpt>` adds real-weight checks).

```bash
python -m squareq.v3.selftest
python -m squareq.v3.build_slab --model klein4b --src transformer.safetensors \
    --out-dir klein4b_squareq --rank 32 --group 32          # int4 (any GPU)
python -m squareq.v3.build_slab ... --format nvfp4          # Blackwell FP4
```

### v3.1 — int8-residual mode: taking the 0.5x point too

`--format int8`: int8 per-row (MSE-swept) of the H256-rotated rank-16 residual
+ the low-rank branch. Measured from REAL slab bytes on the same Krea-2 weights
vs the public ConvRot INT8 checkpoint (`lilcheaty/Krea2-INT8-ConvRot`):

| layer | ConvRot int8 @0.5003x | SquareQ w8 @0.505x |
|---|---|---|
| blocks.0.attn.wq | 0.999958 | **0.999984** |
| blocks.0.mlp.down | 0.999955 | **0.999961** |
| blocks.13.mlp.down | 0.999956 | **0.999959** |
| blocks.27.mlp.gate | 0.999960 | **0.999965** |

Full krea2 slab: cos_w min 0.99990 / mean 0.99996 over 224 layers. Trains in
the same Mojo pipeline (Klein-4B smoke losses statistically identical to the
fp8 arm), same restartable builder, same one-artifact contract. Bytes are
0.9% above theirs (the low-rank branch) — disclosed.

### Cross-stack comparison vs SimpleTuner v4.5.2 + SDNQ (int8-sdnq + Hadamard-256)

Same dataset (51-image identity set), seed 42, rank-16/a16, lr 3e-5, bs 1,
512px, 300 steps, FLUX.2 Klein-4B. Each quantized arm judged vs ITS OWN bf16
(cross-stack loss scales differ).

| stack | quant arm | bytes | s/step (pure training) | quant-vs-own-bf16 loss |
|---|---|---|---|---|
| SimpleTuner (torch) | int8-sdnq+H256, dequant-first* | 0.50x | 0.917 (bf16: 0.676) | -0.39% (noise: int8 is ~lossless) |
| SquareQ (Mojo) | int4 g32+MSE | **0.294x** | **0.859** (fp8 base: 0.695; NVFP4 fwd: 0.785) | +0.63% |

CORRECTION (2026-07-28): an earlier revision listed SquareQ at 1.373 s/step —
that number included inline 1024x1024 sampling every 100 steps inside the
measured window; the SimpleTuner rates did not. Matched windows (training
steps only, above): SquareQ's quantized arm is FASTER than SimpleTuner's
(0.859 vs 0.917), and its fp8 arm matches their bf16 (0.695 vs 0.676).

*Their quantized-matmul ConvRot preset (compiled and uncompiled) failed on
current Triton ("Kernel requires a runtime memory allocation, but no
allocator was set"); the documented dequant-first int8+H256 route ran.

Honest reading, both directions: int8 at 0.5x is effectively lossless and the
torch stack is absolutely faster at this small-model scale. SquareQ's case is
the 0.29x byte class (no competing offering) and large-model residency —
Krea-2 at 1024px trains fully resident and FASTER than fp8 on a 16 GB card,
a configuration neither fp8 nor 0.5x int8 bases can fit.

The production training/inference runtime (pure Mojo: fused kernels, resident
block streaming, the trainers these numbers come from) lives in
[CodeAlexx/mojodiffusion](https://github.com/CodeAlexx/mojodiffusion).


---

# SquareQ v2 — INT8 block-swapping (PyTorch)

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
