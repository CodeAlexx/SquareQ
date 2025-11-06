# SquareQ
# SQUARE‑Q
SquaredQ stores the model as an INT8 slab on disk/CPU and streams one block at a time into VRAM. It computes directly on the INT8 weights with a fused kernel and then evicts the block, so the full model never lives in GPU memory.

Why it’s different: “Most quant libs (BnB/Quanto) quantize then keep the whole model resident in VRAM; classic swapping frameworks swap FP16/BF16. SquaredQ swaps INT8 blocks, which is smaller and training-friendly.”

Why it works for training: SquaredQ keeps base weights frozen in INT8; only small LoRA adapters receive FP32 gradients, so backprop fits on 24 GB.
Personal Note: 
I did this to add to my rust trainer, and provides a proof of concept system. i am happy with the results. 
**Where I’ll take it next (beating or matching SVDQuant, practically):**  
This prototype proves the memory story. If I expand it, the plan is to push *compression without trashing quality* and add a strict **threshold-based controller** so mixed precision decisions are automatic and safe.

- **Target parity, then surpass:** First match SVDQuant’s quality at similar bit-width (W4A8/W4A4) on Flux; then surpass with **learned rounding (AdaRound)** + **SmoothQuant** + **block-aware AutoTune** that’s integrated with streaming. The goal is: *same or better FID/LPIPS with less VRAM*.
- **Threshold system (the one we discussed):** Each layer/block gets a quality budget and numeric guardrails. If the **reconstruction MSE** or **activation outlier score** rises above a threshold, the system **escalates precision** (W4→W6→W8) for that layer only. If it’s well below the floor, it **de-escalates** next build to recover memory. This runs in the slab builder and logs a manifest of final bit‑widths.
  - **Quality thresholds:** per‑layer MSE <= τ_MSE and block‑level FID drift <= τ_FID (sampled).  
  - **Stability thresholds:** activation range / p99 <= τ_ACT; per‑row scale clamp <= τ_SCALE.  
  - **Budget threshold:** global “compression budget” (e.g., target slab size) that the tuner respects while staying within quality/stability guards.
- **Outlier handling:** Prefer **SmoothQuant α‑search** and per‑channel clipping rather than blanket W8. Only escalate when thresholds trigger.
- **Streaming-aware decisions:** Keep frequently‑resident blocks leaner (W4/W6); allow W6/W8 on numerically sensitive heads. The tuner knows which blocks are co‑resident to keep VRAM under cap.
- **Proof that it works:** lock in a tiny acceptance suite: (1) **BP8/AutoTune** meets τ_MSE/τ_FID, (2) end‑to‑end LoRA fine‑tune converges within Δ steps vs BF16, (3) slab size & peak VRAM under configured budgets.
- **Later (optional):** add **per‑group INT4/INT8** for specific MLP mats if quality thresholds allow; keep a fallback ladder (W4→W6→W8) when thresholds trip.

This keeps the approach honest: it won’t down‑bit a layer unless it passes **my thresholds**, and it automatically spends bits where needed. It’s about *beating SVDQuant on training practicality and memory*, while matching its quality with a transparent, testable controller.

Alex. 
--------------------------------------------------------------------------------

Quantized **block swapping** + **training** for diffusion models (Flux) on consumer GPUs.

This repo packages the parts we actually use in practice: an **all‑layers INT8 slab** format (BP8), a **CPU‑stage loader** that never materializes float weights on the GPU, **Triton fused kernels** for matmul, and a **LoRA training path** with VRAM‑aware smoke tests and simple benchmarks.

> **Honest scope:** This is not the fastest INT8 library nor a new quantization algorithm. It’s a practical, working stack to **fine‑tune Flux on 24 GB** by combining quantization **and** block‑level streaming.

---

## Why this exists

### Flux architecture note
Flux carries two transformer stacks: one **“double”** stack of 19 dual‑attention blocks and one **“single”** stack of 38 lighter blocks, plus context/embed heads. SQUARE‑Q targets **all 57 blocks**; base weights live as INT8 buffers while LoRA adapters provide the trainable path.

Training large diffusion transformers on 24 GB usually OOMs. Plain INT8 cuts memory, but keeping the whole model resident still blows VRAM. **SQUARE‑Q** streams **quantized** blocks on demand, so only a few blocks live in VRAM at once. Together with LoRA‑only training, this puts Flux fine‑tuning within reach on a single 3090/4090/A6000.

**What’s actually different:** _quantized_ block swapping **for training** (end‑to‑end).

---

## Key features (practical)

- **All‑layers BP8 slab** (INT8 per‑channel): `qweight:int8` + `scale/zero:fp32` (+ optional `bias:fp32`) for **every** layer (double_blocks, single_blocks, final head). **No float weights** in the slab.
- **CPU‑stage loader:** attaches the slab and replaces Linear/Conv with `QuantLinear` wrappers **before** any GPU dtype cast; only INT8/FP32 buffers land on GPU.
- **Triton fused matmul:** INT8×BF16 → FP32 accumulate with optional fused epilogue (bias/GeLU); autotune tiles; no FP16 dequant tensor.
- **LoRA training path:** `QuantLinearLoRA` adds FP32 adapters (A/B) while base INT8 is frozen; works with BF16 autocast + FP32 grads.
- **VRAM/bench scripts:** small tools that print `[VRAM:after_*]` and compare BP8 vs BF16; contract tests to prevent regressions.

**Not included / not claims:** fastest kernels, novel PTQ, 8‑bit optimizers, LLM coverage. This is diffusion‑focused and training‑first.

---

## Requirements

- Ubuntu 20.04+
- NVIDIA GPU (Ampere+), CUDA 11.8+ (12.x preferred)
- Python 3.9–3.11
- PyTorch ≥ 2.1 with CUDA
- Triton ≥ 2.1

---

## Install

```bash
git clone <your-new-repo-url> squareq
cd squareq
pip install -e .
```

Environment flags that force the fused path (recommended):
```bash
export SQUAREQ_USE_TRITON=1
export SQUAREQ_DISABLE_DEQUANT_FALLBACK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

## Quickstart

### 1) Build an **all‑layers** BP8 slab

```bash
python src/squareq/tools/squareq_build_slab.py   --in  /path/to/flux1-dev.safetensors   --out output/flux1_dev_bp8_all_layers.fpk   --layout rowwise --symmetric --pack-k 64
```
Requirements:
- Quantize **every** layer (double_blocks, single_blocks, final head)
- Store only `{qweight:int8, scale:fp32, zero:fp32, [bias:fp32]}`

### 2) BP8 pipeline smoke (no training)

```bash
python scripts/test_flux_bp8_pipeline.py   --mode bp8   --slab-path output/flux1_dev_bp8_all_layers.fpk   --height 512 --width 512
```
You should see `[VRAM:after_load/forward]` with BP8 significantly below BF16 (e.g., ~11–12 GB vs ~22 GB at 512²).

### 3) LoRA VRAM smoke (training step)

```bash
python scripts/test_flux_bp8_lora_vram.py   --height 384 --width 384   --rank 8   --slab-path output/flux1_dev_bp8_all_layers.fpk
```
Expected: `[VRAM:after_backward]` hits the same peak as forward (activations dominate), confirming real grads flow through LoRA. With all 57 Flux blocks (19 dual‑attn “double” + 38 “single” blocks) and rank=8 at 384² we sit ~21–22 GB on a 24 GB card.

### 4) Kernel micro‑bench

```bash
python runner/bench.py --m 1024 --k 4096 --n 4096 --iters 30 --warmup 5 --gelu --bias
```
Baseline correctness + average forward ms for the fused kernel.

---

## Project layout

```
src/squareq/
  slab/         # slab schema & manifest helpers
  quant/        # loader, quant wrappers, LoRA variants
  kernels/      # Triton fused matmul + prefetch helper
  tools/        # slab builder & BP8 vs BF16 bench
scripts/        # pipeline smokes
runner/         # kernel bench
tests/          # contract & log‑parser tests
```

---

## Usage notes & guardrails
with SquaredQ used, bp8 slab , actually used forward is 11.7 gigs used, back prop at 21.2 gigs. pytorch handles it. there is much room for improvement 

- **BP8 mode:** move model **scaffold** to GPU, then `attach_bp8_slab()`. **Do not call** `.to(dtype=...)` after attaching the slab or you will materialize BF16 weights and lose savings.
- **QuantLinear contract:** no float `weight` Parameter. Buffers only: `qweight:int8`, `scale:fp32`, `zero:fp32`, optional `bias:fp32`.
- **LoRA only:** set optimizer params to LoRA A/B exclusively; leave base frozen.
- **BF16 autocast:** wrap Flux forward in autocast; grads stay FP32 by default.
- **Checkpointing:** enable gradient checkpointing in Flux blocks to keep activations in check.
- **TE/VAE:** for memory tests, keep TE on CPU and run VAE after Flux forward.

---

## Tests

Run the small suite (marks skip CUDA if no GPU):
```bash
pytest -q tests -k squareq --maxfail=1
```
What they check:
- Slab manifest covers all layers (no missing single_blocks/final).
- Loader never materializes float weights; no `.to(dtype=...)` after BP8 attach.
- Quant modules expose INT8 buffers and no float `weight`.
- Log parser: BP8 uses less VRAM than BF16 in your micro‑bench.

---

## Benchmarks (expected ballpark)

- **Memory:** BF16 peak ~22 GB → BP8 peak ~11–12 GB at 512² (latents batch=1).  
- **Speed:** current fused Triton ~1.5–3.4× slower than BF16 (shape‑dependent). Enable autotune, bias/GeLU epilogue, and K‑packing for gains; prefetch helps if you stream blocks.

We do **not** claim SOTA speed; the point is training feasibility on 24 GB.

---

## Roadmap (short)

- Optimize fused kernels (tile autotune, better epilogues, cuBLASLt fallback).
- Slab inspector CLI (manifest print, K‑pack status).
- End‑to‑end LoRA example notebook with quality metrics.
- Optional 8‑bit optimizer states for adapters.

---

## Troubleshooting

- **OOM at load:** your slab likely includes float weights (not all‑layers). Rebuild slab; ensure no float `weight` blobs are written.  
- **BP8 looks like BF16 memory:** you casted dtype after attach (don’t), or you hit a dequant fallback (export the fused env flags).  
- **No grad memory change:** LoRA not attached; check `requires_grad` only on LoRA.  
- **Slow kernels:** confirm autotune ran once; warm up 3–5 iters; keep K packed to 32/64; try bias/GeLU fusion on.

---

## License

TBD by repository owner.

Experimental quantized block-swapping backend extracted written for ai-toolkit. Provides BP8 slab tooling, fused Triton kernels, and LoRA-enabled runtime helpers for Flux-style diffusion models.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
export SQUAREQ_USE_TRITON=1
export SQUAREQ_DISABLE_DEQUANT_FALLBACK=1
python scripts/test_flux_bp8_pipeline.py --mode bp8 --slab-path output/flux1_dev_bp8_all_layers.fpk --height 512 --width 512
```
```
python scripts/test_flux_bp8_lora_vram.py --height 384 --width 384 --rank 8 --slab-path output/flux1_dev_bp8_all_layers.fpk

```
WHY? 
Deterministic VRAM budget (no surprises).
Off-the-shelf stacks “discover” memory at runtime (optimizer states, transient buffers, padding, longer prompts). A slab pins weight bytes and layout up front, so you know exactly what fits—and keep headroom for activations.

Cold-start & reload speed.
One mmapped blob loads fast (fewer fs syscalls, no per-tensor allocator churn). Matters for multi-run experiments and quick back-to-back evals.

Block-swap without re-quantizing.
Your SQUARE-Q slab keeps per-block scales + offsets; you can swap DiT/MMDiT/UNet/attn/MLP blocks and test variants without a fresh quant pass. Off-the-shelf formats rarely preserve that contract.

Fragmentation & allocator sanity.
Thousands of small tensors → allocator fragmentation → sudden OOMs at higher res/seq. A slab is contiguous; far fewer device allocations.

Bigger contexts / higher res on the same 24 GB.
“It runs at 1k tokens / 1024²” is not the same as “it runs with headroom for longer context, larger batch, packed seq, or extra adapters.” Slab shrinks weights so you can spend VRAM on activations (where training actually hurts).

Multi-model workflows.
If you bounce between WAN / Qwen-Image / Flux variants, slab + index map lets you unmap/remap quickly with predictable peaks. Off-the-shelf checkpoints re-hit the graph build and memory spikes each time.

Profiling reproducibility.
With slab you get stable weight layout → stable perf numbers across runs. That’s how you catch true kernel regressions instead of loader noise.

Serving & A/B.
For side-by-side evals, slabs let you pin N variants and just switch views. Off-the-shelf will re-init and repack every time.

## Contents
- `src/squareq/slab`: slab metadata/schema helpers
- `src/squareq/quant`: loader modules and LoRA wrappers
- `src/squareq/kernels`: Triton fused matmul kernels
- `src/squareq/tools`: slab builder and benchmarking utilities
- `scripts/`: VRAM/benchmark smoke tests
- `tests/`: minimal verification suite

## License
TBD – inherited from source repository.
