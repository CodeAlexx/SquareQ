# SquareQ

Experimental quantized block-swapping backend extracted from ai-toolkit. Provides BP8 slab tooling, fused Triton kernels, and LoRA-enabled runtime helpers for Flux-style diffusion models.

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

## Contents
- `src/squareq/slab`: slab metadata/schema helpers
- `src/squareq/quant`: loader modules and LoRA wrappers
- `src/squareq/kernels`: Triton fused matmul kernels
- `src/squareq/tools`: slab builder and benchmarking utilities
- `scripts/`: VRAM/benchmark smoke tests
- `tests/`: minimal verification suite

## License
TBD – inherited from source repository.
