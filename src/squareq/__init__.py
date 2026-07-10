"""SquareQ integration — quantized block swapping for Stagehand.

Provides INT8 slab format, fused Triton kernel, QuantLinear modules,
and scaffold preparation for Stagehand-driven streaming.
"""
from __future__ import annotations

__all__: list[str] = []

# INT4 (SVDQuant W4A16) — the 4-bit sibling of the INT8 builder.
from squareq.svdquant_int4 import (
    quantize_svdquant_w4a16,
    reconstruct_w4a16,
    build_svdquant_int4_slab,
)

__all__ += ["quantize_svdquant_w4a16", "reconstruct_w4a16", "build_svdquant_int4_slab"]
