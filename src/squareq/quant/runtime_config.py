"""
Minimal runtime configuration helpers for SquareQ streaming loaders.

Provides environment-driven defaults for prefetch horizon and CUDA stream pool,
plus a helper to clamp settings for tiny models.
"""
from __future__ import annotations

import os
from typing import Dict

SQUAREQ_PREFETCH_HORIZON = int(os.environ.get("SQUAREQ_PREFETCH_HORIZON", "2"))
SQUAREQ_STREAM_POOL = int(os.environ.get("SQUAREQ_STREAM_POOL", "2"))
SQUAREQ_PIN_MIN_BYTES = int(os.environ.get("SQUAREQ_PIN_MIN_BYTES", "1048576"))
SQUAREQ_TINY_MODEL_BYTES = int(os.environ.get("SQUAREQ_TINY_MODEL_BYTES", "20971520"))


def adjust_for_tiny_model(
    total_slab_bytes: int,
    horizon: int = SQUAREQ_PREFETCH_HORIZON,
    stream_pool: int = SQUAREQ_STREAM_POOL,
    pin_min_bytes: int = SQUAREQ_PIN_MIN_BYTES,
) -> Dict[str, int]:
    """Clamp streaming parameters when the slab is small."""
    is_tiny = total_slab_bytes < SQUAREQ_TINY_MODEL_BYTES
    if is_tiny:
        horizon = min(horizon, 1)
        stream_pool = 1
    return {
        "horizon": horizon,
        "stream_pool": stream_pool,
        "pin_min_bytes": pin_min_bytes,
        "is_tiny": is_tiny,
    }
