# SPDX-License-Identifier: Apache-2.0
"""
Utility kernels for the Squared-Q runtime.

Currently exposes Triton fused matmul kernels used by the slab quantization
pipeline.  Kernels are imported lazily so the package can still be imported
when Triton is not installed.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["require_triton_kernel", "slab_matmul"]


def require_triton_kernel(name: str) -> Any:
    """
    Lazily import a Triton kernel module.  Raises a helpful error when Triton
    is missing so downstream callers can fall back to the reference PyTorch
    implementation instead of crashing with ImportError.

    Args:
        name: Kernel module name under ``squareq.kernels`` (e.g., ``slab_matmul``)

    Returns:
        Imported module.
    """

    try:
        return import_module(f"squareq.kernels.{name}")
    except ModuleNotFoundError as exc:  # pragma: no cover - defensive path
        if exc.name == "triton":
            raise RuntimeError(
                "Squared-Q Triton kernels require the `triton` package. "
                "Install it or disable fused kernels via the "
                "`SQUAREQ_USE_TRITON` environment variable."
            ) from exc
        raise


def slab_matmul() -> Any:
    """
    Convenience accessor for the slab matmul kernel module.

    Returns:
        Imported module exposing ``run_slab_matmul`` and helpers.
    """

    return require_triton_kernel("slab_matmul")
