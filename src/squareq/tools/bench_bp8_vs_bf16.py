#!/usr/bin/env python
"""
Benchmark BF16 vs BP8 memory usage and runtime for the Flux transformer.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Dict, Tuple

import torch

from scripts.test_flux_bp8_pipeline import run_pass


def benchmark(
    mode: str,
    *,
    model_path: str,
    slab_path: str | None,
    height: int,
    width: int,
    iters: int,
) -> Tuple[float, float]:
    peaks = []
    times = []
    for _ in range(iters):
        peak, duration = run_pass(mode, model_path, slab_path, height, width)
        peaks.append(peak)
        times.append(duration)
        torch.cuda.empty_cache()
    return statistics.mean(peaks), statistics.mean(times)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark BF16 vs BP8 VRAM usage.")
    parser.add_argument("--model-path", default="/home/alex/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev")
    parser.add_argument("--slab-path", default="output/flux1_dev_bp8_all_layers.fpk")
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()

    model_path = str(Path(args.model_path).expanduser())
    slab_path = str(Path(args.slab_path).expanduser())

    results: Dict[str, Tuple[float, float]] = {}
    results["bf16"] = benchmark(
        "bf16",
        model_path=model_path,
        slab_path=None,
        height=args.height,
        width=args.width,
        iters=args.iters,
    )
    results["bp8"] = benchmark(
        "bp8",
        model_path=model_path,
        slab_path=slab_path,
        height=args.height,
        width=args.width,
        iters=args.iters,
    )

    print("=== Benchmark Results ===")
    for mode, (peak, duration) in results.items():
        print(f"{mode.upper():<6} peak_vram={peak:.2f} GB | avg_time={duration*1000:.2f} ms")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
