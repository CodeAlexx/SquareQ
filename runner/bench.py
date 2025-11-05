#!/usr/bin/env python
"""Convenience wrapper to run the BP8 vs BF16 benchmark."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run BP8 vs BF16 benchmark")
    parser.add_argument("--model-path", default="/home/alex/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev")
    parser.add_argument("--slab-path", default="output/flux1_dev_bp8_all_layers.fpk")
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "test_flux_bp8_pipeline.py"
    cmd = [
        sys.executable,
        str(script),
        "--mode",
        "bp8",
        "--model-path",
        args.model_path,
        "--slab-path",
        args.slab_path,
        "--height",
        str(args.height),
        "--width",
        str(args.width),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
