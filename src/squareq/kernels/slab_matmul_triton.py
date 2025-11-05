# SPDX-License-Identifier: Apache-2.0
"""
Triton fused matmul kernel for BP8 slab weights.

The kernel accepts per-row scales and zero-points and optionally applies a
LoRA update and ReLU activation.  It is intentionally minimal so that it can
be swapped out once higher-performance CUTLASS kernels land.
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def slab_matmul_kernel(
    Q,
    X,
    Scale,
    ZP,
    LoRA_A,
    LoRA_B,
    Alpha,
    UseLoRA,
    Out,
    M,
    N,
    K,
    Rank,
    stride_qm,
    stride_qk,
    stride_xk,
    stride_xn,
    stride_om,
    stride_on,
    stride_la_m,
    stride_la_r,
    stride_lb_r,
    stride_lb_n,
    apply_relu: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused int8 matmul with optional LoRA and ReLU.

    Parameters mirror the layout of the Python wrapper; strides are provided to
    keep the kernel generic.
    """

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_offset in range(0, K, BLOCK_K):
        q_offset = offs_m[:, None] * stride_qm + (k_offset + offs_k)[None, :]
        x_offset = (k_offset + offs_k)[:, None] * stride_xk + offs_n[None, :]

        q_int8 = tl.load(
            Q + q_offset,
            mask=(offs_m[:, None] < M) & (k_offset + offs_k[None, :] < K),
            other=0,
        ).to(tl.float32)
        scale = tl.load(Scale + offs_m, mask=offs_m < M, other=1.0)
        zp = tl.load(ZP + offs_m, mask=offs_m < M, other=0.0)
        x = tl.load(
            X + x_offset,
            mask=(k_offset + offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0,
        ).to(tl.float32)

        q_dequant = (q_int8 - zp[:, None]) * scale[:, None]
        acc += tl.dot(q_dequant, x)

    if UseLoRA:
        lora_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for r_offset in range(0, Rank, BLOCK_K):
            rank_offsets = r_offset + tl.arange(0, BLOCK_K)
            mask_a = (offs_m[:, None] < M) & (rank_offsets[None, :] < Rank)
            mask_b = (rank_offsets[:, None] < Rank) & (offs_n[None, :] < N)

            la_ptr = LoRA_A + offs_m[:, None] * stride_la_m + rank_offsets[None, :] * stride_la_r
            lb_ptr = LoRA_B + rank_offsets[:, None] * stride_lb_r + offs_n[None, :] * stride_lb_n

            lora_a = tl.load(la_ptr, mask=mask_a, other=0.0)
            lora_b = tl.load(lb_ptr, mask=mask_b, other=0.0)

            lora_acc += tl.dot(lora_a, lora_b)

        acc += lora_acc * Alpha

    if apply_relu:
        acc = tl.maximum(acc, 0.0)

    out_offset = offs_m[:, None] * stride_om + offs_n[None, :]
    tl.store(
        Out + out_offset,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )
