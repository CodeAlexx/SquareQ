"""SquareQ INT4 (SVDQuant W4A16) — weight-only 4-bit quantizer.

The INT4 sibling of builder.py's per-row INT8. Where INT8 is 1 byte/param, this
is 0.5 byte/param + a rank-R low-rank branch that recovers the 8→4-bit error:

    W [out,in] bf16
    → SVD: W ≈ L1 @ L2  (rank R),  L1[out,R]=U_R Σ_R,  L2[R,in]=V_R^T
    → residual = W − L1@L2
    → group-G symmetric int4 of the residual (per-(group,out) scale = max|.|/7)
    → clean ROW-MAJOR pack: qweight[out,in/2] (2 nibbles/byte, lo=even),
      wscales[in/G,out], lora_down[in,R]=L2^T, lora_up[out,R]=L1, smooth[in]=1, bias

WEIGHT-ONLY (W4A16): the consuming Mojo runtime is dequant-first (weights→bf16,
GEMM in bf16; activations STAY bf16), so there is NO activation-int4 and NO
smoothing — `smooth` is ones. If activation smoothing is added later, match
nunchaku's DIVIDE convention (main = (x/smooth)@W^T), not multiply.

Reconstruct (serenitymojo ops/svdquant.mojo, twos_complement / lo_even, smooth=1):
    W_rec[o,k] = int4(nibble)*wscales[k//G,o] + (lora_up @ lora_down^T)[o,k]

This is the SINGLE source of the int4 quant math; the mojodiffusion scripts
(scripts/svdquant_selfquant.py, svdquant_quantize_model.py) import from here.
Gated: serenitymojo/ops/parity/svdquant_ltx2_layer_parity.mojo (real LTX2 layer
W cos 0.9971) + the full-stack forward cos 0.9901. MJ-1095/1096.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import torch

__all__ = [
    "quantize_svdquant_w4a16",
    "reconstruct_w4a16",
    "build_svdquant_int4_slab",
    "NBITS_MAX",
    "DEFAULT_RANK",
    "DEFAULT_GROUP",
]

NBITS_MAX = 7          # symmetric int4 range [-8,7]; scale = max|.|/7 → max→7
DEFAULT_RANK = 32
DEFAULT_GROUP = 64
# W4A4 uses a fatter low-rank (rank 128) so per-OUTPUT weight int4 (which factors
# through the stock CUTLASS int4 GEMM, no group kernel) still clears cos ~0.99 on
# the big-`in` layers. Measured sweep (scripts/svdquant_w4a4_gate.py --sweep):
# r128 per-out = ff.net.2 0.990, ff.net.0 0.994, audio_ff 0.995. MJ-1099.
W4A4_DEFAULT_RANK = 128

# Class-A exclusion by name (norms/embedders/modulation/conditioning stay bf16).
EXCLUDE_SUBSTR = (
    "norm", "embed", "gate_logits", "scale_shift", "adaln",
    "patchify", "proj_out", "time_", "caption", "connector",
)


def quantize_svdquant_w4a16(
    W: torch.Tensor, rank: int = DEFAULT_RANK, group: int = DEFAULT_GROUP,
    *, full_svd: bool = False,
):
    """W [out,in] → (qbyte I8[out,in/2], wscales BF16[in/G,out],
    lora_down BF16[in,R], lora_up BF16[out,R], smooth BF16[in]=ones).

    full_svd uses torch.linalg.svd (exact); default uses torch.svd_lowrank
    (equal rank-R subspace, ~30× faster on wide layers, identical residual pack)."""
    out, inh = W.shape
    assert inh % group == 0, f"in {inh} not divisible by group {group}"
    Wf = W.float()
    if full_svd:
        U, S, Vh = torch.linalg.svd(Wf, full_matrices=False)
        L1 = U[:, :rank] * S[:rank]
        L2 = Vh[:rank, :]
    else:
        q = min(rank + 8, min(out, inh))
        Ul, Sl, Vl = torch.svd_lowrank(Wf, q=q, niter=4)
        L1 = Ul[:, :rank] * Sl[:rank]
        L2 = Vl[:, :rank].t()
    residual = Wf - L1 @ L2
    ngrp = inh // group
    r = residual.view(out, ngrp, group)
    scale = r.abs().amax(dim=2) / NBITS_MAX
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.clamp(torch.round(r / scale[:, :, None]), -8, 7).to(torch.int16).view(out, inh)
    nib = (q & 0xF).to(torch.int16)
    lo = nib[:, 0::2]
    hi = nib[:, 1::2]
    qbyte = (lo | (hi << 4)).to(torch.uint8).view(torch.int8)      # I8 [out,in/2]
    wscales = scale.t().contiguous().to(torch.bfloat16)            # [ngrp,out]
    lora_down = L2.t().contiguous().to(torch.bfloat16)             # [in,R]
    lora_up = L1.contiguous().to(torch.bfloat16)                   # [out,R]
    smooth = torch.ones(inh, dtype=torch.bfloat16)
    return qbyte, wscales, lora_down, lora_up, smooth


def hadamard(n: int) -> torch.Tensor:
    """Normalized Sylvester–Hadamard matrix [n,n] (n a power of 2). H @ H.T = I.
    Data-free QuaRot rotation — the SAME matrix is regenerated online by size, so
    only a `hadamard: sylvester` flag is stored, never the matrix."""
    assert n > 0 and (n & (n - 1)) == 0, f"{n} is not a power of two"
    H = torch.ones(1, 1, dtype=torch.float64)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / (n ** 0.5))


def quantize_svdquant_w4a4(W: torch.Tensor, rank: int = W4A4_DEFAULT_RANK):
    """W [out,in] → QuaRot W4A4 tensors (per-OUTPUT weight scale, so it factors
    through the stock CUTLASS int4 GEMM — no group-scale kernel):

        SVD: W ≈ L1@L2 (rank R)  → residual R = W − L1@L2
        rotate residual: R_rot = R @ H   (H = Sylvester Hadamard [in,in])
        per-output int4:  ws[o] = max_k|R_rot[o,k]|/7 ; q = round(R_rot/ws)∈[-8,7]
        pack lo-even: qweight[out,in/2]

    Online (runtime): y = dequant(  (X@H)_int4 @ qweight^T  )·(xscale⊗ws)
                          + (X @ lora_down) @ lora_up^T  + bias      (low-rank bf16)
    since (X@H)@(R@H)^T = X@R^T and R = W − L1@L2. Quality (sim, real ff layer):
    cos ≈ 0.99/layer, outlier-robust. Returns
    (qbyte I8[out,in/2], wscale BF16[out], lora_down BF16[in,R], lora_up BF16[out,R])."""
    out, inh = W.shape
    assert inh > 0 and (inh & (inh - 1)) == 0, f"in {inh} must be a power of 2 for Hadamard"
    # svd_lowrank (float32) — the rank-R subspace is what matters; full double SVD
    # is ~100x slower and unneeded for the 1344-layer slab regen (gated: worst
    # real ff layer cos 0.988 either way).
    Wf = W.float()
    q = min(rank + 16, min(out, inh))
    Ul, Sl, Vl = torch.svd_lowrank(Wf, q=q, niter=6)
    L1 = (Ul[:, :rank] * Sl[:rank]).double()
    L2 = Vl[:, :rank].t().double()
    R = Wf.double() - L1 @ L2
    H = hadamard(inh)
    Rrot = R @ H                                    # rotate residual along `in`
    ws = Rrot.abs().amax(dim=1) / NBITS_MAX         # [out] per-output scale
    ws = torch.where(ws == 0, torch.ones_like(ws), ws)
    q = torch.clamp(torch.round(Rrot / ws[:, None]), -8, 7).to(torch.int16)
    nib = (q & 0xF)
    lo = nib[:, 0::2]; hi = nib[:, 1::2]
    qbyte = (lo | (hi << 4)).to(torch.uint8).view(torch.int8)   # I8 [out,in/2]
    wscale = ws.to(torch.bfloat16)                              # [out]
    lora_down = L2.t().contiguous().to(torch.bfloat16)          # [in,R]
    lora_up = L1.contiguous().to(torch.bfloat16)                # [out,R]
    return qbyte, wscale, lora_down, lora_up


def _unpack_int4_perout(qbyte):
    out = qbyte.shape[0]; inh = qbyte.shape[1] * 2
    nb = (qbyte.view(torch.uint8).to(torch.int16) & 0xFF)
    lo = nb & 0xF; hi = (nb >> 4) & 0xF
    v = torch.zeros(out, inh, dtype=torch.int16)
    v[:, 0::2] = torch.where(lo >= 8, lo - 16, lo)
    v[:, 1::2] = torch.where(hi >= 8, hi - 16, hi)
    return v.double()


def w4a4_forward(X, qbyte, wscale, lora_down, lora_up):
    """Float reference of the exact W4A4 runtime (for gating): per-token act int4
    on the ROTATED activation, int4 main GEMM, rescale, + bf16 low-rank."""
    inh = qbyte.shape[1] * 2
    H = hadamard(inh)
    Xf = X.double()
    Xrot = Xf @ H
    xs = Xrot.abs().amax(dim=1, keepdim=True) / NBITS_MAX
    xs = torch.where(xs == 0, torch.ones_like(xs), xs)
    Xq = torch.clamp(torch.round(Xrot / xs), -8, 7)             # int4 activation
    Rq = _unpack_int4_perout(qbyte)                            # int4 weight (rotated residual)
    main = (Xq @ Rq.t()) * xs * wscale.double()[None, :]        # int32·(xs⊗ws)
    low = (Xf @ lora_down.double()) @ lora_up.double().t()      # bf16 low-rank (unrotated)
    return main + low


def reconstruct_w4a16(qbyte, wscales, lora_down, lora_up, group=DEFAULT_GROUP):
    """CPU reference reconstruct (matches the Mojo codec) for fidelity checks."""
    out = qbyte.shape[0]
    inh = qbyte.shape[1] * 2
    nb = (qbyte.view(torch.uint8).to(torch.int16) & 0xFF)
    lo = nb & 0xF
    hi = (nb >> 4) & 0xF
    v = torch.zeros(out, inh, dtype=torch.int16)
    v[:, 0::2] = torch.where(lo >= 8, lo - 16, lo)
    v[:, 1::2] = torch.where(hi >= 8, hi - 16, hi)
    sc = wscales.float().t()[:, torch.arange(inh) // group]
    return v.float() * sc + lora_up.float() @ lora_down.float().t()


def _is_class_a(key, shape, group):
    if not key.endswith(".weight") or len(shape) != 2:
        return False
    out, inh = shape
    if inh < 256 or out < 256:
        return False
    if any(s in key for s in EXCLUDE_SUBSTR):
        return False
    return inh % group == 0


# W4A4 runtime supports these `in` (K) sizes (FWHT shared mem fits ≤48KB).
W4A4_FWHT_KS = (2048, 4096, 8192)


def build_svdquant_w4a4_slab(
    src: str, out: str, *, key_prefix: str = "",
    rank: int = W4A4_DEFAULT_RANK, group: int = DEFAULT_GROUP, sample: int = 8,
):
    """Quantize a DiT to a HYBRID W4A4 slab: class-A linears with in∈{2048,4096,
    8192} → QuaRot W4A4 (per-out int4 of the Hadamard-rotated residual, rank R);
    in=16384 (and non-power-of-2 in) → W4A16 group-64 (the FWHT shared-mem blocker,
    48 ff.net.2 layers). Non-class-A pass through verbatim. The manifest tags each
    quantized layer 'w4a4'|'w4a16' so the Mojo block dispatch knows. Returns manifest."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    torch.set_grad_enabled(False)
    f = safe_open(src, "pt")
    keys = list(f.keys())
    tensors: dict = {}
    manifest = {"method": "svdquant_w4a4_hybrid", "rank": rank, "w4a16_rank": DEFAULT_RANK,
                "group": group, "hadamard": "sylvester", "layers": {}, "passthrough": []}
    cos4 = []
    n4 = n16 = pn = 0
    for k in keys:
        shape = list(f.get_slice(k).get_shape())
        in_scope = k.startswith(key_prefix) if key_prefix else True
        if in_scope and _is_class_a(k, shape, group):
            W = f.get_tensor(k)
            out_f, in_f = W.shape
            base = k[:-len(".weight")]
            is_pow2 = (in_f & (in_f - 1)) == 0
            if is_pow2 and in_f in W4A4_FWHT_KS:
                qb, ws, ld, lu = quantize_svdquant_w4a4(W, rank)
                tensors[base + ".qweight"] = qb          # I8 [out,in/2] (per-out, rotated)
                tensors[base + ".wscale"] = ws           # bf16 [out]
                tensors[base + ".lora_down"] = ld        # bf16 [in,R]
                tensors[base + ".lora_up"] = lu          # bf16 [out,R]
                if (base + ".bias") in keys:
                    tensors[base + ".bias"] = f.get_tensor(base + ".bias").to(torch.bfloat16)
                manifest["layers"][k] = {"kind": "w4a4", "in": in_f, "out": out_f, "rank": rank}
                n4 += 1
                if len(cos4) < sample:
                    from .svdquant_int4 import w4a4_forward
                    X = torch.randn(64, in_f)
                    a = w4a4_forward(X, qb, ws, ld, lu).flatten()
                    b = (X.double() @ W.double().t()).flatten()
                    cos4.append((a @ b / (a.norm() * b.norm())).item())
            else:
                qb, wsc, ld, lu, sm = quantize_svdquant_w4a16(W, DEFAULT_RANK, group, full_svd=False)
                tensors[base + ".qweight"] = qb          # I8 [out,in/2] (group-64)
                tensors[base + ".wscales"] = wsc         # bf16 [in/G,out]
                tensors[base + ".lora_down"] = ld
                tensors[base + ".lora_up"] = lu
                tensors[base + ".smooth"] = sm
                if (base + ".bias") in keys:
                    tensors[base + ".bias"] = f.get_tensor(base + ".bias").to(torch.bfloat16)
                manifest["layers"][k] = {"kind": "w4a16", "in": in_f, "out": out_f, "rank": DEFAULT_RANK}
                n16 += 1
        else:
            t = f.get_tensor(k)
            tensors[k] = t.to(torch.bfloat16) if t.dtype in (
                torch.float32, torch.bfloat16, torch.float16) else t
            manifest["passthrough"].append(k)
            pn += 1
    if cos4:
        ct = torch.tensor(cos4)
        manifest["sample_w4a4_cos"] = {"mean": ct.mean().item(), "min": ct.min().item()}
    manifest["w4a4_count"] = n4
    manifest["w4a16_count"] = n16
    manifest["passthrough_count"] = pn
    save_file(tensors, out, metadata={"svdquant_manifest": json.dumps(manifest)})
    Path(out).with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    manifest["slab_bytes"] = os.path.getsize(out)
    return manifest


def build_svdquant_int4_slab(
    src: str, out: str, *, key_prefix: str = "",
    rank: int = DEFAULT_RANK, group: int = DEFAULT_GROUP,
    full_svd: bool = False, sample: int = 10,
):
    """Quantize every class-A linear of a safetensors DiT → one INT4 slab +
    manifest. Non-class-A tensors pass through VERBATIM (dtype preserved) so the
    slab is a complete loadable model. File-based (the video path — the model is
    a raw safetensors, not a loaded nn.Module). Returns the manifest dict."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    torch.set_grad_enabled(False)
    f = safe_open(src, "pt")
    keys = list(f.keys())
    tensors: dict = {}
    manifest = {"method": "svdquant_int4", "quant_bits": 4, "rank": rank,
                "group": group, "quantized": {}, "passthrough": []}
    cos_samples = []
    qn = pn = 0
    for k in keys:
        shape = list(f.get_slice(k).get_shape())
        in_scope = k.startswith(key_prefix) if key_prefix else True
        if in_scope and _is_class_a(k, shape, group):
            W = f.get_tensor(k)
            out_f, in_f = W.shape
            qb, ws, ld, lu, sm = quantize_svdquant_w4a16(W, rank, group, full_svd=full_svd)
            base = k[:-len(".weight")]
            tensors[base + ".qweight"] = qb
            tensors[base + ".wscales"] = ws
            tensors[base + ".lora_down"] = ld
            tensors[base + ".lora_up"] = lu
            tensors[base + ".smooth"] = sm
            if (base + ".bias") in keys:
                tensors[base + ".bias"] = f.get_tensor(base + ".bias").to(torch.bfloat16)
            manifest["quantized"][k] = {"in": in_f, "out": out_f, "rank": rank, "group": group}
            qn += 1
            if len(cos_samples) < sample:
                Wr = reconstruct_w4a16(qb, ws, ld, lu, group)
                a = Wr.double().flatten(); b = W.double().flatten()
                cos_samples.append((a @ b / (a.norm() * b.norm())).item())
        else:
            t = f.get_tensor(k)
            tensors[k] = t.to(torch.bfloat16) if t.dtype in (
                torch.float32, torch.bfloat16, torch.float16) else t
            manifest["passthrough"].append(k)
            pn += 1
    save_file(tensors, out, metadata={"svdquant_manifest": json.dumps(manifest)})
    Path(out).with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    if cos_samples:
        cs = torch.tensor(cos_samples)
        manifest["sample_w_cos"] = {"mean": cs.mean().item(), "min": cs.min().item()}
    manifest["quantized_count"] = qn
    manifest["passthrough_count"] = pn
    manifest["slab_bytes"] = os.path.getsize(out)
    return manifest
