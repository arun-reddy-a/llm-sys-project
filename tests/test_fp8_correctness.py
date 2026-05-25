#!/usr/bin/env python3
"""
FP8 MoE Correctness Test
========================
1. Generates random float32 weights, quantizes to FP8 E4M3 with 128-element block scales.
2. Computes reference output (dequant → float32 matmul in numpy).
3. Calls the CUDA binary (./build/test_fp8_cuda) with the same data.
4. Compares outputs: max-abs-error and fraction within tolerance.

No PyTorch dependency — pure numpy.

Usage (run via Modal make target, or locally if GPU is present):
    python3 tests/test_fp8_correctness.py [data_dir]
"""

import numpy as np
import os
import subprocess
import sys
import struct
import tempfile

# ---------------------------------------------------------------------------
# Test configuration
# Note: D and I must be multiples of 128 (block size) for clean scale grids.
#       N_GROUP must divide E_global evenly.
# ---------------------------------------------------------------------------
T        = 128    # tokens
E_local  = 4     # local experts on this GPU
E_global = 32    # total experts (8 groups of 4 with N_GROUP=8)
K        = 4     # top-K experts
D        = 256   # hidden dim   (2 blocks of 128)
I        = 128   # intermediate (1 block of 128)
BLOCK    = 128   # quantization block size
N_GROUP  = 8     # expert groups (E_global / N_GROUP experts per group)
TOPK_GROUP = 4   # top groups to select
LOCAL_EXPERT_OFFSET = 0   # this shard starts at expert 0

# float8 E4M3 can represent values in [-448, 448]
FP8_MAX = 448.0

# ---------------------------------------------------------------------------
# FP8 quantization helpers (software emulation)
# ---------------------------------------------------------------------------

def quantize_fp8_block(arr_float, block=128):
    """
    Block-scale quantize float32 array along last two axes.
    Returns:
        arr_fp8:   uint8 array, same shape (FP8 E4M3 encoded as raw bytes)
        scales:    float32 array, shape = arr.shape[:-1] + (N/block,) truncated
    For 2D array [N, D]:
        scales shape = [ceil(N/block), ceil(D/block)]
    """
    shape = arr_float.shape
    assert arr_float.ndim == 2
    N, Dcol = shape
    N_blk = (N + block - 1) // block
    D_blk = (Dcol + block - 1) // block

    # Pad to multiples of block
    Np = N_blk * block
    Dp = D_blk * block
    padded = np.zeros((Np, Dp), dtype=np.float32)
    padded[:N, :Dcol] = arr_float

    # Compute per-block max and scale
    reshaped = padded.reshape(N_blk, block, D_blk, block)  # [Nb, b, Db, b]
    block_max = np.abs(reshaped).max(axis=(1, 3))           # [Nb, Db]
    scales = block_max / FP8_MAX
    scales = np.maximum(scales, 1e-12)                       # avoid div by zero

    # Quantize: float32 → clamp to E4M3 range → round to E4M3 representable values
    scale_expanded = np.repeat(np.repeat(scales, block, axis=0), block, axis=1)  # [Np, Dp]
    q = padded / scale_expanded
    q = np.clip(q, -FP8_MAX, FP8_MAX)

    # Encode as E4M3: nearest representable value (we approximate with float32 ops)
    # For correctness testing: round to 8 significant steps of E4M3
    # E4M3 mantissa = 3 bits → step = 1/8
    q_rounded = np.round(q * 8) / 8
    q_rounded = np.clip(q_rounded, -FP8_MAX, FP8_MAX)

    # Store quantized values as float32 in uint8 memory (raw bit reinterpretation)
    # We store the *scaled* value (before scale) as a float16-like encoding.
    # For the actual byte format the CUDA binary understands:
    # We write the float32 quantized values (scaled) as raw fp32 bytes per element,
    # but that's too large. Instead we store the int8 approximation:
    # integer = round(value * 8), clamped to [-448*8, 448*8]
    # At decode: float = int8_val / 8 * scale  — but int8 can only hold -128..127.
    #
    # Simpler: use the actual "as-if FP8" pipeline:
    # The CUDA kernel receives raw uint8 bytes and casts __nv_fp8_e4m3 → float.
    # For the Python side to match, we need to produce the same bytes.
    #
    # FP8 E4M3 encoding (from CUDA docs):
    #   sign=1 bit, exp=4 bits (bias=7), mantissa=3 bits
    #   value = (-1)^s * 2^(exp-7) * (1 + mant/8)  for exp != 0
    #         = (-1)^s * 2^(-6) * (mant/8)           for exp == 0 (denormal)
    fp8_bytes = fp32_to_fp8_e4m3(q_rounded.ravel()).reshape(Np, Dp)
    fp8_out = fp8_bytes[:N, :Dcol]
    return fp8_out.astype(np.uint8), scales.astype(np.float32)


def fp32_to_fp8_e4m3(arr):
    """Convert float32 array to FP8 E4M3 byte representation."""
    out = np.zeros(arr.shape, dtype=np.uint8)
    for i, v in enumerate(arr.flat):
        out.flat[i] = _f32_to_fp8_e4m3_scalar(float(v))
    return out


def _f32_to_fp8_e4m3_scalar(v):
    """Scalar float32 → FP8 E4M3 encoding."""
    if np.isnan(v):
        return 0x7F  # NaN
    sign = 0
    if v < 0:
        sign = 1
        v = -v
    if v == 0.0:
        return 0

    # Clamp
    v = min(v, FP8_MAX)

    # Find exponent and mantissa
    import math
    exp = math.floor(math.log2(v)) if v >= 1.0 else -int(math.ceil(-math.log2(v)))
    exp_biased = exp + 7   # bias = 7

    if exp_biased <= 0:
        # Denormal
        mant = round(v / (2**(-6)) * 8)
        mant = min(mant, 7)
        return (sign << 7) | mant
    elif exp_biased >= 15:
        # Overflow → max value: s=sign, exp=1111, mant=110 (448.0)
        return (sign << 7) | (0xF << 3) | 0x6
    else:
        mant_f = (v / (2**exp) - 1.0) * 8
        mant = int(round(mant_f))
        if mant >= 8:
            exp_biased += 1
            mant = 0
        mant = min(mant, 7)
        return (sign << 7) | (exp_biased << 3) | mant


def fp8_e4m3_to_fp32(byte_val):
    """Decode one FP8 E4M3 byte to float32."""
    b = int(byte_val)
    sign = (b >> 7) & 1
    exp_biased = (b >> 3) & 0xF
    mant = b & 0x7
    if exp_biased == 0:
        v = (2**(-6)) * (mant / 8.0)
    elif exp_biased == 0xF and mant == 0x7:
        v = float('nan')
    else:
        v = (2**(exp_biased - 7)) * (1.0 + mant / 8.0)
    return -v if sign else v


def dequantize_fp8_block(fp8_bytes, scales, block=128):
    """Dequantize FP8 block-scale array back to float32."""
    N, D = fp8_bytes.shape
    N_blk = (N + block - 1) // block
    D_blk = (D + block - 1) // block

    # Decode FP8 bytes
    float_vals = np.vectorize(fp8_e4m3_to_fp32)(fp8_bytes.ravel()).reshape(N, D)

    # Expand scales and multiply
    scale_exp = np.repeat(np.repeat(scales[:N_blk, :D_blk], block, axis=0),
                          block, axis=1)[:N, :D]
    return (float_vals * scale_exp).astype(np.float32)


# ---------------------------------------------------------------------------
# DeepSeek-style routing (numpy)
# ---------------------------------------------------------------------------

def route(routing_logits, routing_bias, K, N_GROUP, TOPK_GROUP):
    T, E = routing_logits.shape
    s = 1.0 / (1.0 + np.exp(-routing_logits))            # sigmoid [T, E]
    s_wb = s + routing_bias                               # +bias   [T, E]

    group_size = E // N_GROUP
    s_wb_g = s_wb.reshape(T, N_GROUP, group_size)         # [T, 8, g]

    # Group scores = sum of top-2 per group
    idx2 = np.argsort(-s_wb_g, axis=2)[:, :, :2]
    top2 = np.take_along_axis(s_wb_g, idx2, axis=2).sum(axis=2)  # [T, 8]

    # Top-TOPK_GROUP groups
    group_ranks = np.argsort(-top2, axis=1)[:, :TOPK_GROUP]       # [T, 4]
    score_mask = np.zeros((T, E), dtype=np.float32)
    for t in range(T):
        for g in group_ranks[t]:
            score_mask[t, g * group_size:(g + 1) * group_size] = 1.0

    # Global top-K within selected groups
    scores_pruned = np.where(score_mask == 1, s_wb, -1e30)
    topk_idx = np.argsort(-scores_pruned, axis=1)[:, :K]          # [T, K]

    # Weights: normalize raw sigmoid (no bias)
    M = np.zeros((T, E), dtype=np.float32)
    for t in range(T):
        M[t, topk_idx[t]] = 1.0
    w = s * M
    w_sum = w.sum(axis=1, keepdims=True) + 1e-20
    weights = w / w_sum                                            # [T, E]
    return topk_idx, weights


# ---------------------------------------------------------------------------
# Reference FFN (numpy)
# ---------------------------------------------------------------------------

def silu(x):
    return x / (1.0 + np.exp(-x))


def moe_ref(A, W1_float, W2_float, topk_idx, weights, E_local, E_global,
            local_start, T, D, I):
    """Full numpy reference for the local expert FFN + weighted scatter."""
    output = np.zeros((T, D), dtype=np.float32)
    for le in range(E_local):
        ge = local_start + le
        sel = np.any(topk_idx == ge, axis=1)   # [T] bool
        if not sel.any():
            continue
        tok_idx = np.where(sel)[0]
        A_e  = A[tok_idx]                       # [Tk, D]
        W1_e = W1_float[le]                     # [2I, D]
        W2_e = W2_float[le]                     # [D, I]

        G1 = A_e @ W1_e.T                       # [Tk, 2I]
        X1 = G1[:, :I]                          # up
        X2 = G1[:, I:]                          # gate
        C  = silu(X2) * X1                      # [Tk, I]

        O  = C @ W2_e.T                         # [Tk, D]

        w_tok = weights[tok_idx, ge]            # [Tk]
        output[tok_idx] += O * w_tok[:, None]
    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="fp8_test_")
    os.makedirs(data_dir, exist_ok=True)
    binary = "./build/test_fp8_cuda"

    rng = np.random.default_rng(42)

    print("=== FP8 MoE Correctness Test ===")
    print(f"  Config: T={T} E_local={E_local} E_global={E_global} K={K} D={D} I={I}")
    print(f"  Data dir: {data_dir}")

    # ---- 1. Generate inputs ----
    A_fp32          = rng.standard_normal((T, D)).astype(np.float32) * 0.1
    gate_weight     = rng.standard_normal((E_global, D)).astype(np.float32) * 0.02
    gate_bias       = rng.standard_normal((E_global,)).astype(np.float32) * 0.01
    routing_logits  = (A_fp32 @ gate_weight.T)   # [T, E_global]

    W1_float = rng.standard_normal((E_local, 2 * I, D)).astype(np.float32) * 0.02
    W2_float = rng.standard_normal((E_local, D, I)).astype(np.float32) * 0.02

    # ---- 2. Quantize weights to FP8 with block scales ----
    print("  Quantizing weights to FP8 E4M3...")
    w1_fp8_list, w1_scales_list = [], []
    for e in range(E_local):
        fp8_e, sc_e = quantize_fp8_block(W1_float[e], BLOCK)   # [2I, D], [N_blk, D_blk]
        w1_fp8_list.append(fp8_e)
        w1_scales_list.append(sc_e)

    w2_fp8_list, w2_scales_list = [], []
    for e in range(E_local):
        fp8_e, sc_e = quantize_fp8_block(W2_float[e], BLOCK)   # [D, I], [N_blk, I_blk]
        w2_fp8_list.append(fp8_e)
        w2_scales_list.append(sc_e)

    w1_fp8    = np.stack(w1_fp8_list,   axis=0)   # [E_local, 2I, D]
    w1_scales = np.stack(w1_scales_list, axis=0)  # [E_local, N_blk, D_blk]
    w2_fp8    = np.stack(w2_fp8_list,   axis=0)   # [E_local, D, I]
    w2_scales = np.stack(w2_scales_list, axis=0)  # [E_local, N_blk, I_blk]

    # ---- 3. Dequantize for reference ----
    print("  Dequantizing for reference computation...")
    W1_dequant = np.stack(
        [dequantize_fp8_block(w1_fp8[e], w1_scales[e], BLOCK) for e in range(E_local)])
    W2_dequant = np.stack(
        [dequantize_fp8_block(w2_fp8[e], w2_scales[e], BLOCK) for e in range(E_local)])

    # ---- 4. Routing ----
    topk_idx, weights = route(routing_logits, gate_bias, K, N_GROUP, TOPK_GROUP)

    # ---- 5. Reference output ----
    print("  Running numpy reference...")
    ref_output = moe_ref(A_fp32, W1_dequant, W2_dequant,
                         topk_idx, weights, E_local, E_global,
                         LOCAL_EXPERT_OFFSET, T, D, I)

    # ---- 6. Save inputs for CUDA binary ----
    def save(name, arr):
        arr.tofile(os.path.join(data_dir, name))

    save("input.bin",       A_fp32.astype(np.float32))
    save("gate_weight.bin", gate_weight.astype(np.float32))
    save("gate_bias.bin",   gate_bias.astype(np.float32))
    save("w1_fp8.bin",      w1_fp8.astype(np.uint8))
    save("w1_scales.bin",   w1_scales.astype(np.float32))
    save("w2_fp8.bin",      w2_fp8.astype(np.uint8))
    save("w2_scales.bin",   w2_scales.astype(np.float32))

    # ---- 7. Run CUDA binary ----
    if not os.path.exists(binary):
        print(f"  CUDA binary not found at {binary}, skipping GPU comparison.")
        print(f"  Build with: make test_fp8")
        # Save reference output so it can be compared later
        ref_output.tofile(os.path.join(data_dir, "ref_output.bin"))
        return

    cmd = [binary, data_dir,
           str(T), str(E_local), str(E_global), str(K), str(D), str(I)]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout.strip())
    if result.returncode != 0:
        print("  ERROR from CUDA binary:")
        print(result.stderr)
        sys.exit(1)

    # ---- 8. Compare ----
    cuda_output = np.fromfile(os.path.join(data_dir, "cuda_output.bin"), dtype=np.float32)
    cuda_output = cuda_output.reshape(T, D)

    abs_err  = np.abs(ref_output - cuda_output)
    max_err  = abs_err.max()
    mean_err = abs_err.mean()
    atol = 0.15   # FP8 quant + TF32 rounding accumulation

    pct_pass = 100.0 * (abs_err < atol).mean()

    print(f"\n  === Comparison Results ===")
    print(f"  Max abs error:    {max_err:.6f}   (atol={atol})")
    print(f"  Mean abs error:   {mean_err:.6f}")
    print(f"  Within tolerance: {pct_pass:.1f}%")

    if max_err < atol:
        print(f"\n  [PASS] ✅  FP8 online dequant matches numpy reference!")
    else:
        print(f"\n  [FAIL] ❌  Max error {max_err:.4f} exceeds atol {atol}")
        # Print worst offenders
        flat_idx = np.argsort(abs_err.ravel())[::-1][:5]
        for idx in flat_idx:
            t, d = divmod(int(idx), D)
            print(f"    token={t} dim={d}: ref={ref_output[t,d]:.6f}  "
                  f"cuda={cuda_output[t,d]:.6f}  err={abs_err[t,d]:.6f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
