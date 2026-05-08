# FlashInfer MLSys 2026 — fused_moe Submission

Competition: [FlashInfer AI Kernel Generation Contest @ MLSys 2026](http://mlsys26.flashinfer.ai/)
Track: **fused_moe** — `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`

## Problem

FP8 block-scale MoE with DeepSeek-V3 routing. Each forward pass runs:

1. Softmax routing → top-8 expert selection → renormalize by sum × 2.5
2. Gather tokens for local experts (32 of 256)
3. GEMM1: `[N_local, 7168] × [32, 4096, 7168]^T` → `[N_local, 4096]` BF16
4. SwiGLU activation
5. FP8 requantize activations
6. GEMM2: `[N_local, 2048] × [32, 7168, 2048]^T` → `[N_local, 7168]` BF16
7. Weighted scatter-add into BF16 output

| Parameter | Value |
|-----------|-------|
| Total experts (E) | 256 |
| Local experts (E_local) | 32 |
| Top-K | 8 |
| Groups / selected groups | 8 / 4 |
| Hidden dim (H) | 7168 |
| Intermediate dim (I) | 2048 |
| GEMM1 output dim | 4096 (gate+up interleaved) |
| Block scale size | 128 |
| Output dtype | bfloat16 |

## Approach

### Routing

Softmax over all 256 experts → top-8 selection → weights renormalized to sum × 2.5. This matches the reference implementation exactly. Tokens are then sorted by expert ID to enable contiguous grouped GEMM.

### Grouped GEMM (custom Triton)

Both GEMM1 and GEMM2 use a single custom Triton FP8 block-scale grouped GEMM kernel:

```
grid = (total_m_tiles, N // 128)
where total_m_tiles = Σ_expert ceil(tokens_per_expert / BLOCK_M)
```

**Tile→expert mapping** is precomputed on CPU before the launch. Each CUDA block knows immediately which expert and which row range it owns — no in-kernel binary search.

**Kernel inner loop** per K-block:
```python
a  = load fp8 [BLOCK_M, 128]     # token tile
a_s = load fp32 [BLOCK_M]        # per-row per-k-block scale
b  = load fp8 [128, 128]         # weight tile (transposed)
b_s = load fp32 []               # per (expert, n-block, k-block) scale
dot = tl.dot(a, b.T, out_dtype=tl.float32)  # FP8 Tensor Cores
acc += a_s[:, None] * b_s * dot
```

Accumulation is in FP32, output cast to BF16 at store time.

**Key parameters**: BLOCK_M=32, BLOCK_N=128, BLOCK_K=128, num_warps=8, num_stages=3 (async double-buffering via Triton's software pipelining).

### FP8 Requantization

A Triton kernel computes per-row-per-128-block amax, derives `scale = amax / 448.0`, and stores the quantized FP8 value and scale. Used between GEMM1 and GEMM2 for the SwiGLU activations.

### Weighted Scatter

Triton `tl.atomic_add` accumulates each expert's output into the pre-allocated BF16 output tensor, weighted by the renormalized routing weight.

## Correctness

Verified against a pure PyTorch FP32 reference (exact block-scale dequantization) on B200 (CUDA 12.8.1):

```
=== Triton vs PyTorch reference (atol=1.0, rtol=0.3) ===

  T=16      max_abs=7.80e-05  max_rel=3.41e+01  match=100.0%  [PASS]
  T=52      max_abs=1.70e-04  max_rel=6.15e+01  match=100.0%  [PASS]
  T=80      max_abs=1.70e-04  max_rel=6.15e+01  match=100.0%  [PASS]
  T=128     max_abs=1.70e-04  max_rel=6.15e+01  match=100.0%  [PASS]
  T=256     max_abs=1.70e-04  max_rel=7.10e+01  match=100.0%  [PASS]
  T=512     max_abs=1.70e-04  max_rel=7.10e+01  match=100.0%  [PASS]
  T=901     max_abs=1.72e-04  max_rel=7.10e+01  match=100.0%  [PASS]
  T=2048    max_abs=1.72e-04  max_rel=9.02e+01  match=100.0%  [PASS]
  T=4096    max_abs=2.63e-04  max_rel=9.61e+01  match=100.0%  [PASS]

ALL PASS
```

Max absolute error < 3e-4 across all tested sequence lengths. (max_rel is large because many output positions are near-zero, making relative error non-informative; the absolute error and 100% match rate are what matter.)

## Performance (B200, CUDA 12.8.1)

Three-way comparison: FlashInfer TRT-LLM baseline, DeepGEMM FP8, and our custom Triton FP8 kernel.

| T | FlashInfer baseline | deep_gemm FP8 | Our Triton FP8 | vs baseline | vs deep_gemm |
|---|---------------------|---------------|----------------|-------------|--------------|
| 52 | 0.179 ms | 0.347 ms | 0.864 ms | 0.21x | 0.40x |
| 56 | 0.240 ms | 0.340 ms | 1.013 ms | 0.24x | 0.34x |
| 80 | 0.269 ms | 0.343 ms | 1.023 ms | 0.26x | 0.34x |
| 901 | 0.699 ms | 0.450 ms | 1.180 ms | 0.59x | 0.38x |
| 2048 | — | 0.576 ms | 1.504 ms | — | 0.38x |
| 4096 | — | 0.880 ms | 2.223 ms | — | 0.40x |
| 11948 | 4.985 ms | 1.564 ms | 5.377 ms | 0.93x | 0.29x |
| 14107 | 6.359 ms | 1.746 ms | **5.909 ms** | **1.08x** | 0.30x |

At small T (< 128), all three kernels are bottlenecked by Python/routing/launch overhead — not GEMM. The FlashInfer baseline is fastest here because it has a highly optimized CUDA routing kernel. Our Triton kernel pays ~0.9ms of fixed Python overhead, making it 4–5x slower at small T.

At large T, GEMM dominates and our kernel pulls within range of the FlashInfer baseline (1.08x at T=14107). deep_gemm is consistently 2.5–3.4x faster than our Triton kernel across all sizes — it uses persistent WGMMA + TMA Blackwell kernels that cannot currently be reproduced in Triton.

## Why not torch.compile?

`torch.compile` (Inductor) cannot generate this kernel because:
1. `nonzero()` (from routing) produces dynamic shapes Inductor can't trace through
2. Inductor has no FP8 block-scale GEMM template — it would fall back to BF16 GEMM
3. There is no grouped GEMM template: 32 separate kernel launches vs. one

## Files

```
solution/triton/kernel.py        — competition entry point (kernel() function)
scripts/run_local.py             — local benchmark
scripts/run_modal.py             — B200 benchmark via Modal
scripts/check_correctness.py     — correctness check vs PyTorch reference
config.toml                      — competition metadata
```

## Run

```bash
# Correctness check (requires B200 via Modal)
python3 -m modal run scripts/run_modal.py --check-correctness

# Benchmark on B200
python3 -m modal run scripts/run_modal.py --warmup 5 --iters 30

# Local benchmark (requires CUDA GPU)
python scripts/run_local.py
```
