#!/usr/bin/env python3
"""
Local benchmark for the FlashInfer MLSys 2026 fused_moe submission.
Tests a range of sequence lengths including small and non-power-of-2 values.

Usage:
    python scripts/run_local.py [--compare-baseline] [--warmup N] [--iters N]
"""
import argparse
import sys
import time
import math
import torch

sys.path.insert(0, ".")
from solution.triton.kernel import kernel as our_kernel

# Problem definition constants
NUM_EXPERTS     = 256
NUM_LOCAL_EXP   = 32
LOCAL_OFFSET    = 0          # first 32 experts are local
HIDDEN          = 7168
INTERMEDIATE    = 2048
GEMM1_OUT       = 4096
TOP_K           = 8
N_GROUP         = 8
TOPK_GROUP      = 4
BLOCK_SIZE      = 128
ROUTED_SCALE    = 2.5        # DeepSeek-V3 default

# Primary focus: medium → large sequence lengths where GEMM dominates.
# Small T (<64) left for later — Python overhead swamps the kernel there.
SEQ_LENS = [
    # Medium (routing overhead manageable)
    52, 54, 56, 58, 59, 62, 80,
    128, 256, 512,
    # Large (GEMM-bound — where deep_gemm should shine)
    901, 1024, 2048, 4096,
    8192, 11948, 14107,
]


def make_inputs(T: int, device="cuda"):
    """Create random FP8 + scale tensors matching the competition API (7-tensor variant)."""
    H, I, G1 = HIDDEN, INTERMEDIATE, GEMM1_OUT
    E, EL, BS = NUM_EXPERTS, NUM_LOCAL_EXP, BLOCK_SIZE

    routing_logits      = torch.randn(T, E,  dtype=torch.float32, device=device)
    gemm1_weights       = torch.randn(EL, G1, H, dtype=torch.float32, device=device).to(torch.float8_e4m3fn)
    gemm1_weights_scale = torch.ones(EL, G1 // BS, H // BS, dtype=torch.float32, device=device)
    gemm2_weights       = torch.randn(EL, H, I,  dtype=torch.float32, device=device).to(torch.float8_e4m3fn)
    gemm2_weights_scale = torch.ones(EL, H // BS, I // BS, dtype=torch.float32, device=device)
    output              = torch.zeros(T, H, dtype=torch.bfloat16, device=device)

    # hidden_states passed as kwargs (may or may not be in competition API)
    hidden_states       = torch.randn(T, H, dtype=torch.float32, device=device).to(torch.float8_e4m3fn)
    hidden_states_scale = torch.ones(H // BS, T, dtype=torch.float32, device=device)

    return dict(
        positional=(routing_logits, gemm1_weights, gemm1_weights_scale,
                    gemm2_weights, gemm2_weights_scale,
                    LOCAL_OFFSET, ROUTED_SCALE, output),
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
    )


def bench_one(T: int, warmup: int, iters: int, compare_baseline: bool):
    inp = make_inputs(T)
    pos_args = inp["positional"]
    hs, hss  = inp["hidden_states"], inp["hidden_states_scale"]

    for _ in range(warmup):
        our_kernel(*pos_args, hidden_states=hs, hidden_states_scale=hss)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        our_kernel(*pos_args, hidden_states=hs, hidden_states_scale=hss)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    t_our   = sum(times) / len(times)
    tps_our = T / (t_our * 1e-3)

    baseline_str = ""
    if compare_baseline:
        try:
            from flashinfer.fused_moe import trtllm_fp8_block_scale_moe as bl_fn
            rl, w1, w1s, w2, w2s, lo, rsf, out = pos_args
            rb = torch.zeros(NUM_EXPERTS, dtype=torch.float32, device=rl.device)
            bl_times = []
            for _ in range(warmup):
                bl_fn(rl, rb, hs, hss, w1, w1s, w2, w2s, lo, rsf, out)
            torch.cuda.synchronize()
            for _ in range(iters):
                t0 = time.perf_counter()
                bl_fn(rl, rb, hs, hss, w1, w1s, w2, w2s, lo, rsf, out)
                torch.cuda.synchronize()
                bl_times.append((time.perf_counter() - t0) * 1e3)
            bl_times.sort()
            t_bl = sum(bl_times) / len(bl_times)
            speedup = t_bl / t_our
            baseline_str = f"  baseline={t_bl:7.3f}ms  speedup={speedup:.2f}x"
        except Exception as e:
            baseline_str = f"  [baseline N/A: {e}]"

    print(f"  T={T:<6}  mean={t_our:7.3f}ms  tok/s={tps_our:>12,.0f}{baseline_str}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warmup",            type=int,  default=3)
    p.add_argument("--iters",             type=int,  default=20)
    p.add_argument("--compare-baseline",  action="store_true")
    p.add_argument("--seq-lens",          type=str,  default="",
                   help="Comma-separated list; overrides defaults")
    args = p.parse_args()

    seq_lens = (
        [int(x) for x in args.seq_lens.split(",") if x.strip()]
        if args.seq_lens else SEQ_LENS
    )

    print(f"\n=== FP8 Block-Scale MoE — Local Benchmark ===")
    print(f"    warmup={args.warmup}  iters={args.iters}  "
          f"compare_baseline={args.compare_baseline}\n")
    header = f"  {'T':<8}  {'mean(ms)':>9}  {'tok/s':>14}"
    if args.compare_baseline:
        header += "  baseline(ms)    speedup"
    print(header)
    print("  " + "-" * (70 if args.compare_baseline else 40))

    for T in seq_lens:
        try:
            bench_one(T, args.warmup, args.iters, args.compare_baseline)
        except Exception as e:
            print(f"  T={T:<6}  ERROR: {e}")

    print()


if __name__ == "__main__":
    main()
