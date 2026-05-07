#!/usr/bin/env python3
"""
Benchmark vLLM's fused_moe Triton kernel with the DeepSeek-V3 MoE config.
E=256, D=7168, I=2048, K=8.  Runs on B200 with bfloat16 (production dtype).
Output format matches bench_moe.cu so the local entrypoint can parse both.
"""
import torch
import time
import sys

SEQ_LENS = [64, 256, 512, 1024, 2048, 4096]
E, D, I_DIM, K = 256, 7168, 2048, 8
WARMUP = int(sys.argv[1]) if len(sys.argv) > 1 else 10
ITERS  = int(sys.argv[2]) if len(sys.argv) > 2 else 50


def bench_vllm(T: int):
    from vllm.model_executor.layers.fused_moe import fused_moe

    dtype  = torch.bfloat16
    hidden = torch.randn(T, D,          dtype=dtype,             device="cuda")
    w1     = torch.randn(E, 2 * I_DIM, D, dtype=dtype,          device="cuda")
    w2     = torch.randn(E, D, I_DIM,  dtype=dtype,             device="cuda")
    router = torch.randn(T, E,          dtype=torch.float32,     device="cuda")

    for _ in range(WARMUP):
        fused_moe(hidden, w1, w2, router, top_k=K, renormalize=True, inplace=False)
    torch.cuda.synchronize()

    lats = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        fused_moe(hidden, w1, w2, router, top_k=K, renormalize=True, inplace=False)
        torch.cuda.synchronize()
        lats.append((time.perf_counter() - t0) * 1000)

    lats.sort()
    t_min  = lats[0]
    t_mean = sum(lats) / len(lats)
    return t_min, t_mean, T / (t_mean * 1e-3)


def main():
    print(f"\n=== vLLM fused_moe Benchmark (BF16 Triton, B200) ===")
    print(f"    warmup={WARMUP}  iters={ITERS}\n")
    print(f"  {'Variant':<16}  {'Config':<44}  {'Min(ms)':>8}  {'Mean(ms)':>9}  {'Tok/s':>12}")
    print("  " + "-" * 100)

    for T in SEQ_LENS:
        t_min, t_mean, tps = bench_vllm(T)
        config = f"T={T},E={E},EL=32,K={K},D={D},I={I_DIM}"
        print(f"  {'vLLM-fused_moe':<16}  {config:<44}  {t_min:>8.3f}  {t_mean:>9.3f}  {tps:>12.0f}")

    print()


if __name__ == "__main__":
    main()
