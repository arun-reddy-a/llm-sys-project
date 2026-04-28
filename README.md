# DeepSeek-V3 MoE Kernel — NVIDIA Blackwell B200

Custom CUDA kernels for the Mixture-of-Experts (MoE) forward pass, optimized for the DeepSeek-V3 architecture on NVIDIA Blackwell B200 GPUs. Starting from a naive sequential baseline, the kernel is systematically improved through profiler-guided architectural changes to reach **400K tokens/sec** — a **1,300× speedup** over the baseline.

## Results

| Implementation | Latency @ T=1024 | Throughput @ T=4096 | Speedup |
|---|---|---|---|
| Naive Baseline | 3254 ms | 306 tok/s | 1× |
| Opt 1 — Tiled GEMM | 2622 ms | 377 tok/s | 1.2× |
| Opt 2 — Fused Routing | 2628 ms | 377 tok/s | 1.2× |
| Opt 3 — Grouped-GEMM | 88.7 ms | 10,387 tok/s | 37× |
| Opt 4 — Async Double-Buffer | 15.4 ms | 67,616 tok/s | 211× |
| **DeepSeek-V3** | **4.4 ms** | **400,403 tok/s** | **739×** |

Architecture: E=256 experts, D=7168, intermediate=2048, top-K=8, T ∈ {64…4096}.

## Project Structure

```
kernels/moe/
  moe_kernels.cuh       # MoE config struct + all kernel declarations
  moe_kernels.cu        # All kernel implementations (Naive → DeepSeek-V3)

benchmarks/
  bench_moe.cu          # Full sweep benchmark (all variants × sequence lengths)
  bench_moe_smoke.cu    # NVTX-instrumented smoke benchmark (for profiling)
  bench_moe_seqlen.cu   # Sequence-length scaling sweep
  bench_moe_4096.cu     # T=4096 peak throughput benchmark

tests/
  test_moe.cu           # Correctness tests for Opt 1–4
  test_moe_deepseek.cu  # Correctness tests for DeepSeek-V3 routing

profiling/
  profile_moe.sh        # 3-stage orchestrator: nsys → ncu → diagnosis
  diagnose_moe.py       # NCU CSV parser with roofline bottleneck classification
  analyze_ncu.py        # Lightweight NCU CSV parser

utils/
  cuda_utils.cuh        # CUDA error checking, GPU timer helpers

Makefile                # Build targets: bench, test, profile
```

## Optimization Summary

### Naive Baseline
Three kernels dispatched per expert in a host loop (gather → GEMM → scatter), producing 768 sequential launches for 256 experts. `gather_kernel` ran at 1.6% SM occupancy with `block_size=1`. NCU: Latency-Bound throughout.

### Opt 1 — Tiled Shared-Memory GEMM
Replaced scalar GEMM with 16×16 tiled shared-memory blocking. GEMM became Compute-Bound at 97% occupancy, but the `gather_kernel` `block_size=1` bottleneck dominated end-to-end time. 20% improvement.

### Opt 2 — Fused Routing Kernel
Merged gate-logits, softmax, and top-K into a single kernel to eliminate DRAM round-trips. Routing overhead dropped, but the gather bottleneck remained. <1% end-to-end improvement.

### Opt 3 — Grouped-GEMM (30× leap)
Replaced the O(E) dispatch loop with a single O(1) Grouped-GEMM. Tokens are pre-sorted by expert ID; all 256 experts run in one unified GPU grid. NCU: 99.6% occupancy, 73% SM throughput. Latency: 2628 ms → 88.7 ms.

### Opt 4 — Async Double-Buffered DMA
Replaced blocking HBM loads with `__pipeline_memcpy_async` (cp.async). Tile i+1 prefetches into a background SMEM buffer while Tensor Cores compute on tile i, eliminating pipeline stalls. Latency: 88.7 ms → 15.4 ms.

### DeepSeek-V3 — Sigmoid+Bias Routing
Replaced softmax+top-K with a Sigmoid+Bias gate, removing the global reduction dependency across experts. Each expert score is computed independently, fully parallelizing token dispatch. Added 128-bit vectorized memory access. Peak: **400,403 tok/s** at T=4096.

## Build & Run

```bash
# Build everything
make all

# Run full benchmark
./build/bench_moe

# Run smoke benchmark (for profiling)
./build/bench_moe_smoke

# Run correctness tests
./build/test_moe
./build/test_moe_deepseek

# Full profiling pipeline (nsys → ncu → diagnosis)
bash profiling/profile_moe.sh DeepSeek
```

## Profiling

Every optimization followed a **measure → identify → fix → re-measure** loop:

1. **Nsight Systems** (`nsys profile --trace=cuda-hw`): timeline gaps, CPU dispatch overhead, idle SM windows.
2. **Nsight Compute** (`ncu --launch-skip 50 --launch-count 20`): roofline classification (Memory-Bound / Compute-Bound / Latency-Bound), occupancy, warp stall breakdown.
3. **`diagnose_moe.py`**: automated bottleneck classification from NCU CSV exports.

> Note: NCU serializes launches and disables caches, inflating kernel durations (e.g. 1.6 ms → 8.3 ms). All latency/throughput numbers are from native NVTX-gated benchmarks.
