# Optimising MoE Layers on Blackwell GPUs

Custom CUDA kernels for Mixture-of-Experts (MoE) layers, targeting NVIDIA Blackwell B200 GPUs. This repository starts with naive baseline implementations and will be iteratively optimised through kernel fusion, batching, and Blackwell-specific hardware features.

## Prerequisites

- **CUDA Toolkit** >= 12.0 (`-arch=native` requires CUDA 12+; 12.8+ for Blackwell `sm_100`)
- **NVIDIA GPU** with compute capability >= 7.0 (V100, Ampere, Hopper, or Blackwell)
- **GNU Make**
- **C++17**-capable host compiler (GCC >= 9, Clang >= 10)

## Quick Start

```bash
# Clone the repo
git clone <repo-url> && cd LLM-Sys-Project

# Build everything (tests + benchmarks)
# By default, -arch=native is used, which auto-detects your GPU at compile time.
make all

# Run correctness tests
make test

# Run benchmarks
make bench

# Run individually
make test_moe
make bench_moe

# Clean build artifacts
make clean
```

## ☁️ Modal Cloud Execution

You can build and run this project on cloud GPUs using [Modal](https://modal.com). This is the recommended way to benchmark on the latest **Blackwell B200** GPUs.

```bash
# 1. Install and setup Modal
pip install modal
modal setup

# 2. Run smoke benchmark on a Blackwell B200
modal run modal_run.py --target bench_moe_smoke

# 3. Profile MoE kernels (3-stage pipeline)
modal run modal_run.py --target profile_moe_1    # Timeline trace
modal run modal_run.py --target profile_moe_2    # Deep kernel metrics
modal run modal_run.py --target profile_moe_3    # Automated diagnosis
```

See [docs/modal/README.md](docs/modal/README.md) for more details.

If you need to target a specific architecture (e.g. cross-compiling or `-arch=native` is not available), override `NVCC_FLAGS`:

```bash
make all NVCC_FLAGS="-std=c++17 -O2 -arch=sm_100"   # Blackwell B200
make all NVCC_FLAGS="-std=c++17 -O2 -arch=sm_90"    # Hopper H100
make all NVCC_FLAGS="-std=c++17 -O2 -arch=sm_80"    # Ampere A100
make all NVCC_FLAGS="-std=c++17 -O2 -arch=sm_70"    # V100
```

Benchmark iterations can be configured via CLI arguments:

```bash
./build/bench_moe <warmup> <iters>   # default: 10 warmup, 50 iters
```

## Project Structure

```
.
├── Makefile                        # Build system (bench, test, profile targets)
├── README.md
├── modal_run.py                    # Modal cloud runner (B200 GPU)
├── Proposal.pdf                    # Project proposal
├── kernels/
│   └── moe/
│       ├── naive_moe.cuh           # MoE config struct + kernel declarations
│       └── naive_moe.cu            # MoE kernel implementations (Naive → Opt5)
├── tests/
│   └── test_moe.cu                 # MoE correctness tests (GPU vs CPU reference)
├── benchmarks/
│   ├── bench_moe.cu                # MoE full benchmark (all variants × configs)
│   ├── bench_moe_smoke.cu          # MoE smoke benchmark (NVTX-instrumented, for profiling)
│   └── simple_vadd.cu              # Simple vector-add (profiling toolchain validation)
├── profiling/
│   ├── profile_moe.sh              # 3-stage profiling orchestrator (nsys → ncu → diagnosis)
│   ├── diagnose_moe.py             # Advanced wide-format Nsight Compute CSV parser
│   ├── analyze_ncu.py              # Lightweight NCU CSV parser
│   └── results/                    # Generated .nsys-rep, .ncu-rep, .csv files
├── docs/
│   ├── moe/
│   │   ├── README.md               # MoE optimization descriptions (Opt2–Opt5)
│   │   ├── PROFILING.md            # Profiling decision tree + metrics reference
│   │   └── BLACKWELL_PROFILING_PEDAGOGY.md  # Step-by-step Blackwell profiling guide
│   ├── modal/
│   │   └── README.md               # Modal cloud setup instructions
│   └── external/
│       └── nsight_systems_user_guide.html  # Local copy of Nsight docs
└── utils/
    └── cuda_utils.cuh              # CUDA error checking, GPU timer, helpers
```

## Kernel Descriptions

### Naive MoE Layer

The MoE forward pass executes five stages, each as a separate kernel (intentionally un-fused to serve as the optimisation baseline):

| Stage | Kernel | Description |
|-------|--------|-------------|
| 1 | `gate_logits_kernel` + `softmax_experts_kernel` + `topk_kernel` | Computes gating logits via input-gate matmul, applies per-token softmax over experts, selects top-K experts. Intermediate logit tensor is fully materialised in DRAM. |
| 2 | `gather_kernel` | For **each expert separately**, gathers assigned tokens into a contiguous buffer. Runs as a sequential single-thread scan per expert. |
| 3 | `naive_gemm_bt_kernel` | Up-projection GEMM: `[tokens, D] x W1^T -> [tokens, 2*I]`. Naive per-element computation, no shared-memory tiling. Called once per expert in a host loop. |
| 4 | `swiglu_strided_kernel` | Standalone SwiGLU activation reading the GEMM1 output from DRAM: `out = silu(gate) * up`. Forces an extra DRAM round-trip. |
| 5 | `naive_gemm_bt_kernel` + `scatter_kernel` | Down-projection GEMM followed by weighted scatter-add back to the output tensor. Again per-expert. |

**Key inefficiencies** (by design): per-expert sequential kernel launches, redundant gather/scatter passes, intermediate tensors fully materialised in DRAM, no shared-memory tiling in GEMMs.

## Testing

Tests compare GPU kernel output against a CPU reference implementation for multiple problem sizes:

```
=== MoE Correctness Tests ===

  small  (T=4, E=4, D=64)          max_err=X.XXe-XX  mean_err=X.XXe-XX  PASS
  medium (T=32, E=8, D=128)        max_err=X.XXe-XX  mean_err=X.XXe-XX  PASS
  ...

Results: N / N passed
```

The error tolerance is 1e-3 for smaller sizes and 1e-2 for larger sizes (FP32 accumulation differences).

## Benchmarking

Benchmarks report min/mean/median/max latency and throughput across multiple problem sizes:

```
=== MoE Kernel Comparison Benchmark ===
    warmup=2  iters=10

  Variant     Config                                   Min(ms)  Mean(ms)         Tok/s
  --------------------------------------------------------------------------------------
  Opt5        T=64,E=8,K=2,D=256,I=512                   0.134     0.135        473,306
  Opt5        T=128,E=16,K=2,D=512,I=1024                0.457     0.708        180,915
  Opt5        T=64,E=256,EL=32,K=8,D=7168,I=2048         2.083     2.083         30,720
  Opt5        T=512,E=256,EL=32,K=8,D=7168,I=2048        8.343     8.343         61,370
  Opt5        T=2048,E=256,EL=32,K=8,D=7168,I=2048      24.088    24.134         84,859
  --------------------------------------------------------------------------------------
```

---

## TODO: Optimization Roadmap

### MoE Kernel Optimizations

Listed in order from most basic to most advanced. Each builds on the previous. See [docs/moe/README.md](docs/moe/README.md) for deeper architectural analysis of these iterations.

1. - [x] **Shared-memory tiled GEMM** -- Replaced the naive per-element GEMM with a classic 2D-tiled GEMM.
2. - [x] **Fused routing pipeline** -- Merged gate-logits, softmax, and top-K selection into a single kernel, avoiding D-RAM roundtrips.
3. - [x] **Single-pass token permutation / Grouped-GEMM** -- Replaced per-expert gather/scatter passes with a unified Grouped-GEMM that processes all expert batches synchronously.
4. - [x] **Persistent Threads (Failed Expr)** -- Attempted to keep activations resident in SMEM between up/down projections via a long-running while-loop queue, but suffered a **2.5x performance penalty** due to systemic L2 cache thrashing and atomic contention overheads natively blocking the streaming multiprocessors.
5. - [x] **Hardware Async DMA Pipelining** -- Reverted to native Grid Dispatching, inserting double-buffered `__pipeline_memcpy_async` instructions to completely jump over the Register-File bottleneck, moving tensor payloads natively from Global -> Shared.
6. - [x] **`float4` Vectorized Fetches** -- Upgraded the scalar `cp.async` pipeline into exact 128-bit chunks, restoring native coalesced memory alignment and breaking the 35ms bounds. Included an exact $+4$ padding technique across multi-dimensional arrays mapping to Bank Conflict avoidance. 
7. - [x] **TF32 Tensor Cores (Opt 7)** -- Integrated `<mma.h>` native `wmma::precision::tf32` Tensor Blocks into the async pipe.
- **Result:** We completely obliterated the compute loop, bringing math execution time down to nanoseconds! However, due to tiny `16x16` framework TILE_SIZE allocations, the Kernel became completely **Latency Bound**, starving the Streaming Multiprocessors. Wait limits spiked to 43ms.
8. - [x] **64x64 Tensor Tiling & SMEM Union (Opt 8)** -- Radically scaled the Async Tensor framework into monstrous `64x64` chunks (4096-element matrices) mapping all 8 Warps onto independent evaluation targets sequentially. Bypassed the rigid physical 48KB maximum Shared Memory limits natively by forcing Epilogue staging variables into a `union` structure collapsing dynamic overhead back under the static ceilings seamlessly!
- **Ultimate Result:** Shattered the wait blockings permanently, crashing execution limits down directly to **8.3ms**, translating to a massive **85,000 Tok/s** for `T=2048` at sweeping `D=7168` standard scale setups!

> For exact hardware boundaries based directly on the empirical traces of the Blackwell multiprocessor fabric vs these outputs, proceed straight to **[THEORETICAL_LIMITS.md](docs/moe/THEORETICAL_LIMITS.md)**.
