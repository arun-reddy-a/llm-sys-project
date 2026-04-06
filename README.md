# Optimising MoE and Sparse Attention Layers on Blackwell GPUs

Custom CUDA kernels for Mixture-of-Experts (MoE) and DeepSeek Sparse Attention (DSA / MLA) layers, targeting NVIDIA Blackwell B200 GPUs. This repository starts with naive baseline implementations and will be iteratively optimised through kernel fusion, batching, and Blackwell-specific hardware features.

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
make test_dsa
make bench_moe
make bench_dsa

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
./build/bench_dsa <warmup> <iters>
```

## Project Structure

```
.
├── Makefile                        # Build system (bench, test, profile targets)
├── README.md
├── modal_run.py                    # Modal cloud runner (B200 GPU)
├── Proposal.pdf                    # Project proposal
├── kernels/
│   ├── moe/
│   │   ├── naive_moe.cuh           # MoE config struct + kernel declarations
│   │   └── naive_moe.cu            # MoE kernel implementations (Naive → Opt5)
│   └── dsa/
│       ├── naive_dsa.cuh           # DSA config struct + kernel declarations
│       └── naive_dsa.cu            # DSA kernel implementations
├── tests/
│   ├── test_moe.cu                 # MoE correctness tests (GPU vs CPU reference)
│   └── test_dsa.cu                 # DSA correctness tests (GPU vs CPU reference)
├── benchmarks/
│   ├── bench_moe.cu                # MoE full benchmark (all variants × configs)
│   ├── bench_moe_smoke.cu          # MoE smoke benchmark (NVTX-instrumented, for profiling)
│   ├── bench_dsa.cu                # DSA latency benchmarks
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

### Naive DSA Layer (DeepSeek Sparse Attention / MLA)

The DSA forward pass processes sparse attention in six stages:

| Stage | Kernel | Description |
|-------|--------|-------------|
| 1 | `kv_gather_*_kernel` / `v_gather_kernel` | Gathers selected KV tokens from the full paged cache into contiguous buffers as a standalone DRAM pass. Three separate launches for K_compressed, K_positional, and V. |
| 2 | `dot_compressed_kernel` | Computes primary attention scores: `score += dot(q_nope, K_c)` over the compressed dimension (Dc). |
| 3 | `dot_positional_kernel` | Computes positional correction: `score += dot(q_pe, K_p)` over the positional dimension (Dp). Separate kernel launch from step 2. |
| 4 | `scale_kernel` | Scales scores by `1/sqrt(Dc + Dp)`. |
| 5 | `softmax_kernel` | Standard two-pass softmax (max-reduce then exp-normalise) over selected KV tokens. Full attention weight matrix materialised. |
| 6 | `output_proj_kernel` | Weighted sum of V: `out = attn_weights * V`. |

**Key inefficiencies** (by design): separate KV gather DRAM pass, two separate dot-product kernel launches, full attention matrix materialisation, no online/tiled softmax, unsorted sparse indices causing non-coalesced memory access.

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
=== MoE Naive Kernel Benchmark ===
    warmup=10  iters=50

  Config                        Min(ms)  Mean(ms)  Med(ms)  Max(ms)     Tok/s
  ---------------------------  --------  --------  --------  --------  ----------
  T=16,E=4,K=2,D=64,I=128        X.XXX     X.XXX     X.XXX     X.XXX       XXXXX
  ...
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
- **Result:** We completely obliterated the compute loop, bringing math execution time down to nanoseconds! However, due to tiny `16x16` framework TILE_SIZE allocations, the Kernel is completely **Latency Bound**, starving the Streaming Multiprocessors. The true path to scale requires feeding `128x128` blocks to satiate the Blackwell DMA schedulers!

### DSA Kernel Optimizations

Listed in order from most basic to most advanced. Each builds on the previous.

1. - [x] **Shared-memory tiling for dot products** -- Tile the K/V dimension in shared memory so that each thread block reuses loaded K/V tiles across multiple query heads. Reduces global memory traffic proportional to tile reuse factor. (completed)

2. - [x] **Fuse two dot products into a single GEMM** -- Concatenate `[q_nope | q_pe]` (Dc+Dp = 576 dims) and `[K_c | K_p]` along the head dimension. Yields a single 576-dim dot product per (query, kv) pair, eliminating one kernel launch and halving score-tensor DRAM traffic.

3. **Batch across queries** -- Replace the per-query serial computation with a batched GEMM over all queries simultaneously (analogous to Grouped-GEMM in MoE). Enables SM packing and amortises launch overhead.

4. **FlashAttention-style online softmax** -- Tile over the S selected KV tokens in chunks, carrying running max/sum accumulators in registers. Never materialises the full `[Q, H, S]` attention weight matrix in DRAM. Memory footprint drops from O(S) to O(tile_size).

5. **Sort sparse indices by page** -- Sort the selected KV indices by page ID before execution so that KV memory accesses are coalesced at page granularity (page_size=64 tokens). Improves L2 cache hit rate and DRAM burst efficiency.

6. **Fuse KV gather into compute pipeline via TMA** -- Replace the standalone gather pass with hardware-managed TMA prefetches driven directly by the sparse index list. The gather and compute overlap in a pipelined fashion, hiding gather latency entirely.

7. **TMEM-based asynchronous KV tile prefetching (Blackwell SM 10.0)** -- Use Tensor Memory hardware to speculatively prefetch upcoming KV tiles while Tensor Cores compute the current tile, achieving multi-stage pipeline overlap without manual cp.async choreography.
