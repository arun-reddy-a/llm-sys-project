# Optimising MoE Layers on Blackwell GPUs

Custom CUDA kernels for Mixture-of-Experts (MoE) layers, targeting NVIDIA Blackwell B200 GPUs. This repository starts with naive baseline implementations and will be iteratively optimised through kernel fusion, batching, and Blackwell-specific hardware features.

## ☁️ Modal Cloud Execution

You can build and run this project on cloud GPUs using [Modal](https://modal.com). This is the recommended way to benchmark on the latest **Blackwell B200** GPUs.

### Standard DeepSeek-V3 Workflow

To benchmark, verify, and profile the production-standard variant:

```bash
# 1. Benchmark Throughput (Peak: ~400k Tok/s)
modal run modal_run.py --target bench_moe

# 2. Verify Correctness (Sigmoid-Bias-Grouped Routing)
modal run modal_run.py --target test_deepseek

# 3. Full Profiling Analysis (NSYS -> NCU -> Diagnosis)
modal run modal_run.py --target profile_moe_full --variant DeepSeek
```

See [docs/modal/README.md](docs/modal/README.md) for more details.

Benchmark iterations can be configured via CLI arguments:

```bash
./build/bench_moe <warmup> <iters>   # default: 10 warmup, 50 iters
```

---

## 🛠 The Iteration & Profiling Methodology

When iteratively accelerating the `moe_forward` layers from the naive baseline up to the massive `64x64` pipeline scales across 8 structural configurations, we relied entirely on hardware-level execution traces to verify bottlenecks natively.

### 1. The Target Benchmark
Operating profile traces on immense token batches (like `T=2048`) intrinsically loops the compiler and buries fundamental computational bottlenecks beneath extremely massive `.sqlite` sizes. To evaluate hardware metrics immediately, we constructed `bench_moe_smoke.cu` replicating exactly a **miniature DeepSeek-V3 Scale**:
* **Tokens (T):** `512`
* **Total Global Experts (E):** `256`
* **Local Active Experts (EL):** `32`
* **Top-K Routing (K):** `8`
* **Dimensional matrices:** `D=7168`, `I=2048`

Locking exactly into these bounds provided fixed, perfectly predictable execution thresholds where our architecture modifications logically scaled from $42.9ms$ down to $8.3ms$.

### 2. The Profiler Pipeline
Executing `modal run modal_run.py --target profile_moe_full` orchestrated our custom 3-Stage trace workflow:
1. **Nsight Systems (`nsys`):** Established Timeline limits. Discovered exactly where the kernel hung synchronously behind the host, identifying the catastrophic Host Launch overhead natively solved via Grouped-GEMMs.
2. **Nsight Compute (`ncu`):** Deep-traced NVTX tags. Investigated deep mathematical thresholds like memory pipelining waits (`__pipeline_wait_prior`), Occupancy, and Instruction-Level Parallelism (ILP).
3. **Automated Breakdown (`diagnose_moe.py`):** Automatically piped massive trace matrices exported from the NCU framework directly into a custom Python script, systematically identifying if the kernel was officially `LATENCY-BOUND`, `COMPUTE-BOUND` or suffering from L2 cache thrashings (which guided our transition fully into `cp.async` DMA allocations).

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
│   ├── test_moe.cu                 # MoE correctness tests
│   └── test_moe_deepseek.cu        # DeepSeek-V3 correctness tests
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
│   │   ├── OPTIMIZATION_REPORT.md  # DeepSeek-V3 Final Profile & Future Architectures
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

Tests bypass Host RAM bottlenecks (OOM) associated with scaling massive $7168$-dimensional matrix tracking on standard CPUs. We structurally test the optimized GPU kernels directly against a mathematical **Naive GPU Reference** implementation. 

```bash
# Run DeepSeek-V3 production tests physically on device
modal run modal_run.py --target test_deepseek
```

Because tests are executed using native B200 Tensor Cores, the max error tolerance naturally absorbs extreme TF32 precision bounds.

## Benchmarking

Benchmarks trace native wall-clock execution against the final DeepSeek-V3 engine:

```text
=== DeepSeek-V3 MoE Kernel Benchmark ===
    warmup=2  iters=10

  Variant       Config                                     Min(ms)  Mean(ms)         Tok/s
  --------------------------------------------------------------------------------------------------------------
  DeepSeek-V3   T=64,E=256,EL=32,K=8,D=7168,I=2048           1.581     1.589         40287
  DeepSeek-V3   T=128,E=256,EL=32,K=8,D=7168,I=2048          1.324     1.328         96400
  DeepSeek-V3   T=256,E=256,EL=32,K=8,D=7168,I=2048          2.077     2.080        123054
  DeepSeek-V3   T=512,E=256,EL=32,K=8,D=7168,I=2048          1.666     1.672        306251
  DeepSeek-V3   T=1024,E=256,EL=32,K=8,D=7168,I=2048         4.407     4.418        231781
  DeepSeek-V3   T=2048,E=256,EL=32,K=8,D=7168,I=2048         5.676     5.682        360436
  DeepSeek-V3   T=4096,E=256,EL=32,K=8,D=7168,I=2048        10.218    10.230        400403
  --------------------------------------------------------------------------------------------------------------
```

### DeepSeek-V3 Performance (Blackwell)
Integrated the production DeepSeek-V3 gating: Sigmoid activation + learned expert biases + grouped expert pruning (8 groups → top-4 groups → top-8 experts). Completely detaches scaling latencies natively off traditional sequential execution limits.

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
9. - [x] **DeepSeek-V3 Production "No-Aux" Routing (Opt 9)** -- Fully standardized the pipeline to the native DeepSeek-V3 architecture: Sigmoid activation + learned expert biases + iterative grouped expert pruning (8 groups → top-4 groups → top-8 experts). Completely isolates scaling math away from traditional limits.
- **Ultimate Result:** Shattered the wait blockings permanently. For identical testing frames (T=512), the kernel evaluates practically instantaneously at **1.67ms** natively. Operating on full maximal evaluation bundles ($T=4096$, $E=256$, $D=7168$, $I=2048$), the framework achieved a peak throughput of **400,403 Tok/s**, directly breaking past all latency bottlenecks scaling on B200 hardware!

> For exact hardware boundaries based directly on the empirical traces of the Blackwell multiprocessor fabric vs these outputs, proceed straight to **[THEORETICAL_LIMITS.md](docs/moe/THEORETICAL_LIMITS.md)**.

