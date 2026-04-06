# MoE Kernel Optimizations

This document chronicles the architectural journey, bottlenecks, and the final highly optimized implementations for the Mixture-of-Experts (MoE) forward pass. The current implementations are consolidated in [`kernels/moe/naive_moe.cu`](../../kernels/moe/naive_moe.cu).

Rather than just listing the optimized states, this README documents *why* we took the path we did, specifically dissecting the performance constraints of the NVIDIA Blackwell (B200) architecture.

## Baseline 

The baseline naive MoE implementation writes all intermediates to global memory, using separate kernel launches for each mathematical phase:
1. `gate_logits_kernel` computes logits.
2. `softmax_experts_kernel` handles probability generation.
3. `topk_kernel` does the expert selection.
4. `gather_kernel` reads non-contiguous memory, pulling tokens needed for one expert into a contiguous buffer.
5. FFN operations (GEMM1, SwiGLU, GEMM2) are dispatched repeatedly per expert via the host.

### The Problem
This architecture suffers from catastrophic launch overhead (launching $E$ individual small GEMMs sequentially on the host), serial blocking (the loop synchronizes the CPU with the GPU constantly), and brutal DRAM bandwidth consumption (streaming full `[T, E]` and intermediate activation arrays to and from HBM).

---

## Optimization 2: Fused Routing

Optimization 2 addresses the front-end of the MoE block by fusing the routing calculation into a single `fused_gate_kernel`.

1. A block cooperatively stages the token inputs into Shared Memory.
2. The threads compute exactly one expert logit each.
3. Thread 0 (or a designated warp) performs a block-level Softmax and Top-K accumulation *entirely inside registers/shared memory*. 
4. The final selection `expert_indices` and `expert_weights` are written to DRAM.

### The Win
The full `[T, E]` logit matrix no longer escapes into DRAM. A sequence of 3 un-optimized launches is condensed into one launch, drastically alleviating L2 cache thrashing on the token routing paths.

---

## Optimization 3: Grouped GEMM

Optimization 3 fixes the devastating per-expert FFN host-launch overhead by utilizing a **Grouped GEMM** paradigm (`moe_forward_opt3`).

1. **Histogram & Atomic Work:** `expert_grouping_kernel` builds an atomic histogram identifying exactly how many tokens require execution on each expert.
2. **Reordering:** `group_reorder_kernel` permutes all tokens into a contiguous chunk in DRAM.
3. **The Grouped GEMM:** Instead of looping on the CPU, we launch a single massive GEMM grid calculation for all tokens across all active experts. The kernel uses pre-computed bounds (`expert_offsets`) to ensure that each thread block correctly maps its rows to the correct subset of the massive, combined Multi-Layer Perceptron weights.

### The Win
This completely excises the repetitive host launching! The entire forward pass runs as a constant $O(1)$ sequence of massive hardware grids, rather than $O(E)$ microscopic grids. 

---

## Optimization 4: Fully Fused Expert Kernel

Optimization 4 is a monolithic architectural experiment. Instead of grouping all tokens and executing massive GEMMs, `fused_moe_kernel` allocates exactly **one block per token**.
Every single block computes the routing logits internally, retrieves its designated Top-K experts, loads exactly the matrices it requires, calculates the mathematical dot-products (GEMV) using local shared memory for the `SwiGLU` projection, and aggregates its final sum to DRAM.

### The Verdict
While highly elegant, eliminating intermediate global memory completely, **it severely limits hardware math execution (arithmetic intensity)**. Matrix-Vector multiplications (GEMV) drastically underutilize Tensor Cores compared to unified Grouped computations. Furthermore, massive shared-memory footprint demands limit block occupancy, capping performance scalability on standard models.

---

## Optimization 5: Async DMA Pipelines (`cp.async`) 

Optimization 5 details the critical findings while targeting optimal Blackwell B200 execution, addressing the latency barriers that Grouped GEMMs exhibit on large expert counts.

### The Persistent Thread Trap
Initially, we assumed that even with Grouped GEMMs, assigning blocks across massive expert grids was incurring CTA hardware scheduling overheads. We designed a **Persistent Thread Pattern**: allocating a fixed grid of 640 CTAs locking into an atomic work queue. 

**Result: 2.5x Performance Degradation.**

By tracking metrics through Nsight Compute, we uncovered that CTA launch overhead at the 35ms timescale was a ghost (costing roughly 0.02% of the runtime). More importantly: sweeping 32GB of $W_1$ and $W_2$ weights via random persistent assignments broke the hardware L2 Cache localities, hammering the Memory Controller. The bottleneck on Blackwell isn't the dispatch—it's the DRAM bandwidth. 

### The Solution: Direct Global-to-Shared DMA
Optimization 5 (`grouped_gemm_blackwell_async_kernel`) drops the Persistent Thread model and leverages true hardware asynchronous copies (`cuda::pipeline` / `cp.async`).
1. We preserve the highly performant standard hardware grid scheduling, ensuring adjacent blocks maintain proper spatial L2 cache residency.
2. We invoke `__pipeline_memcpy_async` instructions.
3. The hardware initiates a **Direct Memory Access (DMA)** request pulling weights out of Global Memory straight into Shared Memory memory-banks, completely bypassing the SM execution datapath and the Register File.

### The Alignment Trap
While async DMA guarantees we avoid the register file, our initial scalar execution resulted in a **57.3ms latency** (slower than the 35ms baseline). Because we instructed `__pipeline_memcpy_async` to fetch exactly `sizeof(float)`, we effectively forced the DMA hardware to execute isolated 4-byte loads (`cp.async.ca.shared.global.b32`). 
Our previous naive `val = A[...]` loop was automatically vectorized by the `nvcc` compiler into 128-bit fetches (`LDG.E 128`). By dropping to rigid scalar async pipelines, we shattered the memory coalescence, generating 4x the required memory transactions. 

---

## Optimization 6: Vectorized `float4` DMA Fetch

To fix the structural alignment breakdown, Optimization 6 scales the memory pipelining to exclusively utilize explicit 16-byte instructions (`cp.async.ca.shared.global.b128`).

1. The block's 256 threads form cooperative data-movement queues. Threads 0-63 process the Active Row Matrix fetch (`As`), while 64-127 map to the Weights Matrix (`Bs`).
2. We invoke `__pipeline_memcpy_async` natively casting all array offsets mathematically to guarantee 16-byte contiguous alignment by slicing the tile dynamically.
3. To permit 128-bit memory instructions without succumbing to crippling shared-memory bank conflicts on the math evaluation side, we pad the multidimensional `Bs` array (`__shared__ float Bs[2][16][16 + 4]`). This $+4$ shift mathematically displaces the matrix to ensure exactly 0 warp stride conflicts when performing the matrix multiplication loop.

### The Win
**Latency reduced to 41.9ms.** Spatial memory bandwidth is completely restored. But we are formally COMPUTE-BOUND on FP32 CUDA cores executing $83 \text{ million}$ scalar multiplications! 

---

## Optimization 7: Tensor Cores (Next Steps)
We have successfully decoupled memory execution, leaving the strict compute kernel. Moving into Opt 7, we must swap the scalar FMA thread loops explicitly into **Warp Matrix Multiply Accumulates (WMMA)** instructions using the `<mma.h>` Tensor Core API.
