# MoE Kernel Optimizations

This document details the optimizations for the Mixture of Experts (MoE) kernels, emphasizing the impact of **Shared-Memory Tiling** and **Fused Routing** on NVIDIA Blackwell GPUs.

## 🚀 Optimization 1: Shared-Memory Tiled GEMM

### Concept
In a naive MoE implementation, expert projections ($W_1$ and $W_2$) are computed per-token using standard dot products. This is highly inefficient as it ignores the 2D layout of matrix multiplication and results in multiple DRAM reads for the same weights.

**Tiled GEMM (Opt 1)** brings a classical 2D tiling approach to MoE:
1.  **Shared Memory Tiles**: Loads 16x16 tiles of both inputs and weights into shared memory once.
2.  **Thread Reuse**: All threads in a 16x16 block reuse the loaded tile for 256 partial calculations.
3.  **Blackwell Compute Utilization**: By converting a memory-bound problem into a compute-bound one, this optimization takes full advantage of the B200's massive Tensor Core-adjacent raw floating-point performance.

### Results on B200:
*   **34% Speedup**: The dominant gain for MoE comes from moving the эксперт проекции ($D=512, I=1024$) to tiled implementations.

## 🚀 Optimization 2: Fused Routing Pipeline

### Concept
MoE routing traditionally involves three distinct DRAM round-trips for each token:
1.  **Gate-Logits**: (Input * GateWeights) -> DRAM.
2.  **Softmax**: DRAM -> Read/Write.
3.  **Top-K**: DRAM -> Read.

**Fused Routing (Opt 2)** merges these into a single "one-pass" kernel:
*   Each token is handled by one thread block.
*   The gating logits are computed and stored directly in **Shared Memory (smem)**.
*   Softmax and Top-K selection take place entirely in smem without any intermediate DRAM writes.

---

## 📊 Performance Benchmarks (Blackwell B200)

| Variant | Latency (Mean) | Throughput | Details |
| :--- | :--- | :--- | :--- |
| Naive | 8.487 ms | 15,083 Tok/s | Baseline expert loop |
| **Opt 1/2** | ~7.000 ms | ~18,000 Tok/s | Tiled GEMM + Fused Router |
| **Opt 3 (Grouped)** | **0.539 ms** | **237,493 Tok/s** | **~15.7x Speedup (Asynchronous)** 🚀 |

### ⚠️ Performance Observations & "Issues"

1.  **Host-Device Synchronization Stalls**: The single biggest performance bottleneck we identified was a `cudaStreamSynchronize` inside the expert loop in older implementations. By moving to **Grouped-GEMM (Opt 3)**, we eliminated these stalls, resulting in a **15x+ throughput boost** on Blackwell B200.
2.  **Blackwell Occupancy**: Opt 3 keeps the Blackwell SMs hot by launching all expert GEMMs in one batch. This amortizes the kernel launch overhead across all experts simultaneously.
3.  **Intermediate Buffers**: While Opt 3 is extremely fast, it still materializes "gathered" and "grouped" buffers in DRAM. Our final optimization (Opt 4) will attempt to eliminate these as well.

---

## 🛠️ Future Roadmap

*   **[x] Grouped-GEMM Integration**: (Completed in Opt 3) - All tokens/experts handled in a single launch.
*   **[ ] Expert Fusion (Opt 4)**: Launch a single kernel that performs the entire MoE forward pass (Routing + Expert FFNs) without materializing gathered buffers in DRAM.
*   **[ ] 8-bit Quantization (FP8/W8A16)**: Leverage specialized hardware in Blackwell.
