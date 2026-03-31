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

| Variant | Latency (Mean) | Throughput | Status |
| :--- | :--- | :--- | :--- |
| Naive | 10.075 ms | 12,705 Tok/s | Baseline |
| **Opt 1 (Tiled GEMM)** | 7.545 ms | 16,964 Tok/s | **~1.34x** |
| **Opt 2 (+ Fused Routing)** | **7.516 ms** | **17,030 Tok/s** | Best |

### ⚠️ Performance Observations & "Issues"

1.  **Expert Dominance**: In current benchmarks, the expert GEMMs (FFN layers) account for over 90% of the execution time. This is why Fused Routing (Opt 2), while more elegant, provides only incremental gains (~0.5%) compared to the Tiled GEMM (Opt 1).
2.  **Sequence Length Scales**: For massive batches (e.g., $T=4096$), the routing overhead becomes more pronounced, which is where Opt 2 will show its full potential.
3.  **Kernel Launch Overhead**: Launching separate gather/scatter kernels for every expert ($E=16$) creates significant overhead. This indicates the next big win is moving to a **Grouped-GEMM** or a single fused expert kernel.

---

## 🛠️ Future Roadmap

*   **[ ] Grouped-GEMM Integration**: Batch all expert computations across all tokens into a single multi-expert GEMM launch. This is the single biggest win for multi-GPU and large batch scales.
*   **[ ] Expert Fusion**: Launch a single kernel that performs the entire MoE forward pass (Routing + Expert FFNs) without materializing gathered buffers in DRAM.
*   **[ ] 8-bit Quantization (FP8/W8A16)**: Leverage the specialized FP8 hardware in Blackwell to double throughput for expert layers.
