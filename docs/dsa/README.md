# DSA Kernel Optimizations

This document details the optimizations implemented for the DeepSeek Sparse Attention (DSA) kernels, specifically the **Flash-style Fusion (Opt 3)** variant, and discusses the performance results observed on NVIDIA Blackwell GPUs.

## 🚀 Optimization 3: Flash Fusion (v2)

### Concept
Traditional attention layers (including our Opt 0-2) suffer from a massive memory bottleneck: they materialize the full attention score matrix `[Q, H, S]` in DRAM. For a sequence length $S=4096$, this matrix is ~16MB per head, which quickly exceeds the L2 cache for large batches.

**Opt 3 (Flash Fusion)** implements the FlashAttention approach by fusing the entire attention pipeline into a single kernel:
1.  **In-kernel Gather**: Fetches KV tokens directly from the paged cache.
2.  **Online Softmax**: Computes softmax values on-the-fly using a running $max$ and $sum\_exp$, avoiding the need to write scores to DRAM.
3.  **Weighted Sum Accumulation**: Simultaneously accumulates the attention output ($O = \sum P_i V_i$) in shared memory.

### Implementation Details: Cooperative Tiling
To handle the massive head dimensions ($D_c=512$, $D_p=64$) of modern models like DeepSeek, we implemented a **Cooperative Tiled** architecture:
*   **Tile Size ($S_{tile}=16$):** We process the sequence in small chunks. 16 tokens of $(K_c + K_p + V)$ require ~72KB of shared memory, which fits comfortably within the B200's 227KB per SM limit.
*   **Block Per Head:** Each thread block is responsible for one query-head pair $(q, h)$.
*   **Collaborative Loading:** All 128 threads in the block work together to fetch a KV tile into shared memory before any computation occurs. This reduces global memory transactions by a factor of 128x compared to a naive thread-local load.

---

## 📊 Performance Benchmarks (Blackwell B200)

| Config ($Dc=256, S=256$) | Throughput | Latency | Speedup |
| :--- | :--- | :--- | :--- |
| Naive | 11,699 Q/s | 0.684 ms | 1.0x |
| **Flash Fusion (Opt 3)** | **23,750 Q/s** | **0.337 ms** | **~2.03x** 🚀 |

### ⚠️ Performance Observations & "Issues"

1.  **DRAM vs. Compute on Blackwell**: On the B200, the **HBM3e bandwidth (8 TB/s)** is so high that "ping-ponging" small buffers through DRAM (like our Naive gather) is relatively cheap. This is why for smaller sequence lengths, Fusion only provides a moderate boost. Fusion's real power is realized when $S > 4096$.
2.  **Tiling Latency**: For very short sequences, the overhead of synchronization (`__syncthreads`) and loop initialization in the tiled kernel is higher than the simple non-tiled pass.
3.  **Shared Memory Limits**: Modern DSL/MLA models use $D_c=512$ or higher. This requires setting `cudaFuncAttributeMaxDynamicSharedMemorySize` in the host code, as the default limit is 48KB.
4.  **Redundant Reads**: In the current implementation, every block (head) re-fetches the same KV-tokens from DRAM. In a future iteration, we can share KV tiles across multiple heads within a single block to reduce memory traffic by another 16x.

---

## 🛠️ Future Roadmap

*   **[ ] Blackwell TMA Integration**: Use the B200's Tensor Memory Accelerator for asynchronous, non-blocking prefetching of KV tiles.
*   **[ ] Multi-Head Sharing**: Group multiple heads into a single block to amortize the cost of reading KV tokens from DRAM.
*   **[ ] Warp-Level Reductions**: Use specialized `mma` instructions or warp-shuffle reductions for even faster dot products.
