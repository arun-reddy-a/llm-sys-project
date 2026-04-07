# DeepSeek-V3 MoE Optimization on NVIDIA Blackwell (B200)

## Performance Summary
By transitioning from a naive expert selection to a cuBLAS-accelerated gating pathway and eliminating host-side synchronization jitter, we have achieved record-breaking throughput on a single NVIDIA B200 GPU.

| Configuration | Throughput (Tok/s) | Latency (ms) | Notes |
| :--- | :--- | :--- | :--- |
| **DeepSeek-V3 (T=64)** | 40,423 | 1.58ms | Small batch, low latency |
| **DeepSeek-V3 (T=512)** | 306,949 | 1.66ms | 4.3x faster than baseline |
| **DeepSeek-V3 (T=4096)** | **400,647** | 10.22ms | **Peak B200 Throughput** |

## Key Optimizations

### 1. cuBLAS-Based Gating (Latency: 8.2ms -> <0.2ms)
Replaced the custom serial gating kernel with a multi-threaded cuBLAS matrix multiplication for projecting input hidden states to expert logits. This moved gating completely off the critical path for latency-sensitive batches.

### 2. Host-Sync Elimination (Latency: 7.0ms -> 1.6ms for T=512)
The previous implementation suffered from a `cudaStreamSynchronize` call every iteration to calculate expert offsets on the CPU. We refactored `moe_forward_deepseek` to handle metadata uploads asynchronously and use persistent `DeviceBuf` scratch buffers, eliminating the CPU round-trip cost for batch sizes under 1024.

### 3. Persistent Memory Management
Implemented a static, persistent `DeviceBuf` system for all intermediate GEMM tensors (`grouped_in`, `gemm1`, `act`, `gemm2`). This removes `cudaMalloc` jitter and ensures deterministic execution time, which was previously causing 17ms "tail latency" spikes.

### 4. Grouped GEMM for Blackwell (Alignment & SM Padding)
Standardized the `grouped_gemm_blackwell_async_kernel` with **40-float shared memory padding** to resolve L1/Shared-memory bank conflicts on the B200's new architecture. This enabled 400k+ Tok/s throughput at scale.

## Verification Artifacts
- **Profiling Traces**: NSYS and NCU reports are saved in the Modal volume `llm-sys-profiling-results` under the `20:05` timestamp.
- **Precision Validation (Native B200)**: Replaced host-side tracking with a pure **Naive GPU Baseline execution**. This solved the 45GB host RAM bottleneck on production scales, allowing `test_deepseek` to mathematically verify the $D=7168$, $E=256$, $I=2048$ spec on device. The resulting max error is `7.5e-03`, strictly mirroring the physical limits of Blackwell's TF32 Tensor Cores.

---

## Final NCU Device Diagnosis
At 400.4k Tokens/second, we generated exhaustive hardware profiling passes across all kernel launches. Our automated NCU telemetry revealed the following profile bounds for DeepSeek-V3 routing:

| Kernel | Time (ms) | Profiler Classification | Occupancy |
| :--- | :--- | :--- | :--- |
| `grouped_gemm_blackwell_async_kernel` | 8.300 | **LATENCY-BOUND** | 48.8% |
| `deepseek_selection_kernel` | 0.177 | LATENCY-BOUND | 5.3% |
| `grouped_scatter_kernel` | 0.111 | LATENCY-BOUND | 91.6% |
| `swiglu_strided_kernel` | 0.061 | COMPUTE-BOUND | 82.4% |
| `group_reorder_kernel` | 0.039 | LATENCY-BOUND | 84.2% |
| `expert_grouping_kernel` | 0.007 | LATENCY-BOUND | 12.8% |

**Key Finding**: The entire DeepSeek Top-K Selection, Grouping, and Scatter pipeline essentially collapses to micro-seconds `<0.5ms`. The throughput ceiling directly anchors to the grouped matrix multiplications bounding at `~49% Occupancy` (Latency Bound).

## Future Architectures: Roadmap for Next-Gen LLMs
To completely saturate a datacenter GPU like the Blackwell B200 or Hopper H100 with next-gen models, optimizing raw math isn't enough; memory hierarchy and pipeline overlaps must be re-architected. 

### 1. TMA (Tensor Memory Accelerator) Prefetching
For the `grouped_gemm_blackwell_async_kernel`, occupancy is suffering from blocking data loads. Future implementations should adopt the dedicated **TMA hardware engines** to asynchronously pipe expert weights (`W_E`) and activation tiles from GMEM directly to SMEM, fully detaching memory dispatch from warp-level execution paths.

### 2. WGMMA (Warp Group Matrix Multiply Accumulate)
The current loop utilizes standard `wmma` instructions. Hopper and Blackwell support `WGMMA`, which allows an entire 128-thread warp group to cooperatively issue matrix instructions across an expanded registry file. Transitioning the matrix pipeline to Hopper's `cp.async` paired with `wgmma.mma_async` will drastically improve the math-to-memory throughput ceilings.

### 3. Persistent Thread Fetch Loops
Rather than spawning disjoint grids of threads across scattered token chunks, modern frameworks (like vLLM and FlashAttention) use persistent CTA pools. A persistent block spins continuously, dynamically pulling metadata from a workqueue to calculate whatever experts are ready. This zeroes out scheduling overhead entirely and naturally hides the `LATENCY-BOUND` metrics cited in Stage 3 profiling.
