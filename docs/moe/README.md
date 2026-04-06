# MoE Kernel Optimizations

This document summarizes the implemented Mixture-of-Experts forward optimizations in this repo and describes what each version actually changes in the current CUDA code. The focus here is accuracy to the implementation in [`kernels/moe/naive_moe.cu`](/home/rrongali/llm-sys-project/kernels/moe/naive_moe.cu), not an idealized future design.

## Baseline

The baseline MoE path materializes nearly every intermediate in global memory and iterates over experts on the host:

1. `gate_logits_kernel`, `softmax_experts_kernel`, and `topk_kernel` run as separate launches.
2. `gather_kernel` scans all tokens for one expert at a time using a single thread.
3. GEMM1, SwiGLU, GEMM2, and scatter are launched once per expert.
4. The host copies each expert's token count back from device memory and synchronizes before deciding whether to launch that expert's FFN path.

That structure is intentionally simple, but it creates three major bottlenecks:

- routing writes and rereads the full `[T, E]` logit tensor from DRAM
- token movement is serialized by a per-expert gather loop
- the FFN path pays repeated host launch and synchronization overhead

## Optimization 2: Fused Routing

Optimization 2 replaces the three-stage routing pipeline with a single kernel, [`fused_gate_kernel`](/home/rrongali/llm-sys-project/kernels/moe/naive_moe.cu#L85). The launch uses one block per token, and shared memory is partitioned into two regions:

- `s_input[D]` caches the token embedding once for the whole block
- `s_logits[E]` stores the gate logits produced for that token

The main effect is that the token vector is loaded from global memory a single time per token rather than once per expert. Relative to the baseline gate-logit kernel, that removes the repeated input reads that previously scaled with `E`.

Inside the kernel:

1. Threads cooperatively load `input[t, :]` into shared memory.
2. Threads with `threadIdx.x < E` compute one expert logit each and write into `s_logits`.
3. Thread 0 performs softmax over experts directly from shared memory.
4. The same thread performs top-K selection and renormalizes the selected routing weights.

This removes two extra kernel launches and avoids materializing the full logit tensor in DRAM. The outputs that still reach global memory are only the final `expert_indices[T, K]` and `expert_weights[T, K]`.

### Consistency notes

The high-level description is mostly consistent with the code, with two caveats:

- softmax and top-K are currently done by a single thread for simplicity, so the kernel is more launch-efficient than throughput-optimal
- the current launch caps the thread count at 256, so this implementation is best matched to the small-to-moderate expert counts used by the benchmark configs in [`benchmarks/bench_moe.cu`](/home/rrongali/llm-sys-project/benchmarks/bench_moe.cu)

## Optimization 3: Grouped GEMM

Optimization 3 removes the per-expert FFN launch loop and converts the expert work into a grouped execution flow. The implementation is in [`moe_forward_opt3`](/home/rrongali/llm-sys-project/kernels/moe/naive_moe.cu#L635).

The grouped path has four phases before the FFN math:

1. Fused routing produces `expert_indices` and `expert_weights`.
2. [`expert_grouping_kernel`](/home/rrongali/llm-sys-project/kernels/moe/naive_moe.cu#L543) uses `atomicAdd` to build a histogram of token assignments per expert and records each token's local position within that expert bucket.
3. A prefix sum over expert counts computes `expert_offsets`, which define the contiguous segment belonging to each expert in the grouped buffers.
4. [`group_reorder_kernel`](/home/rrongali/llm-sys-project/kernels/moe/naive_moe.cu#L553) permutes token rows into a single grouped input matrix and records the original token id for later scatter.

Once tokens are packed contiguously by expert, each FFN stage becomes a single grouped launch:

- one grouped GEMM for `W1`
- one SwiGLU launch over the packed rows
- one grouped GEMM for `W2`
- one grouped scatter back into `[T, D]`

The grouped GEMM kernel, [`grouped_gemm_bt_kernel`](/home/rrongali/llm-sys-project/kernels/moe/naive_moe.cu#L569), determines which expert a given output row belongs to by comparing the row index against `expert_offsets`. It then selects the corresponding expert weight matrix from the concatenated `W1` or `W2` tensor and computes that row's output.

This is a substantial improvement over the baseline because the number of GPU launches no longer scales with `E`. For a fixed forward pass, the kernel sequence is constant-size even if the number of experts grows.

### Consistency notes

Most of the provided description matches the code, but two details should be stated more carefully:

- the histogram and token reordering are device-side and parallel, but the prefix sum is still performed on the CPU after copying counts back, so one host-device synchronization remains
- the implementation eliminates the per-expert host loop, not all host coordination

So a precise wording is: Optimization 3 eliminates the repeated per-expert host synchronizations and reduces the forward pass to a fixed set of launches, but it still contains one CPU-side prefix-sum step in the current version.

## Optimization 4: Fully Fused Expert Kernel

Optimization 4, implemented by [`fused_moe_kernel`](/home/rrongali/llm-sys-project/kernels/moe/naive_moe.cu#L716), moves the entire token forward path into a single kernel. One block is assigned to one token, and shared memory holds the token-local working set:

- input vector `s_x[D]`
- routing logits `s_logits[E]`
- intermediate activation storage `s_act[2 * I]`
- output accumulator `s_y[D]`

Within a block, the kernel performs:

1. cooperative load of the token input
2. routing logits, softmax, and top-K selection
3. per-selected-expert GEMV for `W1`
4. in-kernel SwiGLU
5. per-selected-expert GEMV for `W2`
6. weighted accumulation into the token output

The main benefit is that routing, expert activation, and accumulation are all kept on-chip for the lifetime of the token. This avoids allocating large intermediate global-memory buffers such as grouped token matrices or GEMM outputs.

The tradeoff is compute efficiency. Unlike Optimization 3, which converts expert work into larger GEMMs, this fused kernel performs token-centric GEMVs. That reduces arithmetic intensity and gives the hardware less opportunity to amortize weight loads across multiple tokens. Shared-memory usage also grows with `D + E + 2I + D`, which limits scalability to larger hidden sizes and intermediate sizes.

This makes Optimization 4 a good conceptual "fully fused" design point, but not necessarily the fastest implementation for moderate or large batches.

## Optimization 5: Double-Buffered Grouped GEMM

Optimization 5 keeps the grouped execution structure of Optimization 3 and swaps in a pipelined grouped GEMM kernel, [`grouped_gemm_blackwell_async_kernel`](/home/rrongali/llm-sys-project/kernels/moe/naive_moe.cu#L839), for both `W1` and `W2`.

The kernel allocates:

- one shared-memory tile buffer for activations, `As`
- two shared-memory tile buffers for weights, `Bs[2]`

The intended schedule is:

1. preload tile 0 of the expert weight matrix into buffer 0
2. while computing tile `t` from buffer `t % 2`, stage tile `t + 1` into the alternate buffer
3. alternate buffers across iterations so the next tile is ready when the loop advances

This is the standard software structure for double buffering, and it is the right control flow if we later replace the manual shared-memory fills with true asynchronous copies.

### Consistency notes

The current implementation is not yet a real asynchronous Blackwell TMA pipeline:

- it uses ordinary shared-memory writes, not `cp.async`, TMA, or hardware-managed async transactions
- synchronization is still done with block-wide `__syncthreads()`
- the overlap is therefore structural rather than fully asynchronous

So the strongest accurate claim is:

"Optimization 5 introduces a double-buffered grouped GEMM kernel whose control flow is designed to overlap future tile staging with current tile computation. In the current code this is a software-managed precursor to true async copy/TMA support, not a finished hardware-async implementation."

That also means performance wins from Opt 5 should be described as coming from better pipelining structure and cache behavior, not from guaranteed Blackwell-only async transfer machinery.

### TMA Attempt

We also attempted a true TMA-based version of Optimization 5 using CUDA tensor-map descriptors (`CUtensorMap`), host-side `cuTensorMapEncodeTiled`, and device-side `cp_async_bulk_tensor_*` wrappers with block-scoped barriers. The prototype compiled locally with CUDA 12.9 and was structured around a rank-3 tensor map for the concatenated expert-weight tensor. However, when tested on the remote B200 setup used by [`modal_run.py`](/home/rrongali/llm-sys-project/modal_run.py), the kernel did not reach a stable passing state: the TMA path either hung during execution or required a barrier/transaction protocol that was not yet correct.

For that reason, the experimental TMA code was reverted and the repo currently keeps the earlier software-managed double-buffered implementation. In other words, Opt 5 in the checked-in code should still be understood as a non-TMA, non-hardware-async pipeline. A future TMA version is still plausible, but it likely needs to be developed first as a small standalone tile-copy test before being reintegrated into the grouped GEMM kernel.

## Suggested Wording

If you want a tighter version for the report, this would be consistent with the code:

### Optimization 2: Fused Routing

The routing stage is fused into a single kernel with one thread block per token. The token embedding is loaded cooperatively into shared memory, and each thread computes one expert logit using the cached input. Logits remain on-chip in shared memory, where a single thread performs softmax and top-K selection before writing only the selected expert ids and normalized weights to global memory. Compared with the baseline three-kernel routing pipeline, this removes the intermediate `[T, E]` logit tensor, reduces token input reloads during gating, and cuts kernel launch overhead.

### Optimization 3: Grouped GEMM

The per-expert host loop is replaced by a grouped execution flow. A device-side histogram assigns routed tokens to experts using atomic operations, and a prefix sum computes contiguous offsets for each expert bucket. Tokens are then permuted into grouped buffers so that all rows for a given expert are contiguous. This enables each FFN stage to run as a single grouped GEMM across all active experts instead of launching separate GEMMs per expert. The launch structure becomes constant with respect to the number of experts, although the current implementation still performs the prefix sum on the CPU and therefore retains one host-device synchronization point.

### Optimization 4: Fully Fused Expert Kernel

The full MoE forward pass is fused into a single token-parallel kernel. Each block processes one token and uses shared memory to hold the token input, routing logits, intermediate activations, and output accumulator. Routing, top-K selection, expert matvecs, SwiGLU activation, and weighted accumulation all occur within the same kernel, eliminating large intermediate global-memory buffers. The main drawback is that the expert computation is expressed as GEMV rather than GEMM, which lowers arithmetic intensity and makes the approach more sensitive to shared-memory capacity. As a result, this design is most attractive for smaller batch sizes or as a demonstration of maximal fusion.

### Optimization 5: Double-Buffered Grouped GEMM

The grouped GEMM path is further refined with double-buffered shared-memory tiles for expert weights. Two buffers are used so that the next weight tile can be staged while the current tile is being consumed, creating a pipelined execution structure across the reduction dimension. In the current implementation this is done with software-managed shared-memory buffering rather than true asynchronous copy instructions, but the kernel structure is intentionally aligned with a future TMA or `cp.async` implementation. The optimization preserves the grouped-GEMM execution model of Optimization 3 while improving the memory/computation pipeline within each GEMM kernel.
