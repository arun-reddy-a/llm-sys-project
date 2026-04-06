# DSA Kernel Optimizations

This document summarizes the current DeepSeek Sparse Attention (DSA / MLA) optimization path implemented in
[kernels/dsa/naive_dsa.cu](/home/rrongali/llm-sys-project/kernels/dsa/naive_dsa.cu).

## Baseline

The baseline DSA pipeline materializes several intermediate tensors and launches each stage separately:

1. gather compressed keys, positional keys, and values from the KV cache
2. compute compressed-key dot products
3. compute positional-key dot products
4. scale the scores
5. apply softmax over the selected KV tokens
6. project the attention weights through the gathered values

This structure is simple but expensive because it performs three separate gather launches, two dot-product launches,
an extra score-scaling pass, and a standalone softmax over a fully materialized `[Q, H, S]` score tensor.

## Implemented Optimizations

### Optimization 1: Tiled Dot Products

The first optimization replaces the naive dot-product kernels with shared-memory tiled variants for the compressed and
positional components separately. One block is assigned per `(q, s)` pair, and the selected key slice is loaded into
shared memory once and reused across all heads. This improves reuse of gathered key data and reduces repeated global
memory accesses during score computation.

### Optimization 2: Fused Tiled Dot Products

The second optimization fuses the compressed and positional score computation into a single tiled dot-product kernel.
Instead of launching separate kernels for the two components, the kernel walks a concatenated `(D_c + D_p)` feature
space and accumulates the final score in one pass. This removes one kernel launch and reduces intermediate score
traffic between the two dot-product stages.

### Gather Optimization: Single-Pass KV/V Gather

The original implementation used three separate gather kernels: one for compressed keys, one for positional keys, and
one for values. The current implementation replaces them with a single fused gather kernel. One thread block handles
one selected `(q, s)` entry, loads the sparse KV index once, and cooperatively copies the compressed key, positional
key, and value slices into their output buffers.

This reduces launch overhead and avoids re-reading the sparse index list three times. In practice, this change helped
the large-config `Opt2` path noticeably more than the small-config path.

### Softmax Optimization: Adaptive Scaled Softmax

Score scaling is now folded into the softmax stage so the pipeline no longer launches a separate elementwise scaling
kernel. The softmax path is also adaptive:

- for smaller `S`, a simple one-thread-per-row scaled softmax is used
- for larger `S`, a block-parallel row softmax is used so threads cooperate on the max reduction, exponential pass,
  and normalization

This improves the large-`S` regime where a single-thread softmax becomes a bottleneck.

### Optimization 3: Flash-Style Fused Kernel

Optimization 3 is an alternate design that fuses gather, score computation, online softmax state, and value
accumulation into a single token/head kernel. It tiles over the selected KV sequence in chunks and keeps queries,
key/value tiles, and the output accumulator in shared memory. This avoids materializing the full attention matrix in
global memory.

### Optimization 4: Double-Buffered Flash-Style Kernel

Optimization 4 extends the flash-style design with double-buffered shared-memory staging for KV tiles. It keeps the
same overall algorithm as Optimization 3, but allocates two staging buffers so one tile can be prepared while another
is being consumed. In the current implementation, this is still software-managed double buffering rather than a true
hardware-async TMA pipeline.

## Current Performance Summary

Recent Modal B200 benchmark results show two clear trends:

- `Opt1` and `Opt2` are the best-performing paths for the larger tested DSA regime after the gather and adaptive
  softmax improvements.
- `Opt3` and `Opt4` remain slower for large `S` and large `D_c`, likely because their flash-style fusion increases
  shared-memory pressure and reduces occupancy.

Example results from the latest run:

| Config | Variant | Mean Latency | Throughput |
| :--- | :--- | :--- | :--- |
| `Q=8,H=16,Dc=256,Dp=64,S=256,N=2048` | Naive | `0.496 ms` | `16131 Q/s` |
| `Q=8,H=16,Dc=256,Dp=64,S=256,N=2048` | Opt1 | `0.614 ms` | `13021 Q/s` |
| `Q=8,H=16,Dc=256,Dp=64,S=256,N=2048` | Opt2 | `0.493 ms` | `16238 Q/s` |
| `Q=8,H=16,Dc=256,Dp=64,S=256,N=2048` | Opt3 | `0.387 ms` | `20651 Q/s` |
| `Q=8,H=16,Dc=256,Dp=64,S=256,N=2048` | Opt4 | `0.388 ms` | `20614 Q/s` |
| `Q=8,H=16,Dc=512,Dp=64,S=1024,N=8192` | Naive | `0.944 ms` | `8479 Q/s` |
| `Q=8,H=16,Dc=512,Dp=64,S=1024,N=8192` | Opt1 | `0.958 ms` | `8351 Q/s` |
| `Q=8,H=16,Dc=512,Dp=64,S=1024,N=8192` | Opt2 | `0.957 ms` | `8356 Q/s` |
| `Q=8,H=16,Dc=512,Dp=64,S=1024,N=8192` | Opt3 | `2.338 ms` | `3422 Q/s` |
| `Q=8,H=16,Dc=512,Dp=64,S=1024,N=8192` | Opt4 | `2.322 ms` | `3445 Q/s` |

The small case is still mixed: the adaptive softmax improved the large-`S` regime substantially, but `Opt1` remains
noisier on the smaller configuration. `Opt2` is currently the more stable optimized path.

## Notes

- `N` is the total KV-cache length, not the number of selected tokens. The actual attention work scales with `S`, the
  number of selected KV entries per query.
- `dsa_forward()` currently still routes to `Opt4`, which is useful as a fused design point but is not the strongest
  benchmarked path for the tested large configuration.

## Deferred Follow-Up Work

The next DSA improvements are intentionally deferred for later:

- tune `Opt1` for the small-`S` regime so it does not regress relative to `Opt2`
- revisit the flash-style kernels to reduce shared-memory footprint and improve occupancy
- explore a true TMA-based gather/compute pipeline only after the current non-TMA paths are stable
