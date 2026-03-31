#pragma once

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// DeepSeek Sparse Attention (DSA / MLA) configuration
// ---------------------------------------------------------------------------

struct DsaConfig {
    int num_queries;          // Q  – number of query tokens
    int num_heads;            // H  – number of attention heads
    int head_dim_compressed;  // Dc – compressed KV dimension (e.g. 512)
    int head_dim_positional;  // Dp – positional correction dimension (e.g. 64)
    int num_selected_kv;      // S  – sparse selected KV count per query (e.g. 2048)
    int page_size;            // P  – page size in the paged KV cache (e.g. 64)
    int total_kv_tokens;      // N  – total KV tokens in the cache
};

// ---------------------------------------------------------------------------
// Kernel declarations
// ---------------------------------------------------------------------------

// Gather selected KV tokens from a paged cache into contiguous buffers.
//   kv_cache_compressed  [N, Dc]   (full paged KV cache – compressed part)
//   kv_cache_positional  [N, Dp]   (full paged KV cache – positional part)
//   v_cache              [N, Dc]   (value cache, same dim as compressed K)
//   sparse_indices       [Q, S]    (selected KV token indices per query)
//   gathered_kc          [Q, S, Dc] (output)
//   gathered_kp          [Q, S, Dp] (output)
//   gathered_v           [Q, S, Dc] (output)
void dsa_kv_gather(const float* kv_cache_compressed,
                   const float* kv_cache_positional,
                   const float* v_cache,
                   const int* sparse_indices,
                   float* gathered_kc, float* gathered_kp, float* gathered_v,
                   const DsaConfig& cfg, cudaStream_t stream = 0);

// Dot product (compressed): scores[q,h,s] += dot(q_nope[q,h,:Dc], kc[q,s,:Dc])
//   q_nope   [Q, H, Dc]
//   kc       [Q, S, Dc]
//   scores   [Q, H, S]   (output, additive)
void dsa_dot_compressed(const float* q_nope, const float* kc, float* scores,
                        const DsaConfig& cfg, cudaStream_t stream = 0);

// Dot product (positional): scores[q,h,s] += dot(q_pe[q,h,:Dp], kp[q,s,:Dp])
//   q_pe     [Q, H, Dp]
//   kp       [Q, S, Dp]
//   scores   [Q, H, S]   (output, additive)
void dsa_dot_positional(const float* q_pe, const float* kp, float* scores,
                        const DsaConfig& cfg, cudaStream_t stream = 0);

// Fused dot: computes scores += dot([q_nope|q_pe], [kc|kp]) in one kernel launch.
// This replaces two separate dot kernels and writes scores once.
void dsa_dot_fused(const float* q_nope, const float* q_pe,
                   const float* kc, const float* kp,
                   float* scores,
                   const DsaConfig& cfg, cudaStream_t stream = 0);

// Softmax over the S dimension per (q, h) pair.
//   scores   [Q, H, S]   (in-place)
void dsa_softmax(float* scores, const DsaConfig& cfg, cudaStream_t stream = 0);

// Output projection: out[q,h,d] = sum_s attn[q,h,s] * v[q,s,d]
//   attn   [Q, H, S]
//   v      [Q, S, Dc]
//   output [Q, H, Dc]
void dsa_output_proj(const float* attn, const float* v, float* output,
                     const DsaConfig& cfg, cudaStream_t stream = 0);

// ---------------------------------------------------------------------------
// DSA Forward Implementations
// ---------------------------------------------------------------------------

// 1. BASELINE: Naive kernels (gate_logits -> softmax -> topk -> naive_gemm)
void dsa_forward_naive(const float* q_nope, const float* q_pe,
                       const float* kv_cache_compressed,
                       const float* kv_cache_positional,
                       const float* v_cache,
                       const int* sparse_indices,
                       float* output,
                       const DsaConfig& cfg, cudaStream_t stream = 0);

// 2. OPT 1: Tiled Dot Products (Replace naive dot with shared-memory tiling)
void dsa_forward_opt1(const float* q_nope, const float* q_pe,
                      const float* kv_cache_compressed,
                      const float* kv_cache_positional,
                      const float* v_cache,
                      const int* sparse_indices,
                      float* output,
                      const DsaConfig& cfg, cudaStream_t stream = 0);

// 3. OPT 2: Fused Tiled Dot Products (Fuse compressed and positional into one GEMM)
void dsa_forward_opt2(const float* q_nope, const float* q_pe,
                      const float* kv_cache_compressed,
                      const float* kv_cache_positional,
                      const float* v_cache,
                      const int* sparse_indices,
                      float* output,
                      const DsaConfig& cfg, cudaStream_t stream = 0);

// 4. OPT 3: Flash Fusion (Fuse gather + dot + softmax + output calculation into one kernel)
void dsa_forward_opt3(const float* q_nope, const float* q_pe,
                      const float* kv_cache_compressed,
                      const float* kv_cache_positional,
                      const float* v_cache,
                      const int* sparse_indices,
                      float* output,
                      const DsaConfig& cfg, cudaStream_t stream = 0);

// Defaults to the best available implementation (Opt 3)
void dsa_forward(const float* q_nope, const float* q_pe,
                 const float* kv_cache_compressed,
                 const float* kv_cache_positional,
                 const float* v_cache,
                 const int* sparse_indices,
                 float* output,
                 const DsaConfig& cfg, cudaStream_t stream = 0);
