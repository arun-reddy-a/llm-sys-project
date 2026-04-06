#include "naive_dsa.cuh"
#include "../../utils/cuda_utils.cuh"
#include <cfloat>
#include <cstdio>
#include <cmath>

#define DOT_TILE 128
#define MAX_HEADS_PER_THREAD 8
// ===================================================================
// Kernel: fused KV/V gather
//   One block gathers one selected KV entry for one query. Threads
//   cooperatively copy the compressed key, positional key, and value
//   slices after loading the sparse index once.
// ===================================================================
__global__ void kv_gather_fused_kernel(const float* __restrict__ kv_cache_compressed,
                                       const float* __restrict__ kv_cache_positional,
                                       const float* __restrict__ v_cache,
                                       const int* __restrict__ indices,
                                       float* __restrict__ out_kc,
                                       float* __restrict__ out_kp,
                                       float* __restrict__ out_v,
                                       int Q, int S, int Dc, int Dp) {
    int qs = blockIdx.x;
    if (qs >= Q * S) return;

    int q = qs / S;
    int s = qs % S;
    int tid = threadIdx.x;

    int kv_idx = indices[q * S + s];
    int kc_src = kv_idx * Dc;
    int kp_src = kv_idx * Dp;
    int out_kc_base = qs * Dc;
    int out_kp_base = qs * Dp;

    for (int d = tid; d < Dc; d += blockDim.x) {
        out_kc[out_kc_base + d] = kv_cache_compressed[kc_src + d];
        out_v[out_kc_base + d] = v_cache[kc_src + d];
    }
    for (int d = tid; d < Dp; d += blockDim.x) {
        out_kp[out_kp_base + d] = kv_cache_positional[kp_src + d];
    }
}

// ===================================================================
// Kernel: dot product – compressed (naive, one thread per (q,h,s))
// ===================================================================
__global__ void dot_compressed_kernel(const float* q_nope, const float* kc,
                                      float* scores,
                                      int Q, int H, int S, int Dc) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Q * H * S;
    if (tid >= total) return;

    int s = tid % S;
    int h = (tid / S) % H;
    int q = tid / (S * H);

    float sum = 0.0f;
    for (int d = 0; d < Dc; d++) {
        sum += q_nope[q * H * Dc + h * Dc + d] * kc[q * S * Dc + s * Dc + d];
    }
    scores[q * H * S + h * S + s] += sum;
}

// ===================================================================
// Kernel: tiled dot product – compressed (shared-memory over Dc)
//   One block per (q, s).  Tile loop is outermost so kc tiles are
//   loaded once into shared memory and reused across ALL heads.
//   __syncthreads lives only in the tile loop (not the head loop),
//   so every thread always hits the barrier.
// ===================================================================
__global__ void dot_compressed_tiled_kernel(const float* __restrict__ q_nope,
                                            const float* __restrict__ kc,
                                            float* scores,
                                            int Q, int H, int S, int Dc) {
    int bs = blockIdx.x;
    int q = bs / S;
    int s = bs % S;
    if (q >= Q) return;

    extern __shared__ float s_tile[];
    int tid = threadIdx.x;
    int kc_base = (q * S + s) * Dc;

    int nh = (H + (int)blockDim.x - 1) / (int)blockDim.x;
    if (nh > MAX_HEADS_PER_THREAD) nh = MAX_HEADS_PER_THREAD;
    float sums[MAX_HEADS_PER_THREAD];
    for (int i = 0; i < nh; i++) sums[i] = 0.0f;

    for (int d0 = 0; d0 < Dc; d0 += DOT_TILE) {
        int tsize = Dc - d0;
        if (tsize > DOT_TILE) tsize = DOT_TILE;

        for (int t = tid; t < tsize; t += blockDim.x)
            s_tile[t] = kc[kc_base + d0 + t];
        __syncthreads();

        for (int hi = 0; hi < nh; hi++) {
            int h = tid + hi * (int)blockDim.x;
            if (h < H) {
                int qb = (q * H + h) * Dc + d0;
                for (int t = 0; t < tsize; ++t)
                    sums[hi] += q_nope[qb + t] * s_tile[t];
            }
        }
        __syncthreads();
    }

    for (int hi = 0; hi < nh; hi++) {
        int h = tid + hi * (int)blockDim.x;
        if (h < H)
            scores[q * H * S + h * S + s] += sums[hi];
    }
}

// ===================================================================
// Kernel: dot product – positional (naive, one thread per (q,h,s))
// ===================================================================
__global__ void dot_positional_kernel(const float* q_pe, const float* kp,
                                      float* scores,
                                      int Q, int H, int S, int Dp) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Q * H * S;
    if (tid >= total) return;

    int s = tid % S;
    int h = (tid / S) % H;
    int q = tid / (S * H);

    float sum = 0.0f;
    for (int d = 0; d < Dp; d++) {
        sum += q_pe[q * H * Dp + h * Dp + d] * kp[q * S * Dp + s * Dp + d];
    }
    scores[q * H * S + h * S + s] += sum;
}

// ===================================================================
// Kernel: tiled dot product – positional (same structure as compressed)
// ===================================================================
__global__ void dot_positional_tiled_kernel(const float* __restrict__ q_pe,
                                            const float* __restrict__ kp,
                                            float* scores,
                                            int Q, int H, int S, int Dp) {
    int bs = blockIdx.x;
    int q = bs / S;
    int s = bs % S;
    if (q >= Q) return;

    extern __shared__ float s_tile[];
    int tid = threadIdx.x;
    int kp_base = (q * S + s) * Dp;

    int nh = (H + (int)blockDim.x - 1) / (int)blockDim.x;
    if (nh > MAX_HEADS_PER_THREAD) nh = MAX_HEADS_PER_THREAD;
    float sums[MAX_HEADS_PER_THREAD];
    for (int i = 0; i < nh; i++) sums[i] = 0.0f;

    for (int d0 = 0; d0 < Dp; d0 += DOT_TILE) {
        int tsize = Dp - d0;
        if (tsize > DOT_TILE) tsize = DOT_TILE;

        for (int t = tid; t < tsize; t += blockDim.x)
            s_tile[t] = kp[kp_base + d0 + t];
        __syncthreads();

        for (int hi = 0; hi < nh; hi++) {
            int h = tid + hi * (int)blockDim.x;
            if (h < H) {
                int qb = (q * H + h) * Dp + d0;
                for (int t = 0; t < tsize; ++t)
                    sums[hi] += q_pe[qb + t] * s_tile[t];
            }
        }
        __syncthreads();
    }

    for (int hi = 0; hi < nh; hi++) {
        int h = tid + hi * (int)blockDim.x;
        if (h < H)
            scores[q * H * S + h * S + s] += sums[hi];
    }
}

// ===================================================================
// Kernel: fused tiled dot product – single concatenated (Dc+Dp) GEMM
//   Treats the compressed and positional dimensions as one contiguous
//   vector of length (Dc+Dp).  A single tiling loop loads tiles from
//   kc (for d < Dc) or kp (for d >= Dc) and dots with the matching
//   query component.  One kernel launch, one score write per (q,h,s).
// ===================================================================
__global__ void dot_fused_tiled_kernel(const float* __restrict__ q_nope,
                                       const float* __restrict__ q_pe,
                                       const float* __restrict__ kc,
                                       const float* __restrict__ kp,
                                       float* scores,
                                       int Q, int H, int S, int Dc, int Dp) {
    int bs = blockIdx.x;
    int q = bs / S;
    int s = bs % S;
    if (q >= Q) return;

    extern __shared__ float s_tile[];
    int tid = threadIdx.x;
    int qs = q * S + s;
    int total_dim = Dc + Dp;

    int nh = (H + (int)blockDim.x - 1) / (int)blockDim.x;
    if (nh > MAX_HEADS_PER_THREAD) nh = MAX_HEADS_PER_THREAD;
    float sums[MAX_HEADS_PER_THREAD];
    for (int i = 0; i < nh; i++) sums[i] = 0.0f;

    for (int d0 = 0; d0 < total_dim; d0 += DOT_TILE) {
        int tsize = total_dim - d0;
        if (tsize > DOT_TILE) tsize = DOT_TILE;

        for (int t = tid; t < tsize; t += blockDim.x) {
            int d = d0 + t;
            s_tile[t] = (d < Dc) ? kc[qs * Dc + d]
                                 : kp[qs * Dp + (d - Dc)];
        }
        __syncthreads();

        bool all_compressed = (d0 + tsize <= Dc);
        bool all_positional = (d0 >= Dc);

        for (int hi = 0; hi < nh; hi++) {
            int h = tid + hi * (int)blockDim.x;
            if (h < H) {
                if (all_compressed) {
                    int qb = (q * H + h) * Dc + d0;
                    for (int t = 0; t < tsize; ++t)
                        sums[hi] += q_nope[qb + t] * s_tile[t];
                } else if (all_positional) {
                    int qb = (q * H + h) * Dp + (d0 - Dc);
                    for (int t = 0; t < tsize; ++t)
                        sums[hi] += q_pe[qb + t] * s_tile[t];
                } else {
                    for (int t = 0; t < tsize; ++t) {
                        int d = d0 + t;
                        float qv = (d < Dc)
                            ? q_nope[(q * H + h) * Dc + d]
                            : q_pe[(q * H + h) * Dp + (d - Dc)];
                        sums[hi] += qv * s_tile[t];
                    }
                }
            }
        }
        __syncthreads();
    }

    for (int hi = 0; hi < nh; hi++) {
        int h = tid + hi * (int)blockDim.x;
        if (h < H)
            scores[q * H * S + h * S + s] += sums[hi];
    }
}

// ===================================================================
// Kernel: scaled softmax over S per (q, h) – one thread per (q, h)
// ===================================================================
__global__ void scaled_softmax_kernel(float* scores, float scale, int Q, int H, int S) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= Q * H) return;

    float* row = scores + tid * S;

    float mx = -FLT_MAX;
    for (int s = 0; s < S; s++) {
        float scaled = row[s] * scale;
        row[s] = scaled;
        mx = fmaxf(mx, scaled);
    }

    float sum = 0.0f;
    for (int s = 0; s < S; s++) {
        row[s] = expf(row[s] - mx);
        sum += row[s];
    }
    float inv = 1.0f / sum;
    for (int s = 0; s < S; s++) row[s] *= inv;
}

// ===================================================================
// Kernel: block-parallel scaled softmax over S per (q, h)
//   One block owns one row. Threads cooperatively compute the max,
//   exponentials, sum, and normalization for long rows.
// ===================================================================
__global__ void scaled_softmax_block_kernel(float* scores, float scale, int rows, int S) {
    int row_idx = blockIdx.x;
    if (row_idx >= rows) return;

    float* row = scores + row_idx * S;
    int tid = threadIdx.x;

    __shared__ float red[256];

    float thread_max = -FLT_MAX;
    for (int s = tid; s < S; s += blockDim.x) {
        float scaled = row[s] * scale;
        row[s] = scaled;
        thread_max = fmaxf(thread_max, scaled);
    }
    red[tid] = thread_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) red[tid] = fmaxf(red[tid], red[tid + stride]);
        __syncthreads();
    }
    float mx = red[0];

    float thread_sum = 0.0f;
    for (int s = tid; s < S; s += blockDim.x) {
        float ex = expf(row[s] - mx);
        row[s] = ex;
        thread_sum += ex;
    }
    red[tid] = thread_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) red[tid] += red[tid + stride];
        __syncthreads();
    }
    float inv_sum = 1.0f / red[0];

    for (int s = tid; s < S; s += blockDim.x) row[s] *= inv_sum;
}

// ===================================================================
// Kernel: output projection
//   out[q, h, d] = sum_s attn[q,h,s] * v[q,s,d]
// ===================================================================
__global__ void output_proj_kernel(const float* attn, const float* v,
                                   float* output,
                                   int Q, int H, int S, int Dc) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Q * H * Dc;
    if (tid >= total) return;

    int d = tid % Dc;
    int h = (tid / Dc) % H;
    int q = tid / (Dc * H);

    float sum = 0.0f;
    for (int s = 0; s < S; s++) {
        sum += attn[q * H * S + h * S + s] * v[q * S * Dc + s * Dc + d];
    }
    output[tid] = sum;
}

// ===================================================================
// Host wrappers
// ===================================================================

static int dot_threads(int H) {
    int t = (H < DOT_TILE) ? DOT_TILE : ((H + 31) / 32) * 32;
    if (t > 256) t = 256;
    return t;
}

void dsa_kv_gather(const float* kv_cache_compressed,
                   const float* kv_cache_positional,
                   const float* v_cache,
                   const int* sparse_indices,
                   float* gathered_kc, float* gathered_kp, float* gathered_v,
                   const DsaConfig& cfg, cudaStream_t stream) {
    int Q = cfg.num_queries, S = cfg.num_selected_kv;
    int Dc = cfg.head_dim_compressed, Dp = cfg.head_dim_positional;
    int threads = 256;
    int blocks = Q * S;
    kv_gather_fused_kernel<<<blocks, threads, 0, stream>>>(
        kv_cache_compressed, kv_cache_positional, v_cache, sparse_indices,
        gathered_kc, gathered_kp, gathered_v, Q, S, Dc, Dp);
    CUDA_CHECK(cudaGetLastError());
}

void dsa_dot_compressed(const float* q_nope, const float* kc, float* scores,
                        const DsaConfig& cfg, cudaStream_t stream) {
    int Q = cfg.num_queries;
    int S = cfg.num_selected_kv;
    int H = cfg.num_heads;
    int blocks = Q * S;
    int threads = dot_threads(H);
    size_t smem = DOT_TILE * sizeof(float);
    dot_compressed_tiled_kernel<<<blocks, threads, smem, stream>>>(
        q_nope, kc, scores, Q, H, S, cfg.head_dim_compressed);
    CUDA_CHECK(cudaGetLastError());
}

void dsa_dot_positional(const float* q_pe, const float* kp, float* scores,
                        const DsaConfig& cfg, cudaStream_t stream) {
    int Q = cfg.num_queries;
    int S = cfg.num_selected_kv;
    int H = cfg.num_heads;
    int blocks = Q * S;
    int threads = dot_threads(H);
    size_t smem = DOT_TILE * sizeof(float);
    dot_positional_tiled_kernel<<<blocks, threads, smem, stream>>>(
        q_pe, kp, scores, Q, H, S, cfg.head_dim_positional);
    CUDA_CHECK(cudaGetLastError());
}

void dsa_dot_fused(const float* q_nope, const float* q_pe,
                           const float* kc, const float* kp,
                           float* scores,
                           const DsaConfig& cfg, cudaStream_t stream) {
    int Q = cfg.num_queries;
    int S = cfg.num_selected_kv;
    int H = cfg.num_heads;
    int blocks = Q * S;
    int threads = dot_threads(H);
    size_t smem = DOT_TILE * sizeof(float);
    dot_fused_tiled_kernel<<<blocks, threads, smem, stream>>>(
        q_nope, q_pe, kc, kp, scores, Q, H, S,
        cfg.head_dim_compressed, cfg.head_dim_positional);
    CUDA_CHECK(cudaGetLastError());
}

void dsa_scaled_softmax(float* scores, float scale,
                        const DsaConfig& cfg, cudaStream_t stream) {
    int total = cfg.num_queries * cfg.num_heads;
    if (cfg.num_selected_kv >= 512) {
        int threads = 256;
        scaled_softmax_block_kernel<<<total, threads, 0, stream>>>(
            scores, scale, total, cfg.num_selected_kv);
    } else {
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        scaled_softmax_kernel<<<blocks, threads, 0, stream>>>(
            scores, scale, cfg.num_queries, cfg.num_heads, cfg.num_selected_kv);
    }
    CUDA_CHECK(cudaGetLastError());
}

void dsa_output_proj(const float* attn, const float* v, float* output,
                     const DsaConfig& cfg, cudaStream_t stream) {
    int total = cfg.num_queries * cfg.num_heads * cfg.head_dim_compressed;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    output_proj_kernel<<<blocks, threads, 0, stream>>>(
        attn, v, output,
        cfg.num_queries, cfg.num_heads, cfg.num_selected_kv,
        cfg.head_dim_compressed);
    CUDA_CHECK(cudaGetLastError());
}

// 1. BASELINE: Naive Implementation
void dsa_forward_naive(const float* q_nope, const float* q_pe,
                       const float* kv_cache_compressed,
                       const float* kv_cache_positional,
                       const float* v_cache,
                       const int* sparse_indices,
                       float* output,
                       const DsaConfig& cfg, cudaStream_t stream) {
    int Q = cfg.num_queries, S = cfg.num_selected_kv, H = cfg.num_heads;
    int Dc = cfg.head_dim_compressed, Dp = cfg.head_dim_positional;

    DeviceBuf<float> kc(Q * S * Dc), kp(Q * S * Dp), v(Q * S * Dc);
    dsa_kv_gather(kv_cache_compressed, kv_cache_positional, v_cache, sparse_indices, kc.ptr, kp.ptr, v.ptr, cfg, stream);

    DeviceBuf<float> scores(Q * H * S);
    scores.zero();

    // Naive dot (non-tiled)
    dot_compressed_kernel<<<(Q*H*S+255)/256, 256, 0, stream>>>(q_nope, kc.ptr, scores.ptr, Q, H, S, Dc);
    dot_positional_kernel<<<(Q*H*S+255)/256, 256, 0, stream>>>(q_pe, kp.ptr, scores.ptr, Q, H, S, Dp);

    float sc = 1.0f / sqrtf((float)(Dc + Dp));
    dsa_scaled_softmax(scores.ptr, sc, cfg, stream);
    dsa_output_proj(scores.ptr, v.ptr, output, cfg, stream);
}

// 2. OPT 1: Tiled Dot Products
void dsa_forward_opt1(const float* q_nope, const float* q_pe,
                      const float* kv_cache_compressed,
                      const float* kv_cache_positional,
                      const float* v_cache,
                      const int* sparse_indices,
                      float* output,
                      const DsaConfig& cfg, cudaStream_t stream) {
    int Q = cfg.num_queries, S = cfg.num_selected_kv, H = cfg.num_heads;
    int Dc = cfg.head_dim_compressed, Dp = cfg.head_dim_positional;

    DeviceBuf<float> kc(Q * S * Dc), kp(Q * S * Dp), v(Q * S * Dc);
    dsa_kv_gather(kv_cache_compressed, kv_cache_positional, v_cache, sparse_indices, kc.ptr, kp.ptr, v.ptr, cfg, stream);

    DeviceBuf<float> scores(Q * H * S);
    scores.zero();

    // Separate tiled dots
    dsa_dot_compressed(q_nope, kc.ptr, scores.ptr, cfg, stream);
    dsa_dot_positional(q_pe, kp.ptr, scores.ptr, cfg, stream);

    float sc = 1.0f / sqrtf((float)(Dc + Dp));
    dsa_scaled_softmax(scores.ptr, sc, cfg, stream);
    dsa_output_proj(scores.ptr, v.ptr, output, cfg, stream);
}

// 3. OPT 2: Fused Tiled Dot Products
void dsa_forward_opt2(const float* q_nope, const float* q_pe,
                      const float* kv_cache_compressed,
                      const float* kv_cache_positional,
                      const float* v_cache,
                      const int* sparse_indices,
                      float* output,
                      const DsaConfig& cfg, cudaStream_t stream) {
    int Q = cfg.num_queries, S = cfg.num_selected_kv, H = cfg.num_heads;
    int Dc = cfg.head_dim_compressed, Dp = cfg.head_dim_positional;

    DeviceBuf<float> kc(Q * S * Dc), kp(Q * S * Dp), v(Q * S * Dc);
    dsa_kv_gather(kv_cache_compressed, kv_cache_positional, v_cache, sparse_indices, kc.ptr, kp.ptr, v.ptr, cfg, stream);

    DeviceBuf<float> scores(Q * H * S);
    scores.zero();

    // Fused tiled dot
    dsa_dot_fused(q_nope, q_pe, kc.ptr, kp.ptr, scores.ptr, cfg, stream);

    float sc = 1.0f / sqrtf((float)(Dc + Dp));
    dsa_scaled_softmax(scores.ptr, sc, cfg, stream);
    dsa_output_proj(scores.ptr, v.ptr, output, cfg, stream);
}

#define FLASH_S_TILE 16

// ===================================================================
// FLASH OPTIMIZATION v2: Tiled Sequence-Length + Cooperative Load
// ===================================================================
__global__ void dsa_flash_fused_kernel(const float* __restrict__ q_nope,
                                       const float* __restrict__ q_pe,
                                       const float* __restrict__ kc_cache,
                                       const float* __restrict__ kp_cache,
                                       const float* __restrict__ v_cache,
                                       const int*   __restrict__ sparse_indices,
                                       float*       __restrict__ output,
                                       int Q, int H, int S, int Dc, int Dp, float scale) {
    int q = blockIdx.x / H;
    int h = blockIdx.x % H;
    if (q >= Q) return;

    int tid = threadIdx.x;
    int bdim = blockDim.x;

    // Shared memory layout:
    // [Dc + Dp]        - Query (s_q)
    // [S_TILE][Dc]     - Key Compressed (s_kc)
    // [S_TILE][Dp]     - Key Positional (s_kp)
    // [S_TILE][Dc]     - Values (s_v)
    // [Dc]             - Output Accumulator (s_acc)
    
    extern __shared__ float smem[];
    float* s_q  = smem;
    float* s_kc = s_q  + (Dc + Dp);
    float* s_kp = s_kc + (FLASH_S_TILE * Dc);
    float* s_v  = s_kp + (FLASH_S_TILE * Dp);
    float* s_acc = s_v + (FLASH_S_TILE * Dc);

    // 1. Cooperative Load: Head's Query
    for (int d = tid; d < Dc; d += bdim)
        s_q[d] = q_nope[(q * H + h) * Dc + d];
    for (int d = tid; d < Dp; d += bdim)
        s_q[Dc + d] = q_pe[(q * H + h) * Dp + d];
    
    // Initialize Accumulator
    for (int d = tid; d < Dc; d += bdim)
        s_acc[d] = 0.0f;
    __syncthreads();

    // Online softmax state
    float m_prev = -FLT_MAX;
    float l_prev = 0.0f;

    // 2. Loop over Sequence-Length in tiles
    for (int s0 = 0; s0 < S; s0 += FLASH_S_TILE) {
        int tsize = S - s0;
        if (tsize > FLASH_S_TILE) tsize = FLASH_S_TILE;

        // Cooperative Load Tile: KC, KP, V
        for (int t = 0; t < tsize; t++) {
            int kv_idx = sparse_indices[q * S + s0 + t];
            for (int d = tid; d < Dc; d += bdim)
                s_kc[t * Dc + d] = kc_cache[kv_idx * Dc + d];
            for (int d = tid; d < Dp; d += bdim)
                s_kp[t * Dp + d] = kp_cache[kv_idx * Dp + d];
            for (int d = tid; d < Dc; d += bdim)
                s_v[t * Dc + d] = v_cache[kv_idx * Dc + d];
        }
        __syncthreads();

        // 3. Process each token in the tile
        for (int t = 0; t < tsize; t++) {
            // Compute dot(Q, K) for this token
            float score = 0.0f;
            for (int d = tid; d < Dc; d += bdim)
                score += s_q[d] * s_kc[t * Dc + d];
            for (int d = tid; d < Dp; d += bdim)
                score += s_q[Dc + d] * s_kp[t * Dp + d];

            // Reduction across block
            for (int offset = bdim / 2; offset > 0; offset /= 2)
                score += __shfl_down_sync(0xffffffff, score, offset);
            score = __shfl_sync(0xffffffff, score, 0); 
            score *= scale;

            // Online Softmax step
            float m_curr = fmaxf(m_prev, score);
            float p = expf(score - m_curr);
            float alpha = expf(m_prev - m_curr);
            float l_curr = alpha * l_prev + p;

            // Accumulate V: O = O * alpha + P * V
            for (int d = tid; d < Dc; d += bdim) {
                s_acc[d] = s_acc[d] * alpha + p * s_v[t * Dc + d];
            }

            m_prev = m_curr;
            l_prev = l_curr;
        }
        __syncthreads(); // Wait for tile done before next load
    }

    // Final Normalize and Write
    float inv_l = 1.0f / l_prev;
    for (int d = tid; d < Dc; d += bdim) {
        output[(q * H + h) * Dc + d] = s_acc[d] * inv_l;
    }
}

// 4. OPT 3: Flash Implementation
void dsa_forward_opt3(const float* q_nope, const float* q_pe,
                      const float* kv_cache_compressed,
                      const float* kv_cache_positional,
                      const float* v_cache,
                      const int* sparse_indices,
                      float* output,
                      const DsaConfig& cfg, cudaStream_t stream) {
    int Q = cfg.num_queries, H = cfg.num_heads, S = cfg.num_selected_kv;
    int Dc = cfg.head_dim_compressed, Dp = cfg.head_dim_positional;
    
    int threads = 128; 
    // Total Smem: (Dc+Dp) + (S_TILE * (Dc+Dp+Dc)) + Dc
    // 576 + 16 * (512+64+512) + 512 = 576 + 17408 + 512 = 18496 floats = ~72 KB
    size_t smem = ( (Dc + Dp) + (FLASH_S_TILE * (Dc + Dp + Dc)) + Dc ) * sizeof(float);
    float scale = 1.0f / sqrtf((float)(Dc + Dp));

    // For larger Dc, we might need more shared memory (up to device limit).
    // Shared memory > 48KB requires explicit attribute setting.
    // For larger Dc, we might need more shared memory (up to device limit).
    // Shared memory > 48KB requires explicit attribute setting.
    static int max_smem_set = -1;
    if ((int)smem > max_smem_set) {
        CUDA_CHECK(cudaFuncSetAttribute(
            (const void*)dsa_flash_fused_kernel, 
            cudaFuncAttributeMaxDynamicSharedMemorySize, 
            (int)smem));
        max_smem_set = (int)smem;
    }

    dsa_flash_fused_kernel<<<Q * H, threads, smem, stream>>>(
        q_nope, q_pe, kv_cache_compressed, kv_cache_positional, v_cache,
        sparse_indices, output, Q, H, S, Dc, Dp, scale
    );
    CUDA_CHECK(cudaGetLastError());
}

// ===================================================================
// BLACKWELL SPECIFIC: Double-Buffered Flash Attention
// Overlaps KV-cache pre-fetching for tile N+1 with score computation of tile N.
// ===================================================================

__global__ void dsa_blackwell_pipelined_fused_kernel(const float* __restrict__ q_nope,
                                                    const float* __restrict__ q_pe,
                                                    const float* __restrict__ kc_cache,
                                                    const float* __restrict__ kp_cache,
                                                    const float* __restrict__ v_cache,
                                                    const int*   __restrict__ sparse_indices,
                                                    float*       __restrict__ output,
                                                    int Q, int H, int S, int Dc, int Dp, float scale) {
    int q = blockIdx.x / H;
    int h = blockIdx.x % H;
    if (q >= Q) return;

    int tid = threadIdx.x;
    int bdim = blockDim.x;

    // Shared memory layout (Double-buffered KC, KP, V)
    extern __shared__ float smem[];
    float* s_q  = smem;
    float* s_kc = s_q  + (Dc + Dp); // [2][S_TILE][Dc]
    float* s_kp = s_kc + (2 * FLASH_S_TILE * Dc); // [2][S_TILE][Dp]
    float* s_v  = s_kp + (2 * FLASH_S_TILE * Dp); // [2][S_TILE][Dc]
    float* s_acc = s_v + (2 * FLASH_S_TILE * Dc); // [Dc]

    // 1. Initial Load: Query
    for (int d = tid; d < Dc; d += bdim) s_q[d] = q_nope[(q * H + h) * Dc + d];
    for (int d = tid; d < Dp; d += bdim) s_q[Dc + d] = q_pe[(q * H + h) * Dp + d];
    for (int d = tid; d < Dc; d += bdim) s_acc[d] = 0.0f;
    __syncthreads();

    float m_prev = -FLT_MAX;
    float l_prev = 0.0f;

    // Initial Prefetch: Tile 0 into buffer 0
    {
        int tsize0 = S < FLASH_S_TILE ? S : FLASH_S_TILE;
        for (int t = 0; t < tsize0; t++) {
            int kv_idx = sparse_indices[q * S + t];
            for (int d = tid; d < Dc; d += bdim) s_kc[0 * FLASH_S_TILE * Dc + t * Dc + d] = kc_cache[kv_idx * Dc + d];
            for (int d = tid; d < Dp; d += bdim) s_kp[0 * FLASH_S_TILE * Dp + t * Dp + d] = kp_cache[kv_idx * Dp + d];
            for (int d = tid; d < Dc; d += bdim) s_v[0 * FLASH_S_TILE * Dc + t * Dc + d] = v_cache[kv_idx * Dc + d];
        }
        __syncthreads();
    }

    for (int s0 = 0; s0 < S; s0 += FLASH_S_TILE) {
        int curr_buf = (s0 / FLASH_S_TILE) % 2;
        int next_buf = (curr_buf + 1) % 2;
        int next_s0 = s0 + FLASH_S_TILE;

        int tsize = S - s0;
        if (tsize > FLASH_S_TILE) tsize = FLASH_S_TILE;

        // Async Prefetch Next Tile next_s0 into next_buf
        if (next_s0 < S) {
            int nsize = S - next_s0;
            if (nsize > FLASH_S_TILE) nsize = FLASH_S_TILE;
            for (int t = 0; t < nsize; t++) {
                int kv_idx = sparse_indices[q * S + next_s0 + t];
                for (int d = tid; d < Dc; d += bdim) s_kc[next_buf * FLASH_S_TILE * Dc + t * Dc + d] = kc_cache[kv_idx * Dc + d];
                for (int d = tid; d < Dp; d += bdim) s_kp[next_buf * FLASH_S_TILE * Dp + t * Dp + d] = kp_cache[kv_idx * Dp + d];
                for (int d = tid; d < Dc; d += bdim) s_v[next_buf * FLASH_S_TILE * Dc + t * Dc + d] = v_cache[kv_idx * Dc + d];
            }
        }

        // Compute Tile s0 from curr_buf
        for (int t = 0; t < tsize; t++) {
            float score = 0.0f;
            for (int d = tid; d < Dc; d += bdim) 
                score += s_q[d] * s_kc[curr_buf * FLASH_S_TILE * Dc + t * Dc + d];
            for (int d = tid; d < Dp; d += bdim) 
                score += s_q[Dc + d] * s_kp[curr_buf * FLASH_S_TILE * Dp + t * Dp + d];

            for (int offset = bdim / 2; offset > 0; offset /= 2) score += __shfl_down_sync(0xffffffff, score, offset);
            score = __shfl_sync(0xffffffff, score, 0); 
            score *= scale;

            float m_curr = fmaxf(m_prev, score);
            float p = expf(score - m_curr);
            float alpha = expf(m_prev - m_curr);
            float l_curr = alpha * l_prev + p;

            for (int d = tid; d < Dc; d += bdim)
                s_acc[d] = s_acc[d] * alpha + p * s_v[curr_buf * FLASH_S_TILE * Dc + t * Dc + d];

            m_prev = m_curr;
            l_prev = l_curr;
        }
        __syncthreads(); // Wait for compute done and next fetch to potentially progress
    }

    float inv_l = 1.0f / l_prev;
    for (int d = tid; d < Dc; d += bdim) output[(q * H + h) * Dc + d] = s_acc[d] * inv_l;
}

// 6. OPT 4: Blackwell Pipelined Implementation
void dsa_forward_opt4(const float* q_nope, const float* q_pe,
                      const float* kv_cache_compressed,
                      const float* kv_cache_positional,
                      const float* v_cache,
                      const int* sparse_indices,
                      float* output,
                      const DsaConfig& cfg, cudaStream_t stream) {
    int Q = cfg.num_queries, H = cfg.num_heads, S = cfg.num_selected_kv;
    int Dc = cfg.head_dim_compressed, Dp = cfg.head_dim_positional;
    
    int threads = 128; 
    // Smem: (Dc+Dp) + [2 * S_TILE * (Dc+Dp+Dc)] + Dc
    size_t smem = ( (Dc + Dp) + (2 * FLASH_S_TILE * (Dc + Dp + Dc)) + Dc ) * sizeof(float);
    float scale = 1.0f / sqrtf((float)(Dc + Dp));

    // Shared memory attribute check
    static int max_smem_set = -1;
    if ((int)smem > max_smem_set) {
        CUDA_CHECK(cudaFuncSetAttribute((const void*)dsa_blackwell_pipelined_fused_kernel, 
                   cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        max_smem_set = (int)smem;
    }

    dsa_blackwell_pipelined_fused_kernel<<<Q * H, threads, smem, stream>>>(
        q_nope, q_pe, kv_cache_compressed, kv_cache_positional, v_cache,
        sparse_indices, output, Q, H, S, Dc, Dp, scale
    );
    CUDA_CHECK(cudaGetLastError());
}

void dsa_forward(const float* q_nope, const float* q_pe,
                 const float* kv_cache_compressed,
                 const float* kv_cache_positional,
                 const float* v_cache,
                 const int* sparse_indices,
                 float* output,
                 const DsaConfig& cfg, cudaStream_t stream) {
    dsa_forward_opt4(q_nope, q_pe, kv_cache_compressed, kv_cache_positional, v_cache, sparse_indices, output, cfg, stream);
}
