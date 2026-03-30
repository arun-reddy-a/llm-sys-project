#include "naive_dsa.cuh"
#include "../../utils/cuda_utils.cuh"
#include <cfloat>
#include <cstdio>
#include <cmath>

#define DOT_TILE 128
#define MAX_HEADS_PER_THREAD 8

// ===================================================================
// Kernel: KV gather – compressed
// ===================================================================
__global__ void kv_gather_compressed_kernel(const float* kv_cache, const int* indices,
                                            float* out, int Q, int S, int Dc) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Q * S * Dc;
    if (tid >= total) return;

    int d = tid % Dc;
    int s = (tid / Dc) % S;
    int q = tid / (S * Dc);

    int kv_idx = indices[q * S + s];
    out[tid] = kv_cache[kv_idx * Dc + d];
}

// ===================================================================
// Kernel: KV gather – positional
// ===================================================================
__global__ void kv_gather_positional_kernel(const float* kv_cache, const int* indices,
                                            float* out, int Q, int S, int Dp) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Q * S * Dp;
    if (tid >= total) return;

    int d = tid % Dp;
    int s = (tid / Dp) % S;
    int q = tid / (S * Dp);

    int kv_idx = indices[q * S + s];
    out[tid] = kv_cache[kv_idx * Dp + d];
}

// ===================================================================
// Kernel: V gather
// ===================================================================
__global__ void v_gather_kernel(const float* v_cache, const int* indices,
                                float* out, int Q, int S, int Dc) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Q * S * Dc;
    if (tid >= total) return;

    int d = tid % Dc;
    int s = (tid / Dc) % S;
    int q = tid / (S * Dc);

    int kv_idx = indices[q * S + s];
    out[tid] = v_cache[kv_idx * Dc + d];
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
// Kernel: softmax over S per (q, h) – one thread per (q, h)
// ===================================================================
__global__ void softmax_kernel(float* scores, int Q, int H, int S) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= Q * H) return;

    float* row = scores + tid * S;

    float mx = -FLT_MAX;
    for (int s = 0; s < S; s++) mx = fmaxf(mx, row[s]);

    float sum = 0.0f;
    for (int s = 0; s < S; s++) {
        row[s] = expf(row[s] - mx);
        sum += row[s];
    }
    float inv = 1.0f / sum;
    for (int s = 0; s < S; s++) row[s] *= inv;
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
// Kernel: element-wise scale
// ===================================================================
__global__ void scale_kernel(float* data, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] *= scale;
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

    {
        int n = Q * S * Dc;
        int blocks = (n + threads - 1) / threads;
        kv_gather_compressed_kernel<<<blocks, threads, 0, stream>>>(
            kv_cache_compressed, sparse_indices, gathered_kc, Q, S, Dc);
        CUDA_CHECK(cudaGetLastError());
    }
    {
        int n = Q * S * Dp;
        int blocks = (n + threads - 1) / threads;
        kv_gather_positional_kernel<<<blocks, threads, 0, stream>>>(
            kv_cache_positional, sparse_indices, gathered_kp, Q, S, Dp);
        CUDA_CHECK(cudaGetLastError());
    }
    {
        int n = Q * S * Dc;
        int blocks = (n + threads - 1) / threads;
        v_gather_kernel<<<blocks, threads, 0, stream>>>(
            v_cache, sparse_indices, gathered_v, Q, S, Dc);
        CUDA_CHECK(cudaGetLastError());
    }
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

void dsa_softmax(float* scores, const DsaConfig& cfg, cudaStream_t stream) {
    int total = cfg.num_queries * cfg.num_heads;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    softmax_kernel<<<blocks, threads, 0, stream>>>(
        scores, cfg.num_queries, cfg.num_heads, cfg.num_selected_kv);
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

// ===================================================================
// Full DSA forward pass
// ===================================================================
void dsa_forward(const float* q_nope, const float* q_pe,
                 const float* kv_cache_compressed,
                 const float* kv_cache_positional,
                 const float* v_cache,
                 const int* sparse_indices,
                 float* output,
                 const DsaConfig& cfg, cudaStream_t stream) {
    int Q  = cfg.num_queries;
    int S  = cfg.num_selected_kv;
    int Dc = cfg.head_dim_compressed;
    int Dp = cfg.head_dim_positional;
    int H  = cfg.num_heads;

    DeviceBuf<float> gathered_kc(Q * S * Dc);
    DeviceBuf<float> gathered_kp(Q * S * Dp);
    DeviceBuf<float> gathered_v(Q * S * Dc);

    dsa_kv_gather(kv_cache_compressed, kv_cache_positional, v_cache,
                  sparse_indices,
                  gathered_kc.ptr, gathered_kp.ptr, gathered_v.ptr,
                  cfg, stream);

    DeviceBuf<float> scores(Q * H * S);
    scores.zero();

    dsa_dot_fused(q_nope, q_pe, gathered_kc.ptr, gathered_kp.ptr, scores.ptr, cfg, stream);

    // Scale by 1/sqrt(Dc + Dp)
    {
        float scale_val = 1.0f / sqrtf((float)(Dc + Dp));
        int n = Q * H * S;
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        scale_kernel<<<blocks, threads, 0, stream>>>(scores.ptr, scale_val, n);
        CUDA_CHECK(cudaGetLastError());
    }

    dsa_softmax(scores.ptr, cfg, stream);
    dsa_output_proj(scores.ptr, gathered_v.ptr, output, cfg, stream);
}
