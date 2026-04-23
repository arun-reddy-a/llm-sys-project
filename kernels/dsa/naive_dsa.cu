#include "naive_dsa.cuh"
#include "../../utils/cuda_utils.cuh"
#include <climits>
#include <cfloat>
#include <cstdio>
#include <cmath>
#include <cuda_fp16.h>
#include <mma.h>

#define DOT_TILE 128
#define MAX_HEADS_PER_THREAD 8
#define GEMM_TILE_M 16
#define GEMM_TILE_N 16
#define GEMM_TILE_K 16
#define ONLINE_SOFTMAX_THREADS 128
#define ONLINE_SOFTMAX_TILE 32
// Max (Dc + threads - 1) / threads dims updated by one block thread (Dc used in benchmarks ≤ 2048).
#define ONLINE_SOFTMAX_ACC_REGS 32
#define KV_PREFETCH_ROWS 4

namespace wmma = nvcuda::wmma;

// Page-sort kernel cost only pays off on longer sparse spans (bitonic + copies).
static __host__ __forceinline__ bool dsa_page_sort_amortizes(const DsaConfig& cfg) {
    return cfg.num_selected_kv >= 256;
}

// ---------------------------------------------------------------------------
// Opt 8 / 10 tuning (WMMA score GEMM path used by dsa_forward_opt8)
// 64×64 output tile per thread block (vs 16×16 in Opt3–7) → higher arithmetic
// intensity and fewer block launches vs tiny tiles.
// K inner dimension is 16 to match WMMA K=16 (FP16 tensor cores).
// GEMM8_PAD_K row stride in shared memory lets float4-friendly layouts and reduces
// 32-bank conflicts on the K columns used by WMMA loads.
// ---------------------------------------------------------------------------
#define GEMM8_TILE_M 64
#define GEMM8_TILE_N 64
#define GEMM8_TILE_K 16
#define GEMM8_PAD_K 24

__device__ __forceinline__ void prefetch_l2(const void* ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("prefetch.global.L2 [%0];" :: "l"(ptr));
#else
    (void)ptr;
#endif
}

__device__ __forceinline__ float load_query_concat(const float* q_nope,
                                                   const float* q_pe,
                                                   int q, int h, int d,
                                                   int H, int Dc, int Dp) {
    if (d < Dc) {
        return q_nope[(q * H + h) * Dc + d];
    }
    return q_pe[(q * H + h) * Dp + (d - Dc)];
}

__device__ __forceinline__ float load_k_concat(const float* kc,
                                               const float* kp,
                                               int q, int s, int d,
                                               int S, int Dc, int Dp) {
    if (d < Dc) {
        return kc[(q * S + s) * Dc + d];
    }
    return kp[(q * S + s) * Dp + (d - Dc)];
}

__device__ __forceinline__ float load_sparse_k_concat(const float* kv_cache_compressed,
                                                      const float* kv_cache_positional,
                                                      const int* sparse_indices,
                                                      int q, int s, int d,
                                                      int S, int Dc, int Dp) {
    int kv_idx = sparse_indices[q * S + s];
    if (d < Dc) {
        return kv_cache_compressed[kv_idx * Dc + d];
    }
    return kv_cache_positional[kv_idx * Dp + (d - Dc)];
}

__device__ __forceinline__ float load_sparse_v(const float* v_cache,
                                               const int* sparse_indices,
                                               int q, int s, int d,
                                               int S, int Dc) {
    int kv_idx = sparse_indices[q * S + s];
    return v_cache[kv_idx * Dc + d];
}

__device__ __forceinline__ float load_k_by_idx(const float* kv_cache_compressed,
                                               const float* kv_cache_positional,
                                               int kv_idx, int d,
                                               int Dc, int Dp) {
    if (d < Dc) {
        return kv_cache_compressed[kv_idx * Dc + d];
    }
    return kv_cache_positional[kv_idx * Dp + (d - Dc)];
}

__device__ __forceinline__ float load_v_by_idx(const float* v_cache,
                                               int kv_idx, int d, int Dc) {
    return v_cache[kv_idx * Dc + d];
}

__device__ __forceinline__ void prefetch_k_by_idx(const float* kv_cache_compressed,
                                                  const float* kv_cache_positional,
                                                  int kv_idx, int d,
                                                  int Dc, int Dp) {
    if (d < Dc) {
        prefetch_l2(kv_cache_compressed + kv_idx * Dc + d);
    } else {
        prefetch_l2(kv_cache_positional + kv_idx * Dp + (d - Dc));
    }
}

__device__ __forceinline__ bool sparse_index_after(int lhs, int rhs, int page_size) {
    if (lhs == INT_MAX) return rhs != INT_MAX;
    if (rhs == INT_MAX) return false;

    int lhs_page = lhs / page_size;
    int rhs_page = rhs / page_size;
    if (lhs_page != rhs_page) return lhs_page > rhs_page;
    return lhs > rhs;
}

// ===================================================================
// Kernel: sort sparse indices by page ID per query
//   Reorders each query's selected KV indices by (page_id, token_id)
//   before gather so accesses are more page-coalesced.
// ===================================================================
__global__ void sort_sparse_indices_by_page_kernel(const int* input,
                                                   int* output,
                                                   int Q, int S, int padded_S,
                                                   int page_size) {
    int q = blockIdx.x;
    if (q >= Q || page_size <= 0) return;

    extern __shared__ int s_idx[];

    for (int i = threadIdx.x; i < padded_S; i += blockDim.x) {
        s_idx[i] = (i < S) ? input[q * S + i] : INT_MAX;
    }
    __syncthreads();

    for (int size = 2; size <= padded_S; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int idx = threadIdx.x; idx < padded_S; idx += blockDim.x) {
                int partner = idx ^ stride;
                if (partner > idx) {
                    bool ascending = ((idx & size) == 0);
                    int a = s_idx[idx];
                    int b = s_idx[partner];
                    bool swap = ascending
                        ? sparse_index_after(a, b, page_size)
                        : sparse_index_after(b, a, page_size);
                    if (swap) {
                        s_idx[idx] = b;
                        s_idx[partner] = a;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (int i = threadIdx.x; i < S; i += blockDim.x) {
        output[q * S + i] = s_idx[i];
    }
}

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
// Kernel: fused tiled dot (Dc+Dp in one pass, one block per (q,s))
//   Combines compressed and positional dot products into a single
//   kernel by iterating over the concatenated Dc+Dp dimension.
//   Same one-block-per-(q,s) structure as the separate tiled kernels.
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
    int total_dim = Dc + Dp;

    int nh = (H + (int)blockDim.x - 1) / (int)blockDim.x;
    if (nh > MAX_HEADS_PER_THREAD) nh = MAX_HEADS_PER_THREAD;
    float sums[MAX_HEADS_PER_THREAD];
    for (int i = 0; i < nh; i++) sums[i] = 0.0f;

    for (int d0 = 0; d0 < total_dim; d0 += DOT_TILE) {
        int tsize = total_dim - d0;
        if (tsize > DOT_TILE) tsize = DOT_TILE;

        for (int t = tid; t < tsize; t += blockDim.x)
            s_tile[t] = load_k_concat(kc, kp, q, s, d0 + t, S, Dc, Dp);
        __syncthreads();

        for (int hi = 0; hi < nh; hi++) {
            int h = tid + hi * (int)blockDim.x;
            if (h < H) {
                for (int t = 0; t < tsize; ++t) {
                    float q_val = load_query_concat(q_nope, q_pe, q, h,
                                                    d0 + t, H, Dc, Dp);
                    sums[hi] += q_val * s_tile[t];
                }
            }
        }
        __syncthreads();
    }

    for (int hi = 0; hi < nh; hi++) {
        int h = tid + hi * (int)blockDim.x;
        if (h < H)
            scores[q * H * S + h * S + s] = sums[hi];
    }
}

// ===================================================================
// Kernel: fused batched GEMM over queries for score computation
//   For each query q, computes:
//      scores[q] = concat(q_nope[q], q_pe[q]) * concat(kc[q], kp[q])^T
//   with one 2D tile per (query, head-tile, kv-tile), so all queries are
//   packed into a single launch instead of effectively processing one
//   (q, s) pair at a time.
// ===================================================================
__global__ void dot_fused_batched_gemm_kernel(const float* __restrict__ q_nope,
                                              const float* __restrict__ q_pe,
                                              const float* __restrict__ kc,
                                              const float* __restrict__ kp,
                                              float* scores,
                                              int Q, int H, int S, int Dc, int Dp) {
    int q = blockIdx.z;
    int h = blockIdx.y * GEMM_TILE_M + threadIdx.y;
    int s = blockIdx.x * GEMM_TILE_N + threadIdx.x;
    if (q >= Q) return;

    __shared__ float q_tile[GEMM_TILE_M][GEMM_TILE_K];
    __shared__ float k_tile[GEMM_TILE_N][GEMM_TILE_K + 1];

    float acc = 0.0f;
    int total_dim = Dc + Dp;

    for (int d0 = 0; d0 < total_dim; d0 += GEMM_TILE_K) {
        int dk_q = d0 + threadIdx.x;
        if (h < H && dk_q < total_dim) {
            q_tile[threadIdx.y][threadIdx.x] = load_query_concat(
                q_nope, q_pe, q, h, dk_q, H, Dc, Dp);
        } else {
            q_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int dk_k = d0 + threadIdx.y;
        if (s < S && dk_k < total_dim) {
            k_tile[threadIdx.x][threadIdx.y] = load_k_concat(
                kc, kp, q, s, dk_k, S, Dc, Dp);
        } else {
            k_tile[threadIdx.x][threadIdx.y] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < GEMM_TILE_K; k++) {
            acc += q_tile[threadIdx.y][k] * k_tile[threadIdx.x][k];
        }

        __syncthreads();
    }

    if (h < H && s < S) {
        scores[(q * H + h) * S + s] = acc;
    }
}

// ===================================================================
// Kernel: fused batched GEMM with sparse-index-driven K gather
//   Reads K directly from the full KV cache using sparse indices,
//   removing the standalone gathered K buffers.
// ===================================================================
__global__ void dot_fused_sparse_batched_gemm_kernel(const float* __restrict__ q_nope,
                                                     const float* __restrict__ q_pe,
                                                     const float* __restrict__ kv_cache_compressed,
                                                     const float* __restrict__ kv_cache_positional,
                                                     const int* __restrict__ sparse_indices,
                                                     float* scores,
                                                     int Q, int H, int S, int Dc, int Dp) {
    int q = blockIdx.z;
    int h = blockIdx.y * GEMM_TILE_M + threadIdx.y;
    int s = blockIdx.x * GEMM_TILE_N + threadIdx.x;
    if (q >= Q) return;

    __shared__ float q_tile[GEMM_TILE_M][GEMM_TILE_K];
    __shared__ float k_tile[GEMM_TILE_N][GEMM_TILE_K + 1];
    __shared__ int kv_idx_tile[GEMM_TILE_N];

    if (threadIdx.y == 0) {
        kv_idx_tile[threadIdx.x] = (s < S) ? sparse_indices[q * S + s] : 0;
    }
    __syncthreads();

    float acc = 0.0f;
    int total_dim = Dc + Dp;
    int kv_idx = kv_idx_tile[threadIdx.x];

    for (int d0 = 0; d0 < total_dim; d0 += GEMM_TILE_K) {
        int dk_q = d0 + threadIdx.x;
        if (h < H && dk_q < total_dim) {
            q_tile[threadIdx.y][threadIdx.x] = load_query_concat(
                q_nope, q_pe, q, h, dk_q, H, Dc, Dp);
        } else {
            q_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int dk_k = d0 + threadIdx.y;
        if (s < S && dk_k < total_dim) {
            k_tile[threadIdx.x][threadIdx.y] = load_k_by_idx(
                kv_cache_compressed, kv_cache_positional,
                kv_idx, dk_k, Dc, Dp);
        } else {
            k_tile[threadIdx.x][threadIdx.y] = 0.0f;
        }

        int next_d = d0 + GEMM_TILE_K + threadIdx.y;
        if (s < S && next_d < total_dim && threadIdx.x < KV_PREFETCH_ROWS) {
            prefetch_k_by_idx(kv_cache_compressed, kv_cache_positional,
                              kv_idx, next_d, Dc, Dp);
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < GEMM_TILE_K; k++) {
            acc += q_tile[threadIdx.y][k] * k_tile[threadIdx.x][k];
        }

        __syncthreads();
    }

    if (h < H && s < S) {
        scores[(q * H + h) * S + s] = acc;
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
// Kernel: batched GEMM output projection
//   For each query q, computes out[q] = attn[q] * v[q].
// ===================================================================
__global__ void output_proj_batched_gemm_kernel(const float* __restrict__ attn,
                                                const float* __restrict__ v,
                                                float* __restrict__ output,
                                                int Q, int H, int S, int Dc) {
    int q = blockIdx.z;
    int h = blockIdx.y * GEMM_TILE_M + threadIdx.y;
    int d = blockIdx.x * GEMM_TILE_N + threadIdx.x;
    if (q >= Q) return;

    __shared__ float attn_tile[GEMM_TILE_M][GEMM_TILE_K];
    __shared__ float v_tile[GEMM_TILE_N][GEMM_TILE_K + 1];

    float acc = 0.0f;

    for (int s0 = 0; s0 < S; s0 += GEMM_TILE_K) {
        int s_attn = s0 + threadIdx.x;
        if (h < H && s_attn < S) {
            attn_tile[threadIdx.y][threadIdx.x] = attn[(q * H + h) * S + s_attn];
        } else {
            attn_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int s_v = s0 + threadIdx.y;
        if (d < Dc && s_v < S) {
            v_tile[threadIdx.x][threadIdx.y] = v[(q * S + s_v) * Dc + d];
        } else {
            v_tile[threadIdx.x][threadIdx.y] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < GEMM_TILE_K; k++) {
            acc += attn_tile[threadIdx.y][k] * v_tile[threadIdx.x][k];
        }

        __syncthreads();
    }

    if (h < H && d < Dc) {
        output[(q * H + h) * Dc + d] = acc;
    }
}

// ===================================================================
// Kernel: FlashAttention-style online softmax + output accumulation
//   One block per (q, h). Scores are processed in S-tiles, maintaining
//   running max and normalization state while accumulating the weighted
//   value projection directly into the output vector.
// ===================================================================
__global__ void online_softmax_output_kernel(const float* __restrict__ scores,
                                             const float* __restrict__ v,
                                             float* __restrict__ output,
                                             int Q, int H, int S, int Dc) {
    int qh = blockIdx.x;
    if (qh >= Q * H) return;

    int q = qh / H;
    int h = qh % H;
    int tid = threadIdx.x;

    __shared__ float s_max[ONLINE_SOFTMAX_THREADS];
    __shared__ float s_sum[ONLINE_SOFTMAX_THREADS];
    __shared__ float s_scores[ONLINE_SOFTMAX_TILE];
    __shared__ float s_weight[ONLINE_SOFTMAX_TILE];

    float running_max = -FLT_MAX;
    float running_sum = 0.0f;
    float acc_regs[ONLINE_SOFTMAX_ACC_REGS];

    for (int d = tid; d < Dc; d += blockDim.x) {
        output[(q * H + h) * Dc + d] = 0.0f;
    }
    __syncthreads();

    const float* score_row = scores + (q * H + h) * S;
    float* out_row = output + (q * H + h) * Dc;

    for (int s0 = 0; s0 < S; s0 += ONLINE_SOFTMAX_TILE) {
        int tile_size = S - s0;
        if (tile_size > ONLINE_SOFTMAX_TILE) tile_size = ONLINE_SOFTMAX_TILE;

        float local_max = -FLT_MAX;
        for (int i = tid; i < tile_size; i += blockDim.x) {
            float score = score_row[s0 + i];
            s_scores[i] = score;
            local_max = fmaxf(local_max, score);
        }
        s_max[tid] = local_max;
        __syncthreads();

        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                s_max[tid] = fmaxf(s_max[tid], s_max[tid + offset]);
            }
            __syncthreads();
        }

        float tile_max = s_max[0];
        float new_max = fmaxf(running_max, tile_max);
        float max_scale = (running_sum == 0.0f) ? 0.0f : expf(running_max - new_max);

        for (int i = tid; i < tile_size; i += blockDim.x) {
            s_weight[i] = expf(s_scores[i] - new_max);
        }
        __syncthreads();

        float local_sum = 0.0f;
        for (int i = tid; i < tile_size; i += blockDim.x) {
            local_sum += s_weight[i];
        }
        s_sum[tid] = local_sum;
        __syncthreads();

        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                s_sum[tid] += s_sum[tid + offset];
            }
            __syncthreads();
        }

        float new_sum = running_sum * max_scale + s_sum[0];

        int acc_idx = 0;
        for (int d = tid; d < Dc; d += blockDim.x) {
            acc_regs[acc_idx++] = out_row[d] * running_sum * max_scale;
        }
        for (int i = 0; i < tile_size; i++) {
            const float w = s_weight[i];
            const float* vrow = v + (q * S + (s0 + i)) * Dc;
            acc_idx = 0;
            for (int d = tid; d < Dc; d += blockDim.x) {
                acc_regs[acc_idx++] += w * vrow[d];
            }
        }
        acc_idx = 0;
        for (int d = tid; d < Dc; d += blockDim.x) {
            out_row[d] = acc_regs[acc_idx++] / new_sum;
        }
        __syncthreads();

        running_max = new_max;
        running_sum = new_sum;
        __syncthreads();
    }
}

// ===================================================================
// Kernel: online softmax + output accumulation with sparse-index-driven V gather
//   Reads V directly from the full cache using sparse indices, removing the
//   standalone gathered V buffer.
// ===================================================================
__global__ void online_softmax_output_sparse_kernel(const float* __restrict__ scores,
                                                    const float* __restrict__ v_cache,
                                                    const int* __restrict__ sparse_indices,
                                                    float* __restrict__ output,
                                                    int Q, int H, int S, int Dc) {
    int qh = blockIdx.x;
    if (qh >= Q * H) return;

    int q = qh / H;
    int h = qh % H;
    int tid = threadIdx.x;

    __shared__ float s_max[ONLINE_SOFTMAX_THREADS];
    __shared__ float s_sum[ONLINE_SOFTMAX_THREADS];
    __shared__ float s_scores[ONLINE_SOFTMAX_TILE];
    __shared__ float s_weight[ONLINE_SOFTMAX_TILE];
    __shared__ int s_kv_idx[ONLINE_SOFTMAX_TILE];

    float running_max = -FLT_MAX;
    float running_sum = 0.0f;
    float acc_regs[ONLINE_SOFTMAX_ACC_REGS];

    for (int d = tid; d < Dc; d += blockDim.x) {
        output[(q * H + h) * Dc + d] = 0.0f;
    }
    __syncthreads();

    const float* score_row = scores + (q * H + h) * S;
    float* out_row = output + (q * H + h) * Dc;

    for (int s0 = 0; s0 < S; s0 += ONLINE_SOFTMAX_TILE) {
        int tile_size = S - s0;
        if (tile_size > ONLINE_SOFTMAX_TILE) tile_size = ONLINE_SOFTMAX_TILE;

        for (int i = tid; i < tile_size; i += blockDim.x) {
            s_kv_idx[i] = sparse_indices[q * S + s0 + i];
        }
        __syncthreads();

        float local_max = -FLT_MAX;
        for (int i = tid; i < tile_size; i += blockDim.x) {
            float score = score_row[s0 + i];
            s_scores[i] = score;
            local_max = fmaxf(local_max, score);
        }

        int next_tile_start = s0 + ONLINE_SOFTMAX_TILE;
        int next_idx = next_tile_start + tid;
        if (next_idx < S && tid < ONLINE_SOFTMAX_TILE) {
            int kv_idx = sparse_indices[q * S + next_idx];
            prefetch_l2(v_cache + kv_idx * Dc);
        }
        s_max[tid] = local_max;
        __syncthreads();

        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                s_max[tid] = fmaxf(s_max[tid], s_max[tid + offset]);
            }
            __syncthreads();
        }

        float tile_max = s_max[0];
        float new_max = fmaxf(running_max, tile_max);
        float max_scale = (running_sum == 0.0f) ? 0.0f : expf(running_max - new_max);

        for (int i = tid; i < tile_size; i += blockDim.x) {
            s_weight[i] = expf(s_scores[i] - new_max);
        }
        __syncthreads();

        float local_sum = 0.0f;
        for (int i = tid; i < tile_size; i += blockDim.x) {
            local_sum += s_weight[i];
        }
        s_sum[tid] = local_sum;
        __syncthreads();

        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                s_sum[tid] += s_sum[tid + offset];
            }
            __syncthreads();
        }

        float new_sum = running_sum * max_scale + s_sum[0];

        int acc_idx = 0;
        for (int d = tid; d < Dc; d += blockDim.x) {
            acc_regs[acc_idx++] = out_row[d] * running_sum * max_scale;
        }
        for (int i = 0; i < tile_size; i++) {
            const float w = s_weight[i];
            const int kv = s_kv_idx[i];
            acc_idx = 0;
            for (int d = tid; d < Dc; d += blockDim.x) {
                acc_regs[acc_idx++] += w * load_v_by_idx(v_cache, kv, d, Dc);
            }
        }
        acc_idx = 0;
        for (int d = tid; d < Dc; d += blockDim.x) {
            out_row[d] = acc_regs[acc_idx++] / new_sum;
        }
        __syncthreads();

        running_max = new_max;
        running_sum = new_sum;
        __syncthreads();
    }
}

// ===================================================================
// Kernel: sparse batched GEMM without L2 prefetch (Opt6 isolation)
// ===================================================================
__global__ void dot_fused_sparse_batched_gemm_noprefetch_kernel(
        const float* __restrict__ q_nope,
        const float* __restrict__ q_pe,
        const float* __restrict__ kv_cache_compressed,
        const float* __restrict__ kv_cache_positional,
        const int* __restrict__ sparse_indices,
        float* scores,
        int Q, int H, int S, int Dc, int Dp) {
    int q = blockIdx.z;
    int h = blockIdx.y * GEMM_TILE_M + threadIdx.y;
    int s = blockIdx.x * GEMM_TILE_N + threadIdx.x;
    if (q >= Q) return;

    __shared__ float q_tile[GEMM_TILE_M][GEMM_TILE_K];
    __shared__ float k_tile[GEMM_TILE_N][GEMM_TILE_K + 1];
    __shared__ int kv_idx_tile[GEMM_TILE_N];

    if (threadIdx.y == 0) {
        kv_idx_tile[threadIdx.x] = (s < S) ? sparse_indices[q * S + s] : 0;
    }
    __syncthreads();

    float acc = 0.0f;
    int total_dim = Dc + Dp;
    int kv_idx = kv_idx_tile[threadIdx.x];

    for (int d0 = 0; d0 < total_dim; d0 += GEMM_TILE_K) {
        int dk_q = d0 + threadIdx.x;
        if (h < H && dk_q < total_dim) {
            q_tile[threadIdx.y][threadIdx.x] = load_query_concat(
                q_nope, q_pe, q, h, dk_q, H, Dc, Dp);
        } else {
            q_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int dk_k = d0 + threadIdx.y;
        if (s < S && dk_k < total_dim) {
            k_tile[threadIdx.x][threadIdx.y] = load_k_by_idx(
                kv_cache_compressed, kv_cache_positional,
                kv_idx, dk_k, Dc, Dp);
        } else {
            k_tile[threadIdx.x][threadIdx.y] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < GEMM_TILE_K; k++) {
            acc += q_tile[threadIdx.y][k] * k_tile[threadIdx.x][k];
        }

        __syncthreads();
    }

    if (h < H && s < S) {
        scores[(q * H + h) * S + s] = acc;
    }
}

// ===================================================================
// Kernel: online softmax + sparse V gather without L2 prefetch (Opt6)
// ===================================================================
__global__ void online_softmax_output_sparse_noprefetch_kernel(
        const float* __restrict__ scores,
        const float* __restrict__ v_cache,
        const int* __restrict__ sparse_indices,
        float* __restrict__ output,
        int Q, int H, int S, int Dc) {
    int qh = blockIdx.x;
    if (qh >= Q * H) return;

    int q = qh / H;
    int h = qh % H;
    int tid = threadIdx.x;

    __shared__ float s_max[ONLINE_SOFTMAX_THREADS];
    __shared__ float s_sum[ONLINE_SOFTMAX_THREADS];
    __shared__ float s_scores[ONLINE_SOFTMAX_TILE];
    __shared__ float s_weight[ONLINE_SOFTMAX_TILE];
    __shared__ int s_kv_idx[ONLINE_SOFTMAX_TILE];

    float running_max = -FLT_MAX;
    float running_sum = 0.0f;
    float acc_regs[ONLINE_SOFTMAX_ACC_REGS];

    for (int d = tid; d < Dc; d += blockDim.x) {
        output[(q * H + h) * Dc + d] = 0.0f;
    }
    __syncthreads();

    const float* score_row = scores + (q * H + h) * S;
    float* out_row = output + (q * H + h) * Dc;

    for (int s0 = 0; s0 < S; s0 += ONLINE_SOFTMAX_TILE) {
        int tile_size = S - s0;
        if (tile_size > ONLINE_SOFTMAX_TILE) tile_size = ONLINE_SOFTMAX_TILE;

        for (int i = tid; i < tile_size; i += blockDim.x) {
            s_kv_idx[i] = sparse_indices[q * S + s0 + i];
        }
        __syncthreads();

        float local_max = -FLT_MAX;
        for (int i = tid; i < tile_size; i += blockDim.x) {
            float score = score_row[s0 + i];
            s_scores[i] = score;
            local_max = fmaxf(local_max, score);
        }
        s_max[tid] = local_max;
        __syncthreads();

        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                s_max[tid] = fmaxf(s_max[tid], s_max[tid + offset]);
            }
            __syncthreads();
        }

        float tile_max = s_max[0];
        float new_max = fmaxf(running_max, tile_max);
        float max_scale = (running_sum == 0.0f) ? 0.0f : expf(running_max - new_max);

        for (int i = tid; i < tile_size; i += blockDim.x) {
            s_weight[i] = expf(s_scores[i] - new_max);
        }
        __syncthreads();

        float local_sum = 0.0f;
        for (int i = tid; i < tile_size; i += blockDim.x) {
            local_sum += s_weight[i];
        }
        s_sum[tid] = local_sum;
        __syncthreads();

        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                s_sum[tid] += s_sum[tid + offset];
            }
            __syncthreads();
        }

        float new_sum = running_sum * max_scale + s_sum[0];

        int acc_idx = 0;
        for (int d = tid; d < Dc; d += blockDim.x) {
            acc_regs[acc_idx++] = out_row[d] * running_sum * max_scale;
        }
        for (int i = 0; i < tile_size; i++) {
            const float w = s_weight[i];
            const int kv = s_kv_idx[i];
            acc_idx = 0;
            for (int d = tid; d < Dc; d += blockDim.x) {
                acc_regs[acc_idx++] += w * load_v_by_idx(v_cache, kv, d, Dc);
            }
        }
        acc_idx = 0;
        for (int d = tid; d < Dc; d += blockDim.x) {
            out_row[d] = acc_regs[acc_idx++] / new_sum;
        }
        __syncthreads();

        running_max = new_max;
        running_sum = new_sum;
        __syncthreads();
    }
}

// ===================================================================
// Opt 8 helpers: FP16 conversion + Opt 10 float4 loads on contiguous runs
// ===================================================================

__device__ __forceinline__ half2 opt8_float2_to_half2(float a, float b) {
    return __floats2half2_rn(a, b);
}

// Opt 10: float4 (128-bit) global load on compressed KV rows when [d,d+4) ⊂ [0,Dc).
__device__ __forceinline__ void opt8_load_kv_compressed_half4(const float* kv_c, int kv_idx,
                                                               int d, int Dc, half2 out[2]) {
    const float4* p4 = reinterpret_cast<const float4*>(kv_c + kv_idx * Dc + d);
    const float4 v4 = *p4;
    out[0] = opt8_float2_to_half2(v4.x, v4.y);
    out[1] = opt8_float2_to_half2(v4.z, v4.w);
}

// Opt 10: float4 on V cache row when d+4 <= Dc (dominant bandwidth path after WMMA scores).
__device__ __forceinline__ float4 opt8_load_v_float4(const float* v_cache, int kv_idx,
                                                     int d, int Dc) {
    return *reinterpret_cast<const float4*>(v_cache + kv_idx * Dc + d);
}

// ===================================================================
// Kernel: Opt 8 sparse score GEMM — WMMA FP16 (tensor cores), 64×64 tiles,
//         Opt 10 float4 loads. Same math as dot_fused_sparse_* but
//         accumulates on FP16 operands inside WMMA; results written in FP32.
//
// Block: (32,4,4) = 16 warps. Each warp owns one 16x16 WMMA output tile inside
// the 64x64 (H x S) tile for this block. K is padded to a multiple of 16 so
// virtual zeros extend (Dc+Dp) without a scalar tail (still exact vs padded GEMM).
// ===================================================================
__global__ void dot_fused_sparse_batched_gemm_wmma8_kernel(
        const float* __restrict__ q_nope,
        const float* __restrict__ q_pe,
        const float* __restrict__ kv_cache_compressed,
        const float* __restrict__ kv_cache_positional,
        const int* __restrict__ sparse_indices,
        float* scores,
        int Q, int H, int S, int Dc, int Dp) {
    const int q = blockIdx.z;
    if (q >= Q) return;

    const int h0 = blockIdx.y * GEMM8_TILE_M;
    const int s0 = blockIdx.x * GEMM8_TILE_N;
    const int total_dim = Dc + Dp;
    // Pad K so WMMA always runs full 16-wide K tiles (zeros past total_dim).
    const int K_round = (total_dim + GEMM8_TILE_K - 1) / GEMM8_TILE_K * GEMM8_TILE_K;

    __shared__ half sh_A[GEMM8_TILE_M][GEMM8_PAD_K];
    __shared__ half sh_Bcols[GEMM8_TILE_N][GEMM8_PAD_K];
    __shared__ int sh_kv_idx[GEMM8_TILE_N];
    // Staging for WMMA stores so we can clip partial 64×64 tiles at S/H boundaries.
    __shared__ float sh_C[GEMM8_TILE_M][GEMM8_TILE_N + 1];

    const int tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    const int nthreads = blockDim.x * blockDim.y * blockDim.z;

    if (tid < GEMM8_TILE_N) {
        const int sg = s0 + tid;
        sh_kv_idx[tid] = (sg < S) ? sparse_indices[q * S + sg] : 0;
    }
    __syncthreads();

    const int lane_id = threadIdx.x;
    const int warp_m = threadIdx.y;
    const int warp_n = threadIdx.z;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_c;
    wmma::fill_fragment(frag_c, 0.0f);

    for (int k0 = 0; k0 < K_round; k0 += GEMM8_TILE_K) {
        // Opt 8: promote Q tile to FP16 in shared memory (scalar path; WMMA does the heavy math).
        for (int idx = tid; idx < GEMM8_TILE_M * GEMM8_TILE_K; idx += nthreads) {
            const int kloc = idx % GEMM8_TILE_K;
            const int hloc = idx / GEMM8_TILE_K;
            const int hh = h0 + hloc;
            const int dk = k0 + kloc;
            half valh = __float2half_rn(0.0f);
            if (hh < H && dk < total_dim) {
                const float qv = load_query_concat(q_nope, q_pe, q, hh, dk, H, Dc, Dp);
                valh = __float2half_rn(qv);
            }
            sh_A[hloc][kloc] = valh;
        }

        // Opt 8: FP16 K/V concat rows in shared memory. Opt 10: when this K-slab lies entirely
        // inside the compressed Dc block, issue one float4 (128-bit) load per (s, quad-k) and
        // scatter four __half values — cuts global memory instructions on the KV hot path.
        // Opt 10: float4 needs 16-byte addresses; rows are 4*Dc bytes apart, so require Dc%4==0.
        const bool slab_in_compressed =
            (k0 + GEMM8_TILE_K <= Dc) && ((k0 % 4) == 0) && ((Dc % 4) == 0);
        if (slab_in_compressed) {
            const int vec_elems = GEMM8_TILE_N * (GEMM8_TILE_K / 4);
            for (int idx = tid; idx < vec_elems; idx += nthreads) {
                const int quad = idx % (GEMM8_TILE_K / 4);
                const int sloc = idx / (GEMM8_TILE_K / 4);
                const int sg = s0 + sloc;
                const int dk = k0 + quad * 4;
                if (sg < S) {
                    const int kv_idx = sh_kv_idx[sloc];
                    half2 kpair[2];
                    opt8_load_kv_compressed_half4(kv_cache_compressed, kv_idx, dk, Dc, kpair);
                    sh_Bcols[sloc][quad * 4 + 0] = __low2half(kpair[0]);
                    sh_Bcols[sloc][quad * 4 + 1] = __high2half(kpair[0]);
                    sh_Bcols[sloc][quad * 4 + 2] = __low2half(kpair[1]);
                    sh_Bcols[sloc][quad * 4 + 3] = __high2half(kpair[1]);
                } else {
                    sh_Bcols[sloc][quad * 4 + 0] = __float2half_rn(0.0f);
                    sh_Bcols[sloc][quad * 4 + 1] = __float2half_rn(0.0f);
                    sh_Bcols[sloc][quad * 4 + 2] = __float2half_rn(0.0f);
                    sh_Bcols[sloc][quad * 4 + 3] = __float2half_rn(0.0f);
                }
            }
        } else {
            for (int idx = tid; idx < GEMM8_TILE_N * GEMM8_TILE_K; idx += nthreads) {
                const int kloc = idx % GEMM8_TILE_K;
                const int sloc = idx / GEMM8_TILE_K;
                const int sg = s0 + sloc;
                const int dk = k0 + kloc;
                half valh = __float2half_rn(0.0f);
                if (sg < S && dk < total_dim) {
                    const int kv_idx = sh_kv_idx[sloc];
                    valh = __float2half_rn(
                        load_k_by_idx(kv_cache_compressed, kv_cache_positional, kv_idx, dk, Dc, Dp));
                }
                sh_Bcols[sloc][kloc] = valh;
            }
        }

        // Opt 7: software L2 prefetch of the next K-slab (same as Opt7 scalar GEMM).
        const int next_k = k0 + GEMM8_TILE_K + lane_id;
        if (warp_m == 0 && warp_n == 0 && next_k < total_dim) {
            for (int r = 0; r < KV_PREFETCH_ROWS; ++r) {
                const int rr = s0 + r;
                if (rr < S) {
                    const int kv_pref = sparse_indices[q * S + rr];
                    prefetch_k_by_idx(kv_cache_compressed, kv_cache_positional,
                                      kv_pref, next_k, Dc, Dp);
                }
            }
        }

        __syncthreads();

        // Opt 8: WMMA tensor core MMA on FP16 A/B, FP32 accumulators.
        wmma::load_matrix_sync(frag_a, &sh_A[warp_m * 16][0], GEMM8_PAD_K);
        wmma::load_matrix_sync(frag_b, &sh_Bcols[warp_n * 16][0], GEMM8_PAD_K);
        wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        __syncthreads();
    }

    // Opt 8: write WMMA accumulators to shared first — avoids 16-wide global stores past S or H.
    wmma::store_matrix_sync(&sh_C[warp_m * 16][warp_n * 16], frag_c,
                           GEMM8_TILE_N + 1, wmma::mem_row_major);
    __syncthreads();

    for (int idx = tid; idx < GEMM8_TILE_M * GEMM8_TILE_N; idx += nthreads) {
        const int rh = idx / GEMM8_TILE_N;
        const int rw = idx % GEMM8_TILE_N;
        const int hh = h0 + rh;
        const int ss = s0 + rw;
        if (hh < H && ss < S) {
            scores[(q * H + hh) * S + ss] = sh_C[rh][rw];
        }
    }
}

// ===================================================================
// Kernel: Opt 8 online softmax + sparse V — same numerics as Opt 7 path but
// Opt 10: vectorized float4 loads from v_cache when d+4 <= Dc; retains Opt 7
// L2 prefetch of the next tile's V base pointer.
// ===================================================================
__global__ void online_softmax_output_sparse_vec4_kernel(
        const float* __restrict__ scores,
        const float* __restrict__ v_cache,
        const int* __restrict__ sparse_indices,
        float* __restrict__ output,
        int Q, int H, int S, int Dc) {
    const int qh = blockIdx.x;
    if (qh >= Q * H) return;

    const int q = qh / H;
    const int h = qh % H;
    const int tid = threadIdx.x;

    __shared__ float s_max[ONLINE_SOFTMAX_THREADS];
    __shared__ float s_sum[ONLINE_SOFTMAX_THREADS];
    __shared__ float s_scores[ONLINE_SOFTMAX_TILE];
    __shared__ float s_weight[ONLINE_SOFTMAX_TILE];
    __shared__ int s_kv_idx[ONLINE_SOFTMAX_TILE];

    float running_max = -FLT_MAX;
    float running_sum = 0.0f;
    float acc_regs[ONLINE_SOFTMAX_ACC_REGS];

    for (int d = tid; d < Dc; d += blockDim.x) {
        output[(q * H + h) * Dc + d] = 0.0f;
    }
    __syncthreads();

    const float* score_row = scores + (q * H + h) * S;
    float* out_row = output + (q * H + h) * Dc;
    // Opt 10: float4 V loads need Dc multiple of 4 so every kv row is 16-byte aligned at d%4==0.
    const bool vec_v = (Dc % 4) == 0;

    for (int s0 = 0; s0 < S; s0 += ONLINE_SOFTMAX_TILE) {
        int tile_size = S - s0;
        if (tile_size > ONLINE_SOFTMAX_TILE) tile_size = ONLINE_SOFTMAX_TILE;

        for (int i = tid; i < tile_size; i += blockDim.x) {
            s_kv_idx[i] = sparse_indices[q * S + s0 + i];
        }
        __syncthreads();

        float local_max = -FLT_MAX;
        for (int i = tid; i < tile_size; i += blockDim.x) {
            const float score = score_row[s0 + i];
            s_scores[i] = score;
            local_max = fmaxf(local_max, score);
        }

        const int next_tile_start = s0 + ONLINE_SOFTMAX_TILE;
        const int next_idx = next_tile_start + tid;
        if (next_idx < S && tid < ONLINE_SOFTMAX_TILE) {
            const int kv_idx = sparse_indices[q * S + next_idx];
            prefetch_l2(v_cache + kv_idx * Dc);
        }
        s_max[tid] = local_max;
        __syncthreads();

        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                s_max[tid] = fmaxf(s_max[tid], s_max[tid + offset]);
            }
            __syncthreads();
        }

        const float tile_max = s_max[0];
        const float new_max = fmaxf(running_max, tile_max);
        const float max_scale = (running_sum == 0.0f) ? 0.0f : expf(running_max - new_max);

        for (int i = tid; i < tile_size; i += blockDim.x) {
            s_weight[i] = expf(s_scores[i] - new_max);
        }
        __syncthreads();

        float local_sum = 0.0f;
        for (int i = tid; i < tile_size; i += blockDim.x) {
            local_sum += s_weight[i];
        }
        s_sum[tid] = local_sum;
        __syncthreads();

        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                s_sum[tid] += s_sum[tid + offset];
            }
            __syncthreads();
        }

        const float new_sum = running_sum * max_scale + s_sum[0];

        if (vec_v) {
            // Opt 10: strip-mined by 4 with float4 global loads on V.
            for (int d = tid * 4; d < Dc; d += blockDim.x * 4) {
                if (d + 4 <= Dc) {
                    float acc0 = out_row[d + 0] * running_sum * max_scale;
                    float acc1 = out_row[d + 1] * running_sum * max_scale;
                    float acc2 = out_row[d + 2] * running_sum * max_scale;
                    float acc3 = out_row[d + 3] * running_sum * max_scale;
                    for (int i = 0; i < tile_size; i++) {
                        const float w = s_weight[i];
                        const int kv = s_kv_idx[i];
                        const float4 v4 = opt8_load_v_float4(v_cache, kv, d, Dc);
                        acc0 += w * v4.x;
                        acc1 += w * v4.y;
                        acc2 += w * v4.z;
                        acc3 += w * v4.w;
                    }
                    out_row[d + 0] = acc0 / new_sum;
                    out_row[d + 1] = acc1 / new_sum;
                    out_row[d + 2] = acc2 / new_sum;
                    out_row[d + 3] = acc3 / new_sum;
                } else {
                    for (int dd = d; dd < Dc; ++dd) {
                        float acc = out_row[dd] * running_sum * max_scale;
                        for (int i = 0; i < tile_size; i++) {
                            const float w = s_weight[i];
                            acc += w * load_v_by_idx(v_cache, s_kv_idx[i], dd, Dc);
                        }
                        out_row[dd] = acc / new_sum;
                    }
                }
            }
        } else {
            int acc_idx = 0;
            for (int d = tid; d < Dc; d += blockDim.x) {
                acc_regs[acc_idx++] = out_row[d] * running_sum * max_scale;
            }
            for (int i = 0; i < tile_size; i++) {
                const float w = s_weight[i];
                const int kv = s_kv_idx[i];
                acc_idx = 0;
                for (int d = tid; d < Dc; d += blockDim.x) {
                    acc_regs[acc_idx++] += w * load_v_by_idx(v_cache, kv, d, Dc);
                }
            }
            acc_idx = 0;
            for (int d = tid; d < Dc; d += blockDim.x) {
                out_row[d] = acc_regs[acc_idx++] / new_sum;
            }
        }
        __syncthreads();

        running_max = new_max;
        running_sum = new_sum;
        __syncthreads();
    }
}

// ===================================================================
// Kernel: naive output projection – one thread per (q, h, d)
// ===================================================================
__global__ void output_proj_naive_kernel(const float* attn, const float* v,
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
        sum += attn[(q * H + h) * S + s] * v[(q * S + s) * Dc + d];
    }
    output[(q * H + h) * Dc + d] = sum;
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

static void dsa_sort_sparse_indices_by_page(const int* input, int* output,
                                            const DsaConfig& cfg,
                                            cudaStream_t stream) {
    int padded_s = 1;
    while (padded_s < cfg.num_selected_kv) padded_s <<= 1;

    int threads = 256;
    if (threads > padded_s && padded_s > 0) {
        threads = padded_s;
    }
    if (threads < 1) threads = 1;
    size_t smem = static_cast<size_t>(padded_s) * sizeof(int);
    sort_sparse_indices_by_page_kernel<<<cfg.num_queries, threads, smem, stream>>>(
        input, output, cfg.num_queries, cfg.num_selected_kv, padded_s, cfg.page_size);
    CUDA_CHECK(cudaGetLastError());
}

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
    dim3 block(GEMM_TILE_N, GEMM_TILE_M);
    dim3 grid((S + GEMM_TILE_N - 1) / GEMM_TILE_N,
              (H + GEMM_TILE_M - 1) / GEMM_TILE_M,
              Q);
    dot_fused_batched_gemm_kernel<<<grid, block, 0, stream>>>(
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
    dim3 block(GEMM_TILE_N, GEMM_TILE_M);
    dim3 grid((cfg.head_dim_compressed + GEMM_TILE_N - 1) / GEMM_TILE_N,
              (cfg.num_heads + GEMM_TILE_M - 1) / GEMM_TILE_M,
              cfg.num_queries);
    output_proj_batched_gemm_kernel<<<grid, block, 0, stream>>>(
        attn, v, output,
        cfg.num_queries, cfg.num_heads, cfg.num_selected_kv,
        cfg.head_dim_compressed);
    CUDA_CHECK(cudaGetLastError());
}

void dsa_online_softmax_output(const float* scaled_scores, const float* v,
                               float* output,
                               const DsaConfig& cfg, cudaStream_t stream) {
    int blocks = cfg.num_queries * cfg.num_heads;
    online_softmax_output_kernel<<<blocks, ONLINE_SOFTMAX_THREADS, 0, stream>>>(
        scaled_scores, v, output,
        cfg.num_queries, cfg.num_heads, cfg.num_selected_kv,
        cfg.head_dim_compressed);
    CUDA_CHECK(cudaGetLastError());
}

// Sparse score GEMM with L2 prefetch (Opt7)
static void dsa_dot_fused_from_cache(const float* q_nope, const float* q_pe,
                                     const float* kv_cache_compressed,
                                     const float* kv_cache_positional,
                                     const int* sparse_indices,
                                     float* scores,
                                     const DsaConfig& cfg, cudaStream_t stream) {
    dim3 block(GEMM_TILE_N, GEMM_TILE_M);
    dim3 grid((cfg.num_selected_kv + GEMM_TILE_N - 1) / GEMM_TILE_N,
              (cfg.num_heads + GEMM_TILE_M - 1) / GEMM_TILE_M,
              cfg.num_queries);
    dot_fused_sparse_batched_gemm_kernel<<<grid, block, 0, stream>>>(
        q_nope, q_pe, kv_cache_compressed, kv_cache_positional, sparse_indices,
        scores, cfg.num_queries, cfg.num_heads, cfg.num_selected_kv,
        cfg.head_dim_compressed, cfg.head_dim_positional);
    CUDA_CHECK(cudaGetLastError());
}

// Sparse score GEMM without L2 prefetch (Opt6)
static void dsa_dot_fused_from_cache_noprefetch(const float* q_nope, const float* q_pe,
                                                const float* kv_cache_compressed,
                                                const float* kv_cache_positional,
                                                const int* sparse_indices,
                                                float* scores,
                                                const DsaConfig& cfg, cudaStream_t stream) {
    dim3 block(GEMM_TILE_N, GEMM_TILE_M);
    dim3 grid((cfg.num_selected_kv + GEMM_TILE_N - 1) / GEMM_TILE_N,
              (cfg.num_heads + GEMM_TILE_M - 1) / GEMM_TILE_M,
              cfg.num_queries);
    dot_fused_sparse_batched_gemm_noprefetch_kernel<<<grid, block, 0, stream>>>(
        q_nope, q_pe, kv_cache_compressed, kv_cache_positional, sparse_indices,
        scores, cfg.num_queries, cfg.num_heads, cfg.num_selected_kv,
        cfg.head_dim_compressed, cfg.head_dim_positional);
    CUDA_CHECK(cudaGetLastError());
}

// Online softmax + sparse V with L2 prefetch (Opt7)
static void dsa_online_softmax_output_from_cache(const float* scaled_scores,
                                                 const float* v_cache,
                                                 const int* sparse_indices,
                                                 float* output,
                                                 const DsaConfig& cfg,
                                                 cudaStream_t stream) {
    int blocks = cfg.num_queries * cfg.num_heads;
    online_softmax_output_sparse_kernel<<<blocks, ONLINE_SOFTMAX_THREADS, 0, stream>>>(
        scaled_scores, v_cache, sparse_indices, output,
        cfg.num_queries, cfg.num_heads, cfg.num_selected_kv,
        cfg.head_dim_compressed);
    CUDA_CHECK(cudaGetLastError());
}

// Online softmax + sparse V without L2 prefetch (Opt6)
static void dsa_online_softmax_output_from_cache_noprefetch(const float* scaled_scores,
                                                            const float* v_cache,
                                                            const int* sparse_indices,
                                                            float* output,
                                                            const DsaConfig& cfg,
                                                            cudaStream_t stream) {
    int blocks = cfg.num_queries * cfg.num_heads;
    online_softmax_output_sparse_noprefetch_kernel<<<blocks, ONLINE_SOFTMAX_THREADS, 0, stream>>>(
        scaled_scores, v_cache, sparse_indices, output,
        cfg.num_queries, cfg.num_heads, cfg.num_selected_kv,
        cfg.head_dim_compressed);
    CUDA_CHECK(cudaGetLastError());
}

// Opt 8: WMMA score GEMM (FP16 tensor cores + 64x64 tiles + float4 KV where safe).
static void dsa_dot_fused_from_cache_wmma8(const float* q_nope, const float* q_pe,
                                           const float* kv_cache_compressed,
                                           const float* kv_cache_positional,
                                           const int* sparse_indices,
                                           float* scores,
                                           const DsaConfig& cfg, cudaStream_t stream) {
    dim3 block(32, 4, 4);
    dim3 grid((cfg.num_selected_kv + GEMM8_TILE_N - 1) / GEMM8_TILE_N,
              (cfg.num_heads + GEMM8_TILE_M - 1) / GEMM8_TILE_M,
              cfg.num_queries);
    dot_fused_sparse_batched_gemm_wmma8_kernel<<<grid, block, 0, stream>>>(
        q_nope, q_pe, kv_cache_compressed, kv_cache_positional, sparse_indices,
        scores, cfg.num_queries, cfg.num_heads, cfg.num_selected_kv,
        cfg.head_dim_compressed, cfg.head_dim_positional);
    CUDA_CHECK(cudaGetLastError());
}

// Opt 8: online softmax + sparse V with float4 V loads (Opt 10) and L2 prefetch (Opt 7).
static void dsa_online_softmax_output_from_cache_vec4(const float* scaled_scores,
                                                      const float* v_cache,
                                                      const int* sparse_indices,
                                                      float* output,
                                                      const DsaConfig& cfg,
                                                      cudaStream_t stream) {
    const int blocks = cfg.num_queries * cfg.num_heads;
    online_softmax_output_sparse_vec4_kernel<<<blocks, ONLINE_SOFTMAX_THREADS, 0, stream>>>(
        scaled_scores, v_cache, sparse_indices, output,
        cfg.num_queries, cfg.num_heads, cfg.num_selected_kv,
        cfg.head_dim_compressed);
    CUDA_CHECK(cudaGetLastError());
}

// Fused tiled dot (Dc+Dp in one pass, one block per (q,s)) – Opt2
static void dsa_dot_fused_tiled(const float* q_nope, const float* q_pe,
                                const float* kc, const float* kp,
                                float* scores,
                                const DsaConfig& cfg, cudaStream_t stream) {
    int Q = cfg.num_queries, S = cfg.num_selected_kv, H = cfg.num_heads;
    int blocks = Q * S;
    int threads = dot_threads(H);
    size_t smem = DOT_TILE * sizeof(float);
    dot_fused_tiled_kernel<<<blocks, threads, smem, stream>>>(
        q_nope, q_pe, kc, kp, scores,
        Q, H, S, cfg.head_dim_compressed, cfg.head_dim_positional);
    CUDA_CHECK(cudaGetLastError());
}

// Naive output projection – one thread per (q, h, d)
static void dsa_output_proj_naive(const float* attn, const float* v,
                                  float* output,
                                  const DsaConfig& cfg, cudaStream_t stream) {
    int total = cfg.num_queries * cfg.num_heads * cfg.head_dim_compressed;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    output_proj_naive_kernel<<<blocks, threads, 0, stream>>>(
        attn, v, output,
        cfg.num_queries, cfg.num_heads, cfg.num_selected_kv,
        cfg.head_dim_compressed);
    CUDA_CHECK(cudaGetLastError());
}

// Naive dot product wrappers (use non-tiled kernels)
static void dsa_dot_compressed_naive(const float* q_nope, const float* kc,
                                     float* scores,
                                     const DsaConfig& cfg, cudaStream_t stream) {
    int total = cfg.num_queries * cfg.num_heads * cfg.num_selected_kv;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    dot_compressed_kernel<<<blocks, threads, 0, stream>>>(
        q_nope, kc, scores,
        cfg.num_queries, cfg.num_heads, cfg.num_selected_kv,
        cfg.head_dim_compressed);
    CUDA_CHECK(cudaGetLastError());
}

static void dsa_dot_positional_naive(const float* q_pe, const float* kp,
                                     float* scores,
                                     const DsaConfig& cfg, cudaStream_t stream) {
    int total = cfg.num_queries * cfg.num_heads * cfg.num_selected_kv;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    dot_positional_kernel<<<blocks, threads, 0, stream>>>(
        q_pe, kp, scores,
        cfg.num_queries, cfg.num_heads, cfg.num_selected_kv,
        cfg.head_dim_positional);
    CUDA_CHECK(cudaGetLastError());
}

static void dsa_scale_scores(float* scores, int Q, int H, int S, int Dc, int Dp,
                             cudaStream_t stream) {
    float scale_val = 1.0f / sqrtf((float)(Dc + Dp));
    int n = Q * H * S;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    scale_kernel<<<blocks, threads, 0, stream>>>(scores, scale_val, n);
    CUDA_CHECK(cudaGetLastError());
}

// ===================================================================
// DSA Forward Variants  (Naive + Opt1–Opt8, one per README optimization)
// ===================================================================

// NAIVE: Gather → naive per-element dots → 3-pass softmax → naive output proj
void dsa_forward_naive(const float* q_nope, const float* q_pe,
                       const float* kv_cache_compressed,
                       const float* kv_cache_positional,
                       const float* v_cache,
                       const int* sparse_indices,
                       float* output,
                       const DsaConfig& cfg, cudaStream_t stream) {
    int Q  = cfg.num_queries, S = cfg.num_selected_kv;
    int Dc = cfg.head_dim_compressed, Dp = cfg.head_dim_positional;
    int H  = cfg.num_heads;

    DeviceBuf<float> kc(Q * S * Dc), kp(Q * S * Dp), gv(Q * S * Dc);
    dsa_kv_gather(kv_cache_compressed, kv_cache_positional, v_cache,
                  sparse_indices, kc.ptr, kp.ptr, gv.ptr, cfg, stream);

    DeviceBuf<float> scores(Q * H * S);
    scores.zero();

    dsa_dot_compressed_naive(q_nope, kc.ptr, scores.ptr, cfg, stream);
    dsa_dot_positional_naive(q_pe, kp.ptr, scores.ptr, cfg, stream);

    dsa_scale_scores(scores.ptr, Q, H, S, Dc, Dp, stream);
    dsa_softmax(scores.ptr, cfg, stream);
    dsa_output_proj_naive(scores.ptr, gv.ptr, output, cfg, stream);
}

// OPT 1: Shared-memory tiled dots (reuse kc/kp tile across heads)
void dsa_forward_opt1(const float* q_nope, const float* q_pe,
                      const float* kv_cache_compressed,
                      const float* kv_cache_positional,
                      const float* v_cache,
                      const int* sparse_indices,
                      float* output,
                      const DsaConfig& cfg, cudaStream_t stream) {
    int Q  = cfg.num_queries, S = cfg.num_selected_kv;
    int Dc = cfg.head_dim_compressed, Dp = cfg.head_dim_positional;
    int H  = cfg.num_heads;

    DeviceBuf<float> kc(Q * S * Dc), kp(Q * S * Dp), gv(Q * S * Dc);
    dsa_kv_gather(kv_cache_compressed, kv_cache_positional, v_cache,
                  sparse_indices, kc.ptr, kp.ptr, gv.ptr, cfg, stream);

    DeviceBuf<float> scores(Q * H * S);
    scores.zero();

    dsa_dot_compressed(q_nope, kc.ptr, scores.ptr, cfg, stream);
    dsa_dot_positional(q_pe, kp.ptr, scores.ptr, cfg, stream);

    dsa_scale_scores(scores.ptr, Q, H, S, Dc, Dp, stream);
    dsa_softmax(scores.ptr, cfg, stream);
    dsa_output_proj_naive(scores.ptr, gv.ptr, output, cfg, stream);
}

// OPT 2: Fuse two dots into single GEMM (Dc+Dp in one tiled pass)
void dsa_forward_opt2(const float* q_nope, const float* q_pe,
                      const float* kv_cache_compressed,
                      const float* kv_cache_positional,
                      const float* v_cache,
                      const int* sparse_indices,
                      float* output,
                      const DsaConfig& cfg, cudaStream_t stream) {
    int Q  = cfg.num_queries, S = cfg.num_selected_kv;
    int Dc = cfg.head_dim_compressed, Dp = cfg.head_dim_positional;
    int H  = cfg.num_heads;

    DeviceBuf<float> kc(Q * S * Dc), kp(Q * S * Dp), gv(Q * S * Dc);
    dsa_kv_gather(kv_cache_compressed, kv_cache_positional, v_cache,
                  sparse_indices, kc.ptr, kp.ptr, gv.ptr, cfg, stream);

    DeviceBuf<float> scores(Q * H * S);

    dsa_dot_fused_tiled(q_nope, q_pe, kc.ptr, kp.ptr, scores.ptr, cfg, stream);

    dsa_scale_scores(scores.ptr, Q, H, S, Dc, Dp, stream);
    dsa_softmax(scores.ptr, cfg, stream);
    dsa_output_proj_naive(scores.ptr, gv.ptr, output, cfg, stream);
}

// OPT 3: Batch across queries (fused batched GEMM + batched output proj)
void dsa_forward_opt3(const float* q_nope, const float* q_pe,
                      const float* kv_cache_compressed,
                      const float* kv_cache_positional,
                      const float* v_cache,
                      const int* sparse_indices,
                      float* output,
                      const DsaConfig& cfg, cudaStream_t stream) {
    int Q  = cfg.num_queries, S = cfg.num_selected_kv;
    int Dc = cfg.head_dim_compressed, Dp = cfg.head_dim_positional;
    int H  = cfg.num_heads;

    DeviceBuf<float> kc(Q * S * Dc), kp(Q * S * Dp), gv(Q * S * Dc);
    dsa_kv_gather(kv_cache_compressed, kv_cache_positional, v_cache,
                  sparse_indices, kc.ptr, kp.ptr, gv.ptr, cfg, stream);

    DeviceBuf<float> scores(Q * H * S);

    dsa_dot_fused(q_nope, q_pe, kc.ptr, kp.ptr, scores.ptr, cfg, stream);

    dsa_scale_scores(scores.ptr, Q, H, S, Dc, Dp, stream);
    dsa_softmax(scores.ptr, cfg, stream);
    dsa_output_proj(scores.ptr, gv.ptr, output, cfg, stream);
}

// OPT 4: FlashAttention-style online softmax + output fusion
void dsa_forward_opt4(const float* q_nope, const float* q_pe,
                      const float* kv_cache_compressed,
                      const float* kv_cache_positional,
                      const float* v_cache,
                      const int* sparse_indices,
                      float* output,
                      const DsaConfig& cfg, cudaStream_t stream) {
    int Q  = cfg.num_queries, S = cfg.num_selected_kv;
    int Dc = cfg.head_dim_compressed, Dp = cfg.head_dim_positional;
    int H  = cfg.num_heads;

    DeviceBuf<float> kc(Q * S * Dc), kp(Q * S * Dp), gv(Q * S * Dc);
    dsa_kv_gather(kv_cache_compressed, kv_cache_positional, v_cache,
                  sparse_indices, kc.ptr, kp.ptr, gv.ptr, cfg, stream);

    DeviceBuf<float> scores(Q * H * S);

    dsa_dot_fused(q_nope, q_pe, kc.ptr, kp.ptr, scores.ptr, cfg, stream);

    dsa_scale_scores(scores.ptr, Q, H, S, Dc, Dp, stream);
    dsa_online_softmax_output(scores.ptr, gv.ptr, output, cfg, stream);
}

// OPT 5: Sort sparse indices by page for coalesced access
void dsa_forward_opt5(const float* q_nope, const float* q_pe,
                      const float* kv_cache_compressed,
                      const float* kv_cache_positional,
                      const float* v_cache,
                      const int* sparse_indices,
                      float* output,
                      const DsaConfig& cfg, cudaStream_t stream) {
    int Q  = cfg.num_queries, S = cfg.num_selected_kv;
    int Dc = cfg.head_dim_compressed, Dp = cfg.head_dim_positional;
    int H  = cfg.num_heads;

    DeviceBuf<int> sorted_buf;
    const int* idx_ptr = sparse_indices;
    if (dsa_page_sort_amortizes(cfg)) {
        sorted_buf.alloc(Q * S);
        dsa_sort_sparse_indices_by_page(sparse_indices, sorted_buf.ptr, cfg, stream);
        idx_ptr = sorted_buf.ptr;
    }

    DeviceBuf<float> kc(Q * S * Dc), kp(Q * S * Dp), gv(Q * S * Dc);
    dsa_kv_gather(kv_cache_compressed, kv_cache_positional, v_cache,
                  idx_ptr, kc.ptr, kp.ptr, gv.ptr, cfg, stream);

    DeviceBuf<float> scores(Q * H * S);

    dsa_dot_fused(q_nope, q_pe, kc.ptr, kp.ptr, scores.ptr, cfg, stream);

    dsa_scale_scores(scores.ptr, Q, H, S, Dc, Dp, stream);
    dsa_online_softmax_output(scores.ptr, gv.ptr, output, cfg, stream);
}

// OPT 6: Fuse KV gather into compute (sparse-index-driven, no separate gather)
void dsa_forward_opt6(const float* q_nope, const float* q_pe,
                      const float* kv_cache_compressed,
                      const float* kv_cache_positional,
                      const float* v_cache,
                      const int* sparse_indices,
                      float* output,
                      const DsaConfig& cfg, cudaStream_t stream) {
    int Q  = cfg.num_queries, S = cfg.num_selected_kv;
    int Dc = cfg.head_dim_compressed, Dp = cfg.head_dim_positional;
    int H  = cfg.num_heads;

    DeviceBuf<int> sorted_buf;
    const int* idx_ptr = sparse_indices;
    if (dsa_page_sort_amortizes(cfg)) {
        sorted_buf.alloc(Q * S);
        dsa_sort_sparse_indices_by_page(sparse_indices, sorted_buf.ptr, cfg, stream);
        idx_ptr = sorted_buf.ptr;
    }

    DeviceBuf<float> scores(Q * H * S);

    dsa_dot_fused_from_cache_noprefetch(q_nope, q_pe,
                                        kv_cache_compressed, kv_cache_positional,
                                        idx_ptr,
                                        scores.ptr, cfg, stream);

    dsa_scale_scores(scores.ptr, Q, H, S, Dc, Dp, stream);
    dsa_online_softmax_output_from_cache_noprefetch(scores.ptr, v_cache,
                                                    idx_ptr,
                                                    output, cfg, stream);
}

// OPT 7: Asynchronous L2 KV tile prefetching
void dsa_forward_opt7(const float* q_nope, const float* q_pe,
                      const float* kv_cache_compressed,
                      const float* kv_cache_positional,
                      const float* v_cache,
                      const int* sparse_indices,
                      float* output,
                      const DsaConfig& cfg, cudaStream_t stream) {
    int Q  = cfg.num_queries, S = cfg.num_selected_kv;
    int Dc = cfg.head_dim_compressed, Dp = cfg.head_dim_positional;
    int H  = cfg.num_heads;

    DeviceBuf<int> sorted_buf;
    const int* idx_ptr = sparse_indices;
    if (dsa_page_sort_amortizes(cfg)) {
        sorted_buf.alloc(Q * S);
        dsa_sort_sparse_indices_by_page(sparse_indices, sorted_buf.ptr, cfg, stream);
        idx_ptr = sorted_buf.ptr;
    }

    DeviceBuf<float> scores(Q * H * S);

    dsa_dot_fused_from_cache(q_nope, q_pe,
                             kv_cache_compressed, kv_cache_positional,
                             idx_ptr,
                             scores.ptr, cfg, stream);

    dsa_scale_scores(scores.ptr, Q, H, S, Dc, Dp, stream);
    dsa_online_softmax_output_from_cache(scores.ptr, v_cache,
                                         idx_ptr,
                                         output, cfg, stream);
}

// OPT 8: WMMA FP16 score GEMM (tensor cores) + 64x64 tiles + float4 KV/V loads + Opt7 prefetch
void dsa_forward_opt8(const float* q_nope, const float* q_pe,
                      const float* kv_cache_compressed,
                      const float* kv_cache_positional,
                      const float* v_cache,
                      const int* sparse_indices,
                      float* output,
                      const DsaConfig& cfg, cudaStream_t stream) {
    int Q  = cfg.num_queries, S = cfg.num_selected_kv;
    int Dc = cfg.head_dim_compressed, Dp = cfg.head_dim_positional;
    int H  = cfg.num_heads;

    DeviceBuf<int> sorted_buf;
    const int* idx_ptr = sparse_indices;
    if (dsa_page_sort_amortizes(cfg)) {
        sorted_buf.alloc(Q * S);
        dsa_sort_sparse_indices_by_page(sparse_indices, sorted_buf.ptr, cfg, stream);
        idx_ptr = sorted_buf.ptr;
    }

    DeviceBuf<float> scores(Q * H * S);

    dsa_dot_fused_from_cache_wmma8(q_nope, q_pe,
                                   kv_cache_compressed, kv_cache_positional,
                                   idx_ptr,
                                   scores.ptr, cfg, stream);

    dsa_scale_scores(scores.ptr, Q, H, S, Dc, Dp, stream);
    dsa_online_softmax_output_from_cache_vec4(scores.ptr, v_cache,
                                              idx_ptr,
                                              output, cfg, stream);
}

// Default: delegates to Opt8 (WMMA score GEMM + float4 softmax + L2 prefetch).
void dsa_forward(const float* q_nope, const float* q_pe,
                 const float* kv_cache_compressed,
                 const float* kv_cache_positional,
                 const float* v_cache,
                 const int* sparse_indices,
                 float* output,
                 const DsaConfig& cfg, cudaStream_t stream) {
    dsa_forward_opt8(q_nope, q_pe, kv_cache_compressed, kv_cache_positional,
                     v_cache, sparse_indices, output, cfg, stream);
}
