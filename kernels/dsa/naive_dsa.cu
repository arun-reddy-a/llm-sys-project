#include "naive_dsa.cuh"
#include "../../utils/cuda_utils.cuh"
#include <cfloat>
#include <cstdio>
#include <cmath>

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
// Kernel: dot product – compressed
//   scores[q, h, s] += sum_d q_nope[q,h,d] * kc[q,s,d]
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
// Kernel: dot product – positional
//   scores[q, h, s] += sum_d q_pe[q,h,d] * kp[q,s,d]
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
    int total = cfg.num_queries * cfg.num_heads * cfg.num_selected_kv;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    dot_compressed_kernel<<<blocks, threads, 0, stream>>>(
        q_nope, kc, scores,
        cfg.num_queries, cfg.num_heads, cfg.num_selected_kv,
        cfg.head_dim_compressed);
    CUDA_CHECK(cudaGetLastError());
}

void dsa_dot_positional(const float* q_pe, const float* kp, float* scores,
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
// Full naive DSA forward pass
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

    dsa_dot_compressed(q_nope, gathered_kc.ptr, scores.ptr, cfg, stream);
    dsa_dot_positional(q_pe, gathered_kp.ptr, scores.ptr, cfg, stream);

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
