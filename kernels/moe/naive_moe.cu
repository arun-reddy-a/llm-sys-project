#include "naive_moe.cuh"
#include "../../utils/cuda_utils.cuh"
#include <cfloat>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <mma.h>

#define TILE_SIZE 16

// ===================================================================
// Kernel: gate_logits  –  logits[t, e] = dot(input[t], gate_weight[e])
// ===================================================================
__global__ void gate_logits_kernel(const float* input,
                                   const float* gate_weight,
                                   float* logits,
                                   int T, int E, int D) {
    int t = blockIdx.x;
    int e = threadIdx.x;
    if (t >= T || e >= E) return;

    float sum = 0.0f;
    for (int d = 0; d < D; d++) {
        sum += input[t * D + d] * gate_weight[e * D + d];
    }
    logits[t * E + e] = sum;
}

// ===================================================================
// Kernel: softmax over experts per token (in-place on logits)
// ===================================================================
__global__ void softmax_experts_kernel(float* logits, int T, int E) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;

    float* row = logits + t * E;

    float mx = -FLT_MAX;
    for (int e = 0; e < E; e++) mx = fmaxf(mx, row[e]);

    float sum = 0.0f;
    for (int e = 0; e < E; e++) {
        row[e] = expf(row[e] - mx);
        sum += row[e];
    }
    for (int e = 0; e < E; e++) row[e] /= sum;
}

// ===================================================================
// Kernel: top-K selection per token
// ===================================================================
__global__ void topk_kernel(const float* probs, int* indices, float* weights,
                            int T, int E, int K) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;

    const float* row = probs + t * E;
    int*   out_idx = indices + t * K;
    float* out_wt  = weights + t * K;

    for (int k = 0; k < K; k++) {
        out_idx[k] = -1;
        out_wt[k]  = -FLT_MAX;
    }

    for (int e = 0; e < E; e++) {
        float v = row[e];
        int min_k = 0;
        for (int k = 1; k < K; k++) {
            if (out_wt[k] < out_wt[min_k]) min_k = k;
        }
        if (v > out_wt[min_k]) {
            out_wt[min_k]  = v;
            out_idx[min_k] = e;
        }
    }

    float s = 0.0f;
    for (int k = 0; k < K; k++) s += out_wt[k];
    if (s > 0.0f) {
        for (int k = 0; k < K; k++) out_wt[k] /= s;
    }
}

// ===================================================================
// Optimization 2: Fused gate-logits + softmax + top-K kernel
//   One block per token. Caches input row in shared memory,
//   computes logits in smem, then softmax + topK without DRAM roundtrip.
// ===================================================================
__global__ void fused_gate_kernel(const float* __restrict__ input,
                                  const float* __restrict__ gate_weight,
                                  int* expert_indices, float* expert_weights,
                                  int T, int E, int D, int K) {
    int t = blockIdx.x;
    if (t >= T) return;

    extern __shared__ float smem[];
    float* s_input  = smem;
    float* s_logits = smem + D;

    for (int d = threadIdx.x; d < D; d += blockDim.x)
        s_input[d] = input[t * D + d];
    __syncthreads();

    if ((int)threadIdx.x < E) {
        int e = threadIdx.x;
        float dot = 0.0f;
        for (int d = 0; d < D; d++)
            dot += s_input[d] * gate_weight[e * D + d];
        s_logits[e] = dot;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float mx = -FLT_MAX;
        for (int e = 0; e < E; e++) mx = fmaxf(mx, s_logits[e]);
        float sum = 0.0f;
        for (int e = 0; e < E; e++) {
            s_logits[e] = expf(s_logits[e] - mx);
            sum += s_logits[e];
        }
        for (int e = 0; e < E; e++) s_logits[e] /= sum;

        int*   out_idx = expert_indices + t * K;
        float* out_wt  = expert_weights + t * K;
        for (int k = 0; k < K; k++) {
            out_idx[k] = -1;
            out_wt[k]  = -FLT_MAX;
        }
        for (int e = 0; e < E; e++) {
            float v = s_logits[e];
            int min_k = 0;
            for (int k = 1; k < K; k++)
                if (out_wt[k] < out_wt[min_k]) min_k = k;
            if (v > out_wt[min_k]) {
                out_wt[min_k]  = v;
                out_idx[min_k] = e;
            }
        }
        float s = 0.0f;
        for (int k = 0; k < K; k++) s += out_wt[k];
        if (s > 0.0f)
            for (int k = 0; k < K; k++) out_wt[k] /= s;
    }
}

// ===================================================================
// Kernel: gather tokens for a specific expert (single-thread, naive)
// ===================================================================
__global__ void gather_kernel(const float* input, const int* expert_indices,
                              int expert_id, float* gathered, int* token_map,
                              int* count, int T, int K, int D) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int cnt = 0;
    for (int t = 0; t < T; t++) {
        for (int k = 0; k < K; k++) {
            if (expert_indices[t * K + k] == expert_id) {
                token_map[cnt] = t;
                for (int d = 0; d < D; d++) {
                    gathered[cnt * D + d] = input[t * D + d];
                }
                cnt++;
                break;
            }
        }
    }
    *count = cnt;
}

// ===================================================================
// Kernel: naive GEMM  C[M,N] = A[M,K_] * B[K_,N]  (row-major)
// ===================================================================
__global__ void naive_gemm_kernel_impl(const float* A, const float* B,
                                       float* C, int M, int N, int K_) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K_; k++) {
        sum += A[row * K_ + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

// ===================================================================
// Kernel: transposed-B GEMM  C[M,N] = A[M,K_] * B^T
//   B stored row-major [N, K_], so B^T[k, n] = B[n * K_ + k]
// ===================================================================
__global__ void naive_gemm_bt_kernel(const float* A, const float* B,
                                     float* C, int M, int N, int K_) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K_; k++) {
        sum += A[row * K_ + k] * B[col * K_ + k];
    }
    C[row * N + col] = sum;
}

// ===================================================================
// Optimization 1: Tiled GEMM  C[M,N] = A[M,K_] * B[K_,N]
//   Uses shared memory tiles for data reuse within a thread block.
// ===================================================================
__global__ void tiled_gemm_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int N, int K_) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    int numTiles = (K_ + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K_)
            ? A[row * K_ + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K_ && col < N)
            ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++)
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

// ===================================================================
// Optimization 1: Tiled transposed-B GEMM  C[M,N] = A[M,K_] * B^T
//   B stored row-major [N, K_].  Loads B with coalesced access and
//   stores it transposed in shared memory (with +1 padding to avoid
//   bank conflicts).
// ===================================================================
__global__ void tiled_gemm_bt_kernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int M, int N, int K_) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    int numTiles = (K_ + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K_)
            ? A[row * K_ + a_col] : 0.0f;

        int b_row = blockIdx.x * TILE_SIZE + threadIdx.y;
        int b_col = t * TILE_SIZE + threadIdx.x;
        Bs[threadIdx.x][threadIdx.y] = (b_row < N && b_col < K_)
            ? B[b_row * K_ + b_col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++)
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

// ===================================================================
// Kernel: SwiGLU on contiguous gate/up arrays
// ===================================================================
__global__ void swiglu_simple_kernel(const float* gate, const float* up,
                                     float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float s = g / (1.0f + expf(-g));
    out[i] = s * up[i];
}

// ===================================================================
// Kernel: SwiGLU on [count, 2*I] layout  (gate = cols 0..I-1, up = cols I..2I-1)
// ===================================================================
__global__ void swiglu_strided_kernel(const float* in, float* out,
                                      int count, int I) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = count * I;
    if (idx >= total) return;

    int row = idx / I;
    int col = idx % I;

    float g = in[row * 2 * I + col];
    float u = in[row * 2 * I + I + col];
    float s = g / (1.0f + expf(-g));
    out[row * I + col] = s * u;
}

// ===================================================================
// Kernel: scatter expert output back into global output
// ===================================================================
__global__ void scatter_kernel(const float* expert_out, const int* token_map,
                               const float* expert_weights,
                               const int* expert_indices,
                               int expert_id, float* output,
                               int count, int T, int K, int D) {
    int idx = blockIdx.x;
    if (idx >= count) return;

    int t = token_map[idx];

    float w = 0.0f;
    for (int k = 0; k < K; k++) {
        if (expert_indices[t * K + k] == expert_id) {
            w = expert_weights[t * K + k];
            break;
        }
    }

    // Stride loop handles D > 1024 (max threads per block)
    for (int d = threadIdx.x; d < D; d += blockDim.x)
        atomicAdd(&output[t * D + d], w * expert_out[idx * D + d]);
}

// ===================================================================
// Host wrappers
// ===================================================================

void moe_gate(const float* input, const float* gate_weight,
              int* expert_indices, float* expert_weights,
              const MoeConfig& cfg, cudaStream_t stream) {
    int T = cfg.num_tokens, E = cfg.num_experts, D = cfg.hidden_dim;
    int K = cfg.top_k;

    int threads = ((E + 31) / 32) * 32;
    if (threads < 32)  threads = 32;
    if (threads > 256) threads = 256;
    size_t smem_bytes = (D + E) * sizeof(float);

    fused_gate_kernel<<<T, threads, smem_bytes, stream>>>(
        input, gate_weight, expert_indices, expert_weights, T, E, D, K);
    CUDA_CHECK(cudaGetLastError());
}

void moe_gather(const float* input, const int* expert_indices,
                const float* /*expert_weights*/, int expert_id,
                float* gathered, int* token_map, int* count,
                const MoeConfig& cfg, cudaStream_t stream) {
    gather_kernel<<<1, 1, 0, stream>>>(input, expert_indices, expert_id,
                                        gathered, token_map, count,
                                        cfg.num_tokens, cfg.top_k,
                                        cfg.hidden_dim);
    CUDA_CHECK(cudaGetLastError());
}

void naive_gemm(const float* A, const float* B, float* C,
                int M, int N, int K, cudaStream_t stream) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    tiled_gemm_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

void swiglu(const float* gate, const float* up, float* out,
            int n, cudaStream_t stream) {
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    swiglu_simple_kernel<<<blocks, threads, 0, stream>>>(gate, up, out, n);
    CUDA_CHECK(cudaGetLastError());
}

void moe_scatter(const float* expert_out, const int* token_map,
                 const float* expert_weights, const int* expert_indices,
                 int expert_id, float* output, int count,
                 const MoeConfig& cfg, cudaStream_t stream) {
    if (count == 0) return;
    int D = cfg.hidden_dim;
    int threads = min(D, 1024);
    scatter_kernel<<<count, threads, 0, stream>>>(expert_out, token_map,
                                                   expert_weights, expert_indices,
                                                   expert_id, output, count,
                                                   cfg.num_tokens, cfg.top_k, D);
    CUDA_CHECK(cudaGetLastError());
}

// 1. BASELINE: Naive Implementation
void moe_forward_naive(const float* d_input, const float* d_gate_weight,
                       const float* d_w1, const float* d_w2, float* d_output,
                       const MoeConfig& cfg, cudaStream_t stream) {
    int T = cfg.num_tokens, E = cfg.num_experts, E_local = cfg.num_local_experts;
    int D = cfg.hidden_dim, I = cfg.intermediate_dim, K = cfg.top_k;

    // --- Naive Routing (over all E_global experts) ---
    DeviceBuf<float> d_logits(T * E);
    gate_logits_kernel<<<T, E, 0, stream>>>(d_input, d_gate_weight, d_logits.ptr, T, E, D);
    softmax_experts_kernel<<<(T + 255) / 256, 256, 0, stream>>>(d_logits.ptr, T, E);
    DeviceBuf<int>   expert_indices(T * K);
    DeviceBuf<float> expert_weights(T * K);
    topk_kernel<<<(T+255)/256, 256, 0, stream>>>(d_logits.ptr, expert_indices.ptr, expert_weights.ptr, T, E, K);

    CUDA_CHECK(cudaMemsetAsync(d_output, 0, T * D * sizeof(float), stream));

    DeviceBuf<float> gathered(T * D);
    DeviceBuf<int>   token_map(T);
    DeviceBuf<int>   d_count(1);
    DeviceBuf<float> gemm1_out(T * 2 * I);
    DeviceBuf<float> act_out(T * I);
    DeviceBuf<float> gemm2_out(T * D);

    // Only compute over local experts [0..E_local)
    for (int e = 0; e < E_local; e++) {
        moe_gather(d_input, expert_indices.ptr, expert_weights.ptr, e, gathered.ptr, token_map.ptr, d_count.ptr, cfg, stream);
        int h_count = 0;
        CUDA_CHECK(cudaMemcpyAsync(&h_count, d_count.ptr, sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (h_count == 0) continue;

        // GEMM1: naive
        {
            int M_ = h_count, N_ = 2 * I, K_ = D;
            dim3 block(TILE_SIZE, TILE_SIZE);
            dim3 grid((N_ + TILE_SIZE - 1) / TILE_SIZE, (M_ + TILE_SIZE - 1) / TILE_SIZE);
            naive_gemm_bt_kernel<<<grid, block, 0, stream>>>(gathered.ptr, d_w1 + (size_t)e * 2 * I * D, gemm1_out.ptr, M_, N_, K_);
        }
        swiglu_strided_kernel<<<(h_count * I + 255) / 256, 256, 0, stream>>>(gemm1_out.ptr, act_out.ptr, h_count, I);
        // GEMM2: naive
        {
            int M_ = h_count, N_ = D, K_ = I;
            dim3 block(TILE_SIZE, TILE_SIZE);
            dim3 grid((N_ + TILE_SIZE - 1) / TILE_SIZE, (M_ + TILE_SIZE - 1) / TILE_SIZE);
            naive_gemm_bt_kernel<<<grid, block, 0, stream>>>(act_out.ptr, d_w2 + (size_t)e * D * I, gemm2_out.ptr, M_, N_, K_);
        }
        moe_scatter(gemm2_out.ptr, token_map.ptr, expert_weights.ptr, expert_indices.ptr, e, d_output, h_count, cfg, stream);
    }
}

// 2. OPT 1: Tiled GEMM
void moe_forward_opt1(const float* d_input, const float* d_gate_weight,
                      const float* d_w1, const float* d_w2, float* d_output,
                      const MoeConfig& cfg, cudaStream_t stream) {
    int T = cfg.num_tokens, E = cfg.num_experts, E_local = cfg.num_local_experts;
    int D = cfg.hidden_dim, I = cfg.intermediate_dim, K = cfg.top_k;

    // --- Naive Routing (over all E_global experts) ---
    DeviceBuf<float> d_logits(T * E);
    gate_logits_kernel<<<T, E, 0, stream>>>(d_input, d_gate_weight, d_logits.ptr, T, E, D);
    softmax_experts_kernel<<<(T + 255) / 256, 256, 0, stream>>>(d_logits.ptr, T, E);
    DeviceBuf<int>   expert_indices(T * K);
    DeviceBuf<float> expert_weights(T * K);
    topk_kernel<<<(T+255)/256, 256, 0, stream>>>(d_logits.ptr, expert_indices.ptr, expert_weights.ptr, T, E, K);

    CUDA_CHECK(cudaMemsetAsync(d_output, 0, T * D * sizeof(float), stream));
    DeviceBuf<float> gathered(T * D), gemm1_out(T * 2 * I), act_out(T * I), gemm2_out(T * D);
    DeviceBuf<int> token_map(T), d_count(1);

    for (int e = 0; e < E_local; e++) {
        moe_gather(d_input, expert_indices.ptr, expert_weights.ptr, e, gathered.ptr, token_map.ptr, d_count.ptr, cfg, stream);
        int h_count = 0;
        CUDA_CHECK(cudaMemcpyAsync(&h_count, d_count.ptr, sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (h_count == 0) continue;

        // GEMM1: Tiled
        {
            int M_ = h_count, N_ = 2 * I, K_ = D;
            dim3 block(TILE_SIZE, TILE_SIZE);
            dim3 grid((N_ + TILE_SIZE - 1) / TILE_SIZE, (M_ + TILE_SIZE - 1) / TILE_SIZE);
            tiled_gemm_bt_kernel<<<grid, block, 0, stream>>>(gathered.ptr, d_w1 + (size_t)e * 2 * I * D, gemm1_out.ptr, M_, N_, K_);
        }
        swiglu_strided_kernel<<<(h_count * I + 255) / 256, 256, 0, stream>>>(gemm1_out.ptr, act_out.ptr, h_count, I);
        // GEMM2: Tiled
        {
            int M_ = h_count, N_ = D, K_ = I;
            dim3 block(TILE_SIZE, TILE_SIZE);
            dim3 grid((N_ + TILE_SIZE - 1) / TILE_SIZE, (M_ + TILE_SIZE - 1) / TILE_SIZE);
            tiled_gemm_bt_kernel<<<grid, block, 0, stream>>>(act_out.ptr, d_w2 + (size_t)e * D * I, gemm2_out.ptr, M_, N_, K_);
        }
        moe_scatter(gemm2_out.ptr, token_map.ptr, expert_weights.ptr, expert_indices.ptr, e, d_output, h_count, cfg, stream);
    }
}

// 3. OPT 2: Fused Routing + Tiled GEMM
void moe_forward_opt2(const float* d_input, const float* d_gate_weight,
                      const float* d_w1, const float* d_w2, float* d_output,
                      const MoeConfig& cfg, cudaStream_t stream) {
    int T = cfg.num_tokens, E_local = cfg.num_local_experts;
    int D = cfg.hidden_dim, I = cfg.intermediate_dim, K = cfg.top_k;

    // --- Fused Routing (over all E_global experts) ---
    DeviceBuf<int>   expert_indices(T * K);
    DeviceBuf<float> expert_weights(T * K);
    moe_gate(d_input, d_gate_weight, expert_indices.ptr, expert_weights.ptr, cfg, stream);

    CUDA_CHECK(cudaMemsetAsync(d_output, 0, T * D * sizeof(float), stream));
    DeviceBuf<float> gathered(T * D), gemm1_out(T * 2 * I), act_out(T * I), gemm2_out(T * D);
    DeviceBuf<int> token_map(T), d_count(1);

    for (int e = 0; e < E_local; e++) {
        moe_gather(d_input, expert_indices.ptr, expert_weights.ptr, e, gathered.ptr, token_map.ptr, d_count.ptr, cfg, stream);
        int h_count = 0;
        CUDA_CHECK(cudaMemcpyAsync(&h_count, d_count.ptr, sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (h_count == 0) continue;

        {
            int M_ = h_count, N_ = 2 * I, K_ = D;
            dim3 block(TILE_SIZE, TILE_SIZE), grid((N_ + TILE_SIZE - 1) / TILE_SIZE, (M_ + TILE_SIZE - 1) / TILE_SIZE);
            tiled_gemm_bt_kernel<<<grid, block, 0, stream>>>(gathered.ptr, d_w1 + (size_t)e * 2 * I * D, gemm1_out.ptr, M_, N_, K_);
        }
        swiglu_strided_kernel<<<(h_count * I + 255) / 256, 256, 0, stream>>>(gemm1_out.ptr, act_out.ptr, h_count, I);
        {
            int M_ = h_count, N_ = D, K_ = I;
            dim3 block(TILE_SIZE, TILE_SIZE), grid((N_ + TILE_SIZE - 1) / TILE_SIZE, (M_ + TILE_SIZE - 1) / TILE_SIZE);
            tiled_gemm_bt_kernel<<<grid, block, 0, stream>>>(act_out.ptr, d_w2 + (size_t)e * D * I, gemm2_out.ptr, M_, N_, K_);
        }
        moe_scatter(gemm2_out.ptr, token_map.ptr, expert_weights.ptr, expert_indices.ptr, e, d_output, h_count, cfg, stream);
    }
}

// ===================================================================
// Grouped MoE: Expert Counter + Prefix Sum + Reorder
// ===================================================================

__global__ void expert_grouping_kernel(const int* expert_indices, 
                                       int* expert_counts,
                                       int* token_idx_in_expert,
                                       int T, int K, int E) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= T * K) return;
    int e = expert_indices[tid];
    if (e < 0 || e >= E) return;
    token_idx_in_expert[tid] = atomicAdd(&expert_counts[e], 1);
}

__global__ void group_reorder_kernel(const float* input,
                                     const int* expert_indices,
                                     const int* expert_offsets,
                                     const int* token_idx_in_expert,
                                     float* grouped_input,
                                     int* grouped_token_map,
                                     int T, int K, int D) {
    int tk = blockIdx.x * blockDim.x + threadIdx.x;
    if (tk >= T * K) return;
    int e = expert_indices[tk];
    if (e < 0) return; // Fix: guard against -1 expert indices
    int t = tk / K;
    int dest = expert_offsets[e] + token_idx_in_expert[tk];
    grouped_token_map[dest] = t;
    for (int d = 0; d < D; d++)
        grouped_input[dest * D + d] = input[t * D + d];
}

__global__ void grouped_gemm_bt_kernel(const float* __restrict__ A,
                                       const float* __restrict__ W_ptr,
                                       float* __restrict__ C,
                                       const int* __restrict__ expert_offsets,
                                       int D, int N, int E) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Expert lookup
    int total_rows = expert_offsets[E];
    int e = 0;
    if (row < total_rows) {
        while (e < E - 1 && row >= expert_offsets[e+1]) e++;
    }

    const float* W_e = W_ptr + (size_t)e * N * D;

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    float sum = 0.0f;
    int numTiles = (D + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (row < total_rows && a_col < D) ? A[row * D + a_col] : 0.0f;

        // Load W_e transposed (coalesced B read)
        int b_row = (col / TILE_SIZE) * TILE_SIZE + threadIdx.y;
        int b_col = t * TILE_SIZE + threadIdx.x;
        Bs[threadIdx.x][threadIdx.y] = (col < N && b_row < N && b_col < D) ? W_e[b_row * D + b_col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < total_rows && col < N) {
        C[row * N + col] = sum;
    }
}

__global__ void grouped_scatter_kernel(const float* grouped_out,
                                       const int* grouped_token_map,
                                       const float* expert_weights,
                                       const int* expert_indices,
                                       int* expert_offsets,
                                       float* global_out,
                                       int T, int K, int D, int E) {
    // blockIdx.x = index in the GROUPED buffer
    int tk = blockIdx.x;
    int e = 0;
    while (e < E - 1 && tk >= expert_offsets[e+1]) e++;
    if (tk >= expert_offsets[E]) return;

    int t = grouped_token_map[tk];
    float w = 0.0f;
    for (int k = 0; k < K; k++) {
        if (expert_indices[t * K + k] == e) {
            w = expert_weights[t * K + k];
            break;
        }
    }
    // Stride loop handles D > 1024 (max threads per block)
    for (int d = threadIdx.x; d < D; d += blockDim.x)
        atomicAdd(&global_out[t * D + d], w * grouped_out[tk * D + d]);
}

// 4. OPT 3: Grouped Implementation
void moe_forward_opt3(const float* d_input, const float* d_gate_weight,
                      const float* d_w1, const float* d_w2, float* d_output,
                      const MoeConfig& cfg, cudaStream_t stream) {
    int T = cfg.num_tokens, E = cfg.num_experts, E_local = cfg.num_local_experts;
    int D = cfg.hidden_dim, I = cfg.intermediate_dim, K = cfg.top_k;

    // 1. Fused Routing (over all E_global experts)
    DeviceBuf<int>   expert_indices(T * K);
    DeviceBuf<float> expert_weights(T * K);
    moe_gate(d_input, d_gate_weight, expert_indices.ptr, expert_weights.ptr, cfg, stream);

    // 2. Count tokens per expert (all E_global buckets for correct routing)
    DeviceBuf<int> expert_counts(E + 1);
    expert_counts.zero();
    DeviceBuf<int> token_idx_in_expert(T * K);
    expert_grouping_kernel<<<(T * K + 255) / 256, 256, 0, stream>>>(
        expert_indices.ptr, expert_counts.ptr, token_idx_in_expert.ptr, T, K, E
    );

    // 3. Prefix Sum for offsets (all E_global experts)
    std::vector<int> h_counts(E + 1);
    CUDA_CHECK(cudaMemcpyAsync(h_counts.data(), expert_counts.ptr, E * sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<int> h_offsets(E + 1);
    h_offsets[0] = 0;
    for (int i = 0; i < E; i++) h_offsets[i+1] = h_offsets[i] + h_counts[i];
    DeviceBuf<int> expert_offsets(E + 1);
    expert_offsets.upload(h_offsets.data());

    // Only process tokens assigned to local experts [0..E_local)
    int total_local = h_offsets[E_local];
    if (total_local == 0) return;

    // 4. Reorder tokens (all T*K assignments, but GEMM only uses first total_local rows)
    int total_active = h_offsets[E];
    DeviceBuf<float> grouped_input(total_active * D);
    DeviceBuf<int>   grouped_token_map(total_active);
    group_reorder_kernel<<<(T * K + 255) / 256, 256, 0, stream>>>(
        d_input, expert_indices.ptr, expert_offsets.ptr, token_idx_in_expert.ptr,
        grouped_input.ptr, grouped_token_map.ptr, T, K, D
    );

    CUDA_CHECK(cudaMemsetAsync(d_output, 0, T * D * sizeof(float), stream));
    DeviceBuf<float> g_gemm1_out(total_local * 2 * I), g_act_out(total_local * I), g_gemm2_out(total_local * D);

    // 5. Grouped GEMM 1: W1 [E_local, 2*I, D] – only local experts
    {
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((2 * I + TILE_SIZE - 1) / TILE_SIZE, (total_local + TILE_SIZE - 1) / TILE_SIZE);
        grouped_gemm_bt_kernel<<<grid, block, 0, stream>>>(
            grouped_input.ptr, d_w1, g_gemm1_out.ptr, expert_offsets.ptr, D, 2 * I, E_local);
    }

    // 6. SwiGLU
    swiglu_strided_kernel<<<(total_local * I + 255) / 256, 256, 0, stream>>>(g_gemm1_out.ptr, g_act_out.ptr, total_local, I);

    // 7. Grouped GEMM 2: W2 [E_local, D, I]
    {
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((D + TILE_SIZE - 1) / TILE_SIZE, (total_local + TILE_SIZE - 1) / TILE_SIZE);
        grouped_gemm_bt_kernel<<<grid, block, 0, stream>>>(
            g_act_out.ptr, d_w2, g_gemm2_out.ptr, expert_offsets.ptr, I, D, E_local);
    }

    // 8. Grouped Scatter (only local tokens)
    grouped_scatter_kernel<<<total_local, min(D, 1024), 0, stream>>>(
        g_gemm2_out.ptr, grouped_token_map.ptr, expert_weights.ptr,
        expert_indices.ptr, expert_offsets.ptr, d_output, T, K, D, E_local
    );
}

// ===================================================================
// FUSED EXPERT KERNEL: Routing + Gather + FFN + Scatter in one pass
// One block per token. Threads collaboratively compute GEMV for each expert.
// ===================================================================
__global__ void fused_moe_kernel(const float* __restrict__ input,
                                 const float* __restrict__ gate_weight,
                                 const float* __restrict__ w1,
                                 const float* __restrict__ w2,
                                 float* __restrict__ output,
                                 int T, int E, int D, int I, int K) {
    int t = blockIdx.x;
    if (t >= T) return;

    int tid = threadIdx.x;
    int bdim = blockDim.x;

    // Shared memory layout:
    // [D]          - Token input (s_x)
    // [E]          - Gate Logits (s_logits)
    // [2 * I]      - Intermediate activations (s_act)
    // [D]          - Token output accumulator (s_y)
    // Needs about 512 + 16 + 2048 + 512 = 3088 floats = ~12 KB. Fits!

    extern __shared__ float smem[];
    float* s_x = smem;
    float* s_logits = s_x + D;
    float* s_act = s_logits + E;
    float* s_y = s_act + (2 * I);

    // 1. Cooperative Load: Input token
    for (int d = tid; d < D; d += bdim) s_x[d] = input[t * D + d];
    for (int d = tid; d < D; d += bdim) s_y[d] = 0.0f;
    __syncthreads();

    // 2. Routing: Dot(x, gate_weight)
    if (tid < E) {
        float sum = 0.0f;
        for (int d = 0; d < D; d++) sum += s_x[d] * gate_weight[tid * D + d];
        s_logits[tid] = sum;
    }
    __syncthreads();

    // Softmax + Top-K (Single thread for simplicity, or use warp reduce)
    __shared__ int selected_experts[8]; // MAX_K=8
    __shared__ float selected_weights[8];
    if (tid == 0) {
        float mx = -FLT_MAX;
        for (int e = 0; e < E; e++) mx = fmaxf(mx, s_logits[e]);
        float sum_exp = 0.0f;
        for (int e = 0; e < E; e++) {
            s_logits[e] = expf(s_logits[e] - mx);
            sum_exp += s_logits[e];
        }
        for (int e = 0; e < E; e++) s_logits[e] /= sum_exp;

        // Top-K
        for (int k = 0; k < K; k++) {
            int best_e = -1;
            float best_v = -FLT_MAX;
            for (int e = 0; e < E; e++) {
                if (s_logits[e] > best_v) {
                    best_v = s_logits[e];
                    best_e = e;
                }
            }
            selected_experts[k] = best_e;
            selected_weights[k] = best_v;
            if (best_e >= 0) s_logits[best_e] = -1.0f;
        }
        // Normalize weights
        float w_sum = 0.0f;
        for (int k = 0; k < K; k++) w_sum += selected_weights[k];
        for (int k = 0; k < K; k++) selected_weights[k] /= w_sum;
    }
    __syncthreads();

    // 3. Expert Execution: Loop over K experts
    for (int k = 0; k < K; k++) {
        int e = selected_experts[k];
        if (e < 0) continue;
        float weight = selected_weights[k];

        // --- GEMV 1: s_act[2*I] = W1[e, 2*I, D] * s_x[D] ---
        const float* W1_e = w1 + (size_t)e * 2 * I * D;
        for (int i = tid; i < 2 * I; i += bdim) {
            float sum = 0.0f;
            for (int d = 0; d < D; d++) {
                sum += W1_e[i * D + d] * s_x[d];
            }
            // SwiGLU transition immediately
            if (i < I) {
                // This is first half (gate)
                // We'll store it and compute SwiGLU in second pass or keep state.
                s_act[i] = sum; 
            } else {
                // This is second half (up)
                int act_idx = i - I;
                float g = s_act[act_idx];
                float u = sum;
                float silu = g / (1.0f + expf(-g));
                s_act[act_idx] = silu * u; // Store final activation in first I slots
            }
        }
        __syncthreads();

        // --- GEMV 2: s_y[D] += weight * W2[e, D, I] * s_act[I] ---
        const float* W2_e = w2 + (size_t)e * D * I;
        for (int d = tid; d < D; d += bdim) {
            float sum = 0.0f;
            for (int i = 0; i < I; i++) {
                sum += W2_e[d * I + i] * s_act[i];
            }
            s_y[d] += weight * sum;
        }
        __syncthreads();
    }

    // 4. Final Write
    for (int d = tid; d < D; d += bdim) output[t * D + d] = s_y[d];
}

// 5. OPT 4: Fused Expert Implementation
// Note: fused kernel routes and accesses weights by expert id in smem, so it uses
// E_local for both routing and computation (can't cleanly separate global routing).
void moe_forward_opt4(const float* d_input, const float* d_gate_weight,
                      const float* d_w1, const float* d_w2, float* d_output,
                      const MoeConfig& cfg, cudaStream_t stream) {
    int T = cfg.num_tokens, E_local = cfg.num_local_experts;
    int D = cfg.hidden_dim, I = cfg.intermediate_dim, K = cfg.top_k;

    int threads = 128;
    // smem: D + E_local + 2*I + D (floats)
    size_t smem = (size_t)(D + E_local + 2 * I + D) * sizeof(float);

    // Allow >48KB dynamic shared memory (needed when D=7168: ~72KB)
    CUDA_CHECK(cudaFuncSetAttribute(fused_moe_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));

    fused_moe_kernel<<<T, threads, smem, stream>>>(
        d_input, d_gate_weight, d_w1, d_w2, d_output, T, E_local, D, I, K
    );
    CUDA_CHECK(cudaGetLastError());
}

#include <cuda_pipeline_primitives.h>

// ===================================================================
// OPT 5: Double-buffered grouped GEMM with software-managed pipelining.
// This overlaps tile staging for iteration N+1 with computation for N using
// ordinary shared-memory buffers and block-wide synchronization.
// ===================================================================

__global__ void grouped_gemm_blackwell_async_kernel(const float* __restrict__ A,
                                                   const float* __restrict__ W_ptr,
                                                   float* __restrict__ C,
                                                   const int* __restrict__ expert_offsets,
                                                   const int* __restrict__ m_tile_offsets,
                                                   int D, int N, int E) {
    // Hardware dispatch mapping
    int tile_m_global = blockIdx.y;
    int tile_n = blockIdx.x;

    // Binary search to find which expert this block belongs to
    int e_low = 0, e_high = E - 1;
    int e = 0;
    while (e_low <= e_high) {
        int mid = e_low + (e_high - e_low) / 2;
        if (tile_m_global >= m_tile_offsets[mid]) {
            e = mid;
            e_low = mid + 1;
        } else {
            e_high = mid - 1;
        }
    }

    int tile_m_within_e = tile_m_global - m_tile_offsets[e];
    
    int global_row_base = expert_offsets[e] + tile_m_within_e * 64;
    int global_col_base = tile_n * 64;
    
    int expert_total_rows = expert_offsets[e+1] - expert_offsets[e];
    int row_within_e_base = tile_m_within_e * 64;

    const float* W_e = W_ptr + (size_t)e * N * D;

    // Double buffers for weights (Bs) and activations (As)
    // 64x32 patches padded by 4 floats to maintain 16-byte alignment and offset memory banks
    // Overlay Cs onto As and Bs to avoid exceeding the 48KB default static SMEM structural limit!
    __shared__ union {
        struct {
            float As[2][64][36];
            float Bs[2][64][36];
        };
        float Cs[64][64];
    } smem;

    // Each warp processes a 16x32 block of the 64x64 C Tile (requires 2 fragments of 16x16)
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 8, float> c_frag[2];
    nvcuda::wmma::fill_fragment(c_frag[0], 0.0f);
    nvcuda::wmma::fill_fragment(c_frag[1], 0.0f);

    int numTiles = (D + 32 - 1) / 32;
    int tid = threadIdx.y * blockDim.x + threadIdx.x; // Block is still passed as 16x16 = 256 threads
    int warp_id = tid / 32;
    int warp_row = warp_id / 2; // 0..3 (spans 64 rows structurally)
    int warp_col = warp_id % 2; // 0..1 (spans 64 cols structurally)

    // --- PROLOGUE: Async Prefetch Tile 0 ---
    {
        // Total fetch for A: 64*32 = 2048 floats. Threads: 256. Each fetches 8 floats = 2 float4s.
        int a_idx1 = tid * 2;
        int a_idx2 = tid * 2 + 1;
        
        int a_r1 = a_idx1 / 8; // 8 float4s per 32-wide row
        int a_c1 = (a_idx1 % 8) * 4;
        bool valid_A1 = (row_within_e_base + a_r1 < expert_total_rows && (0 * 32 + a_c1) < D);
        if (valid_A1) __pipeline_memcpy_async(&smem.As[0][a_r1][a_c1], &A[(global_row_base + a_r1) * D + (0 * 32 + a_c1)], sizeof(float4));
        else { smem.As[0][a_r1][a_c1] = 0.0f; smem.As[0][a_r1][a_c1+1] = 0.0f; smem.As[0][a_r1][a_c1+2] = 0.0f; smem.As[0][a_r1][a_c1+3] = 0.0f; }

        int a_r2 = a_idx2 / 8;
        int a_c2 = (a_idx2 % 8) * 4;
        bool valid_A2 = (row_within_e_base + a_r2 < expert_total_rows && (0 * 32 + a_c2) < D);
        if (valid_A2) __pipeline_memcpy_async(&smem.As[0][a_r2][a_c2], &A[(global_row_base + a_r2) * D + (0 * 32 + a_c2)], sizeof(float4));
        else { smem.As[0][a_r2][a_c2] = 0.0f; smem.As[0][a_r2][a_c2+1] = 0.0f; smem.As[0][a_r2][a_c2+2] = 0.0f; smem.As[0][a_r2][a_c2+3] = 0.0f; }

        // Fetch B (transposed conceptually but contiguous in memory)
        int b_r1 = a_r1; int b_c1 = a_c1;
        bool valid_B1 = ((global_col_base + b_r1) < N && (0 * 32 + b_c1) < D);
        if (valid_B1) __pipeline_memcpy_async(&smem.Bs[0][b_r1][b_c1], &W_e[(global_col_base + b_r1) * D + (0 * 32 + b_c1)], sizeof(float4));
        else { smem.Bs[0][b_r1][b_c1] = 0.0f; smem.Bs[0][b_r1][b_c1+1] = 0.0f; smem.Bs[0][b_r1][b_c1+2] = 0.0f; smem.Bs[0][b_r1][b_c1+3] = 0.0f; }

        int b_r2 = a_r2; int b_c2 = a_c2;
        bool valid_B2 = ((global_col_base + b_r2) < N && (0 * 32 + b_c2) < D);
        if (valid_B2) __pipeline_memcpy_async(&smem.Bs[0][b_r2][b_c2], &W_e[(global_col_base + b_r2) * D + (0 * 32 + b_c2)], sizeof(float4));
        else { smem.Bs[0][b_r2][b_c2] = 0.0f; smem.Bs[0][b_r2][b_c2+1] = 0.0f; smem.Bs[0][b_r2][b_c2+2] = 0.0f; smem.Bs[0][b_r2][b_c2+3] = 0.0f; }

        __pipeline_commit();
    }

    // --- MAIN PIPELINE LOOP ---
    for (int t = 0; t < numTiles; t++) {
        int curr_buf = t % 2;
        int next_buf = (t + 1) % 2;
        int next_t = t + 1;

        // Initiate async fetch for Tile N+1
        if (next_t < numTiles) {
            int a_idx1 = tid * 2;
            int a_idx2 = tid * 2 + 1;
            
            int a_r1 = a_idx1 / 8;
            int a_c1 = (a_idx1 % 8) * 4;
            bool valid_A1 = (row_within_e_base + a_r1 < expert_total_rows && (next_t * 32 + a_c1) < D);
            if (valid_A1) __pipeline_memcpy_async(&smem.As[next_buf][a_r1][a_c1], &A[(global_row_base + a_r1) * D + (next_t * 32 + a_c1)], sizeof(float4));
            else { smem.As[next_buf][a_r1][a_c1] = 0.0f; smem.As[next_buf][a_r1][a_c1+1] = 0.0f; smem.As[next_buf][a_r1][a_c1+2] = 0.0f; smem.As[next_buf][a_r1][a_c1+3] = 0.0f; }

            int a_r2 = a_idx2 / 8;
            int a_c2 = (a_idx2 % 8) * 4;
            bool valid_A2 = (row_within_e_base + a_r2 < expert_total_rows && (next_t * 32 + a_c2) < D);
            if (valid_A2) __pipeline_memcpy_async(&smem.As[next_buf][a_r2][a_c2], &A[(global_row_base + a_r2) * D + (next_t * 32 + a_c2)], sizeof(float4));
            else { smem.As[next_buf][a_r2][a_c2] = 0.0f; smem.As[next_buf][a_r2][a_c2+1] = 0.0f; smem.As[next_buf][a_r2][a_c2+2] = 0.0f; smem.As[next_buf][a_r2][a_c2+3] = 0.0f; }

            int b_r1 = a_r1; int b_c1 = a_c1;
            bool valid_B1 = ((global_col_base + b_r1) < N && (next_t * 32 + b_c1) < D);
            if (valid_B1) __pipeline_memcpy_async(&smem.Bs[next_buf][b_r1][b_c1], &W_e[(global_col_base + b_r1) * D + (next_t * 32 + b_c1)], sizeof(float4));
            else { smem.Bs[next_buf][b_r1][b_c1] = 0.0f; smem.Bs[next_buf][b_r1][b_c1+1] = 0.0f; smem.Bs[next_buf][b_r1][b_c1+2] = 0.0f; smem.Bs[next_buf][b_r1][b_c1+3] = 0.0f; }

            int b_r2 = a_r2; int b_c2 = a_c2;
            bool valid_B2 = ((global_col_base + b_r2) < N && (next_t * 32 + b_c2) < D);
            if (valid_B2) __pipeline_memcpy_async(&smem.Bs[next_buf][b_r2][b_c2], &W_e[(global_col_base + b_r2) * D + (next_t * 32 + b_c2)], sizeof(float4));
            else { smem.Bs[next_buf][b_r2][b_c2] = 0.0f; smem.Bs[next_buf][b_r2][b_c2+1] = 0.0f; smem.Bs[next_buf][b_r2][b_c2+2] = 0.0f; smem.Bs[next_buf][b_r2][b_c2+3] = 0.0f; }

            __pipeline_commit();
        }

        // Wait for Tile N to finish loading from GMEM -> SMEM
        __pipeline_wait_prior(0);
        __syncthreads();

        // TF32 Tensor Core Math across all 8 Warps handling the 64x64 output block
        for (int k_idx = 0; k_idx < 32; k_idx += 8) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major> a_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major> b_frag1;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 8, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major> b_frag2;

            nvcuda::wmma::load_matrix_sync(a_frag, &smem.As[curr_buf][warp_row * 16][k_idx], 36);
            nvcuda::wmma::load_matrix_sync(b_frag1, &smem.Bs[curr_buf][warp_col * 32 + 0][k_idx], 36);
            nvcuda::wmma::load_matrix_sync(b_frag2, &smem.Bs[curr_buf][warp_col * 32 + 16][k_idx], 36);

            nvcuda::wmma::mma_sync(c_frag[0], a_frag, b_frag1, c_frag[0]);
            nvcuda::wmma::mma_sync(c_frag[1], a_frag, b_frag2, c_frag[1]);
        }
        __syncthreads(); 
    }
    
    // Distribute results safely from all warps to SMEM buffer
    nvcuda::wmma::store_matrix_sync(&smem.Cs[warp_row * 16][warp_col * 32 + 0], c_frag[0], 64, nvcuda::wmma::mem_row_major);
    nvcuda::wmma::store_matrix_sync(&smem.Cs[warp_row * 16][warp_col * 32 + 16], c_frag[1], 64, nvcuda::wmma::mem_row_major);
    __syncthreads();
    
    // Unloaded safely 4 elements per thread to global memory (each thread covers 1/256th of the 4096 elements)
    // 4096 / 256 = 16 elements. Which means 4 float4s perfectly mapped for aligned global memory pushes!
    // Since we need to write to C, we can just write back normally!
    int linear_idx = tid;
    for (int i = 0; i < 16; i++) {
        int idx = linear_idx * 16 + i;
        int crow = idx / 64;
        int ccol = idx % 64;
        if (row_within_e_base + crow < expert_total_rows && global_col_base + ccol < N) {
            C[(global_row_base + crow) * N + (global_col_base + ccol)] = smem.Cs[crow][ccol];
        }
    }
}

// 6. OPT 5: Double-buffered grouped GEMM implementation
void moe_forward_opt5(const float* d_input, const float* d_gate_weight,
                      const float* d_w1, const float* d_w2, float* d_output,
                      const MoeConfig& cfg, cudaStream_t stream) {
    int T = cfg.num_tokens, E = cfg.num_experts, E_local = cfg.num_local_experts;
    int D = cfg.hidden_dim, I = cfg.intermediate_dim, K = cfg.top_k;

    // Reuse Grouped logic from Opt 3 (Indices/Offsets/Reorder)
    DeviceBuf<int>   expert_indices(T * K);
    DeviceBuf<float> expert_weights(T * K);
    moe_gate(d_input, d_gate_weight, expert_indices.ptr, expert_weights.ptr, cfg, stream);

    DeviceBuf<int> expert_counts(E + 1);
    expert_counts.zero();
    DeviceBuf<int> token_idx_in_expert(T * K);
    expert_grouping_kernel<<<(T * K + 255) / 256, 256, 0, stream>>>(
        expert_indices.ptr, expert_counts.ptr, token_idx_in_expert.ptr, T, K, E
    );

    std::vector<int> h_counts(E + 1);
    CUDA_CHECK(cudaMemcpyAsync(h_counts.data(), expert_counts.ptr, E * sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<int> h_offsets(E + 1);
    h_offsets[0] = 0;
    for (int i = 0; i < E; i++) h_offsets[i+1] = h_offsets[i] + h_counts[i];
    DeviceBuf<int> expert_offsets(E + 1);
    expert_offsets.upload(h_offsets.data());

    int total_local = h_offsets[E_local];
    if (total_local == 0) return;

    // Calculate m_tile_offsets for accurate grouped GEMM bounds checking
    std::vector<int> h_m_tile_offsets(E_local + 1, 0);
    for (int i = 0; i < E_local; i++) {
        int tiles = (h_counts[i] + 64 - 1) / 64;
        h_m_tile_offsets[i+1] = h_m_tile_offsets[i] + tiles;
    }
    DeviceBuf<int> m_tile_offsets(E_local + 1);
    CUDA_CHECK(cudaMemcpyAsync(m_tile_offsets.ptr, h_m_tile_offsets.data(), (E_local + 1) * sizeof(int), cudaMemcpyHostToDevice, stream));

    int total_active = h_offsets[E];
    DeviceBuf<float> grouped_input(total_active * D);
    DeviceBuf<int>   grouped_token_map(total_active);
    group_reorder_kernel<<<(T * K + 255) / 256, 256, 0, stream>>>(
        d_input, expert_indices.ptr, expert_offsets.ptr, token_idx_in_expert.ptr,
        grouped_input.ptr, grouped_token_map.ptr, T, K, D
    );

    CUDA_CHECK(cudaMemsetAsync(d_output, 0, T * D * sizeof(float), stream));
    DeviceBuf<float> g_gemm1_out(total_local * 2 * I), g_act_out(total_local * I), g_gemm2_out(total_local * D);

    // Double-buffered + cp.async Grouped GEMM 1
    {
        dim3 block(256);
        int total_m_tiles = h_m_tile_offsets[E_local];
        dim3 grid((2 * I + 64 - 1) / 64, total_m_tiles);
        if (total_m_tiles > 0) {
            grouped_gemm_blackwell_async_kernel<<<grid, block, 0, stream>>>(
                grouped_input.ptr, d_w1, g_gemm1_out.ptr, expert_offsets.ptr, m_tile_offsets.ptr,
                D, 2 * I, E_local);
        }
    }

    swiglu_strided_kernel<<<(total_local * I + 255) / 256, 256, 0, stream>>>(g_gemm1_out.ptr, g_act_out.ptr, total_local, I);

    // Double-buffered + cp.async Grouped GEMM 2
    {
        dim3 block(256);
        int total_m_tiles = h_m_tile_offsets[E_local];
        dim3 grid((D + 64 - 1) / 64, total_m_tiles);
        if (total_m_tiles > 0) {
            grouped_gemm_blackwell_async_kernel<<<grid, block, 0, stream>>>(
                g_act_out.ptr, d_w2, g_gemm2_out.ptr, expert_offsets.ptr, m_tile_offsets.ptr,
                I, D, E_local);
        }
    }

    grouped_scatter_kernel<<<total_local, min(D, 1024), 0, stream>>>(
        g_gemm2_out.ptr, grouped_token_map.ptr, expert_weights.ptr,
        expert_indices.ptr, expert_offsets.ptr, d_output, T, K, D, E_local
    );
}

void moe_forward(const float* d_input, const float* d_gate_weight,
                 const float* d_w1, const float* d_w2, float* d_output,
                 const MoeConfig& cfg, cudaStream_t stream) {
    moe_forward_opt5(d_input, d_gate_weight, d_w1, d_w2, d_output, cfg, stream);
}
