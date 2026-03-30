#include "naive_moe.cuh"
#include "../../utils/cuda_utils.cuh"
#include <cfloat>
#include <cstdio>

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
    int d   = threadIdx.x;
    if (idx >= count || d >= D) return;

    int t = token_map[idx];

    float w = 0.0f;
    for (int k = 0; k < K; k++) {
        if (expert_indices[t * K + k] == expert_id) {
            w = expert_weights[t * K + k];
            break;
        }
    }

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
    scatter_kernel<<<count, D, 0, stream>>>(expert_out, token_map,
                                             expert_weights, expert_indices,
                                             expert_id, output, count,
                                             cfg.num_tokens, cfg.top_k, D);
    CUDA_CHECK(cudaGetLastError());
}

// ===================================================================
// Full MoE forward pass (with tiled GEMM + fused routing)
// ===================================================================
void moe_forward(const float* d_input, const float* d_gate_weight,
                 const float* d_w1, const float* d_w2, float* d_output,
                 const MoeConfig& cfg, cudaStream_t stream) {
    int T = cfg.num_tokens;
    int E = cfg.num_experts;
    int D = cfg.hidden_dim;
    int I = cfg.intermediate_dim;

    DeviceBuf<int>   expert_indices(T * cfg.top_k);
    DeviceBuf<float> expert_weights(T * cfg.top_k);
    moe_gate(d_input, d_gate_weight, expert_indices.ptr, expert_weights.ptr,
             cfg, stream);

    CUDA_CHECK(cudaMemsetAsync(d_output, 0, T * D * sizeof(float), stream));

    DeviceBuf<float> gathered(T * D);
    DeviceBuf<int>   token_map(T);
    DeviceBuf<int>   d_count(1);
    DeviceBuf<float> gemm1_out(T * 2 * I);
    DeviceBuf<float> act_out(T * I);
    DeviceBuf<float> gemm2_out(T * D);

    int h_count = 0;

    for (int e = 0; e < E; e++) {
        moe_gather(d_input, expert_indices.ptr, expert_weights.ptr, e,
                   gathered.ptr, token_map.ptr, d_count.ptr, cfg, stream);
        CUDA_CHECK(cudaMemcpyAsync(&h_count, d_count.ptr, sizeof(int),
                                    cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (h_count == 0) continue;

        const float* w1_e = d_w1 + (size_t)e * 2 * I * D;
        const float* w2_e = d_w2 + (size_t)e * D * I;

        // GEMM1: gathered[count, D] * w1_e^T -> [count, 2*I]
        {
            int M_ = h_count, N_ = 2 * I, K_ = D;
            dim3 block(TILE_SIZE, TILE_SIZE);
            dim3 grid((N_ + TILE_SIZE - 1) / TILE_SIZE,
                      (M_ + TILE_SIZE - 1) / TILE_SIZE);
            tiled_gemm_bt_kernel<<<grid, block, 0, stream>>>(
                gathered.ptr, w1_e, gemm1_out.ptr, M_, N_, K_);
            CUDA_CHECK(cudaGetLastError());
        }

        // SwiGLU on [count, 2*I] -> [count, I]
        {
            int n = h_count * I;
            int thr = 256;
            int blk = (n + thr - 1) / thr;
            swiglu_strided_kernel<<<blk, thr, 0, stream>>>(
                gemm1_out.ptr, act_out.ptr, h_count, I);
            CUDA_CHECK(cudaGetLastError());
        }

        // GEMM2: act_out[count, I] * w2_e^T -> [count, D]
        {
            int M_ = h_count, N_ = D, K_ = I;
            dim3 block(TILE_SIZE, TILE_SIZE);
            dim3 grid((N_ + TILE_SIZE - 1) / TILE_SIZE,
                      (M_ + TILE_SIZE - 1) / TILE_SIZE);
            tiled_gemm_bt_kernel<<<grid, block, 0, stream>>>(
                act_out.ptr, w2_e, gemm2_out.ptr, M_, N_, K_);
            CUDA_CHECK(cudaGetLastError());
        }

        moe_scatter(gemm2_out.ptr, token_map.ptr, expert_weights.ptr,
                    expert_indices.ptr, e, d_output, h_count, cfg, stream);
    }
}
