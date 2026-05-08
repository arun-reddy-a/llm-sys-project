#include "moe_forward.cuh"
#include "../../utils/cuda_utils.cuh"
#include <cfloat>
#include <cublas_v2.h>

// ---------------------------------------------------------------------------
// DeepSeek-V3 "no-aux" routing selection
//   sigmoid → bias → group top-K → global top-K → sigmoid-normalize.
//   One block per token; single-threaded (O(E)=O(256) is fast).
// ---------------------------------------------------------------------------
__global__ void deepseek_selection_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ gate_bias,
    int* expert_ids, float* expert_wts,
    int T, int E_global, int K,
    int N_GROUP, int TOPK_GROUP,
    float routed_scaling_factor)
{
    int t = blockIdx.x;
    if (t >= T || threadIdx.x != 0) return;

    int group_size = E_global / N_GROUP;
    float s_sigmoid[256], s_wb[256], s_group_scores[8];

    for (int e = 0; e < E_global; e++) {
        float s = 1.0f / (1.0f + expf(-logits[t * E_global + e]));
        s_sigmoid[e] = s;
        s_wb[e]      = s + gate_bias[e];
    }

    for (int gn = 0; gn < N_GROUP; gn++) {
        float top1 = -FLT_MAX, top2 = -FLT_MAX;
        for (int i = 0; i < group_size; i++) {
            float v = s_wb[gn * group_size + i];
            if (v > top1) { top2 = top1; top1 = v; }
            else if (v > top2) { top2 = v; }
        }
        s_group_scores[gn] = top1 + top2;
    }

    int top_groups[8];
    for (int i = 0; i < N_GROUP; i++) top_groups[i] = i;
    for (int i = 0; i < TOPK_GROUP; i++) {
        int best = i;
        for (int j = i + 1; j < N_GROUP; j++) {
            float sj = s_group_scores[top_groups[j]], sb = s_group_scores[top_groups[best]];
            if (sj > sb || (sj == sb && top_groups[j] < top_groups[best])) best = j;
        }
        int tmp = top_groups[i]; top_groups[i] = top_groups[best]; top_groups[best] = tmp;
    }

    int*   out_idx = expert_ids + t * K;
    float* out_wt  = expert_wts + t * K;
    for (int k = 0; k < K; k++) { out_idx[k] = -1; out_wt[k] = -FLT_MAX; }

    for (int i = 0; i < TOPK_GROUP; i++) {
        int gn = top_groups[i];
        for (int ei = 0; ei < group_size; ei++) {
            int e = gn * group_size + ei;
            float v = s_wb[e];
            int min_k = 0;
            for (int k = 1; k < K; k++) if (out_wt[k] < out_wt[min_k]) min_k = k;
            if (v > out_wt[min_k] || (v == out_wt[min_k] && e < out_idx[min_k]))
                { out_wt[min_k] = v; out_idx[min_k] = e; }
        }
    }

    float w_sum = 0.0f;
    for (int k = 0; k < K; k++) {
        int e = out_idx[k];
        if (e >= 0) { out_wt[k] = s_sigmoid[e]; w_sum += s_sigmoid[e]; }
    }
    w_sum += 1e-20f;
    for (int k = 0; k < K; k++)
        out_wt[k] = (out_idx[k] >= 0) ? out_wt[k] / w_sum * routed_scaling_factor : 0.0f;
}

// ---------------------------------------------------------------------------
// Softmax routing (matches competition reference implementation)
//   softmax(logits) → top-K → renormalize weights × routed_scaling_factor.
//   One block per token; runs entirely in shared memory for E ≤ 1024.
// ---------------------------------------------------------------------------
__global__ void softmax_topk_kernel(
    const float* __restrict__ logits,
    int* expert_ids, float* expert_wts,
    int T, int E, int K, float routed_scaling_factor)
{
    int t = blockIdx.x;
    if (t >= T || threadIdx.x != 0) return;

    const float* row = logits + t * E;
    int*   out_idx = expert_ids + t * K;
    float* out_wt  = expert_wts + t * K;

    // Stable softmax
    float mx = -FLT_MAX;
    for (int e = 0; e < E; e++) mx = fmaxf(mx, row[e]);
    float sum = 0.0f;
    // Reuse out_wt as scratch for softmax probs (overwritten at end)
    float probs[256]; // E ≤ 256 for DeepSeek
    for (int e = 0; e < E; e++) { probs[e] = expf(row[e] - mx); sum += probs[e]; }
    for (int e = 0; e < E; e++) probs[e] /= sum;

    // Top-K selection
    for (int k = 0; k < K; k++) { out_idx[k] = -1; out_wt[k] = -FLT_MAX; }
    for (int e = 0; e < E; e++) {
        float v = probs[e];
        int min_k = 0;
        for (int k = 1; k < K; k++) if (out_wt[k] < out_wt[min_k]) min_k = k;
        if (v > out_wt[min_k]) { out_wt[min_k] = v; out_idx[min_k] = e; }
    }

    // Renormalize × routed_scaling_factor
    float w_sum = 0.0f;
    for (int k = 0; k < K; k++) if (out_idx[k] >= 0) w_sum += out_wt[k];
    w_sum += 1e-20f;
    for (int k = 0; k < K; k++)
        out_wt[k] = (out_idx[k] >= 0) ? out_wt[k] / w_sum * routed_scaling_factor : 0.0f;
}

// ---------------------------------------------------------------------------
// Host: moe_gate_deepseek — cuBLAS gate GEMM + DeepSeek selection
// ---------------------------------------------------------------------------
void moe_gate_deepseek(
    const float* input, const float* gate_weight, const float* gate_bias,
    int* expert_ids, float* expert_wts,
    const MoeConfig& cfg, cudaStream_t stream)
{
    int T = cfg.num_tokens, E = cfg.num_experts, D = cfg.hidden_dim;

    static cublasHandle_t handle = nullptr;
    if (!handle) cublasCreate(&handle);
    cublasSetStream(handle, stream);

    static DeviceBuf<float> s_logits;
    s_logits.resize(T * E);

    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                E, T, D,
                &alpha, gate_weight, D, input, D,
                &beta,  s_logits.ptr, E);

    deepseek_selection_kernel<<<T, 1, 0, stream>>>(
        s_logits.ptr, gate_bias, expert_ids, expert_wts,
        T, E, cfg.top_k, cfg.n_group, cfg.topk_group, cfg.routed_scaling_factor);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Host: moe_gate_softmax — softmax routing directly from pre-computed logits
// ---------------------------------------------------------------------------
void moe_gate_softmax(
    const float* routing_logits,
    int* expert_ids, float* expert_wts,
    const MoeConfig& cfg, cudaStream_t stream)
{
    int T = cfg.num_tokens, E = cfg.num_experts;
    softmax_topk_kernel<<<T, 1, 0, stream>>>(
        routing_logits, expert_ids, expert_wts,
        T, E, cfg.top_k, cfg.routed_scaling_factor);
    CUDA_CHECK(cudaGetLastError());
}
