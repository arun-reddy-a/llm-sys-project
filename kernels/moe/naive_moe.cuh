#pragma once

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// MoE configuration
// ---------------------------------------------------------------------------

struct MoeConfig {
    int num_tokens;        // T – number of input tokens
    int num_experts;       // E – total experts
    int top_k;             // K – experts selected per token
    int hidden_dim;        // D – model hidden dimension
    int intermediate_dim;  // I – FFN intermediate dimension (per expert)
};

// ---------------------------------------------------------------------------
// Kernel declarations
// ---------------------------------------------------------------------------

// Gate / routing: computes logits = input * gate_weight^T, then per-token
// softmax over experts, then selects top-K experts.
//   input          [T, D]
//   gate_weight    [E, D]
//   expert_indices [T, K]   (output – selected expert ids)
//   expert_weights [T, K]   (output – gating weights after softmax + topk)
void moe_gate(const float* input, const float* gate_weight,
              int* expert_indices, float* expert_weights,
              const MoeConfig& cfg, cudaStream_t stream = 0);

// Gather tokens assigned to a specific expert into a contiguous buffer.
//   input          [T, D]
//   expert_indices [T, K]
//   expert_weights [T, K]
//   gathered       [max_tokens_per_expert, D]  (output)
//   token_map      [max_tokens_per_expert]      (output – original token idx)
//   count          [1]                           (output – actual count)
void moe_gather(const float* input, const int* expert_indices,
                const float* expert_weights, int expert_id,
                float* gathered, int* token_map, int* count,
                const MoeConfig& cfg, cudaStream_t stream = 0);

// Naive GEMM: C[M,N] = A[M,K] * B[K,N]   (row-major)
void naive_gemm(const float* A, const float* B, float* C,
                int M, int N, int K, cudaStream_t stream = 0);

// SwiGLU activation: out[i] = silu(gate[i]) * up[i]
//   gate, up  [n]    (in-place: reads gate & up, writes out)
//   out       [n]
void swiglu(const float* gate, const float* up, float* out,
            int n, cudaStream_t stream = 0);

// Scatter expert output back, weighted by gating weight.
//   expert_out  [count, D]
//   token_map   [count]
//   weights     [T, K]  (full gating weights)
//   expert_indices [T, K]
//   output      [T, D]  (accumulated)
void moe_scatter(const float* expert_out, const int* token_map,
                 const float* expert_weights, const int* expert_indices,
                 int expert_id, float* output, int count,
                 const MoeConfig& cfg, cudaStream_t stream = 0);

// ---------------------------------------------------------------------------
// Full naive MoE forward pass (host-side orchestrator)
// ---------------------------------------------------------------------------

// Runs the complete MoE layer: gate -> per-expert gather/gemm1/swiglu/gemm2/scatter
//   input        [T, D]
//   gate_weight  [E, D]
//   w1           [E, 2*I, D]  (up-projection; first I rows = gate, next I = up)
//   w2           [E, D, I]    (down-projection)
//   output       [T, D]
void moe_forward(const float* input, const float* gate_weight,
                 const float* w1, const float* w2, float* output,
                 const MoeConfig& cfg, cudaStream_t stream = 0);
