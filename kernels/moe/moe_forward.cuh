#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// ---------------------------------------------------------------------------
// Problem configuration (DeepSeek-V3 MoE)
// ---------------------------------------------------------------------------
struct MoeConfig {
    int   num_tokens;           // T  — sequence length
    int   num_experts;          // E  — total experts (routing dimension, 256)
    int   num_local_experts;    // E_local — experts on this GPU (32)
    int   top_k;                // K  — experts per token (8)
    int   hidden_dim;           // D  — model hidden dimension (7168)
    int   intermediate_dim;     // I  — FFN intermediate dimension (2048)
    int   n_group;              // number of expert groups (8)
    int   topk_group;           // groups selected per token (4)
    float routed_scaling_factor; // weight renormalization scale (2.5)
};

// ---------------------------------------------------------------------------
// Gate (routing) — two variants
// ---------------------------------------------------------------------------

// DeepSeek-V3 production routing:
//   cuBLAS gate GEMM → sigmoid → bias-add → group top-K → global top-K → normalize.
//   input        [T, D]   FP32
//   gate_weight  [E, D]   FP32
//   gate_bias    [E]      FP32
//   expert_ids   [T, K]   int32   (output)
//   expert_wts   [T, K]   FP32    (output, normalized routing weights)
void moe_gate_deepseek(
    const float* input, const float* gate_weight, const float* gate_bias,
    int* expert_ids, float* expert_wts,
    const MoeConfig& cfg, cudaStream_t stream = 0);

// Softmax routing (matches competition reference):
//   softmax(routing_logits, dim=-1) → top-K → renormalize × routed_scaling_factor.
//   routing_logits  [T, E]   FP32   (pre-computed gate logits)
//   expert_ids      [T, K]   int32  (output)
//   expert_wts      [T, K]   FP32   (output, normalized)
void moe_gate_softmax(
    const float* routing_logits,
    int* expert_ids, float* expert_wts,
    const MoeConfig& cfg, cudaStream_t stream = 0);

// ---------------------------------------------------------------------------
// BF16 grouped-GEMM forward — WMMA single-kernel variant
//   Uses a hand-written BF16 WMMA double-buffered grouped GEMM (64×64 tiles)
//   with CPU-precomputed tile→expert mapping.
// ---------------------------------------------------------------------------
void moe_forward_deepseek_bf16(
    const __nv_bfloat16* input,       // [T, D]
    const float*         gate_weight, // [E, D]
    const float*         gate_bias,   // [E]
    const __nv_bfloat16* w1,          // [E_local, 2*I, D]
    const __nv_bfloat16* w2,          // [E_local, D, I]
    __nv_bfloat16*       output,      // [T, D]
    const MoeConfig& cfg, cudaStream_t stream = 0);

// ---------------------------------------------------------------------------
// BF16 grouped-GEMM forward — cuBLAS per-expert variant
//   Issues one cublasGemmEx (BF16 → FP32 accumulate) per local expert.
// ---------------------------------------------------------------------------
void moe_forward_deepseek_bf16_cublas(
    const __nv_bfloat16* input,       // [T, D]
    const float*         gate_weight, // [E, D]
    const float*         gate_bias,   // [E]
    const __nv_bfloat16* w1,          // [E_local, 2*I, D]
    const __nv_bfloat16* w2,          // [E_local, D, I]
    __nv_bfloat16*       output,      // [T, D]
    const MoeConfig& cfg, cudaStream_t stream = 0);

// ---------------------------------------------------------------------------
// FP8 block-scale forward — competition interface
//   Matches the Triton competition entry exactly:
//   softmax routing → gather FP8 tokens → FP8 grouped GEMM1 → SwiGLU →
//   FP8 requant → FP8 grouped GEMM2 → weighted scatter → BF16 output.
//
//   Block size: 128 elements. Scales are per-(row-block, col-block).
//
//   routing_logits      [T, E_global]                FP32   (pre-computed)
//   w1                  [E_local, 2I, D]             FP8 E4M3
//   w1_scale            [E_local, 2I/128, D/128]     FP32
//   w2                  [E_local, D, I]              FP8 E4M3
//   w2_scale            [E_local, D/128, I/128]      FP32
//   local_expert_offset  scalar                       int
//   routed_scaling_factor scalar                      float
//   output              [T, D]                        BF16
//   hidden_states       [T, D]                        FP8 E4M3
//   hidden_states_scale [D/128, T]                    FP32
// ---------------------------------------------------------------------------
void moe_forward_fp8(
    const float*            routing_logits,
    const __nv_fp8_e4m3*    w1,
    const float*            w1_scale,
    const __nv_fp8_e4m3*    w2,
    const float*            w2_scale,
    int                     local_expert_offset,
    float                   routed_scaling_factor,
    __nv_bfloat16*          output,
    const __nv_fp8_e4m3*    hidden_states,
    const float*            hidden_states_scale,
    const MoeConfig&        cfg,
    cudaStream_t            stream = 0);

// ---------------------------------------------------------------------------
// FP8 weights + FP32 input/output + DeepSeek routing (test_fp8_cuda.cu interface).
//   input       [T, D]                 FP32
//   gate_weight [E, D]                 FP32
//   gate_bias   [E]                    FP32
//   w1          [E_local, 2I, D]       FP8 E4M3
//   w1_scale    [E_local, 2I/128, D/128] FP32
//   w2          [E_local, D, I]        FP8 E4M3
//   w2_scale    [E_local, D/128, I/128] FP32
//   output      [T, D]                 FP32
// ---------------------------------------------------------------------------
void moe_forward_deepseek_fp8(
    const float*            input,
    const float*            gate_weight,
    const float*            gate_bias,
    const __nv_fp8_e4m3*    w1,
    const float*            w1_scale,
    const __nv_fp8_e4m3*    w2,
    const float*            w2_scale,
    float*                  output,
    const MoeConfig&        cfg,
    cudaStream_t            stream = 0);

// ---------------------------------------------------------------------------
// FP32 compatibility alias — converts weights internally, calls cuBLAS variant.
// For best performance use the BF16 or FP8 variants directly.
// ---------------------------------------------------------------------------
void moe_forward(
    const float* input, const float* gate_weight, const float* gate_bias,
    const float* w1, const float* w2, float* output,
    const MoeConfig& cfg, cudaStream_t stream = 0);
