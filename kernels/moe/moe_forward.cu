// ---------------------------------------------------------------------------
// TODO: Custom CUDA FP8 Grouped GEMM (moe_forward_fp8 / moe_forward_deepseek_fp8)
//
// Current implementation dequantizes FP8→BF16 before calling cuBLAS, which
// defeats the purpose of FP8 (wastes memory bandwidth, misses tensor-core
// speedup).  The correct path is an inline WGMMA kernel:
//
//  1. Tile layout — same precomputed tile_exp[]/tile_m_start[]/tile_m_end[]
//     approach already used in grouped_gemm_bf16_kernel.
//
//  2. Grid — (total_m_tiles, N/128):  one CTA per [BLOCK_M × 128] output tile.
//     All experts share one launch (no per-expert dispatch).
//
//  3. Inner loop — for each 128-wide K-block:
//       a_frag = LDGSTS FP8 [BLOCK_M, 128]   via TMA / async copy
//       b_frag = LDGSTS FP8 [128, 128]        (transposed weight tile)
//       acc    += a_scale * b_scale * WGMMA(a_frag, b_frag)   // FP32 acc
//     Scales applied to FP32 accumulator, not operands.
//
//  4. Precision — use SM90a WGMMA with .f8f8f32 variant (PTX 8.x+).
//     Requires __CUDA_ARCH__ >= 900.  Alternatively use CUTLASS 3.x
//     CollectiveMainloop with Sm90TmaGmmaRmemAFp8 policy (avoids raw PTX).
//
//  5. Double-buffering — two SMEM ping-pong buffers + cp.async.commit_group /
//     cp.async.wait_group to overlap GMEM→SMEM with WGMMA compute.
//
//  Reference: Triton version in solution/triton/kernel.py — the logic is
//  identical; replace tl.dot+num_stages with WGMMA+TMA.
//  Target: match or beat Triton at all T (currently 1.9× slower at T≥2048).
// ---------------------------------------------------------------------------

#include "moe_forward.cuh"
#include "../../utils/cuda_utils.cuh"
#include <vector>
#include <cublas_v2.h>
#include <cuda_fp8.h>

// ---------------------------------------------------------------------------
// Forward declarations of device-kernel launchers (defined in moe_kernels.cu)
// ---------------------------------------------------------------------------
void launch_expert_grouping(const int*, int*, int*, int, int, cudaStream_t);
void launch_group_reorder_bf16(const __nv_bfloat16*, const int*, const float*,
                                const int*, const int*, __nv_bfloat16*, int*, float*,
                                int, int, int, int, cudaStream_t);
void launch_group_reorder_fp8(const __nv_fp8_e4m3*, const float*,
                               const int*, const float*, const int*, const int*,
                               __nv_bfloat16*, int*, float*, int, int, int, int, int, cudaStream_t);
void launch_bf16_to_fp32(const __nv_bfloat16*, float*, int, cudaStream_t);
void launch_fp32_to_bf16(const float*, __nv_bfloat16*, int, cudaStream_t);
void launch_dequant_fp8(const __nv_fp8_e4m3*, const float*, __nv_bfloat16*, int, int, int, cudaStream_t);
void launch_grouped_gemm_bf16(const __nv_bfloat16*, const __nv_bfloat16*, float*,
                               const int*, const int*, const int*, int, int, int, cudaStream_t);
void launch_swiglu_fp32(const float*, float*, int, int, cudaStream_t);
void launch_swiglu_bf16(const __nv_bfloat16*, __nv_bfloat16*, int, int, cudaStream_t);
void launch_requant_fp8(const __nv_bfloat16*, __nv_fp8_e4m3*, float*, int, int, int, cudaStream_t);
void launch_scatter(const float*, const int*, const float*, float*, int, int, cudaStream_t);

// Forward declarations from moe_routing.cu
void moe_gate_deepseek(const float*, const float*, const float*, int*, float*, const MoeConfig&, cudaStream_t);
void moe_gate_softmax(const float*, int*, float*, const MoeConfig&, cudaStream_t);

// ---------------------------------------------------------------------------
// Shared helper: compute expert offsets + tile metadata from host counts.
// Returns total_local (tokens assigned to local experts) and total_active.
// Fills h_off[E+1], h_texp, h_trbase (tile→expert and tile→row_base arrays).
// ---------------------------------------------------------------------------
static void compute_offsets_and_tiles(
    const std::vector<int>& h_cnt, int E, int E_local, int tile_size,
    std::vector<int>& h_off,
    std::vector<int>& h_texp, std::vector<int>& h_trbase,
    int& total_m_tiles)
{
    h_off.assign(E + 1, 0);
    for (int i = 0; i < E; i++) h_off[i + 1] = h_off[i] + h_cnt[i];
    h_texp.clear(); h_trbase.clear(); total_m_tiles = 0;
    for (int e = 0; e < E_local; e++) {
        int n_tiles = (h_cnt[e] + tile_size - 1) / tile_size;
        for (int t = 0; t < n_tiles; t++) {
            h_texp.push_back(e);
            h_trbase.push_back(h_off[e] + t * tile_size);
        }
        total_m_tiles += n_tiles;
    }
}

// ---------------------------------------------------------------------------
// BF16 WMMA grouped GEMM forward pass.
// ---------------------------------------------------------------------------
void moe_forward_deepseek_bf16(
    const __nv_bfloat16* input,
    const float*         gate_weight,
    const float*         gate_bias,
    const __nv_bfloat16* w1,
    const __nv_bfloat16* w2,
    __nv_bfloat16*       output,
    const MoeConfig& cfg, cudaStream_t stream)
{
    int T = cfg.num_tokens, E = cfg.num_experts, E_local = cfg.num_local_experts;
    int D = cfg.hidden_dim, I = cfg.intermediate_dim, K = cfg.top_k;
    const int TILE = 64;

    static DeviceBuf<int>           s_idx, s_cnt, s_slot, s_off, s_tok, s_texp, s_trbase;
    static DeviceBuf<float>         s_wts, s_ifp32, s_g1fp32, s_g2fp32, s_ofp32, s_wtmap;
    static DeviceBuf<__nv_bfloat16> s_gin, s_g1bf16, s_act;

    // Gate (DeepSeek routing)
    s_ifp32.resize(T * D);
    launch_bf16_to_fp32(input, s_ifp32.ptr, T * D, stream);
    s_idx.resize(T * K); s_wts.resize(T * K);
    moe_gate_deepseek(s_ifp32.ptr, gate_weight, gate_bias, s_idx.ptr, s_wts.ptr, cfg, stream);

    // Count + offsets
    s_cnt.resize(E + 1); s_cnt.zero(stream); s_slot.resize(T * K);
    launch_expert_grouping(s_idx.ptr, s_cnt.ptr, s_slot.ptr, T * K, E, stream);

    static std::vector<int> h_cnt, h_off, h_texp, h_trbase;
    h_cnt.resize(E + 1);
    CUDA_CHECK(cudaMemcpyAsync(h_cnt.data(), s_cnt.ptr, E * sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int total_m_tiles;
    compute_offsets_and_tiles(h_cnt, E, E_local, TILE, h_off, h_texp, h_trbase, total_m_tiles);
    s_off.resize(E + 1); s_off.upload(h_off.data(), stream);
    int total_local = h_off[E_local];
    if (total_local == 0) {
        CUDA_CHECK(cudaMemsetAsync(output, 0, (size_t)T * D * sizeof(__nv_bfloat16), stream));
        return;
    }
    s_texp.resize(total_m_tiles);   s_texp.upload(h_texp.data(), stream);
    s_trbase.resize(total_m_tiles); s_trbase.upload(h_trbase.data(), stream);

    // Gather
    int total_active = h_off[E];
    s_gin.resize(total_active * D); s_tok.resize(total_active); s_wtmap.resize(total_active);
    launch_group_reorder_bf16(input, s_idx.ptr, s_wts.ptr, s_off.ptr, s_slot.ptr,
                               s_gin.ptr, s_tok.ptr, s_wtmap.ptr, T, K, D, T * K, stream);

    // GEMM1 → SwiGLU → GEMM2
    s_g1fp32.resize(total_local * 2 * I);
    launch_grouped_gemm_bf16(s_gin.ptr, w1, s_g1fp32.ptr, s_off.ptr, s_texp.ptr, s_trbase.ptr,
                              D, 2 * I, total_m_tiles, stream);

    s_g1bf16.resize(total_local * 2 * I);
    launch_fp32_to_bf16(s_g1fp32.ptr, s_g1bf16.ptr, total_local * 2 * I, stream);
    s_act.resize(total_local * I);
    launch_swiglu_bf16(s_g1bf16.ptr, s_act.ptr, total_local, I, stream);

    s_g2fp32.resize(total_local * D);
    launch_grouped_gemm_bf16(s_act.ptr, w2, s_g2fp32.ptr, s_off.ptr, s_texp.ptr, s_trbase.ptr,
                              I, D, total_m_tiles, stream);

    // Scatter → FP32 → BF16 output
    s_ofp32.resize(T * D); s_ofp32.zero(stream);
    launch_scatter(s_g2fp32.ptr, s_tok.ptr, s_wtmap.ptr, s_ofp32.ptr, total_local, D, stream);
    launch_fp32_to_bf16(s_ofp32.ptr, output, T * D, stream);
}

// ---------------------------------------------------------------------------
// BF16 cuBLAS per-expert forward pass.
// ---------------------------------------------------------------------------
void moe_forward_deepseek_bf16_cublas(
    const __nv_bfloat16* input,
    const float*         gate_weight,
    const float*         gate_bias,
    const __nv_bfloat16* w1,
    const __nv_bfloat16* w2,
    __nv_bfloat16*       output,
    const MoeConfig& cfg, cudaStream_t stream)
{
    int T = cfg.num_tokens, E = cfg.num_experts, E_local = cfg.num_local_experts;
    int D = cfg.hidden_dim, I = cfg.intermediate_dim, K = cfg.top_k;

    static cublasHandle_t h_cb = nullptr;
    if (!h_cb) { cublasCreate(&h_cb); cublasSetMathMode(h_cb, CUBLAS_DEFAULT_MATH); }
    cublasSetStream(h_cb, stream);

    static DeviceBuf<int>           s_idx, s_cnt, s_slot, s_off, s_tok;
    static DeviceBuf<float>         s_wts, s_ifp32, s_g1, s_afp32, s_g2, s_ofp32, s_wtmap;
    static DeviceBuf<__nv_bfloat16> s_gin, s_abf16;

    // Gate
    s_ifp32.resize(T * D);
    launch_bf16_to_fp32(input, s_ifp32.ptr, T * D, stream);
    s_idx.resize(T * K); s_wts.resize(T * K);
    moe_gate_deepseek(s_ifp32.ptr, gate_weight, gate_bias, s_idx.ptr, s_wts.ptr, cfg, stream);

    // Count + offsets
    s_cnt.resize(E + 1); s_cnt.zero(stream); s_slot.resize(T * K);
    launch_expert_grouping(s_idx.ptr, s_cnt.ptr, s_slot.ptr, T * K, E, stream);

    static std::vector<int> h_cnt, h_off;
    h_cnt.resize(E + 1);
    CUDA_CHECK(cudaMemcpyAsync(h_cnt.data(), s_cnt.ptr, E * sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    h_off.assign(E + 1, 0);
    for (int i = 0; i < E; i++) h_off[i + 1] = h_off[i] + h_cnt[i];
    s_off.resize(E + 1); s_off.upload(h_off.data(), stream);

    int total_local = h_off[E_local];
    if (total_local == 0) {
        CUDA_CHECK(cudaMemsetAsync(output, 0, (size_t)T * D * sizeof(__nv_bfloat16), stream));
        return;
    }

    // Gather
    int total_active = h_off[E];
    s_gin.resize(total_active * D); s_tok.resize(total_active); s_wtmap.resize(total_active);
    launch_group_reorder_bf16(input, s_idx.ptr, s_wts.ptr, s_off.ptr, s_slot.ptr,
                               s_gin.ptr, s_tok.ptr, s_wtmap.ptr, T, K, D, T * K, stream);

    s_ofp32.resize(T * D); s_ofp32.zero(stream);
    s_g1.resize(total_local * 2 * I);
    s_afp32.resize(total_local * I);
    s_abf16.resize(total_local * I);
    s_g2.resize(total_local * D);

    float alpha = 1.0f, beta = 0.0f;

    for (int e = 0; e < E_local; e++) {
        int M_e = h_off[e + 1] - h_off[e];
        if (M_e == 0) continue;
        cublasGemmEx(h_cb, CUBLAS_OP_T, CUBLAS_OP_N, 2 * I, M_e, D, &alpha,
                     w1 + (size_t)e * 2 * I * D, CUDA_R_16BF, D,
                     s_gin.ptr + (size_t)h_off[e] * D, CUDA_R_16BF, D, &beta,
                     s_g1.ptr + (size_t)h_off[e] * 2 * I, CUDA_R_32F, 2 * I,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }

    launch_swiglu_fp32(s_g1.ptr, s_afp32.ptr, total_local, I, stream);
    launch_fp32_to_bf16(s_afp32.ptr, s_abf16.ptr, total_local * I, stream);

    for (int e = 0; e < E_local; e++) {
        int M_e = h_off[e + 1] - h_off[e];
        if (M_e == 0) continue;
        cublasGemmEx(h_cb, CUBLAS_OP_T, CUBLAS_OP_N, D, M_e, I, &alpha,
                     w2 + (size_t)e * D * I, CUDA_R_16BF, I,
                     s_abf16.ptr + (size_t)h_off[e] * I, CUDA_R_16BF, I, &beta,
                     s_g2.ptr + (size_t)h_off[e] * D, CUDA_R_32F, D,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }

    launch_scatter(s_g2.ptr, s_tok.ptr, s_wtmap.ptr, s_ofp32.ptr, total_local, D, stream);
    launch_fp32_to_bf16(s_ofp32.ptr, output, T * D, stream);
}

// ---------------------------------------------------------------------------
// FP8 block-scale forward pass — competition-compatible interface.
//
// Routing:  softmax(routing_logits) → top-K → renormalize (matches Triton ref)
// GEMM1/2:  FP8 token gather (dequant on-gather) → BF16 cuBLAS per expert
//           TODO: replace cuBLAS with WGMMA FP8 kernel for Blackwell throughput
// SwiGLU:   BF16 intermediate → FP8 requantize → BF16 for GEMM2
// Scatter:  weighted atomicAdd → BF16 output
//
// FP8 dequant note:
//   hidden_states_scale layout [D/128, T] matches the Triton kernel convention.
//   w1/w2 scales are [E_local, N/128, D/128].
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
    cudaStream_t            stream)
{
    int T = cfg.num_tokens, E = cfg.num_experts, E_local = cfg.num_local_experts;
    int D = cfg.hidden_dim, I = cfg.intermediate_dim, K = cfg.top_k;
    const int BLOCK = 128; // FP8 quantization block size

    static cublasHandle_t h_cb = nullptr;
    if (!h_cb) { cublasCreate(&h_cb); cublasSetMathMode(h_cb, CUBLAS_DEFAULT_MATH); }
    cublasSetStream(h_cb, stream);

    static DeviceBuf<int>           s_idx, s_cnt, s_slot, s_off, s_tok;
    static DeviceBuf<float>         s_wts, s_g1fp32, s_afp32, s_g2fp32, s_ofp32, s_wtmap;
    static DeviceBuf<__nv_bfloat16> s_gin, s_g1bf16, s_actbf16, s_abf16_g2;
    static DeviceBuf<__nv_fp8_e4m3> s_actfp8;
    static DeviceBuf<float>         s_act_scale;
    // Per-expert dequantized weight buffers (reused across calls)
    static DeviceBuf<__nv_bfloat16> s_w1_bf16, s_w2_bf16;

    // 1. Softmax routing (competition-compatible)
    MoeConfig cfg_route = cfg;
    cfg_route.routed_scaling_factor = routed_scaling_factor;
    s_idx.resize(T * K); s_wts.resize(T * K);
    moe_gate_softmax(routing_logits, s_idx.ptr, s_wts.ptr, cfg_route, stream);

    // Filter to local experts [local_expert_offset, local_expert_offset + E_local)
    // The routing kernel already selected from all E experts; we only process local ones.
    // (Tokens routed to non-local experts simply have no slot in the grouped buffer.)

    // 2. Count + offsets for local experts only
    //    We count only slots where expert_id ∈ [offset, offset+E_local).
    s_cnt.resize(E_local + 1); s_cnt.zero(stream); s_slot.resize(T * K);
    // Re-use expert_grouping but remap global IDs → local IDs
    // Use a simple GPU pass: count only local expert assignments
    struct { int* ids; int* cnt; int* slot; int offset; int E_local; } args;
    // Launch custom kernel to count only local experts:
    {
        // expert_grouping_kernel already handles any e < E_local after offset subtraction
        // We offset the IDs by shifting: local_id = global_id - local_expert_offset
        // We'll handle this with the existing kernel by adjusting what we count.
        // Simple approach: use all-E counts and then slice; cheaper for small E.
        static DeviceBuf<int> s_cnt_all;
        s_cnt_all.resize(E + 1); s_cnt_all.zero(stream);
        static DeviceBuf<int> s_slot_all; s_slot_all.resize(T * K);
        launch_expert_grouping(s_idx.ptr, s_cnt_all.ptr, s_slot_all.ptr, T * K, E, stream);

        static std::vector<int> h_cnt_all;
        h_cnt_all.resize(E + 1);
        CUDA_CHECK(cudaMemcpyAsync(h_cnt_all.data(), s_cnt_all.ptr, E * sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::vector<int> h_off_all(E + 1, 0);
        for (int i = 0; i < E; i++) h_off_all[i + 1] = h_off_all[i] + h_cnt_all[i];

        // Build local offsets: only [offset, offset+E_local] entries matter
        std::vector<int> h_local_cnt(E_local + 1, 0);
        for (int le = 0; le < E_local; le++) h_local_cnt[le] = h_cnt_all[local_expert_offset + le];
        std::vector<int> h_local_off(E_local + 1, 0);
        for (int le = 0; le < E_local; le++) h_local_off[le + 1] = h_local_off[le] + h_local_cnt[le];
        s_off.resize(E_local + 1); s_off.upload(h_local_off.data(), stream);

        int total_local = h_local_off[E_local];
        if (total_local == 0) {
            CUDA_CHECK(cudaMemsetAsync(output, 0, (size_t)T * D * sizeof(__nv_bfloat16), stream));
            return;
        }

        // 3. FP8 gather with dequantization: FP8 hidden states → BF16 grouped
        //    Adjust expert_ids so that entries outside [offset, offset+E_local) write to a dummy slot.
        //    Simplest: pass offsets for all E experts, gather only rows with valid local expert.
        //    We rebuild a full-E offset array pointing local experts into contiguous space.
        std::vector<int> h_full_off(E + 1, -1); // -1 means "non-local, ignore"
        for (int le = 0; le < E_local; le++)
            h_full_off[local_expert_offset + le] = h_local_off[le];
        // For gather, we need to write to slots based on h_full_off.
        // We'll need a dedicated gather that skips non-local experts.
        // Use the FP8 gather kernel after adjusting the offset array.
        static DeviceBuf<int> s_full_off;
        s_full_off.resize(E + 1);
        s_full_off.upload(h_full_off.data(), stream);

        s_gin.resize(total_local * D); s_tok.resize(total_local); s_wtmap.resize(total_local);
        // FP8 gather: dequantize hidden states on-the-fly
        launch_group_reorder_fp8(
            hidden_states, hidden_states_scale,
            s_idx.ptr, s_wts.ptr,
            s_full_off.ptr, s_slot_all.ptr,
            s_gin.ptr, s_tok.ptr, s_wtmap.ptr,
            T, K, D, T * K, BLOCK, stream);

        // 4. GEMM1 per local expert: [M_e, D] × w1[le]^T → [M_e, 2I]  (BF16 → FP32)
        //    Dequantize w1[le] from FP8 → BF16 before cuBLAS.
        s_g1fp32.resize(total_local * 2 * I);
        s_w1_bf16.resize((size_t)E_local * 2 * I * D);

        // Dequant all local w1 experts at once
        for (int le = 0; le < E_local; le++) {
            launch_dequant_fp8(
                w1 + (size_t)le * 2 * I * D,
                w1_scale + (size_t)le * (2 * I / BLOCK) * (D / BLOCK),
                s_w1_bf16.ptr + (size_t)le * 2 * I * D,
                2 * I, D, BLOCK, stream);
        }

        float alpha = 1.0f, beta = 0.0f;
        for (int le = 0; le < E_local; le++) {
            int M_e = h_local_cnt[le];
            if (M_e == 0) continue;
            cublasGemmEx(h_cb, CUBLAS_OP_T, CUBLAS_OP_N, 2 * I, M_e, D, &alpha,
                         s_w1_bf16.ptr + (size_t)le * 2 * I * D, CUDA_R_16BF, D,
                         s_gin.ptr + (size_t)h_local_off[le] * D, CUDA_R_16BF, D, &beta,
                         s_g1fp32.ptr + (size_t)h_local_off[le] * 2 * I, CUDA_R_32F, 2 * I,
                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }

        // 5. SwiGLU → FP8 requantize
        s_g1bf16.resize(total_local * 2 * I);
        launch_fp32_to_bf16(s_g1fp32.ptr, s_g1bf16.ptr, total_local * 2 * I, stream);
        s_actbf16.resize(total_local * I);
        launch_swiglu_bf16(s_g1bf16.ptr, s_actbf16.ptr, total_local, I, stream);

        s_actfp8.resize(total_local * I);
        s_act_scale.resize(total_local * (I / BLOCK));
        launch_requant_fp8(s_actbf16.ptr, s_actfp8.ptr, s_act_scale.ptr, total_local, I, BLOCK, stream);

        // 6. Dequant s_actfp8 → BF16 for GEMM2
        //    (Ideally fuse into GEMM2 kernel; for now dequant separately)
        s_abf16_g2.resize(total_local * I);
        // Dequant act: scale layout [M, I/128], use block-wise scale
        // Custom kernel needed for this layout; use simple fp32 intermediate instead
        s_afp32.resize(total_local * I);
        // Cast FP8 → FP32 via BF16 intermediate (scale already applied at requant)
        // Actually after requant the FP8 is already scaled; dequant = fp8 * scale
        // Launch a per-element dequant kernel reusing launch_dequant_fp8 with 2D layout:
        launch_dequant_fp8(s_actfp8.ptr, s_act_scale.ptr, s_abf16_g2.ptr, total_local, I, BLOCK, stream);

        // 7. GEMM2 per local expert: [M_e, I] × w2[le]^T → [M_e, D]  (BF16 → FP32)
        s_w2_bf16.resize((size_t)E_local * D * I);
        for (int le = 0; le < E_local; le++) {
            launch_dequant_fp8(
                w2 + (size_t)le * D * I,
                w2_scale + (size_t)le * (D / BLOCK) * (I / BLOCK),
                s_w2_bf16.ptr + (size_t)le * D * I,
                D, I, BLOCK, stream);
        }

        s_g2fp32.resize(total_local * D);
        for (int le = 0; le < E_local; le++) {
            int M_e = h_local_cnt[le];
            if (M_e == 0) continue;
            cublasGemmEx(h_cb, CUBLAS_OP_T, CUBLAS_OP_N, D, M_e, I, &alpha,
                         s_w2_bf16.ptr + (size_t)le * D * I, CUDA_R_16BF, I,
                         s_abf16_g2.ptr + (size_t)h_local_off[le] * I, CUDA_R_16BF, I, &beta,
                         s_g2fp32.ptr + (size_t)h_local_off[le] * D, CUDA_R_32F, D,
                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }

        // 8. Weighted scatter → BF16 output
        s_ofp32.resize(T * D); s_ofp32.zero(stream);
        launch_scatter(s_g2fp32.ptr, s_tok.ptr, s_wtmap.ptr, s_ofp32.ptr, total_local, D, stream);
        launch_fp32_to_bf16(s_ofp32.ptr, output, T * D, stream);
    }
}

// ---------------------------------------------------------------------------
// FP32 compatibility alias: converts weights to BF16, calls cuBLAS variant.
// ---------------------------------------------------------------------------
void moe_forward(
    const float* input, const float* gate_weight, const float* gate_bias,
    const float* w1, const float* w2, float* output,
    const MoeConfig& cfg, cudaStream_t stream)
{
    int T = cfg.num_tokens, E_local = cfg.num_local_experts;
    int D = cfg.hidden_dim, I = cfg.intermediate_dim;

    static DeviceBuf<__nv_bfloat16> s_ibf16, s_w1bf16, s_w2bf16, s_obf16;
    s_ibf16.resize(T * D);
    s_w1bf16.resize((size_t)E_local * 2 * I * D);
    s_w2bf16.resize((size_t)E_local * D * I);
    s_obf16.resize(T * D);

    launch_fp32_to_bf16(input, s_ibf16.ptr, T * D, stream);
    launch_fp32_to_bf16(w1,    s_w1bf16.ptr, (size_t)E_local * 2 * I * D, stream);
    launch_fp32_to_bf16(w2,    s_w2bf16.ptr, (size_t)E_local * D * I, stream);

    moe_forward_deepseek_bf16_cublas(
        s_ibf16.ptr, gate_weight, gate_bias,
        s_w1bf16.ptr, s_w2bf16.ptr, s_obf16.ptr, cfg, stream);

    launch_bf16_to_fp32(s_obf16.ptr, output, T * D, stream);
}

// ---------------------------------------------------------------------------
// FP8 weights + FP32 input/output + DeepSeek routing.
//   Same DeepSeek-V3 routing as moe_forward_deepseek_bf16_cublas, but
//   weights are FP8 with per-block scales (dequantized before cuBLAS).
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
    cudaStream_t            stream)
{
    int T = cfg.num_tokens, E = cfg.num_experts, E_local = cfg.num_local_experts;
    int D = cfg.hidden_dim, I = cfg.intermediate_dim, K = cfg.top_k;
    const int BLOCK = 128;

    static cublasHandle_t h_cb2 = nullptr;
    if (!h_cb2) { cublasCreate(&h_cb2); cublasSetMathMode(h_cb2, CUBLAS_DEFAULT_MATH); }
    cublasSetStream(h_cb2, stream);

    static DeviceBuf<int>           s_idx2, s_cnt2, s_slot2, s_off2, s_tok2;
    static DeviceBuf<float>         s_wts2, s_g1_2, s_af32_2, s_g2_2, s_of32_2, s_wt2map;
    static DeviceBuf<__nv_bfloat16> s_ibf16_2, s_gin2, s_af16_2, s_w1bf16_2, s_w2bf16_2;

    // Gate routing (DeepSeek)
    s_idx2.resize(T * K); s_wts2.resize(T * K);
    moe_gate_deepseek(input, gate_weight, gate_bias, s_idx2.ptr, s_wts2.ptr, cfg, stream);

    // Count + offsets
    s_cnt2.resize(E + 1); s_cnt2.zero(stream); s_slot2.resize(T * K);
    launch_expert_grouping(s_idx2.ptr, s_cnt2.ptr, s_slot2.ptr, T * K, E, stream);

    static std::vector<int> h_cnt2, h_off2;
    h_cnt2.resize(E + 1);
    CUDA_CHECK(cudaMemcpyAsync(h_cnt2.data(), s_cnt2.ptr, E * sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    h_off2.assign(E + 1, 0);
    for (int i = 0; i < E; i++) h_off2[i + 1] = h_off2[i] + h_cnt2[i];
    s_off2.resize(E + 1); s_off2.upload(h_off2.data(), stream);

    int total_local = h_off2[E_local];
    if (total_local == 0) {
        CUDA_CHECK(cudaMemsetAsync(output, 0, (size_t)T * D * sizeof(float), stream));
        return;
    }

    // Convert FP32 input → BF16 and gather
    int total_active = h_off2[E];
    s_ibf16_2.resize(T * D);
    launch_fp32_to_bf16(input, s_ibf16_2.ptr, T * D, stream);
    s_gin2.resize(total_active * D); s_tok2.resize(total_active); s_wt2map.resize(total_active);
    launch_group_reorder_bf16(s_ibf16_2.ptr, s_idx2.ptr, s_wts2.ptr, s_off2.ptr, s_slot2.ptr,
                               s_gin2.ptr, s_tok2.ptr, s_wt2map.ptr, T, K, D, T * K, stream);

    // Dequantize FP8 weights → BF16, then cuBLAS GEMM
    s_w1bf16_2.resize((size_t)E_local * 2 * I * D);
    s_w2bf16_2.resize((size_t)E_local * D * I);
    for (int e = 0; e < E_local; e++) {
        launch_dequant_fp8(w1 + (size_t)e * 2 * I * D,
                           w1_scale + (size_t)e * (2 * I / BLOCK) * (D / BLOCK),
                           s_w1bf16_2.ptr + (size_t)e * 2 * I * D,
                           2 * I, D, BLOCK, stream);
        launch_dequant_fp8(w2 + (size_t)e * D * I,
                           w2_scale + (size_t)e * (D / BLOCK) * (I / BLOCK),
                           s_w2bf16_2.ptr + (size_t)e * D * I,
                           D, I, BLOCK, stream);
    }

    s_g1_2.resize(total_local * 2 * I);
    s_af32_2.resize(total_local * I);
    s_af16_2.resize(total_local * I);
    s_g2_2.resize(total_local * D);
    float alpha = 1.0f, beta = 0.0f;

    for (int e = 0; e < E_local; e++) {
        int M_e = h_off2[e + 1] - h_off2[e]; if (M_e == 0) continue;
        cublasGemmEx(h_cb2, CUBLAS_OP_T, CUBLAS_OP_N, 2 * I, M_e, D, &alpha,
                     s_w1bf16_2.ptr + (size_t)e * 2 * I * D, CUDA_R_16BF, D,
                     s_gin2.ptr + (size_t)h_off2[e] * D, CUDA_R_16BF, D, &beta,
                     s_g1_2.ptr + (size_t)h_off2[e] * 2 * I, CUDA_R_32F, 2 * I,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    launch_swiglu_fp32(s_g1_2.ptr, s_af32_2.ptr, total_local, I, stream);
    launch_fp32_to_bf16(s_af32_2.ptr, s_af16_2.ptr, total_local * I, stream);

    for (int e = 0; e < E_local; e++) {
        int M_e = h_off2[e + 1] - h_off2[e]; if (M_e == 0) continue;
        cublasGemmEx(h_cb2, CUBLAS_OP_T, CUBLAS_OP_N, D, M_e, I, &alpha,
                     s_w2bf16_2.ptr + (size_t)e * D * I, CUDA_R_16BF, I,
                     s_af16_2.ptr + (size_t)h_off2[e] * I, CUDA_R_16BF, I, &beta,
                     s_g2_2.ptr + (size_t)h_off2[e] * D, CUDA_R_32F, D,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }

    s_of32_2.resize(T * D); s_of32_2.zero(stream);
    launch_scatter(s_g2_2.ptr, s_tok2.ptr, s_wt2map.ptr, s_of32_2.ptr, total_local, D, stream);
    CUDA_CHECK(cudaMemcpyAsync(output, s_of32_2.ptr, (size_t)T * D * sizeof(float), cudaMemcpyDeviceToDevice, stream));
}
