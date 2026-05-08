#include "moe_forward.cuh"
#include "../../utils/cuda_utils.cuh"
#include <mma.h>
#include <cuda_fp8.h>
#include <cuda_pipeline_primitives.h>

// ---------------------------------------------------------------------------
// Expert grouping: assign each (token, slot) pair to its expert bucket.
// ---------------------------------------------------------------------------
__global__ void expert_grouping_kernel(
    const int* expert_ids, int* expert_counts, int* slot_in_expert,
    int TK, int E)
{
    int tk = blockIdx.x * blockDim.x + threadIdx.x;
    if (tk >= TK) return;
    int e = expert_ids[tk];
    if (e >= 0 && e < E) slot_in_expert[tk] = atomicAdd(&expert_counts[e], 1);
}

// ---------------------------------------------------------------------------
// BF16 token gather into expert-contiguous layout.
// Records tok_map[pos]=token and wt_map[pos]=routing_weight for scatter.
// Vectorized: 8 BF16 per float4 where D is a multiple of 8.
// ---------------------------------------------------------------------------
__global__ void group_reorder_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,
    const int*           __restrict__ expert_ids,
    const float*         __restrict__ expert_wts,
    const int*           __restrict__ offsets,
    const int*           __restrict__ slot_in_expert,
    __nv_bfloat16*       __restrict__ grouped,
    int*                 __restrict__ tok_map,
    float*               __restrict__ wt_map,
    int T, int K, int D)
{
    int tk = blockIdx.x, t = tk / K;
    if (t >= T) return;
    int e   = expert_ids[tk];
    int pos = offsets[e] + slot_in_expert[tk];
    if (threadIdx.x == 0) { tok_map[pos] = t; wt_map[pos] = expert_wts[tk]; }
    for (int d = threadIdx.x * 8; d < D; d += blockDim.x * 8) {
        if (d + 7 < D)
            *((float4*)&grouped[pos * D + d]) = *((const float4*)&input[t * D + d]);
        else
            for (int dd = d; dd < D && dd < d + 8; dd++)
                grouped[pos * D + dd] = input[t * D + dd];
    }
}

// ---------------------------------------------------------------------------
// FP8 E4M3 token gather with block-scale dequantization → BF16 output.
// hidden_states_scale layout: [D/128, T]  (note: transposed relative to tokens)
// ---------------------------------------------------------------------------
__global__ void group_reorder_fp8_kernel(
    const __nv_fp8_e4m3* __restrict__ input,
    const float*         __restrict__ input_scale,   // [D/128, T]
    const int*           __restrict__ expert_ids,
    const float*         __restrict__ expert_wts,
    const int*           __restrict__ offsets,
    const int*           __restrict__ slot_in_expert,
    __nv_bfloat16*       __restrict__ grouped,        // dequantized BF16
    int*                 __restrict__ tok_map,
    float*               __restrict__ wt_map,
    int T, int K, int D, int block_size)
{
    int tk = blockIdx.x, t = tk / K;
    if (t >= T) return;
    int e   = expert_ids[tk];
    int pos = offsets[e] + slot_in_expert[tk];
    if (threadIdx.x == 0) { tok_map[pos] = t; wt_map[pos] = expert_wts[tk]; }
    // Dequantize FP8 → BF16 with per-block scale (scale block = 128 elements)
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        int block_idx = d / block_size;
        float scale = input_scale[block_idx * T + t]; // [D/128, T] layout
        grouped[pos * D + d] = __float2bfloat16(float(input[t * D + d]) * scale);
    }
}

// ---------------------------------------------------------------------------
// Type-cast helpers
// ---------------------------------------------------------------------------
__global__ void bf16_to_fp32_kernel(const __nv_bfloat16* src, float* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __bfloat162float(src[i]);
}

__global__ void fp32_to_bf16_kernel(const float* src, __nv_bfloat16* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2bfloat16(src[i]);
}

// ---------------------------------------------------------------------------
// FP8 weight dequantization → BF16.
// w: [N, K] FP8,  scale: [N/128, K/128] FP32 (per-block)
// out: [N, K] BF16
// ---------------------------------------------------------------------------
__global__ void dequant_fp8_kernel(
    const __nv_fp8_e4m3* __restrict__ w,
    const float*         __restrict__ scale,
    __nv_bfloat16*       __restrict__ out,
    int N, int K, int block_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * K) return;
    int row = idx / K, col = idx % K;
    float s = scale[(row / block_size) * (K / block_size) + (col / block_size)];
    out[idx] = __float2bfloat16(float(w[idx]) * s);
}

// ---------------------------------------------------------------------------
// BF16 async double-buffered grouped GEMM — WMMA tensor cores, 64×64 output tile.
// tile_exp[tile_m]       → expert index (precomputed, no binary search).
// tile_row_base[tile_m]  → global row start for this tile (precomputed).
// K-slice: 32 BF16, covered by 2 × wmma<16,16,16>.
// SMEM: As[2][64][40] + Bs[2][64][40] BF16 ≈ 20 KB.
// ---------------------------------------------------------------------------
__global__ void grouped_gemm_bf16_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ W,
    float*               __restrict__ C,
    const int*           __restrict__ expert_offsets,
    const int*           __restrict__ tile_exp,
    const int*           __restrict__ tile_row_base,
    int D, int N)
{
    int tile_m = blockIdx.y, tile_n = blockIdx.x;
    int e              = tile_exp[tile_m];
    int global_row_base = tile_row_base[tile_m];
    int global_col_base = tile_n * 64;
    int exp_rows        = expert_offsets[e + 1] - global_row_base;

    const __nv_bfloat16* W_e = W + (size_t)e * N * D;

    __shared__ union {
        struct { __nv_bfloat16 As[2][64][40]; __nv_bfloat16 Bs[2][64][40]; };
        float Cs[64][64];
    } smem;

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> c_frag[2];
    nvcuda::wmma::fill_fragment(c_frag[0], 0.0f);
    nvcuda::wmma::fill_fragment(c_frag[1], 0.0f);

    int numTiles = (D + 31) / 32;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int warp_id = tid / 32, warp_row = warp_id / 2, warp_col = warp_id % 2;
    int a_r = tid / 4, a_c = (tid % 4) * 8;

    // Prologue: async-prefetch tile 0
    {
        bool vA = (a_r < exp_rows && a_c < D);
        if (vA) __pipeline_memcpy_async(&smem.As[0][a_r][a_c], &A[(global_row_base + a_r) * D + a_c], sizeof(float4));
        else    *(float4*)&smem.As[0][a_r][a_c] = make_float4(0, 0, 0, 0);
        bool vB = (global_col_base + a_r < N && a_c < D);
        if (vB) __pipeline_memcpy_async(&smem.Bs[0][a_r][a_c], &W_e[(global_col_base + a_r) * D + a_c], sizeof(float4));
        else    *(float4*)&smem.Bs[0][a_r][a_c] = make_float4(0, 0, 0, 0);
        __pipeline_commit();
    }

    for (int t = 0; t < numTiles; t++) {
        int curr = t & 1, next = (t + 1) & 1;
        if (t + 1 < numTiles) {
            int koff = (t + 1) * 32;
            bool vA = (a_r < exp_rows && koff + a_c < D);
            if (vA) __pipeline_memcpy_async(&smem.As[next][a_r][a_c], &A[(global_row_base + a_r) * D + koff + a_c], sizeof(float4));
            else    *(float4*)&smem.As[next][a_r][a_c] = make_float4(0, 0, 0, 0);
            bool vB = (global_col_base + a_r < N && koff + a_c < D);
            if (vB) __pipeline_memcpy_async(&smem.Bs[next][a_r][a_c], &W_e[(global_col_base + a_r) * D + koff + a_c], sizeof(float4));
            else    *(float4*)&smem.Bs[next][a_r][a_c] = make_float4(0, 0, 0, 0);
            __pipeline_commit();
        }
        __pipeline_wait_prior(0);
        __syncthreads();

        for (int k_idx = 0; k_idx < 32; k_idx += 16) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::row_major> a_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __nv_bfloat16, nvcuda::wmma::col_major> b0, b1;
            nvcuda::wmma::load_matrix_sync(a_frag, &smem.As[curr][warp_row * 16][k_idx], 40);
            nvcuda::wmma::load_matrix_sync(b0, &smem.Bs[curr][warp_col * 32 +  0][k_idx], 40);
            nvcuda::wmma::load_matrix_sync(b1, &smem.Bs[curr][warp_col * 32 + 16][k_idx], 40);
            nvcuda::wmma::mma_sync(c_frag[0], a_frag, b0, c_frag[0]);
            nvcuda::wmma::mma_sync(c_frag[1], a_frag, b1, c_frag[1]);
        }
        __syncthreads();
    }

    nvcuda::wmma::store_matrix_sync(&smem.Cs[warp_row * 16][warp_col * 32 +  0], c_frag[0], 64, nvcuda::wmma::mem_row_major);
    nvcuda::wmma::store_matrix_sync(&smem.Cs[warp_row * 16][warp_col * 32 + 16], c_frag[1], 64, nvcuda::wmma::mem_row_major);
    __syncthreads();

    for (int i = 0; i < 16; i++) {
        int idx = tid * 16 + i, crow = idx / 64, ccol = idx % 64;
        if (crow < exp_rows && global_col_base + ccol < N)
            C[(global_row_base + crow) * N + global_col_base + ccol] = smem.Cs[crow][ccol];
    }
}

// ---------------------------------------------------------------------------
// SwiGLU: [M, 2I] (gate=0..I-1, up=I..2I-1) → [M, I]
// ---------------------------------------------------------------------------
__global__ void swiglu_fp32_kernel(const float* in, float* out, int M, int I) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * I) return;
    int m = idx / I, i = idx % I;
    float gate = in[m * 2 * I + i], up = in[m * 2 * I + I + i];
    out[m * I + i] = gate / (1.0f + expf(-gate)) * up;
}

__global__ void swiglu_bf16_kernel(const __nv_bfloat16* in, __nv_bfloat16* out, int M, int I) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * I) return;
    int m = idx / I, i = idx % I;
    float gate = __bfloat162float(in[m * 2 * I + i]);
    float up   = __bfloat162float(in[m * 2 * I + I + i]);
    out[m * I + i] = __float2bfloat16(gate / (1.0f + expf(-gate)) * up);
}

// ---------------------------------------------------------------------------
// FP8 requantization after SwiGLU: per-row per-128-block amax → scale → FP8.
// act_in: [M, I] BF16,  fp8_out: [M, I] FP8,  scale_out: [M, I/128] FP32
// One block per row. Threads stride over I elements.
// ---------------------------------------------------------------------------
__global__ void requant_fp8_kernel(
    const __nv_bfloat16* __restrict__ act_in,
    __nv_fp8_e4m3*       __restrict__ fp8_out,
    float*               __restrict__ scale_out,
    int M, int I, int block_size)
{
    int m = blockIdx.x;
    if (m >= M) return;
    int n_blocks = I / block_size;
    for (int b = 0; b < n_blocks; b++) {
        int base = b * block_size;
        float amax = 0.0f;
        for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
            float v = fabsf(__bfloat162float(act_in[m * I + base + i]));
            amax = fmaxf(amax, v);
        }
        // Warp reduce amax
        for (int mask = 16; mask > 0; mask >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, mask));
        float scale = amax / 448.0f + 1e-12f; // 448 = max FP8 E4M3
        if (threadIdx.x == 0) scale_out[m * n_blocks + b] = scale;
        __syncwarp();
        for (int i = threadIdx.x; i < block_size; i += blockDim.x)
            fp8_out[m * I + base + i] = __nv_fp8_e4m3(__bfloat162float(act_in[m * I + base + i]) / scale);
    }
}

// ---------------------------------------------------------------------------
// Scatter: accumulate grouped GEMM output into global output using wt_map.
// wt_map[pos] = routing weight stored at gather time (no O(K) search).
// ---------------------------------------------------------------------------
__global__ void scatter_kernel(
    const float* grouped_out, const int* tok_map, const float* wt_map,
    float* global_out, int total, int D)
{
    int pos = blockIdx.x;
    if (pos >= total) return;
    int t = tok_map[pos];
    float w = wt_map[pos];
    for (int d = threadIdx.x; d < D; d += blockDim.x)
        atomicAdd(&global_out[t * D + d], w * grouped_out[pos * D + d]);
}

// ---------------------------------------------------------------------------
// Kernel symbols exported for moe_forward.cu
// ---------------------------------------------------------------------------
void launch_expert_grouping(
    const int* expert_ids, int* expert_counts, int* slot_in_expert,
    int TK, int E, cudaStream_t s)
{
    expert_grouping_kernel<<<(TK + 255) / 256, 256, 0, s>>>(
        expert_ids, expert_counts, slot_in_expert, TK, E);
    CUDA_CHECK(cudaGetLastError());
}

void launch_group_reorder_bf16(
    const __nv_bfloat16* input, const int* expert_ids, const float* expert_wts,
    const int* offsets, const int* slot_in_expert,
    __nv_bfloat16* grouped, int* tok_map, float* wt_map,
    int T, int K, int D, int TK, cudaStream_t s)
{
    group_reorder_bf16_kernel<<<TK, 256, 0, s>>>(
        input, expert_ids, expert_wts, offsets, slot_in_expert,
        grouped, tok_map, wt_map, T, K, D);
    CUDA_CHECK(cudaGetLastError());
}

void launch_group_reorder_fp8(
    const __nv_fp8_e4m3* input, const float* input_scale,
    const int* expert_ids, const float* expert_wts,
    const int* offsets, const int* slot_in_expert,
    __nv_bfloat16* grouped, int* tok_map, float* wt_map,
    int T, int K, int D, int TK, int block_size, cudaStream_t s)
{
    group_reorder_fp8_kernel<<<TK, 256, 0, s>>>(
        input, input_scale, expert_ids, expert_wts, offsets, slot_in_expert,
        grouped, tok_map, wt_map, T, K, D, block_size);
    CUDA_CHECK(cudaGetLastError());
}

void launch_bf16_to_fp32(const __nv_bfloat16* src, float* dst, int n, cudaStream_t s) {
    bf16_to_fp32_kernel<<<(n + 255) / 256, 256, 0, s>>>(src, dst, n);
    CUDA_CHECK(cudaGetLastError());
}
void launch_fp32_to_bf16(const float* src, __nv_bfloat16* dst, int n, cudaStream_t s) {
    fp32_to_bf16_kernel<<<(n + 255) / 256, 256, 0, s>>>(src, dst, n);
    CUDA_CHECK(cudaGetLastError());
}

void launch_dequant_fp8(
    const __nv_fp8_e4m3* w, const float* scale, __nv_bfloat16* out,
    int N, int K, int block_size, cudaStream_t s)
{
    dequant_fp8_kernel<<<(N * K + 255) / 256, 256, 0, s>>>(w, scale, out, N, K, block_size);
    CUDA_CHECK(cudaGetLastError());
}

void launch_grouped_gemm_bf16(
    const __nv_bfloat16* A, const __nv_bfloat16* W, float* C,
    const int* expert_offsets, const int* tile_exp, const int* tile_row_base,
    int D, int N, int total_m_tiles, cudaStream_t s)
{
    grouped_gemm_bf16_kernel<<<dim3((N + 63) / 64, total_m_tiles), dim3(16, 16), 0, s>>>(
        A, W, C, expert_offsets, tile_exp, tile_row_base, D, N);
    CUDA_CHECK(cudaGetLastError());
}

void launch_swiglu_fp32(const float* in, float* out, int M, int I, cudaStream_t s) {
    swiglu_fp32_kernel<<<(M * I + 255) / 256, 256, 0, s>>>(in, out, M, I);
    CUDA_CHECK(cudaGetLastError());
}
void launch_swiglu_bf16(const __nv_bfloat16* in, __nv_bfloat16* out, int M, int I, cudaStream_t s) {
    swiglu_bf16_kernel<<<(M * I + 255) / 256, 256, 0, s>>>(in, out, M, I);
    CUDA_CHECK(cudaGetLastError());
}

void launch_requant_fp8(
    const __nv_bfloat16* act_in, __nv_fp8_e4m3* fp8_out, float* scale_out,
    int M, int I, int block_size, cudaStream_t s)
{
    requant_fp8_kernel<<<M, 32, 0, s>>>(act_in, fp8_out, scale_out, M, I, block_size);
    CUDA_CHECK(cudaGetLastError());
}

void launch_scatter(
    const float* grouped_out, const int* tok_map, const float* wt_map,
    float* global_out, int total, int D, cudaStream_t s)
{
    scatter_kernel<<<total, min(D, 1024), 0, s>>>(grouped_out, tok_map, wt_map, global_out, total, D);
    CUDA_CHECK(cudaGetLastError());
}
