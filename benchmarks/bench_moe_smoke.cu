#include "../kernels/moe/naive_moe.cuh"
#include <cstdio>
#include <vector>
#include <string>
#include <chrono>
#include <nvToolsExt.h>
#include <cuda_bf16.h>

static void bench_bf16(const char* name,
                       void (*fn)(const __nv_bfloat16*, const float*, const float*,
                                  const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*,
                                  const MoeConfig&, cudaStream_t),
                       const MoeConfig& cfg, int warmup, int iters)
{
    int T = cfg.num_tokens, E = cfg.num_experts, EL = cfg.num_local_experts;
    int D = cfg.hidden_dim, I = cfg.intermediate_dim;

    __nv_bfloat16 *d_in, *d_w1, *d_w2, *d_out;
    float         *d_gate, *d_bias;
    cudaMalloc(&d_in,   (size_t)T  * D         * sizeof(__nv_bfloat16));
    cudaMalloc(&d_gate, (size_t)E  * D         * sizeof(float));
    cudaMalloc(&d_bias, (size_t)E              * sizeof(float));
    cudaMalloc(&d_w1,   (size_t)EL * 2 * I * D * sizeof(__nv_bfloat16));
    cudaMalloc(&d_w2,   (size_t)EL * D * I     * sizeof(__nv_bfloat16));
    cudaMalloc(&d_out,  (size_t)T  * D         * sizeof(__nv_bfloat16));
    cudaMemset(d_bias, 0, E * sizeof(float));

    nvtxRangePushA("warmup");
    for (int i = 0; i < warmup; i++) fn(d_in, d_gate, d_bias, d_w1, d_w2, d_out, cfg, 0);
    cudaStreamSynchronize(0);
    nvtxRangePop();

    char range_name[64];
    snprintf(range_name, sizeof(range_name), "bench_%s", name);
    nvtxRangePushA(range_name);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) fn(d_in, d_gate, d_bias, d_w1, d_w2, d_out, cfg, 0);
    cudaStreamSynchronize(0);
    auto end = std::chrono::high_resolution_clock::now();
    nvtxRangePop();

    double ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;
    printf("  %-14s  %-30s  %8.3f ms  %12.0f Tok/s\n",
           name, "DS-V3 Scale (T=512)", ms, (T * 1000.0) / ms);

    cudaFree(d_in); cudaFree(d_gate); cudaFree(d_bias);
    cudaFree(d_w1); cudaFree(d_w2); cudaFree(d_out);
}

int main(int argc, char** argv) {
    int warmup = 2, iters = 5;
    std::string variant = (argc > 1) ? argv[1] : "";

    if (variant.empty())
        printf("=== MoE BF16 Smoke Bench (iters=%d) ===\n\n", iters);
    else
        printf("=== MoE BF16 Profiling: %s (iters=%d) ===\n\n", variant.c_str(), iters);

    printf("  Variant         Description                     Latency      Throughput\n");
    printf("  ---------------------------------------------------------------------------------\n");

    MoeConfig ds_v3 = {512, 256, 32, 8, 7168, 2048, 8, 4, 1.0f};

    if (variant.empty() || variant == "WMMA")
        bench_bf16("BF16-WMMA",   moe_forward_deepseek_bf16,         ds_v3, warmup, iters);
    if (variant.empty() || variant == "cuBLAS")
        bench_bf16("BF16-cuBLAS", moe_forward_deepseek_bf16_cublas,  ds_v3, warmup, iters);

    printf("  ---------------------------------------------------------------------------------\n");
    return 0;
}
