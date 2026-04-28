#include "../kernels/moe/moe_kernels.cuh"

#include <cstdio>
#include <vector>
#include <string>
#include <chrono>
#include <nvToolsExt.h>

typedef void (*moe_fn)(const float*, const float*, const float*, const float*, float*, const MoeConfig&, cudaStream_t);

void bench_one(const char* name, moe_fn fn, const MoeConfig& cfg, int warmup, int iters) {
    int T = cfg.num_tokens, E = cfg.num_experts, EL = cfg.num_local_experts;
    int D = cfg.hidden_dim, I = cfg.intermediate_dim, K = cfg.top_k;

    // Alloc
    float *d_in, *d_gate, *d_w1, *d_w2, *d_out;
    cudaMalloc(&d_in, (size_t)T * D * sizeof(float));
    cudaMalloc(&d_gate, (size_t)E * D * sizeof(float));
    cudaMalloc(&d_w1, (size_t)EL * 2 * I * D * sizeof(float));
    cudaMalloc(&d_w2, (size_t)EL * D * I * sizeof(float));
    cudaMalloc(&d_out, (size_t)T * D * sizeof(float));

    // Warmup
    nvtxRangePushA("warmup");
    for (int i = 0; i < warmup; i++) fn(d_in, d_gate, d_w1, d_w2, d_out, cfg, 0);
    cudaStreamSynchronize(0);
    nvtxRangePop();

    // Bench
    char range_name[100];
    sprintf(range_name, "bench_%s", name);
    nvtxRangePushA(range_name);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        sprintf(range_name, "%s_iter_%d", name, i);
        nvtxRangePushA(range_name);
        fn(d_in, d_gate, d_w1, d_w2, d_out, cfg, 0);
        nvtxRangePop();
    }
    cudaStreamSynchronize(0);
    auto end = std::chrono::high_resolution_clock::now();
    nvtxRangePop();

    double ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;
    double toks_per_sec = (T * 1000.0) / ms;

    printf("  %-10s  %-30s  %8.3f ms  %12.0f Tok/s\n", name, "DS-V3 Scale (T=512)", ms, toks_per_sec);

    cudaFree(d_in); cudaFree(d_gate); cudaFree(d_w1); cudaFree(d_w2); cudaFree(d_out);
}

void bench_deepseek(const char* name, const MoeConfig& cfg, int warmup, int iters) {
    int T = cfg.num_tokens, E = cfg.num_experts, EL = cfg.num_local_experts;
    int D = cfg.hidden_dim, I = cfg.intermediate_dim, K = cfg.top_k;

    float *d_in, *d_gate, *d_bias, *d_w1, *d_w2, *d_out;
    cudaMalloc(&d_in, (size_t)T * D * sizeof(float));
    cudaMalloc(&d_gate, (size_t)E * D * sizeof(float));
    cudaMalloc(&d_bias, (size_t)E * sizeof(float));
    cudaMemset(d_bias, 0, E * sizeof(float));
    cudaMalloc(&d_w1, (size_t)EL * 2 * I * D * sizeof(float));
    cudaMalloc(&d_w2, (size_t)EL * D * I * sizeof(float));
    cudaMalloc(&d_out, (size_t)T * D * sizeof(float));

    nvtxRangePushA("warmup");
    for (int i = 0; i < warmup; i++) moe_forward_deepseek(d_in, d_gate, d_bias, d_w1, d_w2, d_out, cfg, 0);
    cudaStreamSynchronize(0);
    nvtxRangePop();

    nvtxRangePushA("bench_DeepSeek");
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        moe_forward_deepseek(d_in, d_gate, d_bias, d_w1, d_w2, d_out, cfg, 0);
    }
    cudaStreamSynchronize(0);
    auto end = std::chrono::high_resolution_clock::now();
    nvtxRangePop();

    double ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;
    printf("  %-10s  %-30s  %8.3f ms  %12.0f Tok/s\n", name, "DS-V3 Scale (T=512)", ms, (T * 1000.0) / ms);

    cudaFree(d_in); cudaFree(d_gate); cudaFree(d_bias); cudaFree(d_w1); cudaFree(d_w2); cudaFree(d_out);
}



int main(int argc, char** argv) {
    int warmup = 2;
    int iters  = 5;
    
    std::string variant = "";
    if (argc > 1) variant = argv[1];

    if (variant == "") {
        printf("=== MoE Smoke Test (Reduced iters=%d) ===\n\n", iters);
    } else {
        printf("=== MoE Profiling: %s (iters=%d) ===\n\n", variant.c_str(), iters);
    }
    printf("  Variant     Description                     Latency      Throughput\n");
    printf("  --------------------------------------------------------------------------------\n");

    // DeepSeek-V3 Scale but fewer tokens for speed
    // {num_tokens, num_experts, num_local_experts, top_k, hidden_dim, intermediate_dim, n_group, topk_group, routed_scaling_factor}
    MoeConfig ds_v3 = {512, 256, 32, 8, 7168, 2048, 8, 4, 1.0f};

    if (variant == "" || variant == "TrueNaive")
        bench_one("TrueNaive", moe_forward_true_naive, ds_v3, warmup, iters);
    if (variant == "" || variant == "Naive")
        bench_one("Naive", moe_forward_naive, ds_v3, warmup, iters);
    if (variant == "" || variant == "Opt1")
        bench_one("Opt1", moe_forward_opt1, ds_v3, warmup, iters);
    if (variant == "" || variant == "Opt2")
        bench_one("Opt2", moe_forward_opt2, ds_v3, warmup, iters);
    if (variant == "" || variant == "Opt3")
        bench_one("Opt3", moe_forward_opt3, ds_v3, warmup, iters);
    if (variant == "" || variant == "Opt4")
        bench_one("Opt4", moe_forward_opt4, ds_v3, warmup, iters);
    if (variant == "" || variant == "Opt5")
        bench_one("Opt5", moe_forward_opt5, ds_v3, warmup, iters);
    if (variant == "" || variant == "DeepSeek")
        bench_deepseek("DeepSeek", ds_v3, warmup, iters);


    printf("  --------------------------------------------------------------------------------\n");
    return 0;
}
