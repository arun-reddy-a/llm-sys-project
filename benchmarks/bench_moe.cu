#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
#include "../kernels/moe/naive_moe.cuh"
#include "../utils/cuda_utils.cuh"

struct BenchConfig {
    const char* label;
    MoeConfig   moe;
};

// DeepSeek-style function pointer (includes gate_bias)
typedef void (*deepseek_fn)(const float*, const float*, const float*, const float*, const float*, float*, const MoeConfig&, cudaStream_t);

static void bench_deepseek(const char* variant_name, deepseek_fn func, const BenchConfig& bc, int warmup, int iters) {
    const MoeConfig& cfg = bc.moe;
    int T = cfg.num_tokens, E = cfg.num_experts, E_local = cfg.num_local_experts;
    int D = cfg.hidden_dim, I = cfg.intermediate_dim;

    size_t input_sz = T * D;
    size_t gate_sz  = (size_t)E * D;
    size_t bias_sz  = E;
    size_t w1_sz    = (size_t)E_local * 2 * I * D;
    size_t w2_sz    = (size_t)E_local * D * I;

    std::vector<float> h_input(input_sz), h_gate(gate_sz), h_bias(bias_sz);
    std::vector<float> h_w1(w1_sz), h_w2(w2_sz);

    random_fill(h_input.data(), input_sz, -0.5f, 0.5f);
    random_fill(h_gate.data(), gate_sz, -0.5f, 0.5f);
    random_fill(h_bias.data(), bias_sz, -0.01f, 0.01f);
    random_fill(h_w1.data(), w1_sz, -0.1f, 0.1f);
    random_fill(h_w2.data(), w2_sz, -0.1f, 0.1f);

    DeviceBuf<float> d_input(input_sz);
    DeviceBuf<float> d_gate(gate_sz);
    DeviceBuf<float> d_bias(bias_sz);
    DeviceBuf<float> d_w1(w1_sz);
    DeviceBuf<float> d_w2(w2_sz);
    DeviceBuf<float> d_output(T * D);

    d_input.upload(h_input.data());
    d_gate.upload(h_gate.data());
    d_bias.upload(h_bias.data());
    d_w1.upload(h_w1.data());
    d_w2.upload(h_w2.data());

    // Warm-up
    for (int i = 0; i < warmup; i++) {
        func(d_input.ptr, d_gate.ptr, d_bias.ptr, d_w1.ptr, d_w2.ptr, d_output.ptr, cfg, 0);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    std::vector<float> times(iters);
    GpuTimer timer;
    for (int i = 0; i < iters; i++) {
        timer.begin();
        func(d_input.ptr, d_gate.ptr, d_bias.ptr, d_w1.ptr, d_w2.ptr, d_output.ptr, cfg, 0);
        timer.end();
        times[i] = timer.elapsed_ms();
    }

    std::sort(times.begin(), times.end());
    float t_min    = times.front();
    float t_max    = times.back();
    float t_median = times[iters / 2];
    float t_mean   = 0.0f;
    for (float t : times) t_mean += t;
    t_mean /= iters;

    float tokens_per_sec = T / (t_mean * 1e-3f);

    printf("  %-12s  %-40s  %8.3f  %8.3f  %12.0f\n",
           variant_name, bc.label, t_min, t_mean, tokens_per_sec);
    fflush(stdout);
}

int main(int argc, char** argv) {
    int warmup = 2;
    int iters  = 10;

    if (argc > 1) warmup = atoi(argv[1]);
    if (argc > 2) iters  = atoi(argv[2]);

    printf("=== DeepSeek-V3 MoE Kernel Benchmark ===\n");
    printf("    warmup=%d  iters=%d\n\n", warmup, iters);
    printf("  %-12s  %-40s  %8s  %8s  %12s\n",
           "Variant", "Config", "Min(ms)", "Mean(ms)", "Tok/s");
    printf("  %s\n", std::string(110, '-').c_str());

    srand(123);

    // DeepSeek-V3 configurations across different sequence lengths
    // {num_tokens, num_experts(global), num_local_experts, top_k, hidden_dim, intermediate_dim, n_group, topk_group, routed_scaling_factor}
    BenchConfig configs[] = {
        {"T=64,E=256,EL=32,K=8,D=7168,I=2048",    {64,   256, 32, 8, 7168, 2048, 8, 4, 1.0f}},
        {"T=128,E=256,EL=32,K=8,D=7168,I=2048",   {128,  256, 32, 8, 7168, 2048, 8, 4, 1.0f}},
        {"T=256,E=256,EL=32,K=8,D=7168,I=2048",   {256,  256, 32, 8, 7168, 2048, 8, 4, 1.0f}},
        {"T=512,E=256,EL=32,K=8,D=7168,I=2048",   {512,  256, 32, 8, 7168, 2048, 8, 4, 1.0f}},
        {"T=1024,E=256,EL=32,K=8,D=7168,I=2048",  {1024, 256, 32, 8, 7168, 2048, 8, 4, 1.0f}},
        {"T=2048,E=256,EL=32,K=8,D=7168,I=2048",  {2048, 256, 32, 8, 7168, 2048, 8, 4, 1.0f}},
        {"T=4096,E=256,EL=32,K=8,D=7168,I=2048",  {4096, 256, 32, 8, 7168, 2048, 8, 4, 1.0f}},
    };

    for (auto& bc : configs) {
        bench_deepseek("DeepSeek-V3", moe_forward_deepseek, bc, warmup, iters);
    }
    printf("  %s\n", std::string(110, '-').c_str());

    // BF16 variant
    printf("\n  %-12s  %-40s  %8s  %8s  %12s\n",
           "Variant", "Config", "Min(ms)", "Mean(ms)", "Tok/s");
    printf("  %s\n", std::string(110, '-').c_str());

    for (auto& bc : configs) {
        const MoeConfig& cfg = bc.moe;
        int T = cfg.num_tokens, E_local = cfg.num_local_experts;
        int D = cfg.hidden_dim, I = cfg.intermediate_dim;

        // Allocate BF16 buffers
        DeviceBuf<__nv_bfloat16> d_input_bf16(T * D);
        DeviceBuf<float>         d_gate(cfg.num_experts * D);
        DeviceBuf<float>         d_bias(cfg.num_experts);
        DeviceBuf<__nv_bfloat16> d_w1_bf16((size_t)E_local * 2 * I * D);
        DeviceBuf<__nv_bfloat16> d_w2_bf16((size_t)E_local * D * I);
        DeviceBuf<__nv_bfloat16> d_output_bf16(T * D);

        // Fill with random values (reinterpret as bf16 — good enough for benchmarking)
        CUDA_CHECK(cudaMemset(d_input_bf16.ptr, 0x3f, T * D * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMemset(d_w1_bf16.ptr,    0x3f, (size_t)E_local * 2 * I * D * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMemset(d_w2_bf16.ptr,    0x3f, (size_t)E_local * D * I * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMemset(d_gate.ptr,        0,   cfg.num_experts * D * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_bias.ptr,        0,   cfg.num_experts * sizeof(float)));

        for (int i = 0; i < warmup; i++)
            moe_forward_deepseek_bf16(d_input_bf16.ptr, d_gate.ptr, d_bias.ptr,
                                      d_w1_bf16.ptr, d_w2_bf16.ptr, d_output_bf16.ptr, cfg, 0);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> times(iters);
        GpuTimer timer;
        for (int i = 0; i < iters; i++) {
            timer.begin();
            moe_forward_deepseek_bf16(d_input_bf16.ptr, d_gate.ptr, d_bias.ptr,
                                      d_w1_bf16.ptr, d_w2_bf16.ptr, d_output_bf16.ptr, cfg, 0);
            timer.end();
            times[i] = timer.elapsed_ms();
        }
        std::sort(times.begin(), times.end());
        float t_min = times.front(), t_mean = 0.f;
        for (float t : times) t_mean += t;
        t_mean /= iters;
        printf("  %-12s  %-40s  %8.3f  %8.3f  %12.0f\n",
               "DS-V3-BF16", bc.label, t_min, t_mean, T / (t_mean * 1e-3f));
        fflush(stdout);
    }
    printf("  %s\n", std::string(110, '-').c_str());

    // BF16 cuBLAS variant
    printf("\n  %-12s  %-40s  %8s  %8s  %12s\n",
           "Variant", "Config", "Min(ms)", "Mean(ms)", "Tok/s");
    printf("  %s\n", std::string(110, '-').c_str());

    for (auto& bc : configs) {
        const MoeConfig& cfg = bc.moe;
        int T = cfg.num_tokens, E_local = cfg.num_local_experts;
        int D = cfg.hidden_dim, I = cfg.intermediate_dim;

        DeviceBuf<__nv_bfloat16> d_input_bf16(T * D);
        DeviceBuf<float>         d_gate(cfg.num_experts * D);
        DeviceBuf<float>         d_bias(cfg.num_experts);
        DeviceBuf<__nv_bfloat16> d_w1_bf16((size_t)E_local * 2 * I * D);
        DeviceBuf<__nv_bfloat16> d_w2_bf16((size_t)E_local * D * I);
        DeviceBuf<__nv_bfloat16> d_output_bf16(T * D);

        CUDA_CHECK(cudaMemset(d_input_bf16.ptr, 0x3f, T * D * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMemset(d_w1_bf16.ptr,    0x3f, (size_t)E_local * 2 * I * D * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMemset(d_w2_bf16.ptr,    0x3f, (size_t)E_local * D * I * sizeof(__nv_bfloat16)));
        CUDA_CHECK(cudaMemset(d_gate.ptr,        0,   cfg.num_experts * D * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_bias.ptr,        0,   cfg.num_experts * sizeof(float)));

        for (int i = 0; i < warmup; i++)
            moe_forward_deepseek_bf16_cublas(d_input_bf16.ptr, d_gate.ptr, d_bias.ptr,
                                             d_w1_bf16.ptr, d_w2_bf16.ptr, d_output_bf16.ptr, cfg, 0);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> times(iters);
        GpuTimer timer;
        for (int i = 0; i < iters; i++) {
            timer.begin();
            moe_forward_deepseek_bf16_cublas(d_input_bf16.ptr, d_gate.ptr, d_bias.ptr,
                                             d_w1_bf16.ptr, d_w2_bf16.ptr, d_output_bf16.ptr, cfg, 0);
            timer.end();
            times[i] = timer.elapsed_ms();
        }
        std::sort(times.begin(), times.end());
        float t_min = times.front(), t_mean = 0.f;
        for (float t : times) t_mean += t;
        t_mean /= iters;
        printf("  %-12s  %-40s  %8.3f  %8.3f  %12.0f\n",
               "DS-BF16-CB", bc.label, t_min, t_mean, T / (t_mean * 1e-3f));
        fflush(stdout);
    }
    printf("  %s\n", std::string(110, '-').c_str());

    printf("\n");
    return 0;
}
