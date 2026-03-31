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

typedef void (*moe_fn)(const float*, const float*, const float*, const float*, float*, const MoeConfig&, cudaStream_t);

static void bench_one(const char* variant_name, moe_fn func, const BenchConfig& bc, int warmup, int iters) {
    const MoeConfig& cfg = bc.moe;
    int T = cfg.num_tokens, E = cfg.num_experts;
    int D = cfg.hidden_dim, I = cfg.intermediate_dim;

    size_t input_sz = T * D;
    size_t gate_sz  = E * D;
    size_t w1_sz    = (size_t)E * 2 * I * D;
    size_t w2_sz    = (size_t)E * D * I;

    std::vector<float> h_input(input_sz), h_gate(gate_sz);
    std::vector<float> h_w1(w1_sz), h_w2(w2_sz);

    random_fill(h_input.data(), input_sz, -0.5f, 0.5f);
    random_fill(h_gate.data(), gate_sz, -0.5f, 0.5f);
    random_fill(h_w1.data(), w1_sz, -0.1f, 0.1f);
    random_fill(h_w2.data(), w2_sz, -0.1f, 0.1f);

    DeviceBuf<float> d_input(input_sz);
    DeviceBuf<float> d_gate(gate_sz);
    DeviceBuf<float> d_w1(w1_sz);
    DeviceBuf<float> d_w2(w2_sz);
    DeviceBuf<float> d_output(T * D);

    d_input.upload(h_input.data());
    d_gate.upload(h_gate.data());
    d_w1.upload(h_w1.data());
    d_w2.upload(h_w2.data());

    // Warm-up
    for (int i = 0; i < warmup; i++) {
        func(d_input.ptr, d_gate.ptr, d_w1.ptr, d_w2.ptr, d_output.ptr, cfg, 0);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    std::vector<float> times(iters);
    GpuTimer timer;
    for (int i = 0; i < iters; i++) {
        timer.begin();
        func(d_input.ptr, d_gate.ptr, d_w1.ptr, d_w2.ptr, d_output.ptr, cfg, 0);
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

    printf("  %-10s  %-28s  %8.3f  %8.3f  %12.0f\n",
           variant_name, bc.label, t_min, t_mean, tokens_per_sec);
}

int main(int argc, char** argv) {
    int warmup = 10;
    int iters  = 50;

    if (argc > 1) warmup = atoi(argv[1]);
    if (argc > 2) iters  = atoi(argv[2]);

    printf("=== MoE Kernel Comparison Benchmark ===\n");
    printf("    warmup=%d  iters=%d\n\n", warmup, iters);
    printf("  %-10s  %-28s  %8s  %8s  %12s\n",
           "Variant", "Config", "Min(ms)", "Mean(ms)", "Tok/s");
    printf("  %s\n", std::string(100, '-').c_str());

    srand(123);

    BenchConfig configs[] = {
        {"T=64,E=8,K=2,D=256,I=512",   {64,  8, 2, 256,  512}},
        {"T=128,E=16,K=2,D=512,I=1024",{128,16, 2, 512, 1024}},
    };

    struct { const char* name; moe_fn fn; } variants[] = {
        {"Naive", moe_forward_naive},
        {"Opt1",  moe_forward_opt1},
        {"Opt2",  moe_forward_opt2},
    };

    for (auto& bc : configs) {
        for (auto& v : variants) {
            bench_one(v.name, v.fn, bc, warmup, iters);
        }
        printf("  %s\n", std::string(100, '-').c_str());
    }

    printf("\n");
    return 0;
}
