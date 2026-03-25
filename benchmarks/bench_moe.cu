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

static void bench_one(const BenchConfig& bc, int warmup, int iters) {
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
        moe_forward(d_input.ptr, d_gate.ptr, d_w1.ptr, d_w2.ptr,
                    d_output.ptr, cfg);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    std::vector<float> times(iters);
    GpuTimer timer;
    for (int i = 0; i < iters; i++) {
        timer.begin();
        moe_forward(d_input.ptr, d_gate.ptr, d_w1.ptr, d_w2.ptr,
                    d_output.ptr, cfg);
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

    printf("  %-28s  %8.3f  %8.3f  %8.3f  %8.3f  %12.0f\n",
           bc.label, t_min, t_mean, t_median, t_max, tokens_per_sec);
}

int main(int argc, char** argv) {
    int warmup = 10;
    int iters  = 50;

    if (argc > 1) warmup = atoi(argv[1]);
    if (argc > 2) iters  = atoi(argv[2]);

    printf("=== MoE Naive Kernel Benchmark ===\n");
    printf("    warmup=%d  iters=%d\n\n", warmup, iters);
    printf("  %-28s  %8s  %8s  %8s  %8s  %12s\n",
           "Config", "Min(ms)", "Mean(ms)", "Med(ms)", "Max(ms)", "Tok/s");
    printf("  %s\n", std::string(96, '-').c_str());

    srand(123);

    BenchConfig configs[] = {
        {"T=16,E=4,K=2,D=64,I=128",    {16,  4, 2,  64,  128}},
        {"T=32,E=8,K=2,D=128,I=256",   {32,  8, 2, 128,  256}},
        {"T=64,E=8,K=2,D=256,I=512",   {64,  8, 2, 256,  512}},
        {"T=128,E=8,K=2,D=256,I=512",  {128, 8, 2, 256,  512}},
        {"T=64,E=8,K=4,D=256,I=512",   {64,  8, 4, 256,  512}},
        {"T=128,E=16,K=2,D=512,I=1024",{128,16, 2, 512, 1024}},
    };

    for (auto& bc : configs) {
        bench_one(bc, warmup, iters);
    }

    printf("\n");
    return 0;
}
