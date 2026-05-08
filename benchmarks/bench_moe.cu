// Benchmark both CUDA BF16 variants at competition T values.
// Usage: ./build/bench_moe [warmup] [iters]
// Output lines tagged DS-V3-BF16 / DS-BF16-CB for easy parsing by bench_all.py.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <string>
#include "../kernels/moe/naive_moe.cuh"
#include "../utils/cuda_utils.cuh"

// Competition + round-number T values for a complete picture.
static const int SEQ_LENS[] = {
    52, 80, 128, 256, 512, 901, 1024, 2048, 4096, 8192, 11948, 14107
};
static const int N_SEQ = sizeof(SEQ_LENS) / sizeof(SEQ_LENS[0]);

// DeepSeek-V3 fixed dims (single-GPU: 32 of 256 local experts).
static const int E  = 256, EL = 32, K = 8;
static const int D  = 7168, I  = 2048;
static const int NG = 8, KG = 4;
static const float RSF = 2.5f;

static void bench_variant(
    const char* tag,
    void (*fn)(const __nv_bfloat16*, const float*, const float*,
               const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*,
               const MoeConfig&, cudaStream_t),
    int T, int warmup, int iters)
{
    DeviceBuf<__nv_bfloat16> d_in(T * D), d_w1((size_t)EL*2*I*D), d_w2((size_t)EL*D*I), d_out(T * D);
    DeviceBuf<float>         d_gate(E * D), d_bias(E);
    d_bias.zero();

    MoeConfig cfg = {T, E, EL, K, D, I, NG, KG, RSF};

    for (int i = 0; i < warmup; i++)
        fn(d_in.ptr, d_gate.ptr, d_bias.ptr, d_w1.ptr, d_w2.ptr, d_out.ptr, cfg, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> times(iters);
    GpuTimer timer;
    for (int i = 0; i < iters; i++) {
        timer.begin();
        fn(d_in.ptr, d_gate.ptr, d_bias.ptr, d_w1.ptr, d_w2.ptr, d_out.ptr, cfg, 0);
        timer.end();
        times[i] = timer.elapsed_ms();
    }
    std::sort(times.begin(), times.end());
    float t_min = times[0], t_mean = 0;
    for (float t : times) t_mean += t;
    t_mean /= iters;

    // Tagged format bench_all.py parses: "TAG  T=N,config  min  mean  tok/s"
    char config[64];
    snprintf(config, sizeof(config), "T=%d,E=%d,EL=%d,K=%d,D=%d,I=%d", T, E, EL, K, D, I);
    printf("  %-12s  %-44s  %8.3f  %8.3f  %12.0f\n",
           tag, config, t_min, t_mean, T / (t_mean * 1e-3f));
    fflush(stdout);
}

int main(int argc, char** argv) {
    int warmup = 2, iters = 10;
    if (argc > 1) warmup = atoi(argv[1]);
    if (argc > 2) iters  = atoi(argv[2]);

    printf("=== CUDA BF16 MoE Benchmark (E=%d/EL=%d/K=%d/D=%d/I=%d) ===\n", E, EL, K, D, I);
    printf("    warmup=%d  iters=%d\n\n", warmup, iters);
    printf("  %-12s  %-44s  %8s  %8s  %12s\n",
           "Variant", "Config", "Min(ms)", "Mean(ms)", "Tok/s");
    printf("  %s\n", std::string(90, '-').c_str());

    for (int ti = 0; ti < N_SEQ; ti++)
        bench_variant("DS-V3-BF16", moe_forward_deepseek_bf16,        SEQ_LENS[ti], warmup, iters);
    printf("  %s\n\n", std::string(90, '-').c_str());

    for (int ti = 0; ti < N_SEQ; ti++)
        bench_variant("DS-BF16-CB", moe_forward_deepseek_bf16_cublas,  SEQ_LENS[ti], warmup, iters);
    printf("  %s\n", std::string(90, '-').c_str());

    return 0;
}
