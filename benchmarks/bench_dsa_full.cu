#include "../kernels/dsa/naive_dsa.cuh"
#include "../utils/cuda_utils.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>

typedef void (*dsa_fn)(const float*, const float*,
                       const float*, const float*, const float*,
                       const int*, float*,
                       const DsaConfig&, cudaStream_t);

struct VariantInfo {
    const char* name;
    const char* description;
    dsa_fn      fn;
};

struct BenchConfig {
    const char* label;
    DsaConfig   dsa;
};

static void gen_sparse_indices(int* indices, int Q, int S, int N) {
    std::vector<int> pool(N);
    for (int q = 0; q < Q; q++) {
        std::iota(pool.begin(), pool.end(), 0);
        for (int i = 0; i < S; i++) {
            int j = i + rand() % (N - i);
            std::swap(pool[i], pool[j]);
        }
        std::sort(pool.begin(), pool.begin() + S);
        for (int s = 0; s < S; s++) indices[q * S + s] = pool[s];
    }
}

static double compute_flops(const DsaConfig& cfg) {
    double Q  = cfg.num_queries;
    double H  = cfg.num_heads;
    double Dc = cfg.head_dim_compressed;
    double Dp = cfg.head_dim_positional;
    double S  = cfg.num_selected_kv;

    double score_flops  = Q * H * S * (2.0 * Dc + 2.0 * Dp);
    double softmax_flops = Q * H * S * 5.0;
    double output_flops = Q * H * S * 2.0 * Dc;

    return score_flops + softmax_flops + output_flops;
}

static void bench_variant(const VariantInfo& vi, const BenchConfig& bc,
                          int warmup, int iters, double total_flops) {
    const DsaConfig& cfg = bc.dsa;
    int Q  = cfg.num_queries,  H  = cfg.num_heads;
    int Dc = cfg.head_dim_compressed, Dp = cfg.head_dim_positional;
    int S  = cfg.num_selected_kv, N = cfg.total_kv_tokens;

    std::vector<float> h_q_nope(Q * H * Dc), h_q_pe(Q * H * Dp);
    std::vector<float> h_kv_c(N * Dc), h_kv_p(N * Dp), h_v(N * Dc);
    std::vector<int>   h_idx(Q * S);

    random_fill(h_q_nope.data(), Q * H * Dc, -0.3f, 0.3f);
    random_fill(h_q_pe.data(),   Q * H * Dp, -0.3f, 0.3f);
    random_fill(h_kv_c.data(),   N * Dc, -0.3f, 0.3f);
    random_fill(h_kv_p.data(),   N * Dp, -0.3f, 0.3f);
    random_fill(h_v.data(),      N * Dc, -0.3f, 0.3f);
    gen_sparse_indices(h_idx.data(), Q, S, N);

    DeviceBuf<float> d_q_nope(Q * H * Dc), d_q_pe(Q * H * Dp);
    DeviceBuf<float> d_kv_c(N * Dc), d_kv_p(N * Dp), d_v(N * Dc);
    DeviceBuf<int>   d_idx(Q * S);
    DeviceBuf<float> d_output(Q * H * Dc);

    d_q_nope.upload(h_q_nope.data());
    d_q_pe.upload(h_q_pe.data());
    d_kv_c.upload(h_kv_c.data());
    d_kv_p.upload(h_kv_p.data());
    d_v.upload(h_v.data());
    d_idx.upload(h_idx.data());

    for (int i = 0; i < warmup; i++)
        vi.fn(d_q_nope.ptr, d_q_pe.ptr, d_kv_c.ptr, d_kv_p.ptr,
              d_v.ptr, d_idx.ptr, d_output.ptr, cfg, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> times(iters);
    GpuTimer timer;
    for (int i = 0; i < iters; i++) {
        timer.begin();
        vi.fn(d_q_nope.ptr, d_q_pe.ptr, d_kv_c.ptr, d_kv_p.ptr,
              d_v.ptr, d_idx.ptr, d_output.ptr, cfg, 0);
        timer.end();
        times[i] = timer.elapsed_ms();
    }

    std::sort(times.begin(), times.end());
    float t_min  = times.front();
    float t_mean = 0.0f;
    for (float t : times) t_mean += t;
    t_mean /= iters;

    double gflops = (total_flops / (t_mean * 1e-3)) * 1e-9;
    double qps    = Q / (t_mean * 1e-3);

    printf("    %-8s  %8.3f  %8.3f  %10.2f  %12.0f\n",
           vi.name, t_min, t_mean, gflops, qps);
}

int main(int argc, char** argv) {
    int warmup = 5;
    int iters  = 20;

    if (argc > 1) warmup = atoi(argv[1]);
    if (argc > 2) iters  = atoi(argv[2]);

    VariantInfo variants[] = {
        {"Naive",  "Separate naive dots, 3-pass softmax, naive output",       dsa_forward_naive},
        {"Opt1",   "Shared-memory tiled dot products",                        dsa_forward_opt1},
        {"Opt2",   "Fuse compressed+positional dots into one GEMM",           dsa_forward_opt2},
        {"Opt3",   "Batch across queries (batched GEMM score+output)",        dsa_forward_opt3},
        {"Opt4",   "FlashAttention online softmax + output fusion",           dsa_forward_opt4},
        {"Opt5",   "Sort sparse indices by page",                             dsa_forward_opt5},
        {"Opt6",   "Fuse KV gather into compute (no separate gather)",        dsa_forward_opt6},
        {"Opt7",   "Asynchronous L2 KV tile prefetching",                     dsa_forward_opt7},
        {"Opt8",   "WMMA FP16 scores + 64x64 tiles + float4 KV/V + L2 prefetch (default dsa_forward)", dsa_forward_opt8},
    };
    int num_variants = sizeof(variants) / sizeof(variants[0]);

    BenchConfig configs[] = {
        {"Small",       {4,  4,  64,  16,  64,  16,  256}},
        {"Medium",      {8,  8,  128, 32,  128, 32,  512}},
        {"Large-Q",     {16, 8,  128, 32,  256, 64,  1024}},
        {"Large-Dc",    {8,  16, 256, 64,  256, 64,  2048}},
        {"DS-V3 Half",  {4,  16, 512, 64,  512, 64,  4096}},
        {"DS-V3 Full",  {8,  16, 512, 64,  1024, 64, 8192}},
    };
    int num_configs = sizeof(configs) / sizeof(configs[0]);

    printf("=== DSA Full Benchmark: All Variants x All Configs ===\n");
    printf("    warmup=%d  iters=%d\n", warmup, iters);
    printf("    GPU: NVIDIA B200 (Blackwell)\n\n");

    printf("--- Optimization Variants ---\n");
    for (int v = 0; v < num_variants; v++) {
        printf("  %-8s  %s\n", variants[v].name, variants[v].description);
    }

    printf("\n--- Configurations ---\n");
    printf("  %-12s  %4s  %4s  %4s  %4s  %5s  %4s  %6s  %12s\n",
           "Name", "Q", "H", "Dc", "Dp", "S", "P", "N", "FLOP/call");
    printf("  %s\n", std::string(72, '-').c_str());
    for (int c = 0; c < num_configs; c++) {
        const DsaConfig& d = configs[c].dsa;
        double flops = compute_flops(d);
        printf("  %-12s  %4d  %4d  %4d  %4d  %5d  %4d  %6d  %12.2e\n",
               configs[c].label, d.num_queries, d.num_heads,
               d.head_dim_compressed, d.head_dim_positional,
               d.num_selected_kv, d.page_size, d.total_kv_tokens, flops);
    }

    srand(123);

    for (int c = 0; c < num_configs; c++) {
        const BenchConfig& bc = configs[c];
        const DsaConfig& d = bc.dsa;
        double flops = compute_flops(d);

        printf("\n======================================================================\n");
        printf("  Config: %s  (Q=%d, H=%d, Dc=%d, Dp=%d, S=%d, P=%d, N=%d)\n",
               bc.label, d.num_queries, d.num_heads,
               d.head_dim_compressed, d.head_dim_positional,
               d.num_selected_kv, d.page_size, d.total_kv_tokens);
        printf("  FLOP/call: %.2e\n", flops);
        printf("======================================================================\n");
        printf("    %-8s  %8s  %8s  %10s  %12s\n",
               "Variant", "Min(ms)", "Mean(ms)", "GFLOP/s", "Q/s");
        printf("    %s\n", std::string(54, '-').c_str());

        for (int v = 0; v < num_variants; v++) {
            bench_variant(variants[v], bc, warmup, iters, flops);
        }
    }

    printf("\n=== Benchmark complete ===\n");
    return 0;
}
