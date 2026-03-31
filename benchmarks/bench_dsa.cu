#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include "../kernels/dsa/naive_dsa.cuh"
#include "../utils/cuda_utils.cuh"

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

typedef void (*dsa_fn)(const float*, const float*, const float*, const float*, const float*, const int*, float*, const DsaConfig&, cudaStream_t);

static void bench_one(const char* variant_name, dsa_fn func, const BenchConfig& bc, int warmup, int iters) {
    const DsaConfig& cfg = bc.dsa;
    int Q  = cfg.num_queries;
    int H  = cfg.num_heads;
    int Dc = cfg.head_dim_compressed;
    int Dp = cfg.head_dim_positional;
    int S  = cfg.num_selected_kv;
    int N  = cfg.total_kv_tokens;

    std::vector<float> h_q_nope(Q * H * Dc);
    std::vector<float> h_q_pe(Q * H * Dp);
    std::vector<float> h_kv_c(N * Dc);
    std::vector<float> h_kv_p(N * Dp);
    std::vector<float> h_v(N * Dc);
    std::vector<int>   h_idx(Q * S);

    random_fill(h_q_nope.data(), Q * H * Dc, -0.3f, 0.3f);
    random_fill(h_q_pe.data(),   Q * H * Dp, -0.3f, 0.3f);
    random_fill(h_kv_c.data(),   N * Dc, -0.3f, 0.3f);
    random_fill(h_kv_p.data(),   N * Dp, -0.3f, 0.3f);
    random_fill(h_v.data(),      N * Dc, -0.3f, 0.3f);
    gen_sparse_indices(h_idx.data(), Q, S, N);

    DeviceBuf<float> d_q_nope(Q * H * Dc);
    DeviceBuf<float> d_q_pe(Q * H * Dp);
    DeviceBuf<float> d_kv_c(N * Dc);
    DeviceBuf<float> d_kv_p(N * Dp);
    DeviceBuf<float> d_v(N * Dc);
    DeviceBuf<int>   d_idx(Q * S);
    DeviceBuf<float> d_output(Q * H * Dc);

    d_q_nope.upload(h_q_nope.data());
    d_q_pe.upload(h_q_pe.data());
    d_kv_c.upload(h_kv_c.data());
    d_kv_p.upload(h_kv_p.data());
    d_v.upload(h_v.data());
    d_idx.upload(h_idx.data());

    // Warm-up
    for (int i = 0; i < warmup; i++) {
        func(d_q_nope.ptr, d_q_pe.ptr, d_kv_c.ptr, d_kv_p.ptr,
             d_v.ptr, d_idx.ptr, d_output.ptr, cfg, 0);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    std::vector<float> times(iters);
    GpuTimer timer;
    for (int i = 0; i < iters; i++) {
        timer.begin();
        func(d_q_nope.ptr, d_q_pe.ptr, d_kv_c.ptr, d_kv_p.ptr,
             d_v.ptr, d_idx.ptr, d_output.ptr, cfg, 0);
        timer.end();
        times[i] = timer.elapsed_ms();
    }

    std::sort(times.begin(), times.end());
    float t_min    = times.front();
    float t_mean   = 0.0f;
    for (float t : times) t_mean += t;
    t_mean /= iters;

    float queries_per_sec = Q / (t_mean * 1e-3f);

    printf("  %-10s  %-40s  %8.3f  %8.3f  %12.0f\n",
           variant_name, bc.label, t_min, t_mean, queries_per_sec);
}

int main(int argc, char** argv) {
    int warmup = 10;
    int iters  = 50;

    if (argc > 1) warmup = atoi(argv[1]);
    if (argc > 2) iters  = atoi(argv[2]);

    printf("=== DSA Kernel Comparison Benchmark ===\n");
    printf("    warmup=%d  iters=%d\n\n", warmup, iters);
    printf("  %-10s  %-40s  %8s  %8s  %12s\n",
           "Variant", "Config", "Min(ms)", "Mean(ms)", "Q/s");
    printf("  %s\n", std::string(115, '-').c_str());

    srand(123);

    BenchConfig configs[] = {
        {"Q=8,H=16,Dc=256,Dp=64,S=256,N=2048",
         {8, 16, 256, 64, 256, 64, 2048}},
        {"Q=8,H=16,Dc=512,Dp=64,S=1024,N=8192",
         {8, 16, 512, 64, 1024, 64, 8192}},
    };

    struct { const char* name; dsa_fn fn; } variants[] = {
        {"Naive", dsa_forward_naive},
        {"Opt1",  dsa_forward_opt1},
        {"Opt2",  dsa_forward_opt2},
        {"Opt3",  dsa_forward_opt3},
    };

    for (auto& bc : configs) {
        for (auto& v : variants) {
            bench_one(v.name, v.fn, bc, warmup, iters);
        }
        printf("  %s\n", std::string(115, '-').c_str());
    }

    printf("\n");
    return 0;
}
