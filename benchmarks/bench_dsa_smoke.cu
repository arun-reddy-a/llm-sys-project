#include "../kernels/dsa/naive_dsa.cuh"
#include "../utils/cuda_utils.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <nvToolsExt.h>

typedef void (*dsa_fn)(const float*, const float*,
                       const float*, const float*, const float*,
                       const int*, float*,
                       const DsaConfig&, cudaStream_t);

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

void bench_one(const char* name, dsa_fn fn, const DsaConfig& cfg,
               int warmup, int iters) {
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

    nvtxRangePushA("warmup");
    for (int i = 0; i < warmup; i++)
        fn(d_q_nope.ptr, d_q_pe.ptr, d_kv_c.ptr, d_kv_p.ptr,
           d_v.ptr, d_idx.ptr, d_output.ptr, cfg, 0);
    cudaStreamSynchronize(0);
    nvtxRangePop();

    char range_name[128];
    sprintf(range_name, "bench_%s", name);
    nvtxRangePushA(range_name);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        sprintf(range_name, "%s_iter_%d", name, i);
        nvtxRangePushA(range_name);
        fn(d_q_nope.ptr, d_q_pe.ptr, d_kv_c.ptr, d_kv_p.ptr,
           d_v.ptr, d_idx.ptr, d_output.ptr, cfg, 0);
        nvtxRangePop();
    }
    cudaStreamSynchronize(0);
    auto end = std::chrono::high_resolution_clock::now();
    nvtxRangePop();

    double ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;
    double queries_per_sec = (Q * 1000.0) / ms;

    printf("  %-10s  %-30s  %8.3f ms  %12.0f Q/s\n",
           name, "DS-V3 Scale (Q=8)", ms, queries_per_sec);
}

int main(int argc, char** argv) {
    int warmup = 2;
    int iters  = 5;

    std::string variant = "";
    if (argc > 1) variant = argv[1];

    if (variant.empty()) {
        printf("=== DSA Smoke Benchmark — All Variants (iters=%d) ===\n\n", iters);
    } else {
        printf("=== DSA Profiling: %s (iters=%d) ===\n\n", variant.c_str(), iters);
    }
    printf("  Variant     Description                     Latency      Throughput\n");
    printf("  --------------------------------------------------------------------------------\n");

    srand(123);

    // DeepSeek-V3 representative config
    // {num_queries, num_heads, head_dim_compressed, head_dim_positional,
    //  num_selected_kv, page_size, total_kv_tokens}
    DsaConfig ds_v3 = {8, 16, 512, 64, 1024, 64, 8192};

    if (variant.empty() || variant == "Naive")
        bench_one("Naive", dsa_forward_naive, ds_v3, warmup, iters);
    if (variant.empty() || variant == "Opt1")
        bench_one("Opt1",  dsa_forward_opt1,  ds_v3, warmup, iters);
    if (variant.empty() || variant == "Opt2")
        bench_one("Opt2",  dsa_forward_opt2,  ds_v3, warmup, iters);
    if (variant.empty() || variant == "Opt3")
        bench_one("Opt3",  dsa_forward_opt3,  ds_v3, warmup, iters);
    if (variant.empty() || variant == "Opt4")
        bench_one("Opt4",  dsa_forward_opt4,  ds_v3, warmup, iters);
    if (variant.empty() || variant == "Opt5")
        bench_one("Opt5",  dsa_forward_opt5,  ds_v3, warmup, iters);
    if (variant.empty() || variant == "Opt6")
        bench_one("Opt6",  dsa_forward_opt6,  ds_v3, warmup, iters);
    if (variant.empty() || variant == "Opt7")
        bench_one("Opt7",  dsa_forward_opt7,  ds_v3, warmup, iters);
    if (variant.empty() || variant == "Opt8")
        bench_one("Opt8",  dsa_forward_opt8,  ds_v3, warmup, iters);

    printf("  --------------------------------------------------------------------------------\n");
    return 0;
}
