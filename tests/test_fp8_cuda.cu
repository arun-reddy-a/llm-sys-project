// test_fp8_cuda.cu
// Reads float32 activations + FP8 weights + block scales + pre-computed routing
// from binary files, runs moe_forward_deepseek_fp8, writes output.
//
// Usage:
//   ./build/test_fp8_cuda <data_dir> <T> <E_local> <E_global> <K> <D> <I>
//
// Expected files in data_dir:
//   input.bin         float32  [T, D]
//   gate_weight.bin   float32  [E_global, D]
//   gate_bias.bin     float32  [E_global]
//   w1_fp8.bin        uint8    [E_local, 2*I, D]   (FP8 raw bytes, E4M3)
//   w1_scales.bin     float32  [E_local, ceil(2I/128), ceil(D/128)]
//   w2_fp8.bin        uint8    [E_local, D, I]      (FP8 raw bytes, E4M3)
//   w2_scales.bin     float32  [E_local, ceil(D/128), ceil(I/128)]
//
// Writes:
//   cuda_output.bin   float32  [T, D]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include "../kernels/moe/naive_moe.cuh"
#include "../utils/cuda_utils.cuh"

// ---------------------------------------------------------------------------
// Simple file I/O helpers
// ---------------------------------------------------------------------------
static std::vector<uint8_t> read_bin(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", path.c_str()); exit(1); }
    fseek(f, 0, SEEK_END);
    size_t sz = ftell(f);
    rewind(f);
    std::vector<uint8_t> buf(sz);
    fread(buf.data(), 1, sz, f);
    fclose(f);
    return buf;
}

static void write_bin(const std::string& path, const void* data, size_t bytes) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "ERROR: cannot write %s\n", path.c_str()); exit(1); }
    fwrite(data, 1, bytes, f);
    fclose(f);
}

template<typename T>
static T* upload(const std::vector<uint8_t>& host_bytes) {
    T* d;
    CUDA_CHECK(cudaMalloc(&d, host_bytes.size()));
    CUDA_CHECK(cudaMemcpy(d, host_bytes.data(), host_bytes.size(), cudaMemcpyHostToDevice));
    return d;
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 8) {
        fprintf(stderr,
            "Usage: %s <data_dir> <T> <E_local> <E_global> <K> <D> <I>\n", argv[0]);
        return 1;
    }

    std::string dir = argv[1];
    int T       = atoi(argv[2]);
    int E_local = atoi(argv[3]);
    int E_global= atoi(argv[4]);
    int K       = atoi(argv[5]);
    int D       = atoi(argv[6]);
    int I       = atoi(argv[7]);

    int W1_N_BLKS = (2 * I + 127) / 128;
    int W1_D_BLKS = (D     + 127) / 128;
    int W2_N_BLKS = (D     + 127) / 128;
    int W2_D_BLKS = (I     + 127) / 128;

    printf("=== FP8 MoE Correctness Test ===\n");
    printf("  T=%d  E_local=%d  E_global=%d  K=%d  D=%d  I=%d\n",
           T, E_local, E_global, K, D, I);

    // ---- Load all inputs ----
    auto h_input       = read_bin(dir + "/input.bin");
    auto h_gate_weight = read_bin(dir + "/gate_weight.bin");
    auto h_gate_bias   = read_bin(dir + "/gate_bias.bin");
    auto h_w1_fp8      = read_bin(dir + "/w1_fp8.bin");
    auto h_w1_scales   = read_bin(dir + "/w1_scales.bin");
    auto h_w2_fp8      = read_bin(dir + "/w2_fp8.bin");
    auto h_w2_scales   = read_bin(dir + "/w2_scales.bin");

    // Validate sizes
    auto check = [](const char* name, size_t got, size_t want){
        if (got != want) {
            fprintf(stderr, "ERROR: %s size %zu != expected %zu\n", name, got, want);
            exit(1);
        }
    };
    check("input",       h_input.size(),       (size_t)T * D * 4);
    check("gate_weight", h_gate_weight.size(),  (size_t)E_global * D * 4);
    check("gate_bias",   h_gate_bias.size(),    (size_t)E_global * 4);
    check("w1_fp8",      h_w1_fp8.size(),       (size_t)E_local * (2*I) * D * 1);
    check("w1_scales",   h_w1_scales.size(),    (size_t)E_local * W1_N_BLKS * W1_D_BLKS * 4);
    check("w2_fp8",      h_w2_fp8.size(),       (size_t)E_local * D * I * 1);
    check("w2_scales",   h_w2_scales.size(),    (size_t)E_local * W2_N_BLKS * W2_D_BLKS * 4);

    // ---- Upload to GPU ----
    float*             d_input       = upload<float>(h_input);
    float*             d_gate_weight = upload<float>(h_gate_weight);
    float*             d_gate_bias   = upload<float>(h_gate_bias);
    __nv_fp8_e4m3*     d_w1_fp8     = upload<__nv_fp8_e4m3>(h_w1_fp8);
    float*             d_w1_scales  = upload<float>(h_w1_scales);
    __nv_fp8_e4m3*     d_w2_fp8    = upload<__nv_fp8_e4m3>(h_w2_fp8);
    float*             d_w2_scales  = upload<float>(h_w2_scales);

    float* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, (size_t)T * D * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_output, 0, (size_t)T * D * sizeof(float)));

    // ---- Run FP8 kernel ----
    MoeConfig cfg;
    cfg.num_tokens         = T;
    cfg.num_experts        = E_global;
    cfg.num_local_experts  = E_local;
    cfg.top_k              = K;
    cfg.hidden_dim         = D;
    cfg.intermediate_dim   = I;
    cfg.n_group            = 8;
    cfg.topk_group         = 4;
    cfg.routed_scaling_factor = 1.0f;

    // Warmup
    moe_forward_deepseek_fp8(d_input, d_gate_weight, d_gate_bias,
                              d_w1_fp8, d_w1_scales,
                              d_w2_fp8, d_w2_scales,
                              d_output, cfg, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed run
    GpuTimer timer;
    timer.begin();
    moe_forward_deepseek_fp8(d_input, d_gate_weight, d_gate_bias,
                              d_w1_fp8, d_w1_scales,
                              d_w2_fp8, d_w2_scales,
                              d_output, cfg, 0);
    timer.end();
    float ms = timer.elapsed_ms();
    printf("  Kernel time: %.3f ms\n", ms);

    // ---- Download and save output ----
    std::vector<float> h_output(T * D);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output,
                          (size_t)T * D * sizeof(float), cudaMemcpyDeviceToHost));
    write_bin(dir + "/cuda_output.bin", h_output.data(), h_output.size() * 4);

    printf("  Output saved to %s/cuda_output.bin\n", dir.c_str());

    // Cleanup
    cudaFree(d_input); cudaFree(d_gate_weight); cudaFree(d_gate_bias);
    cudaFree(d_w1_fp8); cudaFree(d_w1_scales);
    cudaFree(d_w2_fp8); cudaFree(d_w2_scales);
    cudaFree(d_output);
    return 0;
}
