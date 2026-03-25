#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <vector>
#include <algorithm>
#include "../kernels/moe/naive_moe.cuh"
#include "../utils/cuda_utils.cuh"

// ===================================================================
// CPU reference implementation
// ===================================================================

static void cpu_softmax(float* data, int n) {
    float mx = -FLT_MAX;
    for (int i = 0; i < n; i++) mx = fmaxf(mx, data[i]);
    float s = 0.0f;
    for (int i = 0; i < n; i++) { data[i] = expf(data[i] - mx); s += data[i]; }
    for (int i = 0; i < n; i++) data[i] /= s;
}

static void cpu_silu(float x, float& out) {
    out = x / (1.0f + expf(-x));
}

struct CpuMoeResult {
    std::vector<int>   expert_indices;   // [T*K]
    std::vector<float> expert_weights;   // [T*K]
    std::vector<float> output;           // [T*D]
};

static CpuMoeResult cpu_moe_forward(const float* input, const float* gate_weight,
                                     const float* w1, const float* w2,
                                     const MoeConfig& cfg) {
    int T = cfg.num_tokens, E = cfg.num_experts, K = cfg.top_k;
    int D = cfg.hidden_dim, I = cfg.intermediate_dim;

    // Gate logits + softmax + topk
    std::vector<float> logits(T * E);
    for (int t = 0; t < T; t++) {
        for (int e = 0; e < E; e++) {
            float sum = 0.0f;
            for (int d = 0; d < D; d++)
                sum += input[t * D + d] * gate_weight[e * D + d];
            logits[t * E + e] = sum;
        }
        cpu_softmax(&logits[t * E], E);
    }

    std::vector<int>   topk_idx(T * K);
    std::vector<float> topk_wt(T * K);
    for (int t = 0; t < T; t++) {
        // Init with worst
        for (int k = 0; k < K; k++) { topk_idx[t*K+k] = -1; topk_wt[t*K+k] = -FLT_MAX; }
        for (int e = 0; e < E; e++) {
            float v = logits[t * E + e];
            int min_k = 0;
            for (int k = 1; k < K; k++)
                if (topk_wt[t*K+k] < topk_wt[t*K+min_k]) min_k = k;
            if (v > topk_wt[t*K+min_k]) {
                topk_wt[t*K+min_k]  = v;
                topk_idx[t*K+min_k] = e;
            }
        }
        // Re-normalise
        float s = 0.0f;
        for (int k = 0; k < K; k++) s += topk_wt[t*K+k];
        if (s > 0.0f) for (int k = 0; k < K; k++) topk_wt[t*K+k] /= s;
    }

    std::vector<float> output(T * D, 0.0f);

    // Per-expert loop
    for (int e = 0; e < E; e++) {
        // Gather
        std::vector<int> tmap;
        for (int t = 0; t < T; t++) {
            for (int k = 0; k < K; k++) {
                if (topk_idx[t*K+k] == e) { tmap.push_back(t); break; }
            }
        }
        int cnt = (int)tmap.size();
        if (cnt == 0) continue;

        const float* w1_e = w1 + (size_t)e * 2 * I * D;
        const float* w2_e = w2 + (size_t)e * D * I;

        // GEMM1: [cnt, D] * w1_e^T -> [cnt, 2I]
        std::vector<float> g1(cnt * 2 * I, 0.0f);
        for (int m = 0; m < cnt; m++) {
            int t = tmap[m];
            for (int n = 0; n < 2 * I; n++) {
                float sum = 0.0f;
                for (int d = 0; d < D; d++)
                    sum += input[t * D + d] * w1_e[n * D + d];
                g1[m * 2 * I + n] = sum;
            }
        }

        // SwiGLU
        std::vector<float> act(cnt * I);
        for (int m = 0; m < cnt; m++) {
            for (int j = 0; j < I; j++) {
                float g = g1[m * 2 * I + j];
                float u = g1[m * 2 * I + I + j];
                float silu_g;
                cpu_silu(g, silu_g);
                act[m * I + j] = silu_g * u;
            }
        }

        // GEMM2: [cnt, I] * w2_e^T -> [cnt, D]
        std::vector<float> g2(cnt * D, 0.0f);
        for (int m = 0; m < cnt; m++) {
            for (int n = 0; n < D; n++) {
                float sum = 0.0f;
                for (int j = 0; j < I; j++)
                    sum += act[m * I + j] * w2_e[n * I + j];
                g2[m * D + n] = sum;
            }
        }

        // Scatter
        for (int m = 0; m < cnt; m++) {
            int t = tmap[m];
            float w = 0.0f;
            for (int k = 0; k < K; k++) {
                if (topk_idx[t*K+k] == e) { w = topk_wt[t*K+k]; break; }
            }
            for (int d = 0; d < D; d++)
                output[t * D + d] += w * g2[m * D + d];
        }
    }

    CpuMoeResult res;
    res.expert_indices = topk_idx;
    res.expert_weights = topk_wt;
    res.output = output;
    return res;
}

// ===================================================================
// Test runner
// ===================================================================

static bool run_test(const char* name, const MoeConfig& cfg, float tol) {
    printf("  %-30s  ", name);
    int T = cfg.num_tokens, E = cfg.num_experts;
    int D = cfg.hidden_dim, I = cfg.intermediate_dim;

    srand(42);

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

    // CPU reference
    CpuMoeResult cpu = cpu_moe_forward(h_input.data(), h_gate.data(),
                                        h_w1.data(), h_w2.data(), cfg);

    // GPU
    DeviceBuf<float> d_input(input_sz);
    DeviceBuf<float> d_gate(gate_sz);
    DeviceBuf<float> d_w1(w1_sz);
    DeviceBuf<float> d_w2(w2_sz);
    DeviceBuf<float> d_output(T * D);

    d_input.upload(h_input.data());
    d_gate.upload(h_gate.data());
    d_w1.upload(h_w1.data());
    d_w2.upload(h_w2.data());

    moe_forward(d_input.ptr, d_gate.ptr, d_w1.ptr, d_w2.ptr,
                d_output.ptr, cfg);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_output(T * D);
    d_output.download(h_output.data());

    float max_err  = max_abs_error(cpu.output.data(), h_output.data(), T * D);
    float mean_err = mean_abs_error(cpu.output.data(), h_output.data(), T * D);

    bool pass = max_err < tol;
    printf("max_err=%.6e  mean_err=%.6e  %s\n", max_err, mean_err,
           pass ? "PASS" : "FAIL");
    return pass;
}

int main() {
    printf("=== MoE Correctness Tests ===\n\n");

    int passed = 0, total = 0;

    // Small sanity test
    {
        MoeConfig cfg = {4, 4, 2, 64, 128};
        total++; if (run_test("small  (T=4, E=4, D=64)", cfg, 1e-3f)) passed++;
    }
    // Medium test
    {
        MoeConfig cfg = {32, 8, 2, 128, 256};
        total++; if (run_test("medium (T=32, E=8, D=128)", cfg, 1e-3f)) passed++;
    }
    // Larger test
    {
        MoeConfig cfg = {64, 8, 2, 256, 512};
        total++; if (run_test("large  (T=64, E=8, D=256)", cfg, 1e-2f)) passed++;
    }
    // Top-K = 4
    {
        MoeConfig cfg = {16, 8, 4, 128, 256};
        total++; if (run_test("topk4  (T=16, E=8, K=4)", cfg, 1e-2f)) passed++;
    }

    printf("\nResults: %d / %d passed\n", passed, total);
    return (passed == total) ? 0 : 1;
}
