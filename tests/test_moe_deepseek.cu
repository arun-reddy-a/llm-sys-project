#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <vector>
#include <algorithm>
#include "../kernels/moe/moe_kernels.cuh"
#include "../utils/cuda_utils.cuh"

// ===================================================================
// CPU reference implementation for DeepSeek-V3 routing
// ===================================================================

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static inline void cpu_silu(float x, float& out) {
    out = x / (1.0f + expf(-x));
}

struct CpuMoeResult {
    std::vector<int>   expert_indices;   // [T*K]
    std::vector<float> expert_weights;   // [T*K]
    std::vector<float> output;           // [T*D]
};

static CpuMoeResult cpu_moe_forward_deepseek(const float* input, const float* gate_weight, const float* gate_bias,
                                              const float* w1, const float* w2,
                                              const MoeConfig& cfg) {
    int T = cfg.num_tokens, E = cfg.num_experts, K = cfg.top_k;
    int D = cfg.hidden_dim, I = cfg.intermediate_dim;
    int N_GROUP = cfg.n_group;
    int TOPK_GROUP = cfg.topk_group;
    float scaling = cfg.routed_scaling_factor;
    int group_size = E / N_GROUP;

    CpuMoeResult res;
    res.expert_indices.resize(T * K);
    res.expert_weights.resize(T * K);
    res.output.assign(T * D, 0.0f);

    for (int t = 0; t < T; t++) {
        std::vector<float> s_sigmoid(E);
        std::vector<float> s_wb(E);
        for (int e = 0; e < E; e++) {
            float dot = 0.0f;
            for (int d = 0; d < D; d++) dot += input[t * D + d] * gate_weight[e * D + d];
            s_sigmoid[e] = sigmoid(dot);
            s_wb[e] = s_sigmoid[e] + gate_bias[e];
        }

        // Group scoring: sum of top-2 s_wb per group
        std::vector<float> group_scores(N_GROUP);
        for (int gn = 0; gn < N_GROUP; gn++) {
            float g_top1 = -FLT_MAX, g_top2 = -FLT_MAX;
            for (int e_in_g = 0; e_in_g < group_size; e_in_g++) {
                float val = s_wb[gn * group_size + e_in_g];
                if (val > g_top1) { g_top2 = g_top1; g_top1 = val; }
                else if (val > g_top2) { g_top2 = val; }
            }
            group_scores[gn] = g_top1 + g_top2;
        }

        // Select top TOPK_GROUP groups
        std::vector<int> top_groups(N_GROUP);
        for (int i = 0; i < N_GROUP; i++) top_groups[i] = i;
        std::sort(top_groups.begin(), top_groups.end(), [&](int a, int b) {
            return group_scores[a] > group_scores[b];
        });

        // Select global top-K experts from the kept groups
        struct ExpertScore { int id; float score; };
        std::vector<ExpertScore> candidates;
        for (int i = 0; i < TOPK_GROUP; i++) {
            int gn = top_groups[i];
            for (int e_in_g = 0; e_in_g < group_size; e_in_g++) {
                int e = gn * group_size + e_in_g;
                candidates.push_back({e, s_wb[e]});
            }
        }
        std::sort(candidates.begin(), candidates.end(), [](const ExpertScore& a, const ExpertScore& b) {
            return a.score > b.score;
        });

        float w_sum = 0.0f;
        for (int k = 0; k < K; k++) {
            int e = candidates[k].id;
            res.expert_indices[t * K + k] = e;
            res.expert_weights[t * K + k] = s_sigmoid[e];
            w_sum += s_sigmoid[e];
        }
        w_sum += 1e-20f;
        for (int k = 0; k < K; k++) {
            res.expert_weights[t * K + k] = (res.expert_weights[t * K + k] / w_sum) * scaling;
        }
    }

    // FFN execution (standard grouped logic)
    for (int e = 0; e < E; e++) {
        std::vector<int> tmap;
        for (int t = 0; t < T; t++) {
            for (int k = 0; k < K; k++) {
                if (res.expert_indices[t * K + k] == e) { tmap.push_back(t); break; }
            }
        }
        if (tmap.empty()) continue;

        const float* w1_e = w1 + (size_t)e * 2 * I * D;
        const float* w2_e = w2 + (size_t)e * D * I;

        for (int m : tmap) {
            std::vector<float> act(I);
            for (int i = 0; i < I; i++) {
                float g = 0.0f, u = 0.0f;
                for (int d = 0; d < D; d++) {
                    g += input[m * D + d] * w1_e[i * D + d];
                    u += input[m * D + d] * w1_e[(I + i) * D + d];
                }
                float silu_g;
                cpu_silu(g, silu_g);
                act[i] = silu_g * u;
            }

            float w = 0.0f;
            for (int k = 0; k < K; k++) if (res.expert_indices[m * K + k] == e) { w = res.expert_weights[m * K + k]; break; }

            for (int d = 0; d < D; d++) {
                float sum = 0.0f;
                for (int i = 0; i < I; i++) sum += act[i] * w2_e[d * I + i];
                res.output[m * D + d] += w * sum;
            }
        }
    }

    return res;
}

static bool run_deepseek_test(const char* name, const MoeConfig& cfg, float tol) {
    printf("  %-30s  ", name);
    int T = cfg.num_tokens, E = cfg.num_experts, EL = cfg.num_local_experts;
    int D = cfg.hidden_dim, I = cfg.intermediate_dim;

    srand(42);

    size_t input_sz = T * D;
    size_t gate_sz  = E * D;
    size_t bias_sz  = E;
    size_t w1_sz    = (size_t)EL * 2 * I * D;
    size_t w2_sz    = (size_t)EL * D * I;

    std::vector<float> h_input(input_sz), h_gate(gate_sz), h_bias(bias_sz);
    std::vector<float> h_w1(w1_sz), h_w2(w2_sz);

    random_fill(h_input.data(), input_sz, -0.5f, 0.5f);
    random_fill(h_gate.data(), gate_sz, -0.5f, 0.5f);
    random_fill(h_bias.data(), bias_sz, -0.1f, 0.1f);
    random_fill(h_w1.data(), w1_sz, -0.1f, 0.1f);
    random_fill(h_w2.data(), w2_sz, -0.1f, 0.1f);

    // GPU Execution buffers
    DeviceBuf<float> d_input(input_sz), d_gate(gate_sz), d_bias(bias_sz);
    DeviceBuf<float> d_w1(w1_sz), d_w2(w2_sz), d_output(T * D), d_output_naive(T * D);

    d_input.upload(h_input.data());
    d_gate.upload(h_gate.data());
    d_bias.upload(h_bias.data());
    d_w1.upload(h_w1.data());
    d_w2.upload(h_w2.data());

    // 1. Naive GPU Execution (Ground truth without host memory limit)
    moe_forward_deepseek_naive(d_input.ptr, d_gate.ptr, d_bias.ptr, d_w1.ptr, d_w2.ptr, d_output_naive.ptr, cfg, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> h_output_naive(T * D);
    d_output_naive.download(h_output_naive.data());

    // 2. Optimized GPU Execution
    moe_forward_deepseek(d_input.ptr, d_gate.ptr, d_bias.ptr, d_w1.ptr, d_w2.ptr, d_output.ptr, cfg, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> h_output(T * D);
    d_output.download(h_output.data());

    // Compare Naive vs Optimized
    float max_err = max_abs_error(h_output_naive.data(), h_output.data(), T * D);
    bool pass = max_err < tol;
    printf("max_err=%.6e  %s\n", max_err, pass ? "PASS" : "FAIL");
    return pass;
}

int main() {
    printf("=== DeepSeek-V3 MoE Correctness Tests ===\n\n");
    int passed = 0, total = 0;

    // 1. DeepSeek-V3 PRODUCTION Config (D=7168, I=2048, E=256 full scale, using GPU naive reference)
    {
        MoeConfig cfg = {8, 256, 32, 8, 7168, 2048, 8, 4, 1.0f};
        total++; if (run_deepseek_test("DeepSeek Production (Arch)", cfg, 2.0e-2f)) passed++;
    }

    // 2. DeepSeek SMALL
    {
        MoeConfig cfg = {8, 16, 16, 4, 128, 256, 4, 2, 1.0f};
        total++; if (run_deepseek_test("DeepSeek Small (T=8)", cfg, 2.0e-2f)) passed++;
    }

    // 3. DeepSeek MEDIUM
    {
        MoeConfig cfg = {32, 64, 64, 8, 256, 512, 8, 4, 1.0f};
        total++; if (run_deepseek_test("DeepSeek Medium (T=32)", cfg, 2.0e-2f)) passed++;
    }

    printf("\nResults: %d / %d passed\n", passed, total);
    return (passed == total) ? 0 : 1;
}
