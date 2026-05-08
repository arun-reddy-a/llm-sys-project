#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <vector>
#include <algorithm>
#include "../kernels/moe/naive_moe.cuh"
#include "../utils/cuda_utils.cuh"

// ---------------------------------------------------------------------------
// CPU reference: FP32 DeepSeek-V3 routing + FFN (same as test_moe_deepseek.cu)
// ---------------------------------------------------------------------------

static inline float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

static std::vector<float> cpu_moe_forward(
    const float* input, const float* gate_weight, const float* gate_bias,
    const float* w1, const float* w2, const MoeConfig& cfg)
{
    int T = cfg.num_tokens, E = cfg.num_experts, K = cfg.top_k;
    int D = cfg.hidden_dim, I = cfg.intermediate_dim;
    int N_GROUP = cfg.n_group, TOPK_GROUP = cfg.topk_group;
    float scaling = cfg.routed_scaling_factor;
    int group_size = E / N_GROUP;

    std::vector<int>   expert_ids(T * K);
    std::vector<float> expert_wts(T * K);
    std::vector<float> output(T * D, 0.0f);

    for (int t = 0; t < T; t++) {
        std::vector<float> s_sig(E), s_wb(E);
        for (int e = 0; e < E; e++) {
            float dot = 0.0f;
            for (int d = 0; d < D; d++) dot += input[t * D + d] * gate_weight[e * D + d];
            s_sig[e] = sigmoidf(dot);
            s_wb[e]  = s_sig[e] + gate_bias[e];
        }

        std::vector<float> group_scores(N_GROUP);
        for (int gn = 0; gn < N_GROUP; gn++) {
            float top1 = -FLT_MAX, top2 = -FLT_MAX;
            for (int i = 0; i < group_size; i++) {
                float v = s_wb[gn * group_size + i];
                if (v > top1) { top2 = top1; top1 = v; }
                else if (v > top2) top2 = v;
            }
            group_scores[gn] = top1 + top2;
        }

        std::vector<int> top_groups(N_GROUP);
        for (int i = 0; i < N_GROUP; i++) top_groups[i] = i;
        std::sort(top_groups.begin(), top_groups.end(),
                  [&](int a, int b) { return group_scores[a] > group_scores[b]; });

        struct ES { int id; float score; };
        std::vector<ES> cands;
        for (int i = 0; i < TOPK_GROUP; i++) {
            int gn = top_groups[i];
            for (int j = 0; j < group_size; j++)
                cands.push_back({gn * group_size + j, s_wb[gn * group_size + j]});
        }
        std::sort(cands.begin(), cands.end(), [](const ES& a, const ES& b) { return a.score > b.score; });

        float w_sum = 0.0f;
        for (int k = 0; k < K; k++) {
            expert_ids[t * K + k] = cands[k].id;
            expert_wts[t * K + k] = s_sig[cands[k].id];
            w_sum += s_sig[cands[k].id];
        }
        w_sum += 1e-20f;
        for (int k = 0; k < K; k++) expert_wts[t * K + k] = expert_wts[t * K + k] / w_sum * scaling;
    }

    for (int e = 0; e < E; e++) {
        std::vector<int> tmap;
        for (int t = 0; t < T; t++)
            for (int k = 0; k < K; k++)
                if (expert_ids[t * K + k] == e) { tmap.push_back(t); break; }
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
                act[i] = g / (1.0f + expf(-g)) * u;
            }
            float w = 0.0f;
            for (int k = 0; k < K; k++)
                if (expert_ids[m * K + k] == e) { w = expert_wts[m * K + k]; break; }
            for (int d = 0; d < D; d++) {
                float sum = 0.0f;
                for (int i = 0; i < I; i++) sum += act[i] * w2_e[d * I + i];
                output[m * D + d] += w * sum;
            }
        }
    }
    return output;
}

// ---------------------------------------------------------------------------
// Test runner: moe_forward() (FP32 compat alias) vs CPU FP32 reference.
// ---------------------------------------------------------------------------

static bool run_test(const char* name, const MoeConfig& cfg, float tol) {
    printf("  %-40s  ", name);
    int T = cfg.num_tokens, E = cfg.num_experts;
    int D = cfg.hidden_dim, I = cfg.intermediate_dim;
    srand(42);

    std::vector<float> h_input(T * D), h_gate((size_t)E * D), h_bias(E);
    std::vector<float> h_w1((size_t)E * 2 * I * D), h_w2((size_t)E * D * I);
    random_fill(h_input.data(), T * D, -0.5f, 0.5f);
    random_fill(h_gate.data(),  E * D, -0.5f, 0.5f);
    zero_fill(h_bias.data(), E);
    random_fill(h_w1.data(), (size_t)E * 2 * I * D, -0.05f, 0.05f);
    random_fill(h_w2.data(), (size_t)E * D * I,     -0.05f, 0.05f);

    std::vector<float> ref = cpu_moe_forward(
        h_input.data(), h_gate.data(), h_bias.data(),
        h_w1.data(), h_w2.data(), cfg);

    DeviceBuf<float> d_input(T * D), d_gate(E * D), d_bias(E);
    DeviceBuf<float> d_w1((size_t)E * 2 * I * D), d_w2((size_t)E * D * I), d_out(T * D);
    d_input.upload(h_input.data()); d_gate.upload(h_gate.data()); d_bias.zero();
    d_w1.upload(h_w1.data()); d_w2.upload(h_w2.data());

    moe_forward(d_input.ptr, d_gate.ptr, d_bias.ptr, d_w1.ptr, d_w2.ptr, d_out.ptr, cfg);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_out(T * D);
    d_out.download(h_out.data());

    float max_err  = max_abs_error(ref.data(), h_out.data(), T * D);
    float mean_err = mean_abs_error(ref.data(), h_out.data(), T * D);
    bool pass = max_err < tol;
    printf("max_err=%.4e  mean_err=%.4e  %s\n", max_err, mean_err, pass ? "PASS" : "FAIL");
    return pass;
}

int main() {
    printf("=== moe_forward() FP32 Compat Alias Tests ===\n");
    printf("    (GPU: BF16 cuBLAS path vs CPU FP32 reference, tolerance accounts for BF16 rounding)\n\n");

    int passed = 0, total = 0;

    // n_group must divide num_experts; topk_group <= n_group; top_k <= topk_group * (num_experts/n_group)
    { MoeConfig c = {8,  8, 8, 2, 64,  128, 2, 1, 1.0f};
      total++; if (run_test("small  (T=8,  E=8,  D=64)",  c, 0.5f)) passed++; }
    { MoeConfig c = {16, 16, 16, 4, 128, 256, 4, 2, 1.0f};
      total++; if (run_test("medium (T=16, E=16, D=128)", c, 0.5f)) passed++; }
    { MoeConfig c = {32, 32, 32, 8, 256, 512, 8, 4, 2.5f};
      total++; if (run_test("large  (T=32, E=32, D=256)", c, 0.5f)) passed++; }

    printf("\nResults: %d / %d passed\n", passed, total);
    return (passed == total) ? 0 : 1;
}
