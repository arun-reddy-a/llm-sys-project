#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <vector>
#include <algorithm>
#include <numeric>
#include "../kernels/dsa/naive_dsa.cuh"
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

static void cpu_dsa_forward(const float* q_nope, const float* q_pe,
                             const float* kv_compressed, const float* kv_positional,
                             const float* v_cache, const int* sparse_indices,
                             float* output, const DsaConfig& cfg) {
    int Q  = cfg.num_queries;
    int H  = cfg.num_heads;
    int Dc = cfg.head_dim_compressed;
    int Dp = cfg.head_dim_positional;
    int S  = cfg.num_selected_kv;

    float scale = 1.0f / sqrtf((float)(Dc + Dp));

    for (int q = 0; q < Q; q++) {
        // Gather KV
        std::vector<float> kc(S * Dc), kp(S * Dp), v(S * Dc);
        for (int s = 0; s < S; s++) {
            int idx = sparse_indices[q * S + s];
            for (int d = 0; d < Dc; d++) kc[s * Dc + d] = kv_compressed[idx * Dc + d];
            for (int d = 0; d < Dp; d++) kp[s * Dp + d] = kv_positional[idx * Dp + d];
            for (int d = 0; d < Dc; d++) v[s * Dc + d]  = v_cache[idx * Dc + d];
        }

        for (int h = 0; h < H; h++) {
            // Dot products -> scores
            std::vector<float> scores(S, 0.0f);
            for (int s = 0; s < S; s++) {
                float sum_c = 0.0f, sum_p = 0.0f;
                for (int d = 0; d < Dc; d++)
                    sum_c += q_nope[q * H * Dc + h * Dc + d] * kc[s * Dc + d];
                for (int d = 0; d < Dp; d++)
                    sum_p += q_pe[q * H * Dp + h * Dp + d] * kp[s * Dp + d];
                scores[s] = (sum_c + sum_p) * scale;
            }

            cpu_softmax(scores.data(), S);

            // Output projection
            for (int d = 0; d < Dc; d++) {
                float sum = 0.0f;
                for (int s = 0; s < S; s++)
                    sum += scores[s] * v[s * Dc + d];
                output[q * H * Dc + h * Dc + d] = sum;
            }
        }
    }
}

// ===================================================================
// Generate random sparse indices (unique per query, in [0, total_kv))
// ===================================================================
static void gen_sparse_indices(int* indices, int Q, int S, int total_kv) {
    std::vector<int> pool(total_kv);
    for (int q = 0; q < Q; q++) {
        std::iota(pool.begin(), pool.end(), 0);
        // Fisher-Yates partial shuffle
        for (int i = 0; i < S; i++) {
            int j = i + rand() % (total_kv - i);
            std::swap(pool[i], pool[j]);
        }
        std::sort(pool.begin(), pool.begin() + S);
        for (int s = 0; s < S; s++) indices[q * S + s] = pool[s];
    }
}

// ===================================================================
// Test runner
// ===================================================================

static bool run_test(const char* name, const DsaConfig& cfg, float tol) {
    printf("  %-40s  ", name);

    int Q  = cfg.num_queries;
    int H  = cfg.num_heads;
    int Dc = cfg.head_dim_compressed;
    int Dp = cfg.head_dim_positional;
    int S  = cfg.num_selected_kv;
    int N  = cfg.total_kv_tokens;

    srand(42);

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

    // CPU reference
    std::vector<float> cpu_out(Q * H * Dc, 0.0f);
    cpu_dsa_forward(h_q_nope.data(), h_q_pe.data(),
                    h_kv_c.data(), h_kv_p.data(), h_v.data(),
                    h_idx.data(), cpu_out.data(), cfg);

    // GPU
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

    dsa_forward(d_q_nope.ptr, d_q_pe.ptr,
                d_kv_c.ptr, d_kv_p.ptr, d_v.ptr,
                d_idx.ptr, d_output.ptr, cfg);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_output(Q * H * Dc);
    d_output.download(h_output.data());

    float max_err  = max_abs_error(cpu_out.data(), h_output.data(), Q * H * Dc);
    float mean_err = mean_abs_error(cpu_out.data(), h_output.data(), Q * H * Dc);

    bool pass = max_err < tol;
    printf("max=%.6e  mean=%.6e  %s\n", max_err, mean_err,
           pass ? "PASS" : "FAIL");
    return pass;
}

int main() {
    printf("=== DSA Correctness Tests ===\n\n");

    int passed = 0, total = 0;

    // Tiny sanity
    {
        DsaConfig cfg = {2, 2, 32, 8, 16, 8, 64};
        total++; if (run_test("tiny   (Q=2,H=2,Dc=32,S=16)", cfg, 1e-3f)) passed++;
    }
    // Small
    {
        DsaConfig cfg = {4, 4, 64, 16, 64, 16, 256};
        total++; if (run_test("small  (Q=4,H=4,Dc=64,S=64)", cfg, 1e-3f)) passed++;
    }
    // Medium
    {
        DsaConfig cfg = {8, 8, 128, 32, 128, 32, 512};
        total++; if (run_test("medium (Q=8,H=8,Dc=128,S=128)", cfg, 1e-2f)) passed++;
    }
    // Larger (closer to production-like ratios)
    {
        DsaConfig cfg = {4, 4, 128, 32, 256, 64, 1024};
        total++; if (run_test("large  (Q=4,H=4,Dc=128,S=256)", cfg, 1e-2f)) passed++;
    }

    printf("\nResults: %d / %d passed\n", passed, total);
    return (passed == total) ? 0 : 1;
}
