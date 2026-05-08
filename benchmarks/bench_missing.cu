#include <cstdio>
#include <vector>
#include <algorithm>
#include <cuda_bf16.h>
#include "../kernels/moe/naive_moe.cuh"
#include "../utils/cuda_utils.cuh"

typedef void (*bf16_fn)(const __nv_bfloat16*, const float*, const float*,
                        const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*,
                        const MoeConfig&, cudaStream_t);

static void bench_one(const char* vname, bf16_fn fn, const MoeConfig& cfg, int warmup, int iters) {
    int T=cfg.num_tokens, E=cfg.num_experts, EL=cfg.num_local_experts;
    int D=cfg.hidden_dim, I=cfg.intermediate_dim;
    DeviceBuf<__nv_bfloat16> d_in(T*D), d_w1((size_t)EL*2*I*D), d_w2((size_t)EL*D*I), d_out(T*D);
    DeviceBuf<float> d_gate(E*D), d_bias(E);
    d_bias.zero();
    for(int i=0;i<warmup;i++) fn(d_in.ptr,d_gate.ptr,d_bias.ptr,d_w1.ptr,d_w2.ptr,d_out.ptr,cfg,0);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> times(iters);
    GpuTimer timer;
    for(int i=0;i<iters;i++){
        timer.begin();
        fn(d_in.ptr,d_gate.ptr,d_bias.ptr,d_w1.ptr,d_w2.ptr,d_out.ptr,cfg,0);
        timer.end();
        times[i]=timer.elapsed_ms();
    }
    float tmean=0; for(float t:times) tmean+=t; tmean/=iters;
    printf("%s,%d,%.4f,%.0f\n",vname,T,tmean,(float)T/(tmean*1e-3f));
    fflush(stdout);
}

int main() {
    int warmup=1, iters=5;
    printf("variant,T,mean_ms,toks_per_sec\n");

    MoeConfig cfg_2048={2048,256,32,8,7168,2048,8,4,1.0f};
    bench_one("BF16-cuBLAS", moe_forward_deepseek_bf16_cublas, cfg_2048, warmup, iters);

    MoeConfig cfg_4096={4096,256,32,8,7168,2048,8,4,1.0f};
    bench_one("BF16-WMMA",   moe_forward_deepseek_bf16,        cfg_4096, warmup, iters);
    bench_one("BF16-cuBLAS", moe_forward_deepseek_bf16_cublas, cfg_4096, warmup, iters);

    return 0;
}
