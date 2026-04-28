#include <cstdio>
#include <vector>
#include <algorithm>
#include "../kernels/moe/naive_moe.cuh"
#include "../utils/cuda_utils.cuh"

typedef void (*moe_fn)(const float*,const float*,const float*,const float*,float*,const MoeConfig&,cudaStream_t);

static void bench_one(const char* vname, moe_fn fn, const MoeConfig& cfg, int warmup, int iters) {
    int T=cfg.num_tokens, E=cfg.num_experts, EL=cfg.num_local_experts;
    int D=cfg.hidden_dim, I=cfg.intermediate_dim;
    DeviceBuf<float> d_in((size_t)T*D), d_gate((size_t)E*D),
                     d_w1((size_t)EL*2*I*D), d_w2((size_t)EL*D*I), d_out((size_t)T*D);
    for(int i=0;i<warmup;i++) fn(d_in.ptr,d_gate.ptr,d_w1.ptr,d_w2.ptr,d_out.ptr,cfg,0);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> times(iters);
    GpuTimer timer;
    for(int i=0;i<iters;i++){
        timer.begin();
        fn(d_in.ptr,d_gate.ptr,d_w1.ptr,d_w2.ptr,d_out.ptr,cfg,0);
        timer.end();
        times[i]=timer.elapsed_ms();
    }
    float tmean=0; for(float t:times) tmean+=t; tmean/=iters;
    printf("%s,%d,%.4f,%.0f\n",vname,T,tmean,(float)T/(tmean*1e-3f));
    fflush(stdout);
}

int main() {
    int warmup=3, iters=10;
    int Ts[]={64,256,512,1024,2048,4096};
    printf("variant,T,mean_ms,toks_per_sec\n");
    for(int T : Ts) {
        MoeConfig cfg={T,256,32,8,7168,2048,8,4,1.0f};
        // Skip Naive for large T (would take many minutes)
        if(T<=1024) bench_one("Naive", moe_forward_naive, cfg, warmup, iters);
        bench_one("Opt1",  moe_forward_opt1, cfg, warmup, iters);
        bench_one("Opt2",  moe_forward_opt2, cfg, warmup, iters);
        bench_one("Opt3",  moe_forward_opt3, cfg, warmup, iters);
        bench_one("Opt5",  moe_forward_opt5, cfg, warmup, iters);
    }
    return 0;
}
