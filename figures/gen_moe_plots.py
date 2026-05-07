import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,
    "axes.grid":True,"grid.linestyle":"--","grid.alpha":0.45,
    "axes.spines.top":False,"axes.spines.right":False})

# Data: mean latency (ms) @ E=256,EL=32,K=8,D=7168,I=2048 on B200
T  = [64,512,2048]
D  = {"Naive":[40.1,243.7,914.3],"Opt1":[31.2,193.1,722.3],
      "Opt2":[33.0,189.5,731.9],"Opt3":[4.05,12.84,39.4],"Opt4*":[4.92,12.14,40.3]}
VARS   = list(D.keys())
LABELS = ["Naive","Opt 1\n(Tiled GEMM)","Opt 2\n(Fused Routing)","Opt 3\n(Grouped GEMM)","Opt 4\n(Dbl-Buf GEMM)"]
COLS   = ["#3b1f8c","#2e7bb4","#3aada8","#45b86e","#b8e04a"]
x,W    = np.arange(5), 0.62

# Fig1: Latency
fig,axes = plt.subplots(1,3,figsize=(14,4.8))
fig.suptitle("MoE kernel variants (DeepSeek-V3 scale, NVIDIA B200) — Latency (ms)",
             fontsize=13,fontweight="bold",y=1.01)
for i,(ax,t) in enumerate(zip(axes,T)):
    v=[D[k][i] for k in VARS]
    bars=ax.bar(x,v,width=W,color=COLS,zorder=3,edgecolor="white",linewidth=0.6)
    for b,val in zip(bars,v):
        ax.text(b.get_x()+W/2,b.get_height()+max(v)*0.015,
                f"{val:.0f}" if val>=20 else f"{val:.1f}",
                ha="center",va="bottom",fontsize=9,fontweight="bold")
    ax.set_title(f"T = {t} tokens",fontsize=12,fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(LABELS,fontsize=8)
    ax.set_ylabel("Latency (ms)" if i==0 else "")
    ax.set_ylim(0,max(v)*1.20)
plt.tight_layout()
fig.savefig("/home/rrongali/llm-sys-project/figures/moe_latency_by_seqlen.png",dpi=180,bbox_inches="tight")
plt.close(); print("saved latency fig")

# Fig2: Speedup
SVS=VARS[1:]; SL=LABELS[1:]; SC=COLS[1:]; x2=np.arange(4)
fig,axes=plt.subplots(1,3,figsize=(14,4.8))
fig.suptitle("MoE kernel variants — Speedup vs. Naive baseline (NVIDIA B200)",
             fontsize=13,fontweight="bold",y=1.01)
for i,(ax,t) in enumerate(zip(axes,T)):
    sp=[D["Naive"][i]/D[k][i] for k in SVS]
    bars=ax.bar(x2,sp,width=W,color=SC,zorder=3,edgecolor="white",linewidth=0.6)
    ax.axhline(1,color="#3b1f8c",linestyle="--",linewidth=1.2,alpha=0.7,label="Naive (1×)")
    for b,val in zip(bars,sp):
        ax.text(b.get_x()+W/2,b.get_height()+max(sp)*0.02,f"{val:.1f}×",
                ha="center",va="bottom",fontsize=9.5,fontweight="bold")
    ax.set_title(f"T = {t} tokens",fontsize=12,fontweight="bold")
    ax.set_xticks(x2); ax.set_xticklabels(SL,fontsize=8)
    ax.set_ylabel("Speedup vs. Naive" if i==0 else "")
    ax.set_ylim(0,max(sp)*1.25)
    if i==0: ax.legend(fontsize=8,loc="upper left")
plt.tight_layout()
fig.savefig("/home/rrongali/llm-sys-project/figures/moe_speedup_vs_naive.png",dpi=180,bbox_inches="tight")
plt.close(); print("saved speedup fig")
