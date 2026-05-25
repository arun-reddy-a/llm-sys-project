import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Sequences
T = [64, 256, 512, 1024, 2048, 4096]

# Data from our runs and the missing values captured
D_lat = {
    "Naive": [220.55, 824.51, 1628.77, 3254.86, 6698.64, 13401.07],
    "Opt1": [166.61, 670.30, 1307.72, 2622.22, 5331.44, 10860.38],
    "Opt2": [166.49, 665.59, 1332.42, 2628.35, 5333.88, 10859.67],
    "Opt3": [7.31, 26.02, 45.07, 88.72, 181.22, 394.34],
    "Opt5": [1.36, 4.10, 7.89, 15.43, 30.44, 60.58],
    "DS-V3": [1.59, 2.08, 1.67, 4.42, 5.68, 10.23] # From README L100-L106 (Note T=64 is 1.58ms minimum, 1.59 mean. T=128 is skipped as per user)
}

D_thr = {
    "Naive": [290, 310, 314, 315, 306, 306],
    "Opt1": [384, 382, 392, 391, 384, 377],
    "Opt2": [384, 385, 384, 390, 384, 377],
    "Opt3": [8756, 9838, 11361, 11542, 11301, 10387],
    "Opt5": [47036, 62380, 64867, 66375, 67285, 67616],
    "DS-V3": [40287, 123054, 306251, 231781, 360436, 400403]
}

def plot_6_subplots(metric_dict, metric_name, file_name, is_latency=True):
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    fig.suptitle(f"MoE kernel variants — {metric_name}", fontsize=17, fontweight="bold", y=0.98)
    
    variants = list(metric_dict.keys())
    colors = ["#3b1f8c", "#2e7bb4", "#3aada8", "#45b86e", "#b8e04a", "#ff9900"]
    labels = ["Naive", "Opt1\n(Tiled)", "Opt2\n(Fused)", "Opt3\n(Grouped)", "Opt4\n(Dbl-Buf)", "DS-V3\n(Prod)"]
    
    for i, t in enumerate(T):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        valid_vals = [metric_dict[v][i] for v in variants]
        x = np.arange(len(valid_vals))
        
        bars = ax.bar(x, valid_vals, color=colors, edgecolor="white", width=0.7)
        
        for bar, val in zip(bars, valid_vals):
            # Format label
            if is_latency:
                lbl = f"{val:.0f}" if val >= 100 else f"{val:.1f}"
            else:
                lbl = f"{val/1000:.0f}k" if val >= 10000 else f"{val:.0f}"
            
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(valid_vals)*0.015,
                    lbl, ha="center", va="bottom", fontsize=10, fontweight="bold")
                    
        ax.set_title(f"T = {t}", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9.5)
        if col == 0:
            ax.set_ylabel(metric_name, fontsize=12)
        
        # Log scale for latency because Naive is 13,000 and DS-V3 is 10.
        if is_latency:
            ax.set_yscale('log')
            ax.set_ylim(1, max(valid_vals) * 2.5)
        else:
            ax.set_yscale('log')
            ax.set_ylim(100, max(valid_vals) * 2.5)
            
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    # Add some spacing for the suptitle
    plt.subplots_adjust(top=0.92)
    fig.savefig(f"/home/rrongali/llm-sys-project/figures/{file_name}", dpi=180, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    plot_6_subplots(D_lat, "Mean Latency (ms) [Log Scale]", "moe_latency_seqlen.png", is_latency=True)
    plot_6_subplots(D_thr, "Throughput (Tok/s) [Log Scale]", "moe_throughput_seqlen.png", is_latency=False)
    print("Done generating updated plots.")
