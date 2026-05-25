"""Generate explanation figures for the MoE Triton kernel optimisations."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np

BLUE   = "#4C72B0"
ORANGE = "#DD8452"
GREEN  = "#55A868"
RED    = "#C44E52"
PURPLE = "#8172B2"
GREY   = "#8C8C8C"
LIGHT  = "#F0F0F0"
DARK   = "#2C2C2C"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#CCCCCC",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Figure 1: Per-expert dispatch vs Single launch ─────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 5))
fig.suptitle("Optimisation 1: Single Kernel Launch", fontsize=14, fontweight="bold", y=1.01)

LAUNCH_OVERHEAD = 0.8   # relative units
COMPUTE         = 1.2
N_EXPERTS       = 8     # simplified to 8 for clarity

ax = axes[0]
ax.set_title("Before: 32 separate launches (one per expert)", color=RED, pad=6)
x = 0
for i in range(N_EXPERTS):
    ax.barh(0, LAUNCH_OVERHEAD, left=x, height=0.5, color=RED,   alpha=0.85, label="Launch overhead" if i==0 else "")
    ax.barh(0, COMPUTE,         left=x+LAUNCH_OVERHEAD, height=0.5, color=BLUE, alpha=0.85, label="Compute" if i==0 else "")
    ax.text(x + LAUNCH_OVERHEAD/2, 0, f"OH", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    ax.text(x + LAUNCH_OVERHEAD + COMPUTE/2, 0, f"E{i}", ha="center", va="center", fontsize=8, color="white")
    x += LAUNCH_OVERHEAD + COMPUTE
ax.set_xlim(0, x); ax.set_yticks([]); ax.set_xlabel("Time →")
ax.legend(loc="upper right", fontsize=9)
ax.annotate(f"Total: {N_EXPERTS} × (launch + compute) — overhead grows with #experts",
            xy=(0.02, -0.25), xycoords="axes fraction", fontsize=9, color=GREY)

ax2 = axes[1]
ax2.set_title("After: One launch — GPU schedules all experts internally", color=GREEN, pad=6)
ax2.barh(0, LAUNCH_OVERHEAD,          height=0.5, color=RED,   alpha=0.85, label="Launch overhead")
ax2.barh(0, N_EXPERTS * COMPUTE,      left=LAUNCH_OVERHEAD, height=0.5, color=BLUE, alpha=0.85, label="All experts (parallel tiles)")
for i in range(N_EXPERTS):
    cx = LAUNCH_OVERHEAD + i * COMPUTE + COMPUTE / 2
    ax2.text(cx, 0, f"E{i}", ha="center", va="center", fontsize=8, color="white")
    if i < N_EXPERTS - 1:
        ax2.axvline(LAUNCH_OVERHEAD + (i+1)*COMPUTE, color="white", linewidth=1, alpha=0.4)
ax2.set_xlim(0, x); ax2.set_yticks([]); ax2.set_xlabel("Time →")
ax2.legend(loc="upper right", fontsize=9)
ax2.annotate("Total: 1 × launch + all compute — overhead paid once regardless of #experts",
             xy=(0.02, -0.25), xycoords="axes fraction", fontsize=9, color=GREY)

plt.tight_layout()
plt.savefig("figures/fig1_single_launch.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ fig1_single_launch.png")

# ── Figure 2: Precomputed tile→expert map ──────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6))
fig.suptitle("Optimisation 2: Precomputed Tile → Expert Map", fontsize=14, fontweight="bold")
ax.axis("off")
ax.set_xlim(0, 13); ax.set_ylim(-0.5, 7.5)

colors_e = [BLUE, GREEN, RED, PURPLE, ORANGE, "#17BECF"]

# Simplified: 5 experts with small token counts so tiles fit cleanly
experts    = [0,  1,  2,  3,  4,  5 ]
tok_counts = [70, 35, 50, 32, 45, 20 ]
BLOCK_M    = 32

# --- Left panel: sorted token rows ---
ax.text(1.5, 7.2, "Token rows\n(sorted by expert)", ha="center", fontsize=11, fontweight="bold")

row_h = 6.5 / sum(tok_counts) * 32   # height per tile-block
y = 6.8
tile_list = []   # (expert, label, y_center)
for e, cnt in zip(experts, tok_counts):
    n_tiles = (cnt + BLOCK_M - 1) // BLOCK_M
    for t in range(n_tiles):
        te = min((t + 1) * BLOCK_M, cnt)
        ts = t * BLOCK_M
        rect = mpatches.FancyBboxPatch(
            (0.1, y - row_h), 2.8, row_h * 0.88,
            boxstyle="round,pad=0.02",
            facecolor=colors_e[e], alpha=0.80, edgecolor="white", linewidth=1.5)
        ax.add_patch(rect)
        label = f"Expert {e}  (rows {sum(tok_counts[:e])+ts}–{sum(tok_counts[:e])+te-1})"
        ax.text(1.5, y - row_h * 0.56, label,
                ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")
        tile_list.append((e, sum(tok_counts[:e]) + ts, sum(tok_counts[:e]) + te, y - row_h * 0.5))
        y -= row_h

# --- Arrow ---
ax.annotate("", xy=(4.6, 3.5), xytext=(3.1, 3.5),
            arrowprops=dict(arrowstyle="-|>", color=DARK, lw=2.5, mutation_scale=18))
ax.text(3.85, 3.95, "CPU precomputes\nbefore launch", ha="center", fontsize=9, color=DARK)

# --- Right panel: tile array table ---
ax.text(9.0, 7.2, "Precomputed tile metadata array", ha="center", fontsize=11, fontweight="bold")

headers = ["tile", "expert", "m_start", "m_end"]
col_x   = [4.9, 6.3, 7.8, 9.3]
col_w   = [1.2, 1.3, 1.3, 1.3]

# header row
for hdr, cx, cw in zip(headers, col_x, col_w):
    ax.text(cx + cw/2, 6.85, hdr, ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=DARK)
ax.axhline(6.65, xmin=4.85/13, xmax=10.9/13, color=DARK, linewidth=1.2)

row_h2 = 5.8 / len(tile_list)
for i, (e, ts, te, _) in enumerate(tile_list):
    ry = 6.5 - i * row_h2 - row_h2 * 0.5
    rect = mpatches.FancyBboxPatch(
        (4.85, ry - row_h2 * 0.46), 6.1, row_h2 * 0.85,
        boxstyle="round,pad=0.02",
        facecolor=colors_e[e], alpha=0.20, edgecolor=colors_e[e], linewidth=1)
    ax.add_patch(rect)
    for val, cx, cw in zip([i, e, ts, te], col_x, col_w):
        ax.text(cx + cw/2, ry, str(val), ha="center", va="center",
                fontsize=9, color=DARK, fontweight="bold" if val == e else "normal")

# --- CTA box at bottom ---
ax.text(9.0, -0.2,
        "Each CTA: 3 integer loads → expert, m_start, m_end → compute immediately. No binary search.",
        ha="center", fontsize=9, color=GREY, style="italic")

# legend
handles = [mpatches.Rectangle((0,0),1,1, facecolor=colors_e[e], alpha=0.75,
           label=f"Expert {e}") for e in range(len(experts))]
ax.legend(handles=handles, loc="lower right", fontsize=8.5, ncol=3, framealpha=0.9)

plt.tight_layout()
plt.savefig("figures/fig2_tile_map.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ fig2_tile_map.png")

# ── Figure 3: Software pipelining ──────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 5))
fig.suptitle("Optimisation 3: Software Pipelining (num_stages=3)", fontsize=14, fontweight="bold", y=1.01)

FETCH = 1.5
COMP  = 1.2
K_BLOCKS = 6

def draw_timeline(ax, stagger, title, title_color):
    ax.set_title(title, color=title_color, pad=6)
    for kb in range(K_BLOCKS):
        # fetch
        fx = kb * (FETCH + COMP) if stagger == 0 else kb * COMP + max(0, kb - 1) * 0
        if stagger == 0:
            fx = kb * (FETCH + COMP)
            cx = fx + FETCH
        else:
            fx = kb * COMP
            cx = kb * COMP
        ax.barh(1, FETCH, left=fx if stagger==0 else kb*COMP,
                height=0.4, color=ORANGE, alpha=0.85, label="Fetch (GMEM→SMEM)" if kb==0 else "")
        if stagger == 0:
            ax.barh(0, COMP, left=cx, height=0.4, color=BLUE, alpha=0.85, label="WGMMA compute" if kb==0 else "")
            ax.barh(0, FETCH, left=fx, height=0.4, color=LIGHT, alpha=0.5)   # idle
        ax.text((fx if stagger==0 else kb*COMP) + FETCH/2,
                1, f"F{kb}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")

    if stagger == 1:
        # pipelined: fetch k+1 while computing k
        for kb in range(K_BLOCKS):
            ax.barh(0, COMP, left=kb*COMP, height=0.4, color=BLUE, alpha=0.85, label="WGMMA compute" if kb==0 else "")
            ax.text(kb*COMP + COMP/2, 0, f"C{kb}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        for kb in range(K_BLOCKS):
            ax.barh(1, FETCH, left=kb*COMP, height=0.4, color=ORANGE, alpha=0.85)
            ax.text(kb*COMP + FETCH/2, 1, f"F{kb}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")

    ax.set_yticks([0, 1]); ax.set_yticklabels(["Tensor cores", "Mem system"])
    ax.set_xlabel("Time →"); ax.set_xlim(0, K_BLOCKS * (FETCH + COMP) + 0.5)
    ax.legend(loc="upper right", fontsize=9)

# Without pipelining
ax = axes[0]
ax.set_title("Without pipelining: tensor cores idle during fetch", color=RED, pad=6)
x = 0
for kb in range(K_BLOCKS):
    ax.barh(1, FETCH, left=x, height=0.4, color=ORANGE, alpha=0.85, label="Fetch (GMEM→SMEM)" if kb==0 else "")
    ax.barh(0, FETCH, left=x, height=0.4, color=LIGHT,  alpha=0.6,  label="Tensor cores idle" if kb==0 else "")
    ax.text(x + FETCH/2, 1, f"F{kb}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    x += FETCH
    ax.barh(0, COMP, left=x, height=0.4, color=BLUE, alpha=0.85, label="WGMMA compute" if kb==0 else "")
    ax.text(x + COMP/2, 0, f"C{kb}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    x += COMP
ax.set_yticks([0,1]); ax.set_yticklabels(["Tensor cores","Mem system"])
ax.set_xlabel("Time →"); ax.set_xlim(0, x+0.3)
ax.legend(loc="upper right", fontsize=9)

# With pipelining
ax2 = axes[1]
ax2.set_title("With pipelining (num_stages=3): fetch and compute overlap", color=GREEN, pad=6)
# prefill 2 stages, then steady state overlap
prefill = 2
x_fetch = 0; x_comp = prefill * FETCH
for kb in range(K_BLOCKS + prefill):
    if kb < K_BLOCKS + prefill - 1:
        ax2.barh(1, FETCH, left=x_fetch, height=0.4, color=ORANGE, alpha=0.85, label="Fetch" if kb==0 else "")
        ax2.text(x_fetch + FETCH/2, 1, f"F{kb}" if kb < K_BLOCKS else "", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        x_fetch += FETCH
    if kb >= prefill and kb - prefill < K_BLOCKS:
        ax2.barh(0, COMP, left=x_comp - prefill*FETCH + (kb-prefill)*COMP, height=0.4,
                 color=BLUE, alpha=0.85, label="WGMMA compute" if kb==prefill else "")
        ax2.text(x_comp - prefill*FETCH + (kb-prefill)*COMP + COMP/2, 0,
                 f"C{kb-prefill}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
total_w = max(x_fetch, x_comp - prefill*FETCH + K_BLOCKS*COMP) + 0.3
ax2.set_yticks([0,1]); ax2.set_yticklabels(["Tensor cores","Mem system"])
ax2.set_xlabel("Time →"); ax2.set_xlim(0, total_w)
ax2.legend(loc="upper right", fontsize=9)

plt.tight_layout()
plt.savefig("figures/fig3_pipelining.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ fig3_pipelining.png")

# ── Figure 4: Performance comparison ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Benchmark Results (B200, E=256/EL=32/K=8/H=7168/I=2048)", fontsize=14, fontweight="bold")

variants = ["FlashInfer\n(baseline)", "deep_gemm\nFP8", "Our Triton\nFP8", "CUDA\nBF16-WMMA", "CUDA\nBF16-cuBLAS"]
colors_v = [RED, PURPLE, GREEN, GREY, BLUE]

data = {
    901:   [0.699,  0.450,  1.180,  5.169,  1.019],
    14107: [6.359,  1.746,  5.909, 73.629, 11.344],
}

for ax, (T, vals) in zip(axes, data.items()):
    bars = ax.bar(variants, vals, color=colors_v, alpha=0.85, edgecolor="white", linewidth=1.2, width=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + max(vals)*0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    # ratio vs baseline
    bl = vals[0]
    for bar, v, name in zip(bars, vals, variants):
        ratio = bl / v
        color = GREEN if ratio >= 0.9 else (ORANGE if ratio >= 0.5 else RED)
        ax.text(bar.get_x() + bar.get_width()/2, -max(vals)*0.08,
                f"{ratio:.2f}×", ha="center", va="top", fontsize=8, color=color, fontweight="bold")
    ax.set_title(f"T = {T} tokens", pad=8)
    ax.set_ylabel("Latency (ms)")
    ax.set_ylim(-max(vals)*0.12, max(vals)*1.18)
    ax.text(0.5, -0.14, "× = speedup vs FlashInfer baseline (higher = closer to baseline)",
            ha="center", transform=ax.transAxes, fontsize=8, color=GREY)

plt.tight_layout()
plt.savefig("figures/fig4_performance.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ fig4_performance.png")

print("\nAll figures saved to figures/")
