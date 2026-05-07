"""
Clean O(E) → O(1) dispatch figure.
Fixes needed:
  1. "API overhead" label was vertical/overlapping — now horizontal, placed below first gap
  2. Reorder step added as a distinct amber block before the GEMM block
  3. Legend is horizontal (ncol=2) and placed below each panel
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

RED    = "#d94f4f"
GREEN  = "#27ae60"
AMBER  = "#e67e22"
LGRAY  = "#e8ecef"
DARK   = "#1f2937"

fig, axes = plt.subplots(2, 1, figsize=(14, 7),
                         gridspec_kw={"height_ratios": [1, 1], "hspace": 0.30})
fig.suptitle("Grouped-GEMM: From O(E) Dispatches to O(1)",
             fontsize=15, fontweight="bold", y=0.99)

TOTAL_W = 16.0
EXEC_W  = 0.28
GAP_W   = 0.62
N_SHOWN = 18

# ─────────────────────────────────────────────────────────────────────────────
# TOP: Naive — 256 tiny sequential launches
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[0]
ax.set_xlim(0, TOTAL_W)
ax.set_ylim(-0.15, 1.05)
ax.set_yticks([]); ax.set_xticks([])
for spine in ["top", "right", "left"]: ax.spines[spine].set_visible(False)
ax.set_facecolor("white")
ax.set_title("Naive Baseline — 256 Sequential Kernel Launches",
             fontsize=12, fontweight="bold", color=RED, loc="left", pad=6)

x = 0.15
for i in range(N_SHOWN):
    ax.add_patch(FancyBboxPatch((x, 0.18), GAP_W, 0.64,
        boxstyle="square,pad=0", facecolor=LGRAY, edgecolor="none"))
    if i == 0:
        ax.annotate("", xy=(x + GAP_W, -0.04), xytext=(x, -0.04),
                    arrowprops=dict(arrowstyle="<->", color="#aaa", lw=1.0))
        ax.text(x + GAP_W / 2, -0.12, "API overhead",
                ha="center", va="center", fontsize=7.5, color="#888")
    x += GAP_W
    ax.add_patch(FancyBboxPatch((x, 0.18), EXEC_W, 0.64,
        boxstyle="square,pad=0", facecolor=RED, edgecolor="none", alpha=0.9))
    x += EXEC_W

ax.text(x + 0.3, 0.50, "· · ·  ×256", ha="left", va="center",
        fontsize=12, color=RED, fontweight="bold")

ax.text(TOTAL_W - 0.1, 1.02, "1.6% SM occupancy per launch",
        ha="right", va="top", fontsize=10, color=RED, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fdecea", edgecolor=RED, lw=1.1))
ax.text(TOTAL_W - 0.1, 0.72, "L2 cache flushed between launches",
        ha="right", va="top", fontsize=9.5, color="#b05050",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff0f0", edgecolor="#d09090", lw=0.8))

h_exec = mpatches.Patch(color=RED,   label="Kernel execution (per expert)")
h_over = mpatches.Patch(color=LGRAY, label="API dispatch overhead")
ax.legend(handles=[h_exec, h_over], loc="upper left",
          fontsize=9, frameon=False, ncol=2)

# ─────────────────────────────────────────────────────────────────────────────
# BOTTOM: Grouped-GEMM — reorder step + single unified launch
# ─────────────────────────────────────────────────────────────────────────────
ax = axes[1]
ax.set_xlim(0, TOTAL_W)
ax.set_ylim(-0.15, 1.05)
ax.set_yticks([]); ax.set_xticks([])
for spine in ["top", "right", "left"]: ax.spines[spine].set_visible(False)
ax.set_facecolor("white")
ax.set_title("Opt 3: Grouped-GEMM — One Unified Launch",
             fontsize=12, fontweight="bold", color=GREEN, loc="left", pad=6)

SORT_W = 1.1
SORT_X = 0.15
ax.add_patch(FancyBboxPatch((SORT_X, 0.18), SORT_W, 0.64,
    boxstyle="round,pad=0.04", facecolor=AMBER, edgecolor="none"))
ax.text(SORT_X + SORT_W / 2, 0.58, "Reorder",
        ha="center", va="center", fontsize=10, color="white", fontweight="bold")
ax.text(SORT_X + SORT_W / 2, 0.33, "tokens by\nexpert ID",
        ha="center", va="center", fontsize=8, color="white", alpha=0.9)

GAP_ARROW = 0.25
ax.annotate("", xy=(SORT_X + SORT_W + GAP_ARROW, 0.50),
            xytext=(SORT_X + SORT_W + 0.04, 0.50),
            arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.5))

BLOCK_START = SORT_X + SORT_W + GAP_ARROW
BLOCK_W     = N_SHOWN * (EXEC_W + GAP_W) / 2.5
END         = BLOCK_START + BLOCK_W

ax.add_patch(FancyBboxPatch((BLOCK_START, 0.18), BLOCK_W, 0.64,
    boxstyle="round,pad=0.04", facecolor=GREEN, edgecolor="none"))
ax.text(BLOCK_START + BLOCK_W / 2, 0.57,
        "All 256 experts — one unified GEMM grid",
        ha="center", va="center", fontsize=11, color="white", fontweight="bold")
ax.text(BLOCK_START + BLOCK_W / 2, 0.33, "contiguous memory · deep L2 reuse",
        ha="center", va="center", fontsize=9, color="white", alpha=0.85)

ax.add_patch(FancyBboxPatch((END, 0.18), TOTAL_W - END - 0.15, 0.64,
    boxstyle="square,pad=0", facecolor=LGRAY, edgecolor="none", alpha=0.5))
ax.text((END + TOTAL_W - 0.15) / 2, 0.50, "GPU free",
        ha="center", va="center", fontsize=10, color="#999", style="italic")

ax.annotate("", xy=(END, -0.04), xytext=(SORT_X, -0.04),
            arrowprops=dict(arrowstyle="<->", color="#555", lw=1.4))
ax.text((SORT_X + END) / 2, -0.12,
        "same compute work  —  30× less wall time",
        ha="center", va="center", fontsize=9, color="#555")

ax.text(TOTAL_W - 0.1, 1.02, "99.6% SM occupancy · 30× throughput vs Naive",
        ha="right", va="top", fontsize=10, color=GREEN, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#eafaf1", edgecolor=GREEN, lw=1.1))
ax.text(TOTAL_W - 0.1, 0.72, "Single O(1) API call — no per-expert overhead",
        ha="right", va="top", fontsize=9.5, color="#1a7d42",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#f0fff4", edgecolor="#90c0a0", lw=0.8))

h_sort = mpatches.Patch(color=AMBER, label="Token reorder (0.007 ms)")
h_gemm = mpatches.Patch(color=GREEN, label="Grouped-GEMM kernel (all 256 experts)")
ax.legend(handles=[h_sort, h_gemm], loc="upper left",
          fontsize=9, frameon=False, ncol=2)

fig.savefig("/home/rrongali/llm-sys-project/figures/moe_dispatch_clean.png",
            dpi=180, bbox_inches="tight")
plt.close()
print("saved")
