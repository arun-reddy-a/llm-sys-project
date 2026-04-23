#!/usr/bin/env python3
"""
Parse LLM-Sys-Project DSA benchmark text (e.g. results.txt from bench_dsa_full)
and generate poster-style figures: per-config throughput bars, speedup vs Naive,
and a config×variant heatmap.

Dependencies: matplotlib, numpy
  pip install matplotlib numpy

Example:
  python visualize_dsa_results.py --input results.txt --outdir figures_dsa
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

VARIANT_ORDER = ["Naive", "Opt1", "Opt2", "Opt3", "Opt4", "Opt5", "Opt6", "Opt7", "Opt8"]

CONFIG_HEADER_RE = re.compile(r"^\s*Config:\s*(.+?)\s+\(Q=")
ROW_RE = re.compile(
    r"^\s*(Naive|Opt[1-8])\s+"
    r"([\d.]+(?:e[+-]\d+)?)\s+"
    r"([\d.]+(?:e[+-]\d+)?)\s+"
    r"([\d.]+(?:e[+-]\d+)?)\s+"
    r"([\d.]+(?:e[+-]\d+)?)\s*$",
    re.IGNORECASE,
)


def parse_results_txt(path: Path) -> list[dict]:
    """Return list of {name, variants: {variant: {min_ms, mean_ms, gflops, qps}}}."""
    configs: list[dict] = []
    current: dict | None = None
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        m = CONFIG_HEADER_RE.match(line)
        if m:
            if current:
                configs.append(current)
            current = {"name": m.group(1).strip(), "variants": {}}
            continue
        if not current:
            continue
        m = ROW_RE.match(line)
        if not m:
            continue
        raw = m.group(1)
        variant = "Naive" if raw.lower() == "naive" else raw
        current["variants"][variant] = {
            "min_ms": float(m.group(2)),
            "mean_ms": float(m.group(3)),
            "gflops": float(m.group(4)),
            "qps": float(m.group(5)),
        }
    if current:
        configs.append(current)
    return configs


def apply_poster_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fafafa",
            "axes.edgecolor": "#333333",
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "font.family": "sans-serif",
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
        }
    )


def build_matrices(configs: list[dict]) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """names, gflops [n_cfg, n_var], mean_ms, speedup (vs Naive mean)."""
    names = [c["name"] for c in configs]
    n_v = len(VARIANT_ORDER)
    gflops = np.full((len(configs), n_v), np.nan)
    mean_ms = np.full((len(configs), n_v), np.nan)
    speedup = np.full((len(configs), n_v), np.nan)
    for i, c in enumerate(configs):
        vmap = c["variants"]
        naive_mean = vmap.get("Naive", {}).get("mean_ms")
        for j, v in enumerate(VARIANT_ORDER):
            if v not in vmap:
                continue
            gflops[i, j] = vmap[v]["gflops"]
            mean_ms[i, j] = vmap[v]["mean_ms"]
            if naive_mean and naive_mean > 0 and vmap[v]["mean_ms"] > 0:
                speedup[i, j] = naive_mean / vmap[v]["mean_ms"]
    return names, gflops, mean_ms, speedup


def plot_throughput_subplots(
    names: list[str],
    gflops: np.ndarray,
    out_path: Path,
    dpi: int,
    title: str,
):
    n = len(names)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.2 * nrows), squeeze=False)
    axes_flat = axes.ravel()
    x = np.arange(len(VARIANT_ORDER))
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(VARIANT_ORDER)))

    for idx, name in enumerate(names):
        ax = axes_flat[idx]
        row = gflops[idx]
        mask = ~np.isnan(row)
        bars = ax.bar(x[mask], row[mask], color=colors[mask], edgecolor="#222", linewidth=0.4)
        ax.set_title(name, fontweight="semibold")
        ax.set_xticks(x)
        ax.set_xticklabels(VARIANT_ORDER, rotation=35, ha="right")
        ax.set_ylabel("GFLOP/s")
        ymax = np.nanmax(row) * 1.12 if np.any(mask) else 1.0
        ax.set_ylim(0, ymax)
        for b in bars:
            h = b.get_height()
            if np.isfinite(h) and h > 0.05 * ymax:
                ax.annotate(
                    f"{h:.0f}",
                    xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_speedup_subplots(
    names: list[str],
    speedup: np.ndarray,
    out_path: Path,
    dpi: int,
    title: str,
):
    n = len(names)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.2 * nrows), squeeze=False)
    axes_flat = axes.ravel()
    x = np.arange(len(VARIANT_ORDER))
    colors = plt.cm.plasma(np.linspace(0.15, 0.92, len(VARIANT_ORDER)))

    for idx, name in enumerate(names):
        ax = axes_flat[idx]
        row = speedup[idx]
        mask = ~np.isnan(row)
        ax.axhline(1.0, color="#666", linestyle=":", linewidth=1.2, label="Naive baseline")
        ax.bar(x[mask], row[mask], color=colors[mask], edgecolor="#222", linewidth=0.4)
        ax.set_title(name, fontweight="semibold")
        ax.set_xticks(x)
        ax.set_xticklabels(VARIANT_ORDER, rotation=35, ha="right")
        ax.set_ylabel("Speedup vs Naive\n(mean latency)")
        ymax = (np.nanmax(row) * 1.12 + 0.05) if np.any(mask) else 2.0
        ax.set_ylim(0, max(ymax, 1.05))

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(
    names: list[str],
    gflops: np.ndarray,
    out_path: Path,
    dpi: int,
    title: str,
):
    fig, ax = plt.subplots(figsize=(10.5, max(4.0, 0.55 * len(names))))
    data = np.nan_to_num(gflops, nan=0.0)
    im = ax.imshow(data, aspect="auto", cmap="YlGnBu")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("GFLOP/s", rotation=270, labelpad=18)

    ax.set_xticks(np.arange(len(VARIANT_ORDER)))
    ax.set_xticklabels(VARIANT_ORDER, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Kernel variant")
    ax.set_ylabel("Configuration")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = gflops[i, j]
            if not np.isfinite(val):
                continue
            lo, hi = np.nanpercentile(gflops, [35, 85])
            text_color = "white" if val >= hi or (np.isfinite(lo) and val > lo + 0.5 * (hi - lo)) else "#111"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center", color=text_color, fontsize=9)

    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_summary_lines(
    names: list[str],
    gflops: np.ndarray,
    out_path: Path,
    dpi: int,
    title: str,
):
    """Line plot: one curve per variant across configs (good for poster overview)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))
    cmap = plt.cm.tab10
    for j, v in enumerate(VARIANT_ORDER):
        col = gflops[:, j]
        if np.all(np.isnan(col)):
            continue
        ax.plot(x, col, marker="o", linewidth=2, markersize=6, label=v, color=cmap(j % 10))

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("GFLOP/s")
    ax.set_xlabel("Configuration")
    ax.legend(ncol=3, loc="upper left", framealpha=0.92)
    ax.set_title(title, fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize DSA bench results.txt")
    parser.add_argument("--input", "-i", type=Path, default=Path("results.txt"))
    parser.add_argument("--outdir", "-o", type=Path, default=Path("figures_dsa"))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        choices=["png", "pdf", "svg"],
        help="Output image formats",
    )
    args = parser.parse_args()

    configs = parse_results_txt(args.input)
    if not configs:
        raise SystemExit(f"No benchmark tables found in {args.input}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    apply_poster_style()

    names, gflops, _mean_ms, speedup = build_matrices(configs)
    base_title = "DSA kernel variants (bench_dsa_full)"

    for fmt in args.formats:
        plot_throughput_subplots(
            names,
            gflops,
            args.outdir / f"dsa_throughput_by_config.{fmt}",
            args.dpi,
            f"{base_title} — throughput (GFLOP/s)",
        )
        plot_speedup_subplots(
            names,
            speedup,
            args.outdir / f"dsa_speedup_vs_naive.{fmt}",
            args.dpi,
            f"{base_title} — speedup vs Naive (mean latency)",
        )
        plot_heatmap(
            names,
            gflops,
            args.outdir / f"dsa_gflops_heatmap.{fmt}",
            args.dpi,
            f"{base_title} — GFLOP/s heatmap",
        )
        plot_summary_lines(
            names,
            gflops,
            args.outdir / f"dsa_throughput_lines.{fmt}",
            args.dpi,
            f"{base_title} — throughput across configs",
        )

    print(f"Wrote figures to {args.outdir.resolve()} ({', '.join(args.formats)})")
    print(f"Parsed {len(configs)} configurations.")


if __name__ == "__main__":
    main()
