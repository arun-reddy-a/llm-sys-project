#!/usr/bin/env python3
"""
MoE Kernel Bottleneck Diagnosis — Decision Tree Analyzer
=========================================================

Reads Nsight Compute CSV export and walks through the profiling decision tree:

    Roofline Classification
    ├── Memory Bound?  → coalescing, bank conflicts, cache hits, bandwidth
    ├── Compute Bound? → tensor cores, instruction mix, FLOPS
    └── Below Both?    → occupancy, warp stalls, ILP

Usage:
    python3 diagnose_moe.py <ncu_export.csv>

The script produces a per-kernel diagnosis with actionable recommendations.
"""

import csv
import sys
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

# ─── Thresholds (tuned for Blackwell B200) ──────────────────────────────────
# These can be adjusted based on architecture.  The idea is:
#   - If DRAM throughput > 60% peak → memory-bound
#   - If SM throughput   > 60% peak → compute-bound
#   - If both < 40%                 → latency-bound (below both roofs)
MEMORY_BOUND_THRESHOLD = 60.0   # % of peak DRAM throughput
COMPUTE_BOUND_THRESHOLD = 60.0  # % of peak SM throughput
LATENCY_THRESHOLD = 40.0        # below this on BOTH = latency-bound

# Sub-thresholds for specific issues
COALESCING_RATIO_BAD = 4.0      # sectors/request > this = poor coalescing
BANK_CONFLICT_HIGH = 1000       # total conflicts above this is concerning
L1_HIT_RATE_LOW = 50.0          # %
L2_HIT_RATE_LOW = 50.0          # %
OCCUPANCY_LOW = 50.0            # % achieved occupancy
STALL_SIGNIFICANT = 20.0        # % — any single stall reason above this is notable
ILP_LOW = 30.0                  # % issue rate


@dataclass
class KernelProfile:
    """Holds all profiled metrics for a single kernel invocation."""
    name: str = ""
    duration_ns: float = 0

    # Roofline
    sm_throughput_pct: float = 0
    dram_throughput_pct: float = 0
    compute_memory_throughput_pct: float = 0

    # Memory
    global_ld_sectors: float = 0
    global_ld_requests: float = 0
    smem_ld_bank_conflicts: float = 0
    smem_st_bank_conflicts: float = 0
    l1_hit_rate: float = 0
    l2_hit_rate: float = 0
    dram_bytes_read: float = 0
    dram_bytes_write: float = 0

    # Compute
    tensor_core_pct: float = 0
    fadd_count: float = 0
    fmul_count: float = 0
    ffma_count: float = 0
    total_instructions: float = 0

    # Occupancy
    achieved_occupancy: float = 0
    regs_per_thread: float = 0
    block_size: float = 0
    dynamic_smem: float = 0
    static_smem: float = 0

    # Stalls
    stall_long_scoreboard: float = 0   # waiting for GMEM
    stall_short_scoreboard: float = 0  # waiting for L1/smem
    stall_wait: float = 0             # barrier (__syncthreads)
    stall_not_selected: float = 0     # eligible but not scheduled
    stall_math_throttle: float = 0    # math pipe full
    stall_mio_throttle: float = 0     # memory pipe full
    stall_no_instruction: float = 0   # instruction cache miss

    # ILP
    issue_rate: float = 0


# ─── Metric name → KernelProfile field mapping ─────────────────────────────
METRIC_MAP = {
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_throughput_pct",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": "dram_throughput_pct",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": "compute_memory_throughput_pct",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": "global_ld_sectors",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum": "global_ld_requests",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum": "smem_ld_bank_conflicts",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum": "smem_st_bank_conflicts",
    "l1tex__t_sector_hit_rate.pct": "l1_hit_rate",
    "lts__t_sector_hit_rate.pct": "l2_hit_rate",
    "dram__bytes_read.sum": "dram_bytes_read",
    "dram__bytes_write.sum": "dram_bytes_write",
    "gpu__time_duration.sum": "duration_ns",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed": "tensor_core_pct",
    "sm__sass_thread_inst_executed_op_fadd_pred_on.sum": "fadd_count",
    "sm__sass_thread_inst_executed_op_fmul_pred_on.sum": "fmul_count",
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum": "ffma_count",
    "smsp__inst_executed.sum": "total_instructions",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "achieved_occupancy",
    "launch__registers_per_thread": "regs_per_thread",
    "launch__block_size": "block_size",
    "launch__shared_mem_per_block_dynamic": "dynamic_smem",
    "launch__shared_mem_per_block_static": "static_smem",
    "smsp__warps_issue_stalled_long_scoreboard_per_warp_active.pct": "stall_long_scoreboard",
    "smsp__warps_issue_stalled_short_scoreboard_per_warp_active.pct": "stall_short_scoreboard",
    "smsp__warps_issue_stalled_wait_per_warp_active.pct": "stall_wait",
    "smsp__warps_issue_stalled_not_selected_per_warp_active.pct": "stall_not_selected",
    "smsp__warps_issue_stalled_math_pipe_throttle_per_warp_active.pct": "stall_math_throttle",
    "smsp__warps_issue_stalled_mio_throttle_per_warp_active.pct": "stall_mio_throttle",
    "smsp__warps_issue_stalled_no_instruction_per_warp_active.pct": "stall_no_instruction",
    "smsp__issue_active.avg.pct_of_peak_sustained_active": "issue_rate",
}


def parse_csv(path: str) -> list[KernelProfile]:
    """Parse ncu CSV export into a list of KernelProfile objects."""
    profiles = []

    with open(path, newline="") as f:
        reader = csv.reader(f)
        headers = None
        units = None
        for row in reader:
            # Skip comment lines
            if not row or row[0].startswith("=="):
                continue
            if headers is None:
                headers = row
                continue
            if units is None:
                units = row
                continue

            record = dict(zip(headers, row))
            kernel_name = record.get("Kernel Name", "").split("(")[0]
            if not kernel_name:
                continue

            # Find existing profile or create new
            profile = None
            for p in profiles:
                if p.name == kernel_name:
                    profile = p
                    break
            if profile is None:
                profile = KernelProfile(name=kernel_name)
                profiles.append(profile)

            # In wide format (raw), we iterate over the known METRIC_MAP keys
            # and pull them directly from the columns
            for csv_header_name, field_name in METRIC_MAP.items():
                if csv_header_name in record:
                    val_str = record[csv_header_name].strip().replace(",", "")
                    if not val_str or val_str.lower() in ["no data", "n/a"]:
                        continue
                        
                    try:
                        val = float(val_str)
                    except ValueError:
                        continue

                    # Lookup the unit from the units row using the csv_header_name index
                    try:
                        idx = headers.index(csv_header_name)
                        metric_unit = units[idx].strip().lower()
                    except ValueError:
                        metric_unit = ""

                    # Fix duration metric units since NCU natively uses various scales
                    if field_name == "duration_ns":
                        if metric_unit in ["ms", "msecond"]:
                            val *= 1e6
                        elif metric_unit in ["us", "usecond"]:
                            val *= 1e3
                        elif metric_unit in ["s", "second"]:
                            val *= 1e9
                    
                    # Accumulate or max (for metrics like duration, block size, etc.)
                    # We overwrite it (latest launch)
                    setattr(profile, field_name, max(getattr(profile, field_name) or 0.0, val))

    return profiles


def classify_bottleneck(p: KernelProfile) -> str:
    """Classify a kernel as memory-bound, compute-bound, or latency-bound."""
    if p.dram_throughput_pct >= MEMORY_BOUND_THRESHOLD:
        return "MEMORY-BOUND"
    if p.sm_throughput_pct >= COMPUTE_BOUND_THRESHOLD:
        return "COMPUTE-BOUND"
    if (p.dram_throughput_pct < LATENCY_THRESHOLD and
            p.sm_throughput_pct < LATENCY_THRESHOLD):
        return "LATENCY-BOUND"
    # Mixed / moderate
    if p.dram_throughput_pct > p.sm_throughput_pct:
        return "MEMORY-BOUND (moderate)"
    return "COMPUTE-BOUND (moderate)"


def diagnose_memory(p: KernelProfile) -> list[str]:
    """Deep-dive memory-bound diagnosis."""
    issues = []

    # Coalescing
    if p.global_ld_requests > 0:
        ratio = p.global_ld_sectors / p.global_ld_requests
        if ratio > COALESCING_RATIO_BAD:
            issues.append(
                f"  ⚠ Poor coalescing: {ratio:.1f} sectors/request "
                f"(ideal ≤ 4). Strided or misaligned access pattern."
            )
        else:
            issues.append(
                f"  ✓ Coalescing OK: {ratio:.1f} sectors/request"
            )

    # Bank conflicts
    total_conflicts = p.smem_ld_bank_conflicts + p.smem_st_bank_conflicts
    if total_conflicts > BANK_CONFLICT_HIGH:
        issues.append(
            f"  ⚠ Shared memory bank conflicts: "
            f"{p.smem_ld_bank_conflicts:.0f} load + "
            f"{p.smem_st_bank_conflicts:.0f} store = {total_conflicts:.0f} total. "
            f"Consider padding shared arrays (+1 column)."
        )
    else:
        issues.append(
            f"  ✓ Bank conflicts low: {total_conflicts:.0f} total"
        )

    # Cache hit rates
    if p.l1_hit_rate < L1_HIT_RATE_LOW:
        issues.append(
            f"  ⚠ L1 hit rate: {p.l1_hit_rate:.1f}% (low). "
            f"Working set may exceed L1 capacity or access is irregular."
        )
    else:
        issues.append(f"  ✓ L1 hit rate: {p.l1_hit_rate:.1f}%")

    if p.l2_hit_rate < L2_HIT_RATE_LOW:
        issues.append(
            f"  ⚠ L2 hit rate: {p.l2_hit_rate:.1f}% (low). "
            f"Large working set or streaming access pattern."
        )
    else:
        issues.append(f"  ✓ L2 hit rate: {p.l2_hit_rate:.1f}%")

    # Achieved bandwidth
    if p.duration_ns > 0:
        total_bytes = p.dram_bytes_read + p.dram_bytes_write
        bw_gbps = total_bytes / p.duration_ns  # bytes/ns = GB/s
        issues.append(
            f"  ℹ Achieved DRAM bandwidth: {bw_gbps:.1f} GB/s "
            f"({p.dram_bytes_read/1e6:.1f} MB read + {p.dram_bytes_write/1e6:.1f} MB write "
            f"in {p.duration_ns/1e6:.3f} ms)"
        )

    return issues


def diagnose_compute(p: KernelProfile) -> list[str]:
    """Deep-dive compute-bound diagnosis."""
    issues = []

    # Tensor core utilization
    if p.tensor_core_pct < 5.0:
        issues.append(
            f"  ⚠ Tensor cores not utilized ({p.tensor_core_pct:.1f}%). "
            f"Using FP32 CUDA cores only. Consider mma.sync / wmma for dense GEMM."
        )
    else:
        issues.append(
            f"  ✓ Tensor core utilization: {p.tensor_core_pct:.1f}%"
        )

    # Instruction mix
    total_fp = p.fadd_count + p.fmul_count + p.ffma_count
    if total_fp > 0:
        ffma_pct = 100.0 * p.ffma_count / total_fp
        issues.append(
            f"  ℹ Instruction mix: "
            f"FADD={p.fadd_count:.0f}, FMUL={p.fmul_count:.0f}, "
            f"FFMA={p.ffma_count:.0f} ({ffma_pct:.1f}% FMA)"
        )
        if ffma_pct < 50.0:
            issues.append(
                f"  ⚠ Low FMA ratio ({ffma_pct:.1f}%). Compiler may not be "
                f"fusing multiply-add. Check for unnecessary separate add/mul ops."
            )

    # Achieved FLOPS
    if p.duration_ns > 0:
        flops = (p.fadd_count + p.fmul_count + 2 * p.ffma_count)
        gflops = flops / p.duration_ns  # FLOP/ns = GFLOP/s
        issues.append(f"  ℹ Achieved: {gflops:.1f} GFLOP/s")

    return issues


def diagnose_latency(p: KernelProfile) -> list[str]:
    """Deep-dive latency-bound diagnosis (below both roofs)."""
    issues = []

    # Occupancy
    if p.achieved_occupancy < OCCUPANCY_LOW:
        issues.append(
            f"  ⚠ Low occupancy: {p.achieved_occupancy:.1f}%"
        )
        # What's limiting it?
        limiters = []
        if p.regs_per_thread > 0:
            limiters.append(f"regs/thread={p.regs_per_thread:.0f}")
        if p.block_size > 0:
            limiters.append(f"block_size={p.block_size:.0f}")
        total_smem = p.dynamic_smem + p.static_smem
        if total_smem > 0:
            limiters.append(f"smem={total_smem:.0f}B (dyn={p.dynamic_smem:.0f}, stat={p.static_smem:.0f})")
        if limiters:
            issues.append(f"    Limiters: {', '.join(limiters)}")
            if p.regs_per_thread > 64:
                issues.append(
                    f"    → Register pressure high ({p.regs_per_thread:.0f}). "
                    f"Try __launch_bounds__ or reducing local variables."
                )
            if total_smem > 48 * 1024:
                issues.append(
                    f"    → Shared memory high ({total_smem/1024:.1f} KB). "
                    f"Exceeds default 48KB limit; using setMaxDynamicSharedMemorySize."
                )
    else:
        issues.append(f"  ✓ Occupancy: {p.achieved_occupancy:.1f}%")

    # Warp stall breakdown
    stalls = [
        ("Long Scoreboard (GMEM wait)", p.stall_long_scoreboard),
        ("Short Scoreboard (L1/SMEM)", p.stall_short_scoreboard),
        ("Barrier (__syncthreads)", p.stall_wait),
        ("Not Selected (scheduler)", p.stall_not_selected),
        ("Math Pipe Throttle", p.stall_math_throttle),
        ("MIO Throttle (memory)", p.stall_mio_throttle),
        ("No Instruction (I-cache)", p.stall_no_instruction),
    ]

    issues.append("  Warp stall breakdown:")
    for name, val in sorted(stalls, key=lambda x: -x[1]):
        marker = "⚠" if val > STALL_SIGNIFICANT else " "
        bar_len = int(val / 2)  # scale for display
        bar = "█" * bar_len
        issues.append(f"    {marker} {val:5.1f}%  {bar:<25s}  {name}")

    # Dominant stall diagnosis
    dominant = max(stalls, key=lambda x: x[1])
    if dominant[1] > STALL_SIGNIFICANT:
        issues.append(f"  → Dominant stall: {dominant[0]} ({dominant[1]:.1f}%)")
        if "Long Scoreboard" in dominant[0]:
            issues.append(
                "    Fix: Increase occupancy, add prefetching, or "
                "use async copy (cp.async / TMA) to hide latency."
            )
        elif "Short Scoreboard" in dominant[0]:
            issues.append(
                "    Fix: Reduce shared memory bank conflicts, "
                "or increase ILP to hide L1 latency."
            )
        elif "Barrier" in dominant[0]:
            issues.append(
                "    Fix: Reduce __syncthreads() calls. Consider "
                "warp-level synchronization or rearranging compute."
            )
        elif "Math Pipe" in dominant[0]:
            issues.append(
                "    Fix: Already compute-saturated on math pipe. "
                "Need algorithmic reduction in FLOP count or upgrade to Tensor Cores."
            )
        elif "MIO" in dominant[0]:
            issues.append(
                "    Fix: Memory instruction pipe full. Reduce memory "
                "instruction count via vectorized loads (float4)."
            )

    # ILP
    if p.issue_rate < ILP_LOW:
        issues.append(
            f"  ⚠ Low ILP: issue rate = {p.issue_rate:.1f}%. "
            f"Try loop unrolling, reducing dependencies between instructions."
        )
    else:
        issues.append(f"  ✓ Issue rate (ILP): {p.issue_rate:.1f}%")

    return issues


def print_diagnosis(profiles: list[KernelProfile]):
    """Walk the decision tree for each kernel and print results."""

    # Sort by duration (longest first = most impactful)
    profiles_sorted = sorted(profiles, key=lambda p: -p.duration_ns)

    print("=" * 72)
    print("  MoE KERNEL BOTTLENECK DIAGNOSIS")
    print("  Decision Tree: Roofline → Memory/Compute/Latency → Root Cause")
    print("=" * 72)

    for i, p in enumerate(profiles_sorted):
        if p.duration_ns <= 0 and p.sm_throughput_pct <= 0:
            continue  # skip kernels with no data

        classification = classify_bottleneck(p)

        dur_ms = p.duration_ns / 1e6 if p.duration_ns > 0 else 0
        print(f"\n{'─' * 72}")
        print(f"  #{i+1}  {p.name}")
        print(f"      Duration: {dur_ms:.3f} ms")
        print(f"      SM Throughput: {p.sm_throughput_pct:.1f}% | "
              f"DRAM Throughput: {p.dram_throughput_pct:.1f}%")
        print(f"      Classification: *** {classification} ***")
        print()

        if "MEMORY" in classification:
            print("  ┌── Memory-Bound Analysis ──────────────────────────────────┐")
            for line in diagnose_memory(p):
                print(f"  │{line}")
            print("  └─────────────────────────────────────────────────────────────┘")

        if "COMPUTE" in classification:
            print("  ┌── Compute-Bound Analysis ─────────────────────────────────┐")
            for line in diagnose_compute(p):
                print(f"  │{line}")
            print("  └─────────────────────────────────────────────────────────────┘")

        if "LATENCY" in classification:
            print("  ┌── Latency-Bound Analysis ─────────────────────────────────┐")
            for line in diagnose_latency(p):
                print(f"  │{line}")
            print("  └─────────────────────────────────────────────────────────────┘")

        # Always print latency analysis for context even if not classified as latency-bound
        if "LATENCY" not in classification:
            print("  ┌── Occupancy & Stalls (supplementary) ────────────────────┐")
            for line in diagnose_latency(p):
                print(f"  │{line}")
            print("  └─────────────────────────────────────────────────────────────┘")

    # ── Summary table ───────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  SUMMARY TABLE")
    print(f"{'=' * 72}")
    print(f"  {'Kernel':<45s} {'Time(ms)':>8s} {'Class':<25s} {'Occupancy':>9s}")
    print(f"  {'─'*45} {'─'*8} {'─'*25} {'─'*9}")
    for p in profiles_sorted:
        # Clean kernel name for display
        short_name = p.name.split("<")[0].split("(")[0]
        dur_ms = p.duration_ns / 1e6 if p.duration_ns > 0 else 0
        cls = classify_bottleneck(p)
        print(f"  {short_name[:45]:<45s} {dur_ms:8.3f} {cls:<25s} {p.achieved_occupancy:8.1f}%")

    # ── Top recommendation ──────────────────────────────────────────────────
    if profiles_sorted:
        top = profiles_sorted[0]
        cls = classify_bottleneck(top)
        print(f"\n{'━' * 72}")
        print(f"  🎯 FOCUS AREA: {top.name}")
        print(f"     This kernel has the longest duration ({top.duration_ns/1e6:.3f} ms).")
        print(f"     Classification: {cls}")
        if "MEMORY" in cls:
            print(f"     → Priority: Check coalescing and cache hit rates.")
            print(f"     → Consider: async copy (TMA), vectorized loads (float4),")
            print(f"       or algorithmic restructuring to reduce data movement.")
        elif "COMPUTE" in cls:
            print(f"     → Priority: Enable Tensor Cores (mma.sync / WMMA).")
            print(f"     → Consider: Reducing FP32 instruction count via FMA fusion.")
        elif "LATENCY" in cls:
            print(f"     → Priority: Increase occupancy or hide latency.")
            print(f"     → Consider: Reduce register usage, add prefetching,")
            print(f"       increase block size, or use persistent-thread patterns.")
        print(f"{'━' * 72}")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <ncu_export.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)

    profiles = parse_csv(csv_path)
    if not profiles:
        print("No kernel profiles found in CSV. Check the file format.")
        sys.exit(1)

    print(f"\nParsed {len(profiles)} unique kernel(s) from {csv_path}\n")
    print_diagnosis(profiles)


if __name__ == "__main__":
    main()
