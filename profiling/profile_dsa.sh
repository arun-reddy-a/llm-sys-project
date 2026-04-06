#!/usr/bin/env bash
# ============================================================================
# DSA Kernel Profiling Script — Full Decision Tree
# ============================================================================
#
# Usage:
#   ./profiling/profile_dsa.sh [--variant <name>] [--stage <1|2|3>] [--output-dir <dir>]
#
# Stages:
#   1 = Nsight Systems (timeline — find the slow kernel)
#   2 = Nsight Compute (deep-dive — roofline, memory, compute, occupancy)
#   3 = Summary report (parse ncu output into a readable diagnosis)
#
# Examples:
#   ./profiling/profile_dsa.sh --variant Opt4 --stage 1
#   ./profiling/profile_dsa.sh --variant Opt3 --stage 2
#   ./profiling/profile_dsa.sh --stage 3           # summary from last ncu run
#   ./profiling/profile_dsa.sh                     # run all stages for Opt4
#
# Requirements: nsys, ncu (NVIDIA Nsight Systems/Compute), nvcc
# ============================================================================

set -euo pipefail

# ─── Defaults ───────────────────────────────────────────────────────────────
VARIANT="Opt4"
STAGE="all"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/profiling/results"
BUILD_DIR="${PROJECT_ROOT}/build"
BENCH_BIN="${BUILD_DIR}/bench_dsa"

# ─── Parse args ─────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant)  VARIANT="$2"; shift 2 ;;
        --stage)    STAGE="$2";   shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help)
            head -25 "$0" | tail -20
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "${OUTPUT_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PREFIX="${OUTPUT_DIR}/dsa_${VARIANT}_${TIMESTAMP}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          DSA Profiling — Full Decision Tree                 ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Variant   : ${VARIANT}"
echo "║  Stage     : ${STAGE}"
echo "║  Output    : ${OUTPUT_DIR}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ─── Build profiling binary ────────────────────────────────────────────────
build_profile_binary() {
    echo "▸ Building DSA profile binary..."
    mkdir -p "${BUILD_DIR}"

    nvcc -std=c++17 -O2 -arch=native -lineinfo \
        -o "${BENCH_BIN}" \
        "${PROJECT_ROOT}/benchmarks/bench_dsa.cu" \
        "${PROJECT_ROOT}/kernels/dsa/naive_dsa.cu"

    echo "  ✓ Built: ${BENCH_BIN} (with -lineinfo for source correlation)"
}

# ============================================================================
# STAGE 1: Nsight Systems — Timeline profiling (find the slow kernel)
# ============================================================================
run_nsys() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  STAGE 1: Nsight Systems — Timeline Trace"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "  Goal: Identify which DSA kernel(s) dominate wall-clock time."
    echo "  DSA pipeline: Gather → Dot(Compressed) → Dot(Positional) → Softmax → OutProj"
    echo "  Or fused: FlashFused / BlackwellPipelined (single kernel)"
    echo ""

    local NSYS_OUT="${PREFIX}_nsys"

    nsys profile \
        --output="${NSYS_OUT}" \
        --force-overwrite=true \
        --trace=cuda,nvtx,osrt \
        --sample=none \
        --cudabacktrace=all \
        --stats=true \
        "${BENCH_BIN}" 5 20

    echo ""
    echo "  ▸ Timeline report saved: ${NSYS_OUT}.nsys-rep"
    echo "  ▸ Open in Nsight Systems GUI: nsys-ui ${NSYS_OUT}.nsys-rep"
    echo ""

    # Extract kernel summary stats
    echo "  ┌─────────────────────────────────────────────────────────┐"
    echo "  │  Top CUDA Kernels by Total GPU Time                    │"
    echo "  └─────────────────────────────────────────────────────────┘"
    nsys stats --report cuda_gpu_kern_sum "${NSYS_OUT}.nsys-rep" 2>/dev/null || true
    echo ""

    echo "  ┌─────────────────────────────────────────────────────────┐"
    echo "  │  CUDA API Call Summary (host-side overhead)             │"
    echo "  └─────────────────────────────────────────────────────────┘"
    nsys stats --report cuda_api_sum "${NSYS_OUT}.nsys-rep" 2>/dev/null || true
    echo ""
}

# ============================================================================
# STAGE 2: Nsight Compute — Deep Kernel Analysis
# ============================================================================
#
# DSA-specific considerations:
#   - Attention kernels are typically MEMORY-BOUND (lots of KV cache reads)
#   - Flash-style fused kernels may shift to LATENCY-BOUND (high smem, low occupancy)
#   - Dot product kernels with small Dp (64) may be COMPUTE-BOUND per element
#   - Gather kernels are almost certainly MEMORY-BOUND (random access pattern)
# ============================================================================
run_ncu() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  STAGE 2: Nsight Compute — Deep Kernel Profiling"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    local NCU_OUT="${PREFIX}_ncu"

    # ── Metric sets organized by the decision tree ──────────────────────────

    local ROOFLINE_METRICS=(
        "sm__throughput.avg.pct_of_peak_sustained_elapsed"
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"
    )

    local MEMORY_METRICS=(
        # Coalescing — critical for gather kernel (random KV cache access)
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"
        "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"

        # Bank Conflicts — critical for tiled dot kernels using smem
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"

        # Cache Hit Rates — sparse indices cause random access → L2 misses
        "l1tex__t_sector_hit_rate.pct"
        "lts__t_sector_hit_rate.pct"

        # Achieved Bandwidth
        "dram__bytes_read.sum"
        "dram__bytes_write.sum"
        "gpu__time_duration.sum"
    )

    local COMPUTE_METRICS=(
        # Tensor Core Utilization (currently unused in DSA, but good to check)
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"

        # Instruction Mix
        "sm__sass_thread_inst_executed_op_fadd_pred_on.sum"
        "sm__sass_thread_inst_executed_op_fmul_pred_on.sum"
        "sm__sass_thread_inst_executed_op_ffma_pred_on.sum"

        # Achieved FLOPS
        "sm__sass_thread_inst_executed_op_fadd_pred_on.sum.per_second"
        "sm__sass_thread_inst_executed_op_ffma_pred_on.sum.per_second"
        "smsp__inst_executed.sum"
    )

    local LATENCY_METRICS=(
        # Occupancy — flash kernels use huge smem, may limit occupancy
        "sm__warps_active.avg.pct_of_peak_sustained_active"

        # What's limiting occupancy?
        "launch__registers_per_thread"
        "launch__block_size"
        "launch__shared_mem_per_block_dynamic"
        "launch__shared_mem_per_block_static"

        # Warp Stall Reasons
        "smsp__warps_issue_stalled_long_scoreboard_per_warp_active.pct"
        "smsp__warps_issue_stalled_short_scoreboard_per_warp_active.pct"
        "smsp__warps_issue_stalled_wait_per_warp_active.pct"
        "smsp__warps_issue_stalled_not_selected_per_warp_active.pct"
        "smsp__warps_issue_stalled_math_pipe_throttle_per_warp_active.pct"
        "smsp__warps_issue_stalled_mio_throttle_per_warp_active.pct"
        "smsp__warps_issue_stalled_no_instruction_per_warp_active.pct"

        # ILP
        "smsp__issue_active.avg.pct_of_peak_sustained_active"
    )

    # Combine all metrics
    local ALL_METRICS=()
    ALL_METRICS+=("${ROOFLINE_METRICS[@]}")
    ALL_METRICS+=("${MEMORY_METRICS[@]}")
    ALL_METRICS+=("${COMPUTE_METRICS[@]}")
    ALL_METRICS+=("${LATENCY_METRICS[@]}")

    local METRICS_CSV
    METRICS_CSV=$(IFS=,; echo "${ALL_METRICS[*]}")

    echo "  Profiling with ${#ALL_METRICS[@]} metrics..."
    echo "  DSA kernel types to profile:"
    echo "    • kv_gather_fused_kernel          (sparse gather from KV cache)"
    echo "    • dot_compressed_kernel            (naive dot — compressed)"
    echo "    • dot_positional_kernel             (naive dot — positional)"
    echo "    • dot_compressed_tiled_kernel       (tiled dot — compressed)"
    echo "    • dot_positional_tiled_kernel       (tiled dot — positional)"
    echo "    • dot_fused_tiled_kernel            (fused compressed+positional dot)"
    echo "    • scaled_softmax_kernel             (naive softmax)"
    echo "    • scaled_softmax_block_kernel       (block-parallel softmax)"
    echo "    • output_proj_kernel                (attention × values)"
    echo "    • dsa_flash_fused_kernel            (flash-style fully fused)"
    echo "    • dsa_blackwell_pipelined_fused_kernel (double-buffered flash)"
    echo ""
    echo "  This may take several minutes per kernel invocation."
    echo ""

    ncu --set full \
        --metrics "${METRICS_CSV}" \
        --launch-skip 10 \
        --launch-count 50 \
        --target-processes all \
        --export "${NCU_OUT}" \
        --force-overwrite \
        --page raw \
        "${BENCH_BIN}" 10 20 \
        2>&1 | tee "${NCU_OUT}_console.log"

    echo ""
    echo "  ▸ Nsight Compute report saved: ${NCU_OUT}.ncu-rep"
    echo "  ▸ Console log: ${NCU_OUT}_console.log"
    echo "  ▸ Open in Nsight Compute GUI: ncu-ui ${NCU_OUT}.ncu-rep"
    echo ""

    # Dump CSV for automated analysis
    ncu --import "${NCU_OUT}.ncu-rep" \
        --csv \
        --page raw \
        > "${NCU_OUT}.csv" 2>/dev/null || true

    echo "  ▸ CSV export: ${NCU_OUT}.csv"
}

# ============================================================================
# STAGE 3: Automated Diagnosis
# ============================================================================
run_diagnosis() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  STAGE 3: Automated Bottleneck Diagnosis (DSA)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    local NCU_CSV
    # Find the most recent dsa ncu CSV
    NCU_CSV=$(ls -t "${OUTPUT_DIR}"/dsa_*_ncu.csv 2>/dev/null | head -1)
    if [[ -z "${NCU_CSV}" ]]; then
        echo "  ✗ No ncu CSV found. Run stage 2 first."
        return 1
    fi

    echo "  Analyzing: ${NCU_CSV}"
    echo ""

    # Reuse the same diagnosis script — it's kernel-agnostic
    python3 "${PROJECT_ROOT}/profiling/diagnose_moe.py" "${NCU_CSV}"
}


# ============================================================================
# Dispatch
# ============================================================================

build_profile_binary

case "${STAGE}" in
    1)   run_nsys ;;
    2)   run_ncu ;;
    3)   run_diagnosis ;;
    all)
        run_nsys
        run_ncu
        run_diagnosis
        ;;
    *)
        echo "Unknown stage: ${STAGE}. Use 1, 2, 3, or all."
        exit 1
        ;;
esac

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  DSA Profiling complete. Results in: ${OUTPUT_DIR}"
echo "═══════════════════════════════════════════════════════════════"
