#!/usr/bin/env bash
# ============================================================================
# MoE Kernel Profiling Script — Full Decision Tree
# ============================================================================
#
# Usage:
#   ./profiling/profile_moe.sh [--variant <name>] [--stage <1|2|3>] [--output-dir <dir>]
#
# Stages:
#   1 = Nsight Systems (timeline — find the slow kernel)
#   2 = Nsight Compute (deep-dive — roofline, memory, compute, occupancy)
#   3 = Summary report (parse ncu output into a readable diagnosis)
#
# Examples:
#   ./profiling/profile_moe.sh --variant Opt5 --stage 1
#   ./profiling/profile_moe.sh --variant Opt3 --stage 2
#   ./profiling/profile_moe.sh --stage 3           # summary from last ncu run
#   ./profiling/profile_moe.sh                     # run all stages for Opt5
#
# Requirements: nsys, ncu (NVIDIA Nsight Systems/Compute), nvcc
# ============================================================================

set -euo pipefail

# ─── Defaults ───────────────────────────────────────────────────────────────
VARIANT="Opt5"
STAGE="all"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/profiling/results"
BUILD_DIR="${PROJECT_ROOT}/build"
BENCH_BIN="${BUILD_DIR}/bench_moe"

# ─── Parse args ─────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant)  VARIANT="$2"; shift 2 ;;
        --stage)    STAGE="$2";   shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --bench-bin) BENCH_BIN="$2"; shift 2 ;;
        -h|--help)
            head -25 "$0" | tail -20
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "${OUTPUT_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PREFIX="${OUTPUT_DIR}/moe_${VARIANT}_${TIMESTAMP}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          MoE Profiling — Full Decision Tree                 ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Variant   : ${VARIANT}"
echo "║  Stage     : ${STAGE}"
echo "║  Output    : ${OUTPUT_DIR}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ─── Build profiling binary ────────────────────────────────────────────────
# We build a special target that runs a SINGLE variant with enough iterations
# for profiling (no need to compare all 6 variants for profiling)
build_profile_binary() {
    if [[ -f "${BENCH_BIN}" ]]; then
        echo "▸ Benchmark binary already exists: ${BENCH_BIN}"
        return 0
    fi
    echo "▸ Building profile binary..."
    mkdir -p "${BUILD_DIR}"

    # Build the regular benchmark binary
    nvcc -std=c++17 -O2 -arch=native -lineinfo -lnvToolsExt \
        -o "${BENCH_BIN}" \
        "${PROJECT_ROOT}/benchmarks/bench_moe.cu" \
        "${PROJECT_ROOT}/kernels/moe/naive_moe.cu"

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
    echo "  Goal: Identify which kernel(s) dominate wall-clock time."
    echo "  This answers: 'Where should I focus optimization effort?'"
    echo ""

    local NSYS_OUT="${PREFIX}_nsys"
    local BENCH_ARGS=""
    # bench_moe_smoke doesn't accept CLI args (hardcoded warmup/iters).
    # bench_moe accepts: <warmup> <iters>
    if [[ ! "${BENCH_BIN}" == *"smoke"* ]]; then
        BENCH_ARGS="2 3"
    fi

    # Use hardware-based tracing on Blackwell (B200)
    # NOTE: cuda-hw REPLACES cuda — they cannot be combined.
    nsys profile \
        --output="${NSYS_OUT}" \
        --force-overwrite=true \
        --trace=cuda,nvtx,osrt \
        --sample=none \
        --cudabacktrace=all \
        "${BENCH_BIN}" "${VARIANT}"

    # Wait for nsys to finalize the report file
    sleep 5

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
# This is the CORE of the decision tree:
#
#   check Roofline
#     → Memory bound?
#         → coalescing, bank conflicts, cache hit rates, bandwidth
#     → Compute bound?
#         → tensor core util, instruction mix, achieved FLOPS
#     → Below both roofs?
#         → occupancy, warp stalls, ILP
# ============================================================================
run_ncu() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  STAGE 2: Nsight Compute — Deep Kernel Profiling"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    local NCU_OUT="${PREFIX}_ncu"

    # ── Metric sets organized by the decision tree ──────────────────────────
    #
    # GROUP A: Roofline classification (memory vs compute bound)
    # GROUP B: Memory-bound deep dive
    # GROUP C: Compute-bound deep dive
    # GROUP D: Latency-bound (below both roofs)

    local ROOFLINE_METRICS=(
        # ── Roofline classification ──
        "sm__throughput.avg.pct_of_peak_sustained_elapsed"     # Overall SM throughput
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed" # DRAM throughput vs peak
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"
    )

    local MEMORY_METRICS=(
        # ── Coalescing ──
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"       # Actual sectors loaded
        "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"      # Requests made
        # coalescing ratio = sectors / (requests * ideal_sectors_per_request)

        # ── Bank Conflicts ──
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"  # Shared mem load conflicts
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"  # Shared mem store conflicts

        # ── Cache Hit Rates ──
        "l1tex__t_sector_hit_rate.pct"                         # L1 hit rate
        "lts__t_sector_hit_rate.pct"                           # L2 hit rate

        # ── Achieved Bandwidth ──
        "dram__bytes_read.sum"                                 # Bytes read from DRAM
        "dram__bytes_write.sum"                                # Bytes written to DRAM
        "gpu__time_duration.sum"                               # Kernel duration (ns)
        # BW = (bytes_read + bytes_write) / duration
    )

    local COMPUTE_METRICS=(
        # ── Tensor Core Utilization ──
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"

        # ── Instruction Mix ──
        "sm__sass_thread_inst_executed_op_fadd_pred_on.sum"    # FP32 add
        "sm__sass_thread_inst_executed_op_fmul_pred_on.sum"    # FP32 mul
        "sm__sass_thread_inst_executed_op_ffma_pred_on.sum"    # FP32 FMA (= 2 FLOP)

        # ── Achieved FLOPS ──
        "sm__sass_thread_inst_executed_op_fadd_pred_on.sum.per_second"
        "sm__sass_thread_inst_executed_op_ffma_pred_on.sum.per_second"
        "smsp__inst_executed.sum"                              # Total instructions
    )

    local LATENCY_METRICS=(
        # ── Occupancy ──
        "sm__warps_active.avg.pct_of_peak_sustained_active"    # Achieved occupancy

        # ── What's limiting occupancy? ──
        "launch__registers_per_thread"                         # Regs per thread
        "launch__block_size"                                   # Block size
        "launch__shared_mem_per_block_dynamic"                 # Dynamic smem
        "launch__shared_mem_per_block_static"                  # Static smem

        # ── Warp Stall Reasons ──
        "smsp__warps_issue_stalled_long_scoreboard_per_warp_active.pct"   # Waiting for GMEM
        "smsp__warps_issue_stalled_short_scoreboard_per_warp_active.pct"  # Waiting for L1/smem
        "smsp__warps_issue_stalled_wait_per_warp_active.pct"              # Barrier sync
        "smsp__warps_issue_stalled_not_selected_per_warp_active.pct"      # Ready but not picked
        "smsp__warps_issue_stalled_math_pipe_throttle_per_warp_active.pct" # Math pipe full
        "smsp__warps_issue_stalled_mio_throttle_per_warp_active.pct"      # Memory pipe full
        "smsp__warps_issue_stalled_no_instruction_per_warp_active.pct"    # I-cache miss

        # ── ILP (Instruction-Level Parallelism) ──
        "smsp__issue_active.avg.pct_of_peak_sustained_active"  # Issue rate
    )

    # Combine all metrics
    local ALL_METRICS=()
    ALL_METRICS+=("${ROOFLINE_METRICS[@]}")
    ALL_METRICS+=("${MEMORY_METRICS[@]}")
    ALL_METRICS+=("${COMPUTE_METRICS[@]}")
    ALL_METRICS+=("${LATENCY_METRICS[@]}")

    local METRICS_CSV
    METRICS_CSV=$(IFS=,; echo "${ALL_METRICS[*]}")

    # ── Reference: MoE kernel names for manual filtering ─────────────────
    # These are NOT passed to ncu (we profile all kernels).
    # Use with --kernel-name if you want to target a specific kernel:
    #   ncu --kernel-name "grouped_gemm_bt_kernel" ...
    local KERNEL_FILTERS=(
        # Grouped GEMM variants (the main compute kernels)
        "grouped_gemm_bt_kernel"
        "grouped_gemm_blackwell_async_kernel"
        # Tiled GEMM variants
        "tiled_gemm_kernel"
        "tiled_gemm_bt_kernel"
        # Naive GEMM
        "naive_gemm_bt_kernel"
        "naive_gemm_kernel_impl"
        # Fused expert kernel
        "fused_moe_kernel"
        # Routing
        "fused_gate_kernel"
        "gate_logits_kernel"
        # Gather/Scatter
        "gather_kernel"
        "scatter_kernel"
        "grouped_scatter_kernel"
        "group_reorder_kernel"
        "expert_grouping_kernel"
        # Activation
        "swiglu_strided_kernel"
        "swiglu_simple_kernel"
        # Softmax/TopK
        "softmax_experts_kernel"
        "topk_kernel"
    )

    echo "  Profiling with ${#ALL_METRICS[@]} metrics across ${#KERNEL_FILTERS[@]} kernel types..."
    echo "  This may take several minutes per kernel invocation."
    echo ""

    # Run ncu — profile kernels, skip warmup launches
    # --launch-skip: skip warmup kernel launches
    # --launch-count: profile only a few launches for each kernel
    # We skip the first ~50 launches (warmup) and profile the next 20.
    ncu --set full \
        --metrics "${METRICS_CSV}" \
        --launch-skip 2 \
        --launch-count 20 \
        --target-processes all \
        --export "${NCU_OUT}" \
        --force-overwrite \
        --page raw \
        "${BENCH_BIN}" "${VARIANT}" \
        2>&1 | tee "${NCU_OUT}_console.log"

    echo ""
    echo "  ▸ Nsight Compute report saved: ${NCU_OUT}.ncu-rep"
    echo "  ▸ Console log: ${NCU_OUT}_console.log"
    echo "  ▸ Open in Nsight Compute GUI: ncu-ui ${NCU_OUT}.ncu-rep"
    echo ""

    # Also dump CSV for automated analysis
    ncu --import "${NCU_OUT}.ncu-rep" \
        --csv \
        --page raw \
        > "${NCU_OUT}.csv" 2>/dev/null || true

    echo "  ▸ CSV export: ${NCU_OUT}.csv"
}

# ============================================================================
# STAGE 3: Automated Diagnosis — Parse ncu output through the decision tree
# ============================================================================
run_diagnosis() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  STAGE 3: Automated Bottleneck Diagnosis"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    local NCU_CSV
    # Find the most recent ncu CSV
    NCU_CSV=$(ls -t "${OUTPUT_DIR}"/moe_*_ncu.csv 2>/dev/null | head -1)
    if [[ -z "${NCU_CSV}" ]]; then
        echo "  ✗ No ncu CSV found. Run stage 2 first."
        return 1
    fi

    echo "  Analyzing: ${NCU_CSV}"
    echo ""

    # Run the Python diagnosis script
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
echo "  Profiling complete. Results in: ${OUTPUT_DIR}"
echo "═══════════════════════════════════════════════════════════════"
