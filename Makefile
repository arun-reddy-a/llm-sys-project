NVCC       ?= nvcc
NVCC_FLAGS  = -std=c++17 -O3 -arch=native -lnvToolsExt
DEBUG_FLAGS = -std=c++17 -G -g -arch=native

BUILD_DIR  = build
VARIANT   ?=

# Source files
MOE_SRC = kernels/moe/naive_moe.cu
DSA_SRC = kernels/dsa/naive_dsa.cu

# Targets
TEST_MOE  = $(BUILD_DIR)/test_moe
TEST_DSA  = $(BUILD_DIR)/test_dsa
BENCH_MOE = $(BUILD_DIR)/bench_moe
BENCH_DSA = $(BUILD_DIR)/bench_dsa

.PHONY: all tests benchmarks test bench clean debug_tests \
        test_moe test_dsa bench_moe bench_dsa bench_dsa_smoke bench_dsa_full \
        check_tools profile_dsa_1 profile_dsa_2 profile_dsa_3 profile_dsa_full

all: tests benchmarks

tests: $(TEST_MOE) $(TEST_DSA)

benchmarks: $(BENCH_MOE) $(BENCH_DSA)

# ---------- Build rules ----------

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TEST_MOE): tests/test_moe.cu $(MOE_SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ tests/test_moe.cu $(MOE_SRC)

$(TEST_DSA): tests/test_dsa.cu $(DSA_SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ tests/test_dsa.cu $(DSA_SRC)

$(BENCH_MOE): benchmarks/bench_moe.cu $(MOE_SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ benchmarks/bench_moe.cu $(MOE_SRC)

$(BENCH_DSA): benchmarks/bench_dsa.cu $(DSA_SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ benchmarks/bench_dsa.cu $(DSA_SRC)

# Debug builds
debug_tests: | $(BUILD_DIR)
	$(NVCC) $(DEBUG_FLAGS) -o $(BUILD_DIR)/test_moe_dbg tests/test_moe.cu $(MOE_SRC)
	$(NVCC) $(DEBUG_FLAGS) -o $(BUILD_DIR)/test_dsa_dbg tests/test_dsa.cu $(DSA_SRC)

# ---------- Convenience run targets ----------

test: tests
	@echo "==============================="
	@echo "  Running MoE tests"
	@echo "==============================="
	./$(TEST_MOE)
	@echo ""
	@echo "==============================="
	@echo "  Running DSA tests"
	@echo "==============================="
	./$(TEST_DSA)

test_moe: $(TEST_MOE)
	./$(TEST_MOE)

test_dsa: $(TEST_DSA)
	./$(TEST_DSA)

bench: benchmarks
	@echo "==============================="
	@echo "  Running MoE benchmark"
	@echo "==============================="
	./$(BENCH_MOE)
	@echo ""
	@echo "==============================="
	@echo "  Running DSA benchmark"
	@echo "==============================="
	./$(BENCH_DSA)

bench_moe: $(BENCH_MOE)
	./$(BENCH_MOE)

bench_dsa: $(BENCH_DSA)
	./$(BENCH_DSA)

# ---------- DSA Smoke Benchmark (variant-selectable, NVTX-instrumented) ----------

bench_dsa_smoke: $(DSA_SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o build/bench_dsa_smoke benchmarks/bench_dsa_smoke.cu $(DSA_SRC)
	./build/bench_dsa_smoke $(VARIANT)

bench_dsa_full: $(DSA_SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o build/bench_dsa_full benchmarks/bench_dsa_full.cu $(DSA_SRC)
	./build/bench_dsa_full

check_tools:
	@nsys --version
	@ncu --version

# ---------- DSA Profiling (3-stage pipeline) ----------
# Uses bench_dsa_smoke (NVTX-instrumented) as the profiling workload.

# Stage 1: Timeline Trace (Nsight Systems)
profile_dsa_1: bench_dsa_smoke
	mkdir -p profiling/results
	nsys profile \
		--output=profiling/results/dsa_$(VARIANT)_$(RUN_ID)_nsys \
		--force-overwrite=true \
		--trace=cuda,nvtx,osrt \
		--sample=none \
		--stats=true \
		./build/bench_dsa_smoke $(VARIANT)

# Stage 2: Deep Dive (Nsight Compute — roofline, memory, compute, occupancy)
profile_dsa_2: bench_dsa_smoke
	mkdir -p profiling/results
	ncu --set full \
		--target-processes all \
		--export profiling/results/dsa_$(VARIANT)_$(RUN_ID)_ncu \
		--force-overwrite \
		./build/bench_dsa_smoke $(VARIANT) \
		2>&1 | tee profiling/results/dsa_$(VARIANT)_$(RUN_ID)_ncu_console.log
	@echo ""
	@echo "Exporting CSV for analysis..."
	ncu --import profiling/results/dsa_$(VARIANT)_$(RUN_ID)_ncu.ncu-rep \
		--csv --page raw > profiling/results/dsa_$(VARIANT)_$(RUN_ID)_ncu.csv

# Stage 3: Automated Diagnosis (parse NCU metrics)
profile_dsa_3:
	@echo "=== DSA Profiling Stage 3: Automated Diagnosis ==="
	@echo "CSV: profiling/results/dsa_$(VARIANT)_$(RUN_ID)_ncu.csv"
	@if [ -f profiling/diagnose_dsa.py ]; then \
		python3 profiling/diagnose_dsa.py profiling/results/dsa_$(VARIANT)_$(RUN_ID)_ncu.csv; \
	elif [ -f profiling/analyze_ncu.py ]; then \
		python3 profiling/analyze_ncu.py profiling/results/dsa_$(VARIANT)_$(RUN_ID)_ncu.csv; \
	else \
		echo "No diagnosis script found. Inspect the CSV manually."; \
	fi

# All 3 stages in sequence
profile_dsa_full: profile_dsa_1 profile_dsa_2 profile_dsa_3

clean:
	rm -rf $(BUILD_DIR)
