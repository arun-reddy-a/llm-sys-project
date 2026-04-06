NVCC       ?= nvcc
NVCC_FLAGS  = -std=c++17 -O2 -arch=native
DEBUG_FLAGS = -std=c++17 -G -g -arch=native

BUILD_DIR  = build
VARIANT   ?= Opt5

# Source files
MOE_SRC = kernels/moe/naive_moe.cu

# Targets
TEST_MOE  = $(BUILD_DIR)/test_moe
TEST_DEEPSEEK = $(BUILD_DIR)/test_deepseek
BENCH_MOE = $(BUILD_DIR)/bench_moe
SIMPLE_VADD = $(BUILD_DIR)/simple_vadd

.PHONY: all tests benchmarks test bench clean debug_tests profile_vadd_nsys profile_vadd_ncu check_tools test_deepseek

all: tests benchmarks

tests: $(TEST_MOE) $(TEST_DEEPSEEK)

benchmarks: $(BENCH_MOE)

# ---------- Build rules ----------

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TEST_MOE): tests/test_moe.cu $(MOE_SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ tests/test_moe.cu $(MOE_SRC)

$(TEST_DEEPSEEK): tests/test_moe_deepseek.cu $(MOE_SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ tests/test_moe_deepseek.cu $(MOE_SRC)

$(BENCH_MOE): benchmarks/bench_moe.cu $(MOE_SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ benchmarks/bench_moe.cu $(MOE_SRC)

bench_moe_smoke: $(MOE_SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -lnvToolsExt -o build/bench_moe_smoke benchmarks/bench_moe_smoke.cu $(MOE_SRC)

# Debug builds (with device-side debugging)
debug_tests: | $(BUILD_DIR)
	$(NVCC) $(DEBUG_FLAGS) -o $(BUILD_DIR)/test_moe_dbg tests/test_moe.cu $(MOE_SRC)

# ---------- Convenience run targets ----------

test: tests
	@echo "==============================="
	@echo "  Running MoE tests"
	@echo "==============================="
	./$(TEST_MOE)
	./$(TEST_DEEPSEEK)

test_deepseek: $(TEST_DEEPSEEK)
	./$(TEST_DEEPSEEK)

bench: benchmarks
	@echo "==============================="
	@echo "  Running MoE benchmark"
	@echo "==============================="
	./$(BENCH_MOE)

test_moe: $(TEST_MOE)
	./$(TEST_MOE)

bench_moe: $(BENCH_MOE)
	./$(BENCH_MOE)

check_tools:
	@nsys --version
	@ncu --version
	@nsys profile --help | grep "cuda-hw" || echo "cuda-hw NOT in help"

simple_vadd: $(SIMPLE_VADD)
	./$(SIMPLE_VADD)

$(SIMPLE_VADD): benchmarks/simple_vadd.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ benchmarks/simple_vadd.cu

# --- Simple Vector Add Profiling ---

profile_vadd_nsys: $(SIMPLE_VADD)
	mkdir -p profiling/results
	nsys profile \
		--output=profiling/results/simple_vadd_nsys \
		--force-overwrite=true \
		--trace=cuda,nvtx,osrt \
		--sample=none \
		--stats=true \
		./$(SIMPLE_VADD)

profile_vadd_nsys_hw: $(SIMPLE_VADD)
	mkdir -p profiling/results
	nsys profile \
		--output=profiling/results/simple_vadd_nsys_hw \
		--force-overwrite=true \
		--trace=cuda-hw,nvtx,osrt \
		--sample=none \
		--stats=true \
		./$(SIMPLE_VADD)

profile_vadd_ncu: $(SIMPLE_VADD)
	mkdir -p profiling/results
	ncu --set full \
		--target-processes all \
		--export profiling/results/simple_vadd_ncu \
		--force-overwrite \
		./$(SIMPLE_VADD)
	@echo "\n📊 Exporting CSV for analysis..."
	ncu --import profiling/results/simple_vadd_ncu.ncu-rep --csv --page raw > profiling/results/simple_vadd_ncu.csv
	@python3 profiling/analyze_ncu.py profiling/results/simple_vadd_ncu.csv

# ---------- MoE Profiling (3-stage pipeline) ----------
# Uses bench_moe_smoke (NVTX-instrumented) as the profiling workload.
# See docs/moe/PROFILING.md for the full decision tree.

# Stage 1: Timeline Trace (Nsight Systems — find the slow kernel)
profile_moe_1: bench_moe_smoke
	./profiling/profile_moe.sh --stage 1 --variant $(VARIANT) --run-id "$(RUN_ID)" --bench-bin ./build/bench_moe_smoke

# Stage 2: Deep Dive (Nsight Compute — roofline, memory, compute, occupancy)
profile_moe_2: bench_moe_smoke
	./profiling/profile_moe.sh --stage 2 --variant $(VARIANT) --run-id "$(RUN_ID)" --bench-bin ./build/bench_moe_smoke

# Stage 3: Automated Diagnosis (parse NCU metrics into recommendations)
profile_moe_3:
	./profiling/profile_moe.sh --stage 3 --variant $(VARIANT) --run-id "$(RUN_ID)"

# All 3 stages in sequence
profile_moe_full: profile_moe_1 profile_moe_2 profile_moe_3

clean:
	rm -rf $(BUILD_DIR)
