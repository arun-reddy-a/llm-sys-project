NVCC       ?= nvcc
NVCC_FLAGS  = -std=c++17 -O3 -arch=native -lcublas -lnvToolsExt
DEBUG_FLAGS = -std=c++17 -G -g -arch=native

BUILD_DIR  = build
VARIANT   ?= Opt5

# Source files
MOE_SRC = kernels/moe/moe_kernels.cu

# Targets
TEST_MOE  = $(BUILD_DIR)/test_moe
TEST_DEEPSEEK = $(BUILD_DIR)/test_deepseek
BENCH_MOE = $(BUILD_DIR)/bench_moe
BENCH_MOE_SEQLEN = $(BUILD_DIR)/bench_moe_seqlen
TEST_FP8 = $(BUILD_DIR)/test_fp8_cuda
SIMPLE_VADD = $(BUILD_DIR)/simple_vadd

.PHONY: all tests benchmarks test bench clean debug_tests profile_vadd_nsys profile_vadd_ncu check_tools test_deepseek test_fp8 test_fp8_correctness

all: tests benchmarks

tests: $(TEST_MOE) $(TEST_DEEPSEEK)

benchmarks: $(BENCH_MOE) $(BENCH_MOE_SEQLEN)

# ---------- Build rules ----------

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TEST_MOE): tests/test_moe.cu $(MOE_SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ tests/test_moe.cu $(MOE_SRC)

$(TEST_DEEPSEEK): tests/test_moe_deepseek.cu $(MOE_SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ tests/test_moe_deepseek.cu $(MOE_SRC)

$(BENCH_MOE): benchmarks/bench_moe.cu $(MOE_SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ benchmarks/bench_moe.cu $(MOE_SRC)

$(BENCH_MOE_SEQLEN): benchmarks/bench_moe_seqlen.cu $(MOE_SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ benchmarks/bench_moe_seqlen.cu $(MOE_SRC)

# FP8 correctness test binary
FP8_SRC = kernels/moe/fp8_moe.cu
$(TEST_FP8): tests/test_fp8_cuda.cu $(MOE_SRC) $(FP8_SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ tests/test_fp8_cuda.cu $(MOE_SRC) $(FP8_SRC)

test_fp8: $(TEST_FP8)
	@echo "FP8 binary built: $(TEST_FP8)"

test_fp8_correctness: $(TEST_FP8)
	@echo "==============================="
	@echo "  Running FP8 correctness test"
	@echo "==============================="
	pip3 install -q numpy 2>/dev/null || true
	python3 tests/test_fp8_correctness.py /tmp/fp8_test_data

bench_moe_smoke: $(MOE_SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -lnvToolsExt -o build/bench_moe_smoke benchmarks/bench_moe_smoke.cu $(MOE_SRC)
	VARIANT=$(VARIANT) ./build/bench_moe_smoke

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

bench_moe_seqlen: $(BENCH_MOE_SEQLEN)
	./$(BENCH_MOE_SEQLEN)

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
	./profiling/profile_moe.sh --stage 3 --variant $(VARIANT) --run-id "$(RUN_ID)" --bench-bin ./build/bench_moe_smoke

# All 3 stages in sequence
profile_moe_full: profile_moe_1 profile_moe_2 profile_moe_3

clean:
	rm -rf $(BUILD_DIR)

bench_moe_4096: benchmarks/bench_moe_4096.cu $(MOE_SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ benchmarks/bench_moe_4096.cu $(MOE_SRC)

run_bench_moe_4096: bench_moe_4096
	./bench_moe_4096

bench_missing: benchmarks/bench_missing.cu $(MOE_SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ benchmarks/bench_missing.cu $(MOE_SRC)

run_bench_missing: bench_missing
	./bench_missing
