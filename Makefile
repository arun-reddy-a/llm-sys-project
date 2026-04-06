NVCC       ?= nvcc
NVCC_FLAGS  = -std=c++17 -O2 -arch=native
DEBUG_FLAGS = -std=c++17 -G -g -arch=native

BUILD_DIR  = build

# Source files
MOE_SRC = kernels/moe/naive_moe.cu
DSA_SRC = kernels/dsa/naive_dsa.cu

# Targets
TEST_MOE  = $(BUILD_DIR)/test_moe
TEST_DSA  = $(BUILD_DIR)/test_dsa
BENCH_MOE = $(BUILD_DIR)/bench_moe
BENCH_DSA = $(BUILD_DIR)/bench_dsa

.PHONY: all tests benchmarks test bench clean debug_tests

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

# Debug builds (with device-side debugging)
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

test_moe: $(TEST_MOE)
	./$(TEST_MOE)

test_dsa: $(TEST_DSA)
	./$(TEST_DSA)

bench_moe: $(BENCH_MOE)
	./$(BENCH_MOE)

bench_dsa: $(BENCH_DSA)
	./$(BENCH_DSA)

# ---------- Profiling targets (MoE) ----------

profile_moe: $(BENCH_MOE)
	./profiling/profile_moe.sh --stage all

profile_moe_nsys: $(BENCH_MOE)
	./profiling/profile_moe.sh --stage 1

profile_moe_ncu: $(BENCH_MOE)
	./profiling/profile_moe.sh --stage 2

profile_moe_diag:
	./profiling/profile_moe.sh --stage 3

# ---------- Profiling targets (DSA) ----------

profile_dsa: $(BENCH_DSA)
	./profiling/profile_dsa.sh --stage all

profile_dsa_nsys: $(BENCH_DSA)
	./profiling/profile_dsa.sh --stage 1

profile_dsa_ncu: $(BENCH_DSA)
	./profiling/profile_dsa.sh --stage 2

profile_dsa_diag:
	./profiling/profile_dsa.sh --stage 3

clean:
	rm -rf $(BUILD_DIR)
