NVCC       ?= nvcc
NVCC_FLAGS  = -std=c++17 -O2 -arch=native
DEBUG_FLAGS = -std=c++17 -G -g -arch=native

BUILD_DIR  = build

# Source files
MOE_SRC = kernels/moe/naive_moe.cu

# Targets
TEST_MOE  = $(BUILD_DIR)/test_moe
BENCH_MOE = $(BUILD_DIR)/bench_moe

.PHONY: all tests benchmarks test bench clean debug_tests

all: tests benchmarks

tests: $(TEST_MOE)

benchmarks: $(BENCH_MOE)

# ---------- Build rules ----------

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TEST_MOE): tests/test_moe.cu $(MOE_SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ tests/test_moe.cu $(MOE_SRC)

$(BENCH_MOE): benchmarks/bench_moe.cu $(MOE_SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ benchmarks/bench_moe.cu $(MOE_SRC)

# Debug builds (with device-side debugging)
debug_tests: | $(BUILD_DIR)
	$(NVCC) $(DEBUG_FLAGS) -o $(BUILD_DIR)/test_moe_dbg tests/test_moe.cu $(MOE_SRC)

# ---------- Convenience run targets ----------

test: tests
	@echo "==============================="
	@echo "  Running MoE tests"
	@echo "==============================="
	./$(TEST_MOE)

bench: benchmarks
	@echo "==============================="
	@echo "  Running MoE benchmark"
	@echo "==============================="
	./$(BENCH_MOE)

test_moe: $(TEST_MOE)
	./$(TEST_MOE)

bench_moe: $(BENCH_MOE)
	./$(BENCH_MOE)

# ---------- Profiling targets (MoE) ----------

profile_moe: $(BENCH_MOE)
	./profiling/profile_moe.sh --stage all

profile_moe_nsys: $(BENCH_MOE)
	./profiling/profile_moe.sh --stage 1

profile_moe_ncu: $(BENCH_MOE)
	./profiling/profile_moe.sh --stage 2

profile_moe_diag:
	./profiling/profile_moe.sh --stage 3

clean:
	rm -rf $(BUILD_DIR)
