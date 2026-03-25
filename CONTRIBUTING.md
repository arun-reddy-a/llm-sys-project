# Contributing

This guide walks you through the full contribution workflow — from cloning the repo to pushing an optimised kernel.

## 1. Get the Code

**First time?** Clone the repository:

```bash
git clone <repo-url> && cd LLM-Sys-Project
```

**Already have the repo?** Pull the latest changes before starting any work:

```bash
git pull origin main
```

## 2. Build and Run Baseline Tests & Benchmarks

Before touching any code, compile everything and verify that the existing kernels are correct and that you have a baseline benchmark to compare against.

```bash
# Build all targets (tests + benchmarks)
make all

# Run correctness tests — all must PASS
make test

# Run benchmarks and SAVE the output
make bench | tee baseline_bench.txt
```

> **Tip:** You can also run MoE and DSA targets independently:
>
> ```bash
> make test_moe
> make test_dsa
> make bench_moe | tee baseline_bench_moe.txt
> make bench_dsa | tee baseline_bench_dsa.txt
> ```

Keep the baseline benchmark output — you will need it later to prove your changes are an improvement.

## 3. Implement Your Kernel Optimisation

Pick a TODO item from the **Optimization Roadmap** in `README.md` and implement it in the appropriate kernel file:

- **MoE kernels** — `kernels/moe/`
- **DSA kernels** — `kernels/dsa/`

Follow the existing code style and naming conventions. If your optimisation adds new files, update the `Makefile` accordingly.

## 4. Re-run Tests & Benchmarks

After making your changes, rebuild and verify correctness:

```bash
make clean && make all

# All tests must still PASS
make test
```

Then re-run the benchmarks:

```bash
make bench | tee optimised_bench.txt
```

Compare the results against your saved baseline:

```bash
diff baseline_bench.txt optimised_bench.txt
```

## 5. Verify Improvement

**Your code may only be pushed if the benchmark numbers improve** (lower latency and/or higher throughput) compared to the baseline you recorded in step 2.

Specifically, check that:

- All correctness tests still **PASS**.
- Latency (min/mean/median) for the affected kernel configs is **lower** than the baseline.
- Throughput (Tok/s) for the affected kernel configs is **higher** than the baseline.

If your changes regress performance or break tests, go back to step 3 and iterate.

## 6. Update the README

Once you have confirmed that your optimisation passes tests and improves benchmarks, mark the corresponding TODO item as done in `README.md`.

For example, change:

```
1. **Shared-memory tiled GEMM** -- Replace the naive ...
```

to:

```
1. ~~**Shared-memory tiled GEMM**~~ ✅ -- Replace the naive ...
```

## 7. Commit and Push

```bash
git add -A
git commit -m "Optimise <kernel>: <short description of what you did>"
git push origin main
```

## Contribution Checklist

Use this checklist before pushing:

- [ ] Pulled the latest code (`git pull`)
- [ ] Recorded baseline benchmark numbers **before** making changes
- [ ] All correctness tests pass (`make test`)
- [ ] Benchmark shows improvement over the baseline
- [ ] Updated the corresponding TODO item in `README.md` as done
- [ ] Committed with a clear, descriptive message
