# Blackwell Profiling Walkthrough

A step-by-step guide for getting your first CUDA kernel profiled on a Blackwell B200 GPU using Modal. Start here if you've never used Nsight Systems or Nsight Compute before.

## Prerequisites

- Modal account configured (`modal setup`)
- Project cloned with `modal_run.py` present

## Step 1: Start with a simple kernel

We use `benchmarks/simple_vadd.cu` — a 64M-element vector addition. It's deliberately trivial so the profiling output is easy to interpret.

```bash
modal run modal_run.py --target simple_vadd
```

Expected output: `Launching simple_vadd with N=67108864, blocks=262144` → `SUCCESS!`

## Step 2: Timeline trace (Nsight Systems)

Run a basic Nsight Systems trace to see the full execution timeline:

```bash
modal run modal_run.py --target profile_vadd_nsys
```

This shows:
- How long `cudaMalloc` and `cudaMemcpy` take on the host side
- The actual GPU execution time of `simple_vadd`
- The generated `.nsys-rep` file (for GUI analysis)

**Blackwell-specific**: You can also use hardware tracing for lower overhead:

```bash
modal run modal_run.py --target profile_vadd_nsys_hw
```

This uses `--trace=cuda-hw` instead of `--trace=cuda`. They are **mutually exclusive** — `cuda-hw` replaces `cuda` with silicon-level instrumentation.

## Step 3: Deep metrics (Nsight Compute)

Now find out *why* the kernel performs the way it does:

```bash
modal run modal_run.py --target profile_vadd_ncu
```

This collects the full metric set and runs `analyze_ncu.py` to extract the key numbers.

### What we observed on B200

| Metric | Value | Interpretation |
|---|---|---|
| DRAM Throughput | ~41% | 3.2 TB/s effective (of 8 TB/s peak) |
| SM Throughput | ~52% | SM is instruction-issue limited |
| Compute/Memory | ~13% | Kernel is **memory-bound** |

### The Blackwell lesson

At 8 TB/s HBM3e bandwidth, the B200 is so fast that **naive scalar loads cannot saturate the memory bus**. The SM hits its instruction-issue rate ceiling before exhausting bandwidth.

**Fix**: Use vectorized memory accesses (`float4` loads — 16 bytes per instruction instead of 4) and loop unrolling. This is mandatory for bandwidth-bound kernels on Blackwell.

## Step 4: Apply to MoE kernels

With the toolchain validated, move to the full auto-fetching MoE profiling pipeline:

```bash
# This triggers a timeline trace, deep metrics extraction, and auto-downloads the raw .csv and .ncu-rep logs back to your machine.
modal run modal_run.py --target profile_moe_full
```

Then you run the diagnosis locally on your downloaded trace files:
```bash
python3 profiling/diagnose_moe.py profiling/local_results/moe_Opt5_<RUN_ID>_ncu.csv
```

See `docs/moe/PROFILING.md` for the full decision tree and metric reference.
