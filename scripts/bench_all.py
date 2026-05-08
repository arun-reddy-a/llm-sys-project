#!/usr/bin/env python3
"""
Unified MoE benchmark — all variants, same T values, one table.

Variants compared:
  FlashInfer TRT-LLM  — production FP8 baseline (requires flashinfer)
  deep_gemm FP8       — DeepSeek grouped GEMM (requires deep_gemm)
  Our Triton FP8      — solution/triton/kernel.py
  CUDA BF16-WMMA      — kernels/moe: moe_forward_deepseek_bf16
  CUDA BF16-cuBLAS    — kernels/moe: moe_forward_deepseek_bf16_cublas

Usage:
  python3 scripts/bench_all.py [--warmup N] [--iters N] [--seq-lens T1,T2,...]
  make bench_all
"""
import argparse, re, subprocess, sys, time, os
import torch

# ── Problem config (DeepSeek-V3, 32 of 256 local experts) ─────────────────
E, EL, H, I_DIM, G1 = 256, 32, 7168, 2048, 4096
K, NG, KG, BS, RSF  = 8, 8, 4, 128, 2.5
LOCAL_OFFSET         = 0

# Match competition + README table (covers small, medium, large).
DEFAULT_T = [52, 80, 512, 901, 2048, 4096, 11948, 14107]

COLS      = ["FlashInfer", "deep_gemm", "Triton-FP8", "CUDA-WMMA", "CUDA-cuBLAS"]

# ── Input factory ──────────────────────────────────────────────────────────
def make_inputs(T):
    w1  = torch.randn(EL, G1,    H,     dtype=torch.float32, device="cuda").to(torch.float8_e4m3fn)
    w1s = torch.ones( EL, G1//BS, H//BS, dtype=torch.float32, device="cuda")
    w2  = torch.randn(EL, H,     I_DIM, dtype=torch.float32, device="cuda").to(torch.float8_e4m3fn)
    w2s = torch.ones( EL, H//BS, I_DIM//BS, dtype=torch.float32, device="cuda")
    rl  = torch.randn(T, E, dtype=torch.float32, device="cuda")
    out = torch.zeros(T, H, dtype=torch.bfloat16, device="cuda")
    hs  = torch.randn(T, H, dtype=torch.float32, device="cuda").to(torch.float8_e4m3fn)
    hss = torch.ones( H//BS, T, dtype=torch.float32, device="cuda")
    return rl, w1, w1s, w2, w2s, out, hs, hss

# ── Generic CUDA-event timer ───────────────────────────────────────────────
def bench(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t.append((time.perf_counter() - t0) * 1e3)
    return sum(t) / len(t)   # mean ms

# ── CUDA benchmark via subprocess ─────────────────────────────────────────
_ROW = re.compile(
    r'^\s*(DS-V3-BF16|DS-BF16-CB)\s+T=(\d+),\S+\s+([\d.]+)\s+([\d.]+)'
)

def run_cuda_bench(warmup, iters):
    """Build bench_moe if needed, run it, return {T: {'WMMA': ms, 'cuBLAS': ms}}."""
    print("  [cuda] building bench_moe ...", flush=True)
    r = subprocess.run(["make", "bench_moe"], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [cuda] build FAILED:\n{r.stderr}", file=sys.stderr)
        return {}

    r = subprocess.run(
        ["./build/bench_moe", str(warmup), str(iters)],
        capture_output=True, text=True,
    )
    out = {}
    for line in r.stdout.splitlines():
        m = _ROW.match(line)
        if not m:
            continue
        tag, T, mean_ms = m.group(1), int(m.group(2)), float(m.group(4))
        out.setdefault(T, {})
        out[T]["WMMA" if tag == "DS-V3-BF16" else "cuBLAS"] = mean_ms
    return out

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup",   type=int, default=3)
    ap.add_argument("--iters",    type=int, default=20)
    ap.add_argument("--seq-lens", type=str, default="",
                    help="Comma-separated T values; defaults to competition set")
    args = ap.parse_args()

    seq_lens = (
        [int(x) for x in args.seq_lens.split(",") if x.strip()]
        if args.seq_lens else DEFAULT_T
    )

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    # ── Load Python kernels ────────────────────────────────────────────────
    from solution.triton.kernel import kernel as triton_kernel

    try:
        from flashinfer.fused_moe import trtllm_fp8_block_scale_moe as _bl
        def flashinfer_fn(rl, hs, hss, w1, w1s, w2, w2s):
            return _bl(rl, None, hs, hss, w1, w1s, w2, w2s,
                       E, K, NG, KG, I_DIM, LOCAL_OFFSET, EL, RSF)
        has_flashinfer = True
        print("  [ok] FlashInfer baseline available")
    except Exception as e:
        has_flashinfer = False
        print(f"  [--] FlashInfer not available: {e}")

    try:
        import deep_gemm as _dg   # noqa: F401
        has_deepgemm = True
        print("  [ok] deep_gemm available")
    except Exception:
        has_deepgemm = False
        print("  [--] deep_gemm not available")

    # ── CUDA kernels ───────────────────────────────────────────────────────
    cuda = run_cuda_bench(args.warmup, args.iters)

    # ── Per-T Python benchmarks ────────────────────────────────────────────
    table = {T: {} for T in seq_lens}

    print(f"\n  Benchmarking Python kernels (warmup={args.warmup}, iters={args.iters}) ...")
    for T in seq_lens:
        print(f"    T={T} ...", end=" ", flush=True)
        rl, w1, w1s, w2, w2s, out, hs, hss = make_inputs(T)
        pos = (rl, w1, w1s, w2, w2s, LOCAL_OFFSET, RSF, out)

        if has_flashinfer:
            try:
                table[T]["FlashInfer"] = bench(
                    lambda: flashinfer_fn(rl, hs, hss, w1, w1s, w2, w2s),
                    args.warmup, args.iters,
                )
            except Exception as e:
                print(f"[flashinfer err: {e}]", end=" ")

        try:
            table[T]["Triton-FP8"] = bench(
                lambda: triton_kernel(*pos, hidden_states=hs, hidden_states_scale=hss),
                args.warmup, args.iters,
            )
        except Exception as e:
            print(f"[triton err: {e}]", end=" ")

        if T in cuda:
            table[T]["CUDA-WMMA"]   = cuda[T].get("WMMA")
            table[T]["CUDA-cuBLAS"] = cuda[T].get("cuBLAS")

        print("done")

    # ── Print comparison table ─────────────────────────────────────────────
    COL_W  = 14
    border = "=" * (7 + (COL_W + 2) * len(COLS))
    print(f"\n{border}")
    print(f"  FP8 Block-Scale MoE — Full Comparison  (B200, E={E}/EL={EL}/K={K}/H={H}/I={I_DIM})")
    print(f"  warmup={args.warmup}  iters={args.iters}")
    print(f"{border}")
    hdr = f"  {'T':>6}" + "".join(f"  {c:>{COL_W}}" for c in COLS)
    print(hdr)
    print("-" * len(hdr))

    for T in seq_lens:
        row = f"  {T:>6}"
        for c in COLS:
            v = table[T].get(c)
            row += f"  {f'{v:.3f} ms':>{COL_W}}" if v is not None else f"  {'—':>{COL_W}}"
        print(row)
    print("-" * len(hdr))

    # ── Speedup vs FlashInfer ──────────────────────────────────────────────
    if any(table[T].get("FlashInfer") for T in seq_lens):
        print(f"\n  Speedup vs FlashInfer baseline (>1.0x = faster than baseline):")
        print(f"  {'T':>6}" + "".join(f"  {c:>{COL_W}}" for c in COLS[1:]))
        print("-" * (7 + (COL_W + 2) * (len(COLS) - 1)))
        for T in seq_lens:
            bl = table[T].get("FlashInfer")
            if not bl:
                continue
            row = f"  {T:>6}"
            for c in COLS[1:]:
                v = table[T].get(c)
                row += f"  {f'{bl/v:.2f}x':>{COL_W}}" if v else f"  {'—':>{COL_W}}"
            print(row)

    print()


if __name__ == "__main__":
    main()
