import modal
import os
import sys
import json
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Modal Configuration:
# - Target B200 GPU for Blackwell architecture.
# - Use a CUDA-enabled development image.
# - Mount the local workspace for access to kernels and benchmarks.
# ---------------------------------------------------------------------------

RESULTS_FILE = "results.jsonl"

app = modal.App("llm-sys-kernels")

# Image definition:
# - Based on NVIDIA CUDA 12.8.1 with Ubuntu 24.04 (matching your mini-sglang env).
# - Provides CUDA headers and libraries.
# - Includes developmental tools for kernels and profiling.
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "build-essential", "wget", "gnupg", "libnuma-dev", "libicu-dev", "software-properties-common")
    .run_commands(
        "rm -f /etc/apt/sources.list.d/cuda*.list",
        "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
        "apt-get install -y nsight-systems-2025.1.3 nsight-compute-2025.1.1",
        "ln -sf /opt/nvidia/nsight-systems/2025.1.3/target-linux-x64/nsys /usr/local/bin/nsys",
        "ln -sf /opt/nvidia/nsight-compute/2025.1.1/ncu /usr/local/bin/ncu",
    )
    # Add local project files into /workspace
    .add_local_dir(
        ".",
        remote_path="/workspace",
        ignore=[".git", "build", "__pycache__", "*.pyc"],
    )
)

# Persist profiling results across runs
results_vol = modal.Volume.from_name("llm-sys-profiling-results", create_if_missing=True)

# Separate image for vLLM comparison — has torch + vLLM Triton kernels but no nsight tools.
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "build-essential", "libnuma-dev")
    .pip_install("torch", extra_index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("vllm")
    .add_local_dir(
        ".",
        remote_path="/workspace",
        ignore=[".git", "build", "__pycache__", "*.pyc"],
    )
)


def log_result(target: str, result: dict):
    """Append a structured JSON Lines entry to results.jsonl.
    
    Each line is a self-contained JSON object:
    {
        "timestamp": "2026-04-06T18:30:00Z",
        "target": "bench_moe_smoke",
        "exit_code": 0,
        "gpu": "B200",
        "image": "nvidia/cuda:12.8.1-devel-ubuntu24.04",
        "stdout": "..."
    }
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "target": target,
        "exit_code": result["returncode"],
        "gpu": "B200",
        "image": "nvidia/cuda:12.8.1-devel-ubuntu24.04",
        "stdout": result["stdout"],
    }
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


@app.function(
    image=image,
    gpu="B200:1",      # Target Blackwell GPU
    volumes={"/workspace/profiling/results": results_vol},
    timeout=600,       # 10 min max run time
)
def run_make_target(target: str, run_id: str = "", variant: str = "") -> dict:
    """Run a specific Makefile target in the remote container."""
    print(f"\n🚀 [modal] Executing: 'make {target}'")
    print(f"   Environment: nvidia/cuda:12.8.1-devel-ubuntu24.04")
    print(f"   GPU        : B200 (single)")
    if run_id: print(f"   Run ID     : {run_id}")
    if variant: print(f"   Variant    : {variant}")

    # Run the make command in /workspace and stream output
    import subprocess
    import sys
    import os

    env = os.environ.copy()
    if run_id:
        env["RUN_ID"] = run_id
    if variant:
        env["VARIANT"] = variant

    process = subprocess.Popen(
        ["make", target],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd="/workspace",
        env=env,
        bufsize=1
    )

    full_output = []
    print(f"--- START OUTPUT for 'make {target}' ---")
    for line in process.stdout:
        print(line, end="", flush=True)
        full_output.append(line)
    process.wait()
    print(f"\n--- END OUTPUT (Exit Code: {process.returncode}) ---")

    return {
        "stdout": "".join(full_output),
        "returncode": process.returncode
    }


@app.function(
    image=vllm_image,
    gpu="B200:1",
    timeout=600,
)
def run_vllm_bench() -> dict:
    """Run vLLM fused_moe benchmark with DeepSeek-V3 config."""
    import subprocess, sys
    print("\n🚀 [modal] Running vLLM fused_moe benchmark (BF16 Triton, B200)")
    process = subprocess.Popen(
        ["python3", "benchmarks/bench_compare.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd="/workspace",
        bufsize=1,
    )
    full_output = []
    for line in process.stdout:
        print(line, end="", flush=True)
        full_output.append(line)
    process.wait()
    return {"stdout": "".join(full_output), "returncode": process.returncode}


def _parse_bench_line(line: str):
    """Extract (T, min_ms, toks_per_sec) from a bench output line."""
    try:
        if "T=" not in line:
            return None
        parts = line.split()
        T = int(parts[1].split("T=")[1].split(",")[0])
        t_min = float(parts[-3])
        tps   = float(parts[-1])
        return T, t_min, tps
    except (IndexError, ValueError):
        return None


@app.local_entrypoint()
def main(target: str = "bench_moe_smoke", variant: str = "Opt5"):
    """
    Local entrypoint to run benchmarks or profiling on cloud GPUs.

    Usage:
        modal run modal_run.py --target bench_moe_smoke
        modal run modal_run.py --target profile_moe_1 --variant Opt5
    
    Results are logged to results.jsonl, and if profiling, raw files are fetched locally.
    """
    print("\n===========================================================")
    print("=                                                         =")
    print("=  LLM System Kernels — Modal Cloud Runner                =")
    print("=                                                         =")
    print("===========================================================")

    # ── comparison mode ────────────────────────────────────────────────────
    if target == "compare":
        print("\n  Running comparison: our DeepSeek-V3 kernel vs vLLM fused_moe")
        print("  (two B200 instances in parallel)\n")

        our_future  = run_make_target.spawn("bench_moe", "", "")
        vllm_future = run_vllm_bench.spawn()

        our_res  = our_future.get()
        vllm_res = vllm_future.get()

        # Parse both outputs
        our_rows  = {}
        vllm_rows = {}
        for line in our_res["stdout"].splitlines():
            if "DeepSeek-V3" in line:
                parsed = _parse_bench_line(line)
                if parsed:
                    our_rows[parsed[0]] = parsed[1:]
        for line in vllm_res["stdout"].splitlines():
            if "vLLM" in line:
                parsed = _parse_bench_line(line)
                if parsed:
                    vllm_rows[parsed[0]] = parsed[1:]

        SEQ_LENS = [64, 256, 512, 1024, 2048, 4096]
        print("\n" + "=" * 96)
        print("  DeepSeek-V3 MoE: Ours (FP32 CUDA) vs vLLM fused_moe (BF16 Triton) — B200")
        print("=" * 96)
        print(f"  {'T':<6}  {'Ours min(ms)':<14} {'Ours tok/s':<14} {'vLLM min(ms)':<14} {'vLLM tok/s':<14} {'Ratio'}")
        print("  " + "-" * 82)
        for T in SEQ_LENS:
            o = our_rows.get(T)
            v = vllm_rows.get(T)
            o_min = f"{o[0]:.3f}" if o else "—"
            o_tps = f"{o[1]:,.0f}"  if o else "—"
            v_min = f"{v[0]:.3f}" if v else "—"
            v_tps = f"{v[1]:,.0f}"  if v else "—"
            if o and v:
                ratio = v[1] / o[1]
                note  = "vLLM faster" if ratio > 1 else "Ours faster"
                ratio_str = f"{ratio:.2f}× ({note})"
            else:
                ratio_str = "—"
            print(f"  {T:<6}  {o_min:<14} {o_tps:<14} {v_min:<14} {v_tps:<14} {ratio_str}")
        print("=" * 96)
        print("  Note: FP32 vs BF16 — not perfectly apples-to-apples.")
        print("  vLLM uses Triton autotuning; ours is handwritten CUDA.")
        return

    # ── normal mode ────────────────────────────────────────────────────────
    targets = target.split(",")
    failed = []

    # Generate unique run ID for tracking output files
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    for t in targets:
        t = t.strip()
        res = run_make_target.remote(t, run_id, variant)

        if res["returncode"] == 0:
            print(f"   ✅ Target '{t}' PASSED.")
        else:
            print(f"   ❌ Target '{t}' FAILED (exit code {res['returncode']}).")
            failed.append(t)

        # Log every run as structured JSON
        log_result(t, res)
        print(f"   📝 Logged to {RESULTS_FILE}")

    if not failed and any("profile" in t for t in targets):
        import subprocess
        print(f"\n📥 Fetching profiling results for RUN_ID: {run_id}...")
        local_dir = "profiling/local_results"
        os.makedirs(local_dir, exist_ok=True)
        
        # We explicitly request the files we expect to be generated
        base_name = f"moe_{variant}_{run_id}"
        expected_files = [
            f"{base_name}_nsys.nsys-rep",
            f"{base_name}_nsys.sqlite",
            f"{base_name}_ncu.ncu-rep",
            f"{base_name}_ncu.csv",
            f"{base_name}_ncu_console.log"
        ]
        
        for file in expected_files:
            try:
                subprocess.run(
                    ["modal", "volume", "get", "llm-sys-profiling-results", file, f"{local_dir}/{file}"],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                print(f"   ✅ Downloaded: {file}")
            except subprocess.CalledProcessError:
                # Some files might not exist depending on the stage executed, which is expected.
                pass
                
        print(f"\n   To analyze locally, run:")
        print(f"   python3 profiling/diagnose_moe.py {local_dir}/{base_name}_ncu.csv")

    if failed:
        print(f"\n==============================================")
        print(f"❌ FAILED targets: {failed}")
        sys.exit(1)
    else:
        print(f"\n==============================================")
        print(f"✅ All targets completed successfully! 🎉")
