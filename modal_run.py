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
def run_make_target(target: str) -> dict:
    """Run a specific Makefile target in the remote container."""
    print(f"\n🚀 [modal] Executing: 'make {target}'")
    print(f"   Environment: nvidia/cuda:12.8.1-devel-ubuntu24.04")
    print(f"   GPU        : B200 (single)")

    # Run the make command in /workspace and stream output
    import subprocess
    import sys

    process = subprocess.Popen(
        ["make", target],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd="/workspace",
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


@app.local_entrypoint()
def main(target: str = "bench_moe_smoke"):
    """
    Local entrypoint to run benchmarks or profiling on cloud GPUs.

    Usage:
        modal run modal_run.py --target bench_moe_smoke
        modal run modal_run.py --target profile_moe_1
    
    Results are logged to results.jsonl (one JSON object per line).
    """
    print("\n===========================================================")
    print("=                                                         =")
    print("=  LLM System Kernels — Modal Cloud Runner                =")
    print("=                                                         =")
    print("===========================================================")

    targets = target.split(",")
    failed = []

    for t in targets:
        t = t.strip()
        res = run_make_target.remote(t)

        if res["returncode"] == 0:
            print(f"   ✅ Target '{t}' PASSED.")
        else:
            print(f"   ❌ Target '{t}' FAILED (exit code {res['returncode']}).")
            failed.append(t)

        # Log every run as structured JSON
        log_result(t, res)
        print(f"   📝 Logged to {RESULTS_FILE}")

    if failed:
        print(f"\n==============================================")
        print(f"❌ FAILED targets: {failed}")
        sys.exit(1)
    else:
        print(f"\n==============================================")
        print(f"✅ All targets completed successfully! 🎉")
