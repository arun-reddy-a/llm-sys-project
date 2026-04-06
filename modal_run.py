import modal
import os
import subprocess
import sys

# Define the Modal App
app = modal.App("llm-sys-kernels")

# ---------------------------------------------------------------------------
# Image Configuration — Built once and cached
# ---------------------------------------------------------------------------
# Using nvidia/cuda:12.8.0-devel-ubuntu22.04:
# - Provides nvcc 12.8.
# - Provides CUDA headers and libraries. 
# - Includes developmental tools for kernels.
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential", "wget", "gnupg")
    .run_commands(
        "apt-get update && apt-get install -y nsight-systems-cli nsight-compute",
    )
    # Add local project files into /workspace
    .add_local_dir(
        ".",
        remote_path="/workspace",
        ignore=[".git", "build", "__pycache__", "*.pyc"],
    )
)

# ---------------------------------------------------------------------------
# Remote GPU Execution
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="B200:1",      # Target Blackwell GPU
    timeout=600,       # 10 min max run time
)
def run_make_target(target: str) -> dict:
    """Run a specific Makefile target in the remote container."""
    print(f"\n🚀 [modal] Executing: 'make {target}'")
    print(f"   Environment: nvidia/cuda:12.8.0-devel-ubuntu22.04")
    print(f"   GPU        : B200 (single)")
    
    # Run the make command in /workspace
    # subprocess.run handles output streaming back to the caller's terminal.
    result = subprocess.run(
        ["make", target],
        cwd="/workspace",
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"   ✅ Target '{target}' PASSED.")
    else:
        print(f"   ❌ Target '{target}' FAILED (exit code {result.returncode}).")
        
    return {
        "target": target,
        "rc": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr
    }

# ---------------------------------------------------------------------------
# Local Entrypoint — Coordinating the flow
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(target: str = "all"):
    """
    Main entry point for Modal runs.
    """
    targets = target.split(",")
    print(f"\n{'='*60}")
    print(f"  LLM System Kernels — Modal Cloud Runner")
    print(f"{'='*60}")
    
    failed = []
    for t in targets:
        t = t.strip()
        res = run_make_target.remote(t)
        
        # Print the remote stdout to local console
        if res["stdout"]:
            print(res["stdout"])
        if res["stderr"]:
            print(res["stderr"], file=sys.stderr)
            
        if res["rc"] != 0:
            failed.append(t)
        
        # If we ran benchmarks, populate results.text locally
        # Update local results.text if relevant
        if t in ["bench", "bench_moe", "bench_dsa", "all", "profile_moe", "profile_moe_diag"]:
            print(f"\n📝 Populating results.text locally...")
            with open("results.text", "a") as f:
                f.write(f"\n--- Result for 'make {t}' on Blackwell B200 ---\n")
                f.write(res["stdout"])
            print(f"   ✓ results.text updated.")
            
    print("\n" + "="*60)
    if failed:
        print(f"❌ FAILED targets: {failed}")
        sys.exit(1)
    else:
        print("✅ All targets completed successfully! 🎉")
