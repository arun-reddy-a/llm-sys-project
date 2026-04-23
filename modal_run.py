import modal
import os
import sys
import json
from datetime import datetime, timezone

RESULTS_FILE = "results.jsonl"

app = modal.App("llm-sys-dsa-kernels")

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
    .add_local_dir(
        ".",
        remote_path="/workspace",
        ignore=[".git", "build", "__pycache__", "*.pyc"],
    )
)

results_vol = modal.Volume.from_name("llm-sys-profiling-results", create_if_missing=True)


def log_result(target: str, result: dict):
    """Append a structured JSON Lines entry to results.jsonl."""
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
    gpu="B200:1",
    volumes={"/workspace/profiling/results": results_vol},
    timeout=600,
)
def run_make_target(target: str, run_id: str = "", variant: str = "") -> dict:
    """Run a specific Makefile target in the remote container."""
    print(f"\n  [modal] Executing: 'make {target}'")
    print(f"   Environment: nvidia/cuda:12.8.1-devel-ubuntu24.04")
    print(f"   GPU        : B200 (single)")
    if run_id: print(f"   Run ID     : {run_id}")
    if variant: print(f"   Variant    : {variant}")

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


@app.local_entrypoint()
def main(target: str = "bench_dsa_smoke", variant: str = ""):
    """
    Local entrypoint to run DSA benchmarks or profiling on cloud GPUs.

    Usage:
        modal run modal_run.py --target bench_dsa_smoke
        modal run modal_run.py --target bench_dsa_smoke --variant Opt4
        modal run modal_run.py --target profile_dsa_full --variant Opt4
        modal run modal_run.py --target test
        modal run modal_run.py --target bench_dsa_full
        modal run modal_run.py --target bench_dsa_smoke --variant Opt8
    """
    print("\n===========================================================")
    print("=                                                         =")
    print("=  LLM System DSA Kernels — Modal Cloud Runner            =")
    print("=                                                         =")
    print("===========================================================")

    targets = target.split(",")
    failed = []

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    for t in targets:
        t = t.strip()
        res = run_make_target.remote(t, run_id, variant)

        if res["returncode"] == 0:
            print(f"   Target '{t}' PASSED.")
        else:
            print(f"   Target '{t}' FAILED (exit code {res['returncode']}).")
            failed.append(t)

        log_result(t, res)
        print(f"   Logged to {RESULTS_FILE}")

    if not failed and any("profile" in t for t in targets):
        import subprocess
        print(f"\n   Fetching profiling results for RUN_ID: {run_id}...")
        local_dir = "profiling/local_results"
        os.makedirs(local_dir, exist_ok=True)

        v = variant if variant else "all"
        base_name = f"dsa_{v}_{run_id}"
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
                print(f"   Downloaded: {file}")
            except subprocess.CalledProcessError:
                pass

    if failed:
        print(f"\n==============================================")
        print(f"FAILED targets: {failed}")
        sys.exit(1)
    else:
        print(f"\n==============================================")
        print(f"All targets completed successfully!")
