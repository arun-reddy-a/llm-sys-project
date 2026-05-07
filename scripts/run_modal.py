#!/usr/bin/env python3
"""Run the competition kernel on a B200 via Modal and compare to the FlashInfer baseline."""
import modal
import sys

PYTHON = "/opt/conda/envs/py312/bin/python3"
PIP    = f"{PYTHON} -m pip install --quiet"

probe_image = modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:20260401-2c675fb")

image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:20260401-2c675fb")
    .run_commands(
        # 1. PyTorch (must come first — deep_gemm build depends on it)
        f"{PIP} torch==2.7.0 --index-url https://download.pytorch.org/whl/cu132",
        # 2. FlashInfer wheel for CUDA 13.2 / Torch 2.7
        f"{PIP} flashinfer-python --find-links https://flashinfer.ai/whl/cu132/torch2.7/",
        # 3. DeepGEMM — build from source (needs torch in path)
        f"{PYTHON} -m pip install --quiet git+https://github.com/deepseek-ai/DeepGEMM.git",
        # 4. Triton (bundled with torch but pin to match)
        f"{PIP} triton",
    )
    .add_local_dir(".", remote_path="/workspace",
                   ignore=[".git", "build", "__pycache__", "*.pyc"])
)

app = modal.App("flashinfer-moe-competition")


@app.function(image=probe_image, gpu="B200:1", timeout=120)
def probe_env() -> str:
    import subprocess
    cmds = [
        "find /opt /usr /root -name 'python3*' -type f 2>/dev/null | head -10",
        "find /opt /usr /root -name 'deep_gemm*' 2>/dev/null | head -10",
        "find /opt /usr /root -path '*/site-packages/flashinfer*' -maxdepth 8 2>/dev/null | head -5",
        "conda run -n py312 python3 -c 'import flashinfer, deep_gemm; print(flashinfer.__file__, deep_gemm.__file__)' 2>&1 || true",
        "conda run -n base  python3 -c 'import flashinfer; print(flashinfer.__file__)' 2>&1 || true",
        # Try all pythons in /opt/conda
        "for py in /opt/conda/envs/*/bin/python3 /opt/conda/bin/python3; do echo \"==$py==\"; $py -c 'import flashinfer; print(flashinfer.__file__)' 2>&1 || true; done",
        "ls /opt/conda/envs/ 2>/dev/null",
    ]
    out = []
    for c in cmds:
        r = subprocess.run(c, shell=True, capture_output=True, text=True, executable="/bin/bash")
        out.append(f"$ {c}\n{r.stdout}{r.stderr}\n")
    result = "\n".join(out)
    print(result)
    return result


@app.function(image=image, gpu="B200:1", timeout=600)
def run_bench(warmup: int = 3, iters: int = 30, compare_baseline: bool = False) -> str:
    import subprocess
    result = subprocess.run(
        [PYTHON, "scripts/run_local.py",
         f"--warmup={warmup}", f"--iters={iters}",
         *(["--compare-baseline"] if compare_baseline else [])],
        cwd="/workspace",
        capture_output=True, text=True,
    )
    out = result.stdout + result.stderr
    print(out)
    return out


@app.local_entrypoint()
def main(warmup: int = 3, iters: int = 30, compare_baseline: bool = False, probe: bool = False):
    if probe:
        print(probe_env.remote())
        return
    result = run_bench.remote(warmup, iters, compare_baseline)
    print(result)
