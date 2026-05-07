#!/usr/bin/env python3
"""Run the competition kernel on a B200 via Modal and compare to the FlashInfer baseline."""
import modal
import sys

probe_image = modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:20260401-2c675fb")

image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:20260401-2c675fb")
    .run_commands(
        # deep_gemm is listed in EVALUATION.md but may need to be built from source
        "pip show deep-gemm 2>/dev/null || pip install --quiet git+https://github.com/deepseek-ai/DeepGEMM.git || true",
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
def run_bench(warmup: int = 3, iters: int = 30, compare_baseline: bool = True) -> str:
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "scripts/run_local.py",
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
