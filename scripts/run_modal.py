#!/usr/bin/env python3
"""Run the competition kernel on a B200 via Modal and compare to the FlashInfer baseline."""
import modal

PYTHON = "python3"

image = (
    # CUDA 12.8 dev image — has verified cu128 wheels for torch/flashinfer
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "build-essential", "clang", "libnuma-dev")
    .pip_install("torch", extra_index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        "flashinfer-python",
        extra_index_url="https://flashinfer.ai/whl/cu128/torch2.7/",
    )
    .run_commands(
        # wheel must be present before --no-build-isolation so setuptools can find torch
        "pip install --quiet wheel && pip install --quiet --no-build-isolation git+https://github.com/deepseek-ai/DeepGEMM.git",
    )
    .add_local_dir(".", remote_path="/workspace",
                   ignore=[".git", "build", "__pycache__", "*.pyc"])
)

app = modal.App("flashinfer-moe-competition")


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
def main(warmup: int = 3, iters: int = 30, compare_baseline: bool = False):
    result = run_bench.remote(warmup, iters, compare_baseline)
    print(result)
