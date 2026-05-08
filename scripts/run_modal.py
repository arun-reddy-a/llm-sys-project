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


@app.function(image=image, gpu="B200:1", timeout=120)
def probe_triton() -> str:
    import subprocess
    r = subprocess.run([
        "python3", "-c",
        "import triton; print('triton', triton.__version__); "
        "import torch, triton, triton.language as tl\n"
        "@triton.jit\n"
        "def _t(p, o): x = tl.load(p + tl.arange(0,16)); tl.store(o + tl.arange(0,16), x)\n"
        "a=torch.zeros(16,dtype=torch.float8_e4m3fn,device='cuda'); b=torch.empty_like(a)\n"
        "_t[1,](a,b); print('fp8 load/store: OK')\n"
        "@triton.jit\n"
        "def _d(p,q,o):\n"
        "  a=tl.load(p+tl.arange(0,16)[:,None]*16+tl.arange(0,16)[None,:])\n"
        "  b=tl.load(q+tl.arange(0,16)[:,None]*16+tl.arange(0,16)[None,:])\n"
        "  c=tl.dot(a,b,out_dtype=tl.float32); tl.store(o+tl.arange(0,16)[:,None]*16+tl.arange(0,16)[None,:],c)\n"
        "a8=torch.ones(16,16,dtype=torch.float8_e4m3fn,device='cuda')\n"
        "c=torch.empty(16,16,dtype=torch.float32,device='cuda')\n"
        "_d[1,](a8,a8,c); print('fp8 tl.dot: OK, result[0,0]=',c[0,0].item())"
    ], capture_output=True, text=True)
    return r.stdout + r.stderr


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


@app.function(image=image, gpu="B200:1", timeout=300)
def run_correctness() -> str:
    import subprocess
    result = subprocess.run(
        [PYTHON, "scripts/check_correctness.py"],
        cwd="/workspace",
        capture_output=True, text=True,
    )
    out = result.stdout + result.stderr
    print(out)
    return out


@app.local_entrypoint()
def main(warmup: int = 3, iters: int = 30, compare_baseline: bool = False,
         probe_tri: bool = False, check_correctness: bool = False):
    if probe_tri:
        print(probe_triton.remote())
        return
    if check_correctness:
        print(run_correctness.remote())
        return
    result = run_bench.remote(warmup, iters, compare_baseline)
    print(result)
