#!/usr/bin/env python3
"""Run the competition kernel on a B200 via Modal and compare to the FlashInfer baseline."""
import modal
import sys

PYTHON = "python3"

probe_image = modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:20260401-2c675fb")

image = (
    # Use CUDA 12.8 dev image — has verified cu128 wheels for torch/flashinfer
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "build-essential", "clang", "libnuma-dev")
    .pip_install("torch", extra_index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        "flashinfer-python",
        extra_index_url="https://flashinfer.ai/whl/cu128/torch2.7/",
    )
    .run_commands(
        # wheel must be present before --no-build-isolation so setuptools can build the package
        "pip install --quiet wheel && pip install --quiet --no-build-isolation git+https://github.com/deepseek-ai/DeepGEMM.git",
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
    # Probe deep_gemm 2.5 API with a small live test
    probe = subprocess.run(
        [PYTHON, "-c", """
import deep_gemm, torch, traceback
print('deep_gemm version:', getattr(deep_gemm,'__version__','?'))
dev = 'cuda'
# Problem dims matching competition
M, K, N, G = 64, 7168, 4096, 32
# Build contiguous grouped data: 2 tokens per expert, sorted
tokens_per_expert = 2
N_local = M
# per-row expert IDs: 0,0,1,1,...,31,31
exp_ids = torch.arange(G, device=dev).repeat_interleave(tokens_per_expert).to(torch.int32)

x   = torch.randn(N_local, K, device=dev).to(torch.float8_e4m3fn)
xs  = torch.ones(N_local, K//128, device=dev)   # [M, K//128]
w   = torch.randn(G, N, K, device=dev).to(torch.float8_e4m3fn)
ws  = torch.ones(G, N//128, K//128, device=dev) # [G, N//128, K//128]
out = torch.zeros(N_local, N, dtype=torch.bfloat16, device=dev)

fn = deep_gemm.m_grouped_fp8_gemm_nt_contiguous
print('trying m_grouped_fp8_gemm_nt_contiguous...')

# Try 1: old-style per-row m_indices
try:
    fn((x,xs),(w,ws),out,exp_ids); print('SUCCESS with per-row exp_ids')
except Exception as e: print('fail1:', e)

# Try 2: CSR grouped_layout [G+1] cumsum
try:
    # Each expert gets 2 tokens: [0,2,4,...,64]
    gl = torch.arange(0, G+1, device=dev, dtype=torch.int32)*tokens_per_expert
    fn((x,xs),(w,ws),out,gl); print('SUCCESS with CSR grouped_layout')
except Exception as e: print('fail2:', e)

# Try 3: m_grouped_fp8_gemm_nt_masked (per-row group IDs)
try:
    fn2 = deep_gemm.m_grouped_fp8_gemm_nt_masked
    fn2((x,xs),(w,ws),out,exp_ids,N_local); print('SUCCESS masked')
except Exception as e: print('fail3 masked:', e)
"""],
        capture_output=True, text=True,
    )
    print("=== deep_gemm probe ===")
    print(probe.stdout + probe.stderr)
    result = subprocess.run(
        [PYTHON, "scripts/run_local.py",
         f"--warmup={warmup}", f"--iters={iters}",
         *(["--compare-baseline"] if compare_baseline else [])],
        cwd="/workspace",
        capture_output=True, text=True,
    )
    out = result.stdout + result.stderr
    print(out)
    return probe.stdout + probe.stderr + "\n" + out


@app.local_entrypoint()
def main(warmup: int = 3, iters: int = 30, compare_baseline: bool = False, probe: bool = False):
    if probe:
        print(probe_env.remote())
        return
    result = run_bench.remote(warmup, iters, compare_baseline)
    print(result)
