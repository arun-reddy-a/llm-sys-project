# Running LLM System Kernels on Modal Cloud GPUs

This directory documents how to use [Modal](https://modal.com) to build and run the CUDA kernels in this project on cloud GPUs.

## 🛠 Setup

1. **Install Modal**
   ```bash
   pip install modal
   ```

2. **Authenticate**
   ```bash
   modal setup
   ```

## 🚀 Running on Modal

The `modal_run.py` script in the project root acts as the entry point for cloud execution. It builds the container image with the necessary CUDA development environment and executes `make` targets on a remote GPU.

### Usage

Run the following command to build and run targets on a **Blackwell B200** GPU:

```bash
# Build and run everything
modal run modal_run.py --target all

# Run specific targets
modal run modal_run.py --target test
modal run modal_run.py --target bench
```

### 📦 Container Environment

The Modal container uses:
- **Base Image**: `nvidia/cuda:12.1.1-devel-ubuntu22.04` (provides `nvcc`, headers, and libraries).
- **GPU**: `B200:1` (Blackwell architecture).

## 💡 Key Modal Details

Based on MagicDec's implementation, several critical patterns are applied here:

1. **Serverless Orchestration**: You write Python to coordinate cloud execution. The `@app.function` decorator tells Modal to run that code on a GPU.
2. **Layer Caching**: The container image is built once and cached. Subsequent runs that don't change the environment start in seconds.
3. **Local Mounts**: `.add_local_dir(".")` ensures your local source code is available at `/workspace` in the container.
4. **Interactive Output**: `subprocess.run` output is streamed back to your terminal, providing a local-like development experience.
5. **Cost Efficiency**: Modal charges only for the seconds the GPU is active.

### Multi-GPU (Coming Soon)
For kernels requiring multiple GPUs (e.g., using `torchrun` and NCCL), use `gpu="A100-80GB:x"` where `x` is the number of GPUs. Ensure you initialize NCCL correctly by setting the device *before* calling `dist.init_process_group`.
