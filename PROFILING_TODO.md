# Profiling Pipeline TODOs

- [ ] **Nsight Systems Optimization**: Add `--capture-range=nvtx` and `--capture-range-end=stop` to `nsys profile` inside `profile_moe.sh`. This will prevent `nsys` from tracing the Python/binary setup and memory allocations during warmup, drastically shrinking the `.sqlite` file size and removing `cudaMalloc` noise from the API summary.
