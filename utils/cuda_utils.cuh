#pragma once

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Error checking
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                    \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// GPU timer using CUDA events
// ---------------------------------------------------------------------------

struct GpuTimer {
    cudaEvent_t start, stop;

    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }
    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void begin(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(start, stream));
    }
    void end(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(stop, stream));
    }
    float elapsed_ms() {
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

// ---------------------------------------------------------------------------
// Host-side random initialisation helpers
// ---------------------------------------------------------------------------

inline void random_fill(float* data, int n, float lo = -1.0f, float hi = 1.0f) {
    for (int i = 0; i < n; i++) {
        data[i] = lo + static_cast<float>(rand()) / RAND_MAX * (hi - lo);
    }
}

inline void zero_fill(float* data, int n) {
    for (int i = 0; i < n; i++) data[i] = 0.0f;
}

// ---------------------------------------------------------------------------
// Comparison helpers (max / mean absolute error)
// ---------------------------------------------------------------------------

inline float max_abs_error(const float* a, const float* b, int n) {
    float mx = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

inline float mean_abs_error(const float* a, const float* b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += fabsf(a[i] - b[i]);
    }
    return static_cast<float>(sum / n);
}

// ---------------------------------------------------------------------------
// Device memory RAII wrapper (simple)
// ---------------------------------------------------------------------------

template <typename T>
struct DeviceBuf {
    T* ptr = nullptr;
    int count = 0;

    DeviceBuf() = default;
    explicit DeviceBuf(int n) : count(n) {
        CUDA_CHECK(cudaMalloc(&ptr, (size_t)n * sizeof(T)));
    }
    ~DeviceBuf() { if (ptr) cudaFree(ptr); }

    void resize(int n) {
        if (n > count) {
            if (ptr) CUDA_CHECK(cudaFree(ptr));
            count = n;
            CUDA_CHECK(cudaMalloc(&ptr, (size_t)n * sizeof(T)));
        }
    }

    void upload(const T* host, cudaStream_t stream = 0) {
        CUDA_CHECK(cudaMemcpyAsync(ptr, host, (size_t)count * sizeof(T), cudaMemcpyHostToDevice, stream));
    }
    void download(T* host, cudaStream_t stream = 0) const {
        CUDA_CHECK(cudaMemcpyAsync(host, ptr, (size_t)count * sizeof(T), cudaMemcpyDeviceToHost, stream));
    }
    void zero(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaMemsetAsync(ptr, 0, (size_t)count * sizeof(T), stream));
    }

    DeviceBuf(const DeviceBuf&) = delete;
    DeviceBuf& operator=(const DeviceBuf&) = delete;
};
