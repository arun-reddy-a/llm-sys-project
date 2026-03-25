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
        CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
    }
    ~DeviceBuf() { if (ptr) cudaFree(ptr); }

    void alloc(int n) {
        if (ptr) cudaFree(ptr);
        count = n;
        CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
    }
    void upload(const T* host) {
        CUDA_CHECK(cudaMemcpy(ptr, host, count * sizeof(T), cudaMemcpyHostToDevice));
    }
    void download(T* host) const {
        CUDA_CHECK(cudaMemcpy(host, ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
    }
    void zero() {
        CUDA_CHECK(cudaMemset(ptr, 0, count * sizeof(T)));
    }

    DeviceBuf(const DeviceBuf&) = delete;
    DeviceBuf& operator=(const DeviceBuf&) = delete;
};
