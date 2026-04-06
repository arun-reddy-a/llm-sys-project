#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// Extremely simple vector addition kernel for pedagogical profiling
__global__ void simple_vadd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char** argv) {
    int n = 1 << 26; // 64M elements (~768MB total for a+b+c)
    if (argc > 1) n = 1 << atoi(argv[1]);

    size_t size = n * sizeof(float);
    
    // Host allocation
    std::vector<float> h_a(n, 1.0f);
    std::vector<float> h_b(n, 2.0f);
    std::vector<float> h_c(n, 0.0f);

    // Device allocation
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy to device
    cudaMemcpy(d_a, h_a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), size, cudaMemcpyHostToDevice);

    // Launch configuration
    int threads_per_block = 256;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    std::cout << "Launching simple_vadd with N=" << n << ", blocks=" << blocks_per_grid << std::endl;

    // Run kernel
    simple_vadd<<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    // Copy back
    cudaMemcpy(h_c.data(), d_c, size, cudaMemcpyDeviceToHost);

    // Verify
    bool success = true;
    for (int i = 0; i < 10; ++i) {
        if (h_c[i] != 3.0f) {
            success = false;
            break;
        }
    }

    if (success) std::cout << "SUCCESS!" << std::endl;
    else std::cout << "FAILURE!" << std::endl;

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
