#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N 10000000  // Vector size = 10 million
#define BLOCK_SIZE 256

__global__ void vector_add(float *a, float *b, float *c, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void vector_add_cpu(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    size_t size = N * sizeof(float);

    // set up host arrays
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    
    // Initialize host arrays with random values
    for (int i = 0; i < N; i++) {
        h_a[i] = rand() / (float)RAND_MAX;  // Random values between 0 and 1
        h_b[i] = rand() / (float)RAND_MAX;
    }

    // allocate device memory for the a and b arrays
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // copy host arrays to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // set up cuda event for timing purposes
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.0f;

    // run 10 times 
    float runs = 10.0;
    float cpu_ms = 0.0;
    float gpu_ms = 0.0;
    for (int i = 0; i < (int) runs; i++) {
        // time the naive cpu kernel
        cudaEventRecord(start);
        vector_add_cpu(h_a, h_b, h_c, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        printf(">> Naive CPU Kernel Run %d Time: %f ms\n", i + 1, ms);
        cpu_ms += ms;

        // time the naive gpu kernel
        cudaEventRecord(start);
        vector_add<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        printf(">> Naive GPU Kernel Run %d Time: %f ms\n", i + 1, ms);
        gpu_ms += ms;
    }

    printf("----------------------------------------\n");
    printf(">> Average CPU Kernel Runtime %f ms\n",cpu_ms / runs);
    printf(">> Average GPU Kernel Runtime %f ms\n", gpu_ms / runs);

    // Add cleanup code at the end of main()
    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}