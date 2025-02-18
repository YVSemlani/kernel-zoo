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

    // define grid and block dimensions
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
	
	// run the vector addition kernel
    vector_add<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
	
	// copy device array back to the host
	// only C because A and B are the same
	
	cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);
    
    printf("Kernel execution complete!\n");

    return 0;
}