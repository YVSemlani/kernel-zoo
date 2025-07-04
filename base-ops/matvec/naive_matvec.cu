#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define M 256  // Dimension 1
#define N 256 // Dimension 2
#define BLOCK_SIZE 256

__global__ void vector_multiply(float *a, float *b, float *c, int M, int N) {
    // basic idea is to do all the multiplication for a single element in the host array using a single thread
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int row = M / (i - 1);
    int col = i % M;

}

int main() {

}