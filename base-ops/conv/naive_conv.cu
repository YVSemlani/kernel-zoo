#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// naive implementation of a 2D convolution kernel in CUDA for understanding

// VARIABLES
// input tensor - shape (batch, in_channels, height, width)
// kernel/weight tensor - shape (num_filters, in_channels, kernel_height, kernel_width) - num_filters = out_channels
// stride, padding, kernel shape, input shape, output shape
// we can calculate the output shape in the kernel but elect to pass it as an input parameter so processing can be done on the host
// output comes out as (batch, num_filters, output_height, output_width)

// APPROACH
// each thread attends to a single element of the output tensor
// computes dot product of kernel and relevant portion of the input
// insert to relevant element of the output tensor

// works up to 1024 filters and no other constraints on shape


#define BATCH_DIM 1
#define IN_CHANNELS 3
#define INPUT_HEIGHT 224
#define INPUT_WIDTH 224
#define NUM_FILTERS 64
#define KERNEL_HEIGHT 3
#define KERNEL_WIDTH 3
#define STRIDE 1
#define PADDING 1
#define OUTPUT_HEIGHT ((INPUT_HEIGHT + 2 * PADDING - KERNEL_HEIGHT) / STRIDE + 1)
#define OUTPUT_WIDTH ((INPUT_WIDTH + 2 * PADDING - KERNEL_WIDTH) / STRIDE + 1)

__global__ void naive_conv(
        float *input,           // input tensor (batch, in_channels, height, width)
        float *kernel,          // kernel tensor (num_filters, in_channels, kernel_height, kernel_width)
        float *output,          // output tensor (batch, num_filters, output_height, output_width)
                                
        // input shape
        int batch_dim,              
        int in_channels,
        int input_height,
        int input_width,

        // kernel shape
        int num_filters,        // number of conv filters == number of output channels
        int kernel_height,
        int kernel_width,

        // convolution params
        int stride,
        int padding,

        // output shape
        int output_height,
        int output_width

) {

    // 3D block grid, 2D thread grid
    // get block and thread indices
    int bx = blockIdx.x; // output width
    int by = blockIdx.y; // output height
    int bz = blockIdx.z; // batch index

    int tx = threadIdx.x; // filter

    // make sure we're within bounds of the input, kernel, and output tensors
    if (bz >= batch_dim || by >= output_height || bx >= output_width || tx >= num_filters) {
        return;
    }

    // use the output x and y indices to get the input x and y indices
    int input_x = bx * stride - padding;
    int input_y = by * stride - padding;

    int input_idx = bz * (in_channels * input_height * input_width); // we'll dynamically update this to hit the correct input channel

    // handle filter sizes over 1024 creating a for loop for the output index
    for (int filter_idx = tx; filter_idx < num_filters; filter_idx += 1024) {
        // iterate over the input tensor until we've exhausted the kernel elements and input channels
        float dot_product = 0.0f;

        // get our output index
        int output_idx = bz * (num_filters * output_height * output_width) + filter_idx * (output_height * output_width) + by * (output_width) + bx;

        // get our starting index for the kernel tensor
        int kernel_idx = filter_idx * (in_channels * kernel_height * kernel_width); // always start with the first input channel and top-left of the kernel

        // C, H, and W are relative to the starting index of the input tensor
        for (int C = 0; C < in_channels; C++) {
            for (int H = 0; H < kernel_height; H++) {
                for (int W = 0; W < kernel_width; W++) {
                    // check if we're at a padding element 
                    // this is done by taking the starting x and y and seeing if we're at a negative index (left side padding of the tensor) or if we're at an index greater than the input width or height (right side padding of the tensor)
                    if (input_x + W >= input_width || input_y + H >= input_height || input_x + W < 0 || input_y + H < 0) {
                        continue; // don't add anything to the output element b/c +0 is the same as not adding anything
                    }

                    // add the product of the input and kernel element directly to the output element
                    dot_product += input[input_idx + C * (input_height * input_width) + (input_y + H) * (input_width) + (input_x + W)] * kernel[kernel_idx + C * (kernel_height * kernel_width) + H * (kernel_width) + W];
                }
            }
        }

        // set the output element to the dot product
        output[output_idx] = dot_product;
    }
}


// main method to run the kernel
int main() {

    size_t size_input = BATCH_DIM * IN_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH * sizeof(float);
    size_t size_kernel = NUM_FILTERS * IN_CHANNELS * KERNEL_HEIGHT * KERNEL_WIDTH * sizeof(float);
    size_t size_output = BATCH_DIM * NUM_FILTERS * OUTPUT_HEIGHT * OUTPUT_WIDTH * sizeof(float);

    // set up host arrays
    float *h_input = (float*)malloc(size_input);
    float *h_kernel = (float*)malloc(size_kernel);
    float *h_output = (float*)malloc(size_output);

    
    // Initialize host arrays with random values
    // modify this when comparing against other implementations so you use the same weights and inputs
    for (int i = 0; i < BATCH_DIM * IN_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH; i++) {
        h_input[i] = rand() / (float)RAND_MAX;
    }

    // Initialize kernel
    for (int i = 0; i < NUM_FILTERS * IN_CHANNELS * KERNEL_HEIGHT * KERNEL_WIDTH; i++) {
        h_kernel[i] = rand() / (float)RAND_MAX;
    }

    // set up timing variables
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.0f;

    // start timing
    cudaEventRecord(start);

    // allocate device memory for the input, kernel, and output tensors
    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, size_input);
    cudaMalloc(&d_kernel, size_kernel);
    cudaMalloc(&d_output, size_output);

    // stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> GPU allocation time: %f ms\n", ms);

    // start timing
    cudaEventRecord(start);

    // copy host arrays to device
    cudaMemcpy(d_input, h_input, size_input, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, size_kernel, cudaMemcpyHostToDevice);

    // stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Host to device transfer time: %f ms\n", ms);

    // define grid and block dimensions
    dim3 gridDim(OUTPUT_WIDTH, OUTPUT_HEIGHT, BATCH_DIM);

    // set the block size to 1024 if the number of filters is greater than 1024, otherwise set it to the number of filters
    int BLOCK_SIZE;
    if (NUM_FILTERS > 1024) {
        BLOCK_SIZE = 1024;
    } else {
        BLOCK_SIZE = NUM_FILTERS;
    }

    // start timing
    cudaEventRecord(start);

    // warm up
    for (int i = 0; i < 10; i++) {
        naive_conv<<<gridDim, BLOCK_SIZE>>>(d_input, d_kernel, d_output, 
            BATCH_DIM, IN_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH, 
            NUM_FILTERS, KERNEL_HEIGHT, KERNEL_WIDTH, 
            STRIDE, PADDING, 
            OUTPUT_HEIGHT, OUTPUT_WIDTH);
    }

    // stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Warm up time: %f ms\n", ms);

    // start timing
    cudaEventRecord(start);
	
	// run the vector addition kernel
    naive_conv<<<gridDim, BLOCK_SIZE>>>(d_input, d_kernel, d_output, 
        BATCH_DIM, IN_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH, 
        NUM_FILTERS, KERNEL_HEIGHT, KERNEL_WIDTH, 
        STRIDE, PADDING, 
        OUTPUT_HEIGHT, OUTPUT_WIDTH);

    // stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Kernel execution time: %f ms\n", ms);

    // start timing
    cudaEventRecord(start);
	
	// copy device array back to the host
	// only C because A and B are the same
	
	cudaMemcpy(h_output, d_output, size_output, cudaMemcpyDeviceToHost);

    // stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Device to host transfer time: %f ms\n", ms);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

    // Free host memory
    free(h_input);
    free(h_kernel);
    free(h_output);
    
    printf("Kernel execution complete!\n");

    return 0;
}