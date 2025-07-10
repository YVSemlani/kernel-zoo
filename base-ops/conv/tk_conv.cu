#include "kittens.cuh"
using namespace kittens;

#define NUM_WARPS 8  // multiple warps per block
#define NUM_THREADS (NUM_WARPS * kittens::WARP_THREADS)

// conv parameters

#define BATCH_DIM 1
#define IN_CHANNELS 3
#define INPUT_HEIGHT 256
#define INPUT_WIDTH 256
#define NUM_FILTERS 8
#define KERNEL_HEIGHT 3
#define KERNEL_WIDTH 3
#define STRIDE 1
#define PADDING 0
#define OUTPUT_HEIGHT ((INPUT_HEIGHT + 2 * PADDING - KERNEL_HEIGHT) / STRIDE + 1)
#define OUTPUT_WIDTH ((INPUT_WIDTH + 2 * PADDING - KERNEL_WIDTH) / STRIDE + 1)

/* 

we're going to be using an im2col based approach to do the convolution
however, we're going to do this with asynchronous full tensor parallelism 
this is primarily useful so that loads don't block the tensor cores from running

my initial assumption is that we'll load in large tiles of the input matrix (probably the size of at least one row of the output matrix)
we'll then async convert each kernel tile in the input matrix into a column in the im2col matrix
we can then do matrix multiplication at our desired size?


kernel shape is initially num_filter x in_channels x kernel_height x kernel_width
but the seperated dimensions make it so that we can't load in 16x16 tiles of the kernel matrix
so we combine the input chanenls and kernel dimensions into a single dimension and keep the number of filter (assuming larger than 16)

*/

struct conv_globals {
    using input_gl = gl<float, -1, -1, -1, -1, st_fl<48, 48>>; // a 80x80 tile of the input matrix gives us a 1/3 of a 16x16 tile of the output matrix given a 5x5 kernel
    using kernel_gl = gl<float, -1, -1, -1, -1, st_fl<IN_CHANNELS, KERNEL_HEIGHT * KERNEL_WIDTH>>; // loading all input channels of the kernel into a single tile so we can update the output tile in a single warp
    using output_gl = gl<float, -1, -1, -1, -1, st_fl<16, 16>>; // output matrix is done in tiles of 16
    
    input_gl input;
    kernel_gl kernel;
    output_gl output;
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void tk_conv(const __grid_constant__ conv_globals g) {

    // setup shared memory
    extern __shared__ alignment_dummy __shm[]; // allocates a dynamic amount of shared memory
    shared_allocator al((int*)&__shm[0]); // create a shared memory allocator and point it to the starting address of the shared memory
    st_fl<80, 80> (&input_s) = al.allocate<st_fl<80, 80>>(); // allocate memory for an 80x80 tile of the input matrix
    st_fl<16, IN_CHANNELS * KERNEL_HEIGHT * KERNEL_WIDTH> (&kernel_s) = al.allocate<st_fl<16, IN_CHANNELS * KERNEL_HEIGHT * KERNEL_WIDTH>>(); // load in 16 filters at a time
    st_fl<16, 16> (&output_s) = al.allocate<st_fl<16, 16>>(); // allocate memory for a 16x16 tile of the output matrix

    // load from HBM to shared
    load(input_s, g.input, {0, 0, 0, 0});
    load(kernel_s, g.kernel, {0, 0, 0, 0});
    __syncthreads();


}