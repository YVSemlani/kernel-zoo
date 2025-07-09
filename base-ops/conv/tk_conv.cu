#include "kittens.cuh"
using namespace kittens;

#define NUM_WARPS 8  // multiple warps per block
#define NUM_THREADS (NUM_WARPS * kittens::WARP_THREADS)

// conv parameters

#define BATCH_DIM 1
#define IN_CHANNELS 3
#define INPUT_HEIGHT 1024
#define INPUT_WIDTH 1024
#define NUM_FILTERS 64
#define KERNEL_HEIGHT 3
#define KERNEL_WIDTH 3
#define STRIDE 1
#define PADDING 1
#define OUTPUT_HEIGHT ((INPUT_HEIGHT + 2 * PADDING - KERNEL_HEIGHT) / STRIDE + 1)
#define OUTPUT_WIDTH ((INPUT_WIDTH + 2 * PADDING - KERNEL_WIDTH) / STRIDE + 1)

// tile sizes + useful constants

#define OUTPUT_TILE_HEIGHT 16
#define OUTPUT_TILE_WIDTH 16
#define INPUT_TILE_HEIGHT (OUTPUT_TILE_HEIGHT * STRIDE + KERNEL_HEIGHT - 1)
#define INPUT_TILE_WIDTH (OUTPUT_TILE_WIDTH * STRIDE + KERNEL_WIDTH - 1)
#define IM2COL_HEIGHT (KERNEL_HEIGHT * KERNEL_WIDTH)
#define IM2COL_WIDTH ((INPUT_TILE_HEIGHT / KERNEL_HEIGHT) * (INPUT_TILE_WIDTH / KERNEL_WIDTH))

// number of tiles in the output
#define TILE_ROWS (OUTPUT_HEIGHT / OUTPUT_TILE_HEIGHT)
#define TILE_COLS (OUTPUT_WIDTH / OUTPUT_TILE_WIDTH)
#define TOTAL_TILES (TILE_ROWS * TILE_COLS)

// global memory layouts

struct conv_globals {
    using input_gl = gl<float, -1, -1, -1, -1, st_fl<INPUT_TILE_HEIGHT, INPUT_TILE_WIDTH>>;
    using kernel_gl = gl<float, -1, -1, -1, -1, st_fl<IN_CHANNELS, KERNEL_HEIGHT * KERNEL_WIDTH>>; // We load all input channels of the kernel into a single tile so we can update the output tile in a single warp
    using output_gl = gl<float, -1, -1, -1, -1, st_fl<OUTPUT_TILE_HEIGHT, OUTPUT_TILE_WIDTH>>;
    
    input_gl input;
    kernel_gl kernel;
    output_gl output;
};

// have each warp handle a tile of the input
// mapping from input to output:
// input bounds = output bounds * stride +- padding

__global__ __launch_bounds__(NUM_THREADS, 1)
void tk_conv(const __grid_constant__ conv_globals g) {

    // each warp handles a tile of output
    // filter index, batch index, and spatial coordinates in the output tile are all fixed inside the warp
    
    int warpid = kittens::warpid();
    int spatial_tile_id = blockIdx.z * NUM_WARPS + warpid;  // each block handles NUM_WARPS tiles
    
    // bounds check - some warps may not have work if total tiles don't divide evenly
    if (spatial_tile_id >= TOTAL_TILES) {
        return;
    }
    
    int output_row = spatial_tile_id / TILE_COLS;
    int output_col = spatial_tile_id % TILE_COLS;

    int bx = blockIdx.x; // batch index
    int by = blockIdx.y; // filter index

    // setup shared memory
    extern __shared__ alignment_dummy __shm[]; // allocates a dynamic amount of shared memory
    shared_allocator al((int*)&__shm[0]); // create a shared memory allocator and point it to the starting address of the shared memory

    // allocate shared memory for input, kernel, output
    st_fl<INPUT_TILE_HEIGHT, INPUT_TILE_WIDTH> (&input_s) = al.allocate<st_fl<INPUT_TILE_HEIGHT, INPUT_TILE_WIDTH>>();
    st_fl<IN_CHANNELS, KERNEL_HEIGHT * KERNEL_WIDTH> (&kernel_s) = al.allocate<st_fl<IN_CHANNELS, KERNEL_HEIGHT * KERNEL_WIDTH>>();
    st_fl<OUTPUT_TILE_HEIGHT, OUTPUT_TILE_WIDTH> (&output_s) = al.allocate<st_fl<OUTPUT_TILE_HEIGHT, OUTPUT_TILE_WIDTH>>();

    // identify the starting x and y coordinates of the input tile
    // will be negative when we're in a padding region
    int input_start_x = output_col * OUTPUT_TILE_WIDTH * STRIDE - PADDING;
    int input_start_y = output_row * OUTPUT_TILE_HEIGHT * STRIDE - PADDING;

    // load the kernel into shared memory
    // the kernel stays constant across all input channels
    // it's a 2D array with dimensions [IN_CHANNELS, KERNEL_HEIGHT * KERNEL_WIDTH]
    load(kernel_s, g.kernel, {by, 0, 0, 0});

    // setup accumulator register
    rt_fl<OUTPUT_TILE_HEIGHT, OUTPUT_TILE_WIDTH> output_reg;
    zero(output_reg);

    // loop over the input channels
    for (int c = 0; c < IN_CHANNELS; c++) {
        // zero the input tile in shared memory so we don't have to actively mask out the padding region
        zero(input_s);

        /*
                PADDING

                We calculate the valid region and mask out the padding region prior to running the convolution operation
        */
        
        // Calculate the overlap between desired region and actual input bounds
        int global_start_row = max(0, input_start_y);
        int global_end_row = min(input_start_y + INPUT_TILE_HEIGHT, INPUT_HEIGHT);
        int global_start_col = max(0, input_start_x);
        int global_end_col = min(input_start_x + INPUT_TILE_WIDTH, INPUT_WIDTH);
        
        // Check if there's any valid region to load
        if (global_end_row > global_start_row && global_end_col > global_start_col) {
            // Calculate where this data should go in the local tile
            int local_start_row = global_start_row - input_start_y;
            int local_start_col = global_start_col - input_start_x;
            
            // Calculate dimensions of the region to load
            int load_height = global_end_row - global_start_row;
            int load_width = global_end_col - global_start_col;
            
            // Create a subtile view of the destination
            auto valid_region = subtile_inplace<load_height, load_width>(
                input_s, {local_start_row, local_start_col}
            );
            
            // Load into the correct position
            load(valid_region, g.input, {bx, c, global_start_row, global_start_col});
        }

        // load the kernel for this input channel into a register
        rt_fl<KERNEL_HEIGHT *KERNEL_WIDTH> kernel_reg;
        load(kernel_reg, kernel_s[c]);
        /*
            CONVOLUTION

            we're going to reformat the input tile into the im2col format so we can perform the convolution operation across the entire tile
            this also makes it easier to update the output accumulator register
        */

        // setup im2col matrix in shared memory    
        st_fl<IM2COL_HEIGHT, IM2COL_WIDTH> (&im2col_s) = al.allocate<st_fl<IM2COL_HEIGHT, IM2COL_WIDTH>>();
        zero(im2col_s);

        // reformat the input tile into the im2col format
        for (int ih = 0; ih < INPUT_TILE_HEIGHT - KERNEL_HEIGHT + 1; ih++) {
            for (int iw = 0; iw < INPUT_TILE_WIDTH - KERNEL_WIDTH + 1; iw++) {
                // get the kernel sized subtile of the input tile which starts at the coordinates ih, iw
                auto input_subtile = subtile_inplace<KERNEL_HEIGHT, KERNEL_WIDTH>(input_s, {ih, iw});

                // flatten and load the subtile into im2col matrix
                for (int kh = 0; kh < KERNEL_HEIGHT; kh++) {
                    for (int kw = 0; kw < KERNEL_WIDTH; kw++) {
                        im2col_s[kh * KERNEL_HEIGHT + kw][ih * (INPUT_TILE_WIDTH - KERNEL_WIDTH + 1)+ iw] = input_subtile[kh][kw];
                    }
                }
            }
        }

        // load the im2col matrix into a register
        rt_fl<IM2COL_HEIGHT, IM2COL_WIDTH> im2col_reg;
        load(im2col_reg, im2col_s);

        // matrix multiply the im2col matrix with the kernel register
        // kernel_reg = kernel_reg * im2col_reg
        // kernel_reg is the same shape as our output
        mma_AB(kernel_reg, kernel_reg, im2col_reg);

        // add the result to the output register
        // unsure if we need to do reshape here or how to do it..?
        add(output_reg, output_reg, kernel_reg);
    }

    // store the output register into shared memory
    store(output_s, output_reg);

    // store the output tile into global memory
    store(g.output, output_s, {bx, by, output_row, output_col});
}