// reimplement addvec kernel in thunderkittens

#include "kittens.cuh"
using namespace kittens;
#define NUM_THREADS (kittens::WARP_THREADS) // use 1 warp

#define N 10000000 // vector size 10M

// define tile sizes
#define _row 16
#define _col 16

// vector addition of two 1D vectors can be made to be 2D by adding an arbitrary extra dimension
// this makes it easier for us to use optimal 16x16 tile sizes

struct important_layouts {
    using _gl = gl<float, -1, -1, -1, -1, st_fl<16, 16>>; // global layout for A, B, C
    _gl a, b, c;
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void tk_addvec(const __grid_constant__ important_layouts g) {

    // shared memory
    extern __shared__ alignment_dummy __shm[]; // allocates a dynamic amount of shared memory
    shared_allocator al((int*)&__shm[0]); // create a shared memory allocator and point it to the starting address of the shared memory
    st_fl<16, 16> (&a_s) = al.allocate<st_fl<16, 16>>(); // allocate memory for a 16x16 tile of A
    st_fl<16, 16> (&b_s) = al.allocate<st_fl<16, 16>>(); // allocate memory for a 16x16 tile of B
    st_fl<16, 16> (&c_s) = al.allocate<st_fl<16, 16>>(); // allocate memory for a 16x16 tile of C

    // Calculate tile coordinates from block indices
    int tile_col = blockIdx.x;
    int tile_row = blockIdx.y;

    // load from HBM to shared
    load(a_s, g.a, {tile_row, tile_col, 0, 0});
    load(b_s, g.b, {tile_row, tile_col, 0, 0});
    __syncthreads();

    // allocate place in register memory for A and B
    rt_fl<16, 16> a_reg_fl;
    rt_fl<16, 16> b_reg_fl;
    __syncthreads();

    // load from shared to register
    load(a_reg_fl, a_s);
    load(b_reg_fl, b_s);
    __syncthreads();

    // add A and B
    add(a_reg_fl, a_reg_fl, b_reg_fl);
    __syncthreads();

    // store from register to shared
    store(c_s, a_reg_fl);
    __syncthreads();

    // store from shared to HBM
    store(g.c, c_s, {tile_row, tile_col, 0, 0});
    __syncthreads();
}

void dispatch_micro( float *d_a, float *d_b, float *d_c ) {
    using _gl = gl<float, -1, -1, -1, -1, st_fl<_row, _col>>;
    using globals = important_layouts;

    // For 1D vector: calculate how many tiles fit along one dimension
    unsigned long total_elements = N;
    unsigned long elements_per_tile = _row * _col;  // 256
    unsigned long total_tiles = (total_elements + elements_per_tile - 1) / elements_per_tile;  // ~39,063
    
    // Arrange as a "strip" - 1 row, many columns
    unsigned long num_rows = 1;
    unsigned long num_cols = total_tiles;

    // create the global layouts
    _gl  a_arg{d_a, num_rows, num_cols, _row, _col};
    _gl  b_arg{d_b, num_rows, num_cols, _row, _col};  
    _gl  c_arg{d_c, num_rows, num_cols, _row, _col};

    globals g{a_arg, b_arg, c_arg};

    unsigned long mem_size = 50480; 
    cudaFuncSetAttribute(tk_addvec, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    // Launch 1D grid with correct number of tiles
    tk_addvec<<<total_tiles, 32, mem_size>>>(g);
    cudaDeviceSynchronize();
}
#include "harness.impl"