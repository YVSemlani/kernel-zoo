/*
    basic attention kernel in ThunderKittens for practice
    ignoring softmax for now as well as the scaling factor

    relevant dimensions:
        - batch 1
        - seqlen 128
        - head dim 256

    assuming Q, K, and V are the inputs to the attention kernel, we have:
        - Q: 1x128x256
        - K: 1x128x256
        - V: 1x128x256

    and treat the 4th dim as arbitary since we're not doing MHA
    
    we get one row of Q, a column of K^T, and a column of V
    
*/

#include "kittens.cuh"
using namespace kittens;
#define NUM_WARPS 16  // 16 warps per block
#define NUM_THREADS (NUM_WARPS * kittens::WARP_THREADS)

// dimensions

constexpr int BATCH_DIM = 1;
constexpr int SEQ_LEN = 128;
constexpr int HEAD_DIM = 256;

struct attn_globals {
    using Q_gl = gl<bf16, -1, -1, -1, -1, st_bf<16, 256>>; // 16x256 Q tile
    using K_gl = gl<bf16, -1, -1, -1, -1, st_bf<16, 256>>; // 16x256 K tile
    using V_gl = gl<bf16, -1, -1, -1, -1, st_bf<16, 16>>; // 16x16 V tile
    using output_gl = gl<float, -1, -1, -1, -1, st_fl<16, 16>>; // 16x16 output tile

    Q_gl Q;
    K_gl K;
    V_gl V;
    output_gl output; // 16x16 output tile that we fill in each warp
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void tk_attn(const __grid_constant__ attn_globals g) {

    // setup shared memory
    extern __shared__ alignment_dummy __shm[]; // allocates a dynamic amount of shared memory
    shared_allocator al((int*)&__shm[0]); // create a shared memory allocator and point it to the starting address of the shared memory

    // allocate memory for the Q, K, V, and output
    st_bf<16, 256> (&Q_s) = al.allocate<st_bf<16, 256>>(); // 16x256 Q tile
    st_bf<16, 256> (&K_s) = al.allocate<st_bf<16, 256>>(); // 16x256 K tile
    st_bf<16, 16> (&V_s) = al.allocate<st_bf<16, 16>>(); // 16x16 V tile

    // each warp will generate one square in the output
    // with our shapes we need 8 * 16 warps to generate the output
    // each block will have 16 warps and generate a single row of the output
    // so our indexing is blockIdx.x up to 8 and then warp_id up to 16
    // thus our blockIdx determine the row of the output (since there are 8 rows in the output)
    // and our warp_id determines the column of the output (since there are 16 columns in the output)

    int out_col = kittens::warpid(); // this is the column of the output
    int out_row = blockIdx.x; // this is the row of the output

    // load the Q into shared memory
    // we load K when inside the loop
    load(Q_s, g.Q, {0, 0, out_row, 0});

    // no need to sync here because we load to the registers after the register space is allocated
    // theres already a thread sync barrier after the register space is allocated

    // create space for our Q row, K column, and attention intermediate in the registers
    rt_bf<16, 256> Q_row;        // bf16 for MMA
    rt_bf<16, 256> K_col;        // bf16 for MMA
    rt_bf<16, 16, kittens::ducks::rt_layout::col> V_block;  // bf16 for MMA, column layout
    rt_fl<16, 16> attn_scores; // float for attention scores
    rt_bf<16, 16> attn_scores_bf16; // bf16 for MMA
    rt_fl<16, 16> O_block; // float for output accumulation
    
    __syncthreads(); // make sure the register space is allocated before we load anything to it

    zero(O_block); // zero the output block so nothing is added from previous passes 

    // Load Q and convert to bf16
    load(Q_row, Q_s); // load the Q row into temporary float registers
    __syncthreads();

    // iterate over the sequence dimension
    for (int seq_idx = 0; seq_idx < SEQ_LEN; seq_idx += 16) {
        // load the K column and V block into shared memory and then the registers
        load(K_s, g.K, {0, 0, seq_idx / 16, 0});
        load(V_s, g.V, {0, 0, seq_idx / 16, out_col});
        __syncthreads();

        // load the K column and V block into temporary float registers, then convert to bf16
        load(K_col, K_s);
        load(V_block, V_s);
        __syncthreads();

        // zero the attention intermediate so nothing is added from previous passes
        zero(attn_scores);
        zero(attn_scores_bf16);
        __syncthreads();

        // multiply the Q row and K column (transpose is handled by the mma_ABt)
        mma_ABt(attn_scores, Q_row, K_col, attn_scores);
        __syncthreads();

        // copy the attention scores to the attention intermediate
        copy(attn_scores_bf16, attn_scores);
        __syncthreads();

        // multiply the attn intermediate with the V block and add to the output block
        mma_AB(O_block, attn_scores_bf16, V_block, O_block);
        __syncthreads();
    }

    // store the output block to the output
    store(g.output, O_block, {0, 0, out_row, out_col});
    __syncthreads();
}

void dispatch_micro( float *d_a, float *d_b, float *d_c, float *d_output) {
    using globals = attn_globals;

    // create the global layouts
    globals::Q_gl  Q_arg{reinterpret_cast<__nv_bfloat16*>(d_a), 1, BATCH_DIM, SEQ_LEN, HEAD_DIM};
    globals::K_gl  K_arg{reinterpret_cast<__nv_bfloat16*>(d_b), 1, BATCH_DIM, SEQ_LEN, HEAD_DIM};  
    globals::V_gl  V_arg{reinterpret_cast<__nv_bfloat16*>(d_c), 1, BATCH_DIM, SEQ_LEN, HEAD_DIM};
    globals::output_gl output_arg{d_output, 1, BATCH_DIM, SEQ_LEN, HEAD_DIM};

    globals g{Q_arg, K_arg, V_arg, output_arg};

    unsigned long mem_size = 100960; 
    cudaFuncSetAttribute(tk_attn, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    // Launch 1D grid with correct number of tiles
    tk_attn<<<SEQ_LEN/16, NUM_THREADS, mem_size>>>(g);
    cudaDeviceSynchronize();
}
#include "harness.impl"