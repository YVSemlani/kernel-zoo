/* 

    kernel for the dynamic chunking layer of HNet
*/

// each warp handles 16 bytes of the input
// blocks handle 16 * NUM_WARPS bytes of the input

// * denotes element wise multiplication

#include "kittens.cuh"
using namespace kittens;

// dimensions

constexpr int BATCH_DIM = 1;
constexpr int SEQ_LEN = 128;
constexpr int HEAD_DIM = 1024; // d_model = d_k 

// grid dimensions
#define NUM_WARPS 8
#define NUM_THREADS (NUM_WARPS * kittens::WARP_THREADS) // 8 * 32 threads per block
#define NUM_BLOCKS (SEQ_LEN / (NUM_WARPS * 16)) // 128 blocks @ 8 warps per block -> 1024 bytes per block

// inputs are bfloat16

struct dc_globals {
    using x_gl = gl<bf16, -1, -1, -1, -1, st_bf<16, 16>>; // 16x16 tile (loading 16 bytes per warp)
    using weights_gl = gl<bf16, -1, -1, -1, -1, st_bf<16, 16>>; // 16x16 tile corresponding to the input spatial location
    using p_gl = gl<bf16, -1, -1, -1, -1, sv_bf<16>>; // 16 element vector of p values corresponding to the 16 bytes of the block
    using b_gl = gl<bf16, -1, -1, -1, -1, sv_bf<16>>; // 16 element vector of b values

    x_gl x;
    weights_gl W_q, W_k;
    p_gl p;
    b_gl b;
}

__global__ __launch_bounds__(NUM_THREADS, 1)
void tk_dc(const __grid_constant__ dc_globals g) {

    // setup shared memory
    extern __shared__ alignment_dummy __shm[]; // allocates a dynamic amount of shared memory
    shared_allocator al((int*)&__shm[0]); // create a shared memory allocator and point it to the starting address of the shared memory

    int byte_id = blockIdx.x * NUM_WARPS + kittens::warpid(); // this is the chunk of 16 bytes that we're processing
    // block 0 warp 0 processing bytes 0-15
    // block 0 warp 1 processing bytes 16-31
    // block 1 warp 0 processing bytes 1024-1039

    // setup shared memory space for Q and K blocks
    st_bf<16, 16> (&W_q_s) = al.allocate<st_bf<16, 16>>(); // allocate memory for a 16x16 tile of W_q
    st_bf<16, 16> (&W_k_s) = al.allocate<st_bf<16, 16>>(); // allocate memory for a 16x16 tile of W_k

    // load the x and weights from HBM to shared
    load(g.x, g.x, {0, 0, byte_id, 0});

    // setup registers
    rt_bf<16, 16> x_r;
    rt_bf<16, 16, kittens::ducks::rt_layout::col> W_q_r;
    rt_bf<16, 16, kittens::ducks::rt_layout::col> W_k_r;

    rt_bf<16, 16> Q_r; // 16x16 tile of the Q block
    rt_bf<16, 16> K_r; // 16x16 tile of the K block

    rv_bf<16> cos_sim; // store the row wise dot products
    rv_bf<16> q_norm; // store the q row wise norm values
    rv_bf<16> k_norm; // store the k row wise norm values
    rv_bf<16> norm; // store the overall norm values
    rv_bf<16> p; // store the row wise p values

    // zero accumulators
    zero(Q_r);
    zero(K_r);
    zero(cos_sim);
    zero(q_norm);
    zero(k_norm);
    zero(norm);
    zero(p);

    // load x to registers
    load(x_r, g.x, {0, 0, byte_id, 0});

    // iterate over the HEAD_DIM dimension in chunks of 16
    for (int i = 0; i < HEAD_DIM; i += 16) {
        for (int j = 0; j < SEQ_LEN / 16; j++) {

            // load weights tiles from HBM to shared to register
            load(W_q_s, g.W_q, {0, 0, j, byte_id});
            load(W_q_r, W_q_s);

            load(W_k_s, g.W_k, {0, 0, j, byte_id});
            load(W_k_r, W_k_s);

            // multiply x with the weights and accumulate to Q_r and K_r
            mma_AB(Q_r, x_r, W_q_r, Q_r); // Q_r is our 16x16 accumulator which represents a tile of the Q block
            mma_AB(K_r, x_r, W_k_r, K_r); // K_r is our 16x16 accumulator which represents a tile of the K block
        }

        // similarity scores
        cos_sim = fma_AxBtC::op(Q_r, K_r, cos_sim); // Q_r * K_r + cos_sim 

        // norms
        q_norm = fma_AxBtC::op(Q_r, Q_r, q_norm); // Q_r * Q_r + q_norm
        k_norm = fma_AxBtC::op(K_r, K_r, k_norm); // K_r * K_r + k_norm

    }

    // convert to p_t scores
    // p_t = 1/2 (1 -(Q_t x K_(t-1)^T) / (||Q_t|| * ||K_(t-1)||) )
    // p = 1/2 (1 - cos_sim / (k_norm * q_norm))

    sqrt(q_norm, q_norm);
    sqrt(k_norm, k_norm);
    mul(norm, k_norm, q_norm); // k_norm * q_norm
    div(p, cos_sim, norm); // cos_sim / (norm)
    sub(p, 1, p); // 1 - p
    mul(p, p, 0.5); // 0.5 * (1 - p)

    store(p, g.p, {0, 0, byte_id, 0});

    rv_bf<16> b_r; // accumulator for boundary token values
    geq(b_r, p, bf16(0.5)); // b_r = p >= 0.5
    store(b_r, g.b, {0, 0, byte_id, 0}); // store the boundary token values

    // update the x values
    mul(x_r, x_r, b_r); // x_r = x_r * b_r, broadcasting is handled by TK
    store(x_r, g.x, {0, 0, byte_id, 0}); // store the updated x values

}

void dispatch_micro(float *d_x, float *d_W_q, float *d_W_k, float *d_p, float *d_b) {
    using globals = dc_globals;

    // create the global layouts
    globals::x_gl  x_arg{reinterpret_cast<__nv_bfloat16*>(d_x), 1, BATCH_DIM, SEQ_LEN, HEAD_DIM};
    globals::weights_gl  W_q_arg{reinterpret_cast<__nv_bfloat16*>(d_W_q), 1, BATCH_DIM, SEQ_LEN, HEAD_DIM};  
    globals::weights_gl  W_k_arg{reinterpret_cast<__nv_bfloat16*>(d_W_k), 1, BATCH_DIM, SEQ_LEN, HEAD_DIM};
    globals::p_gl  p_arg{reinterpret_cast<__nv_bfloat16*>(d_p), 1, 1, BATCH_DIM, SEQ_LEN};
    globals::b_gl  b_arg{reinterpret_cast<__nv_bfloat16*>(d_b), 1, 1, BATCH_DIM, SEQ_LEN};

    globals g{x_arg, W_q_arg, W_k_arg, p_arg, b_arg};

    unsigned long mem_size = 100960; 
    cudaFuncSetAttribute(tk_dc, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    // Launch 1D grid with correct number of tiles
    tk_dc<<<NUM_BLOCKS, NUM_THREADS, mem_size>>>(g);
    cudaDeviceSynchronize();
}
#include "harness.impl"