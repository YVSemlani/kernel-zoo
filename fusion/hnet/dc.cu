/* 

    kernel for the dynamic chunking layer of HNet
*/

// each warp handles 16 bytes of the input
// blocks handle 16 * NUM_WARPS bytes of the input

// * denotes element wise multiplication
// accumulators in float to avoid overflows

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

// dimensions
// commented out unused vars when running in Python

// constexpr int BATCH_DIM = 1;
constexpr int SEQ_LEN = 8192;
constexpr int HEAD_DIM = 1024; // d_model = d_k 

// grid dimensions
#define NUM_WARPS 8
#define NUM_THREADS (NUM_WARPS * kittens::WARP_THREADS) // 8 * 32 threads per block
#define NUM_BLOCKS (SEQ_LEN / (NUM_WARPS * 16)) // 128 blocks @ 8 warps per block -> 1024 bytes per block

// inputs are bfloat16

// define global layouts
using x_gl = gl<bf16, -1, -1, -1, -1, st_bf<16, 16>>; // 16x16 tile (loading 16 bytes per warp)
using weights_gl = gl<bf16, -1, -1, -1, -1, st_bf<16, 16>>; // 16x16 tile corresponding to the input spatial location
using p_gl = gl<bf16, -1, -1, -1, -1, sv_bf<16>>; // 16 element vector of p values corresponding to the 16 bytes of the block
using b_gl = gl<bf16, -1, -1, -1, -1, sv_bf<16>>; // 16 element vector of b values

struct dc_globals {
    // input vars
    x_gl x_q, x_k;
    weights_gl W_q, W_k;
    p_gl p;
    b_gl b;

    // grid - number of thread blocks we are launching
    dim3 grid() { return dim3(NUM_BLOCKS); }
    // block - number of threads in a thread block  
    dim3 block() { return dim3(NUM_THREADS); }
    // Safe shared memory size for H100
    size_t dynamic_shared_memory() { return 224000; }
};

// Sqrt operation following ThunderKittens pattern
struct sqrt_op {
    template<typename T> __device__ static inline T op(const T &a) { return sqrtf(a); }
};
template<> __device__ inline float2 sqrt_op::op<float2>(const float2 &a) { 
    return make_float2(sqrtf(a.x), sqrtf(a.y)); 
}
template<> __device__ inline bf16 sqrt_op::op<bf16>(const bf16 &a) { 
    return __float2bfloat16(sqrtf(__bfloat162float(a))); 
}

// B_t operation following ThunderKittens pattern  
struct b_t_op {
    template<typename T> __device__ static inline T op(const T &a) { 
        return a > 0.5f ? 1.0f : 0.0f; 
    }
};
template<> __device__ inline float2 b_t_op::op<float2>(const float2 &a) { 
    return make_float2(a.x > 0.5f ? 1.0f : 0.0f, a.y > 0.5f ? 1.0f : 0.0f); 
}
template<> __device__ inline bf16 b_t_op::op<bf16>(const bf16 &a) { 
    float f = __bfloat162float(a);
    return __float2bfloat16(f > 0.5f ? 1.0f : 0.0f); 
}

// Clamp operation: clamps values to [min_val, max_val]
struct clamp_op {
    template<typename T> __device__ static inline T op(const T &a) { 
        return max(min(a, 1.0f), 0.0f); 
    }
};
// Specialize for float2 (vectorized)
template<> __device__ inline float2 clamp_op::op<float2>(const float2 &a) { 
    return make_float2(
        max(min(a.x, 1.0f), 0.0f),
        max(min(a.y, 1.0f), 0.0f)
    ); 
}
// Specialize for bf16 (if needed, but we're clamping floats)
template<> __device__ inline bf16 clamp_op::op<bf16>(const bf16 &a) { 
    float f = __bfloat162float(a);
    f = max(min(f, 1.0f), 0.0f);
    return __float2bfloat16(f); 
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
    st_bf<16, 16> (&x_q_s) = al.allocate<st_bf<16, 16>>(); // allocate memory for a 16x16 tile of x
    st_bf<16, 16> (&x_k_s) = al.allocate<st_bf<16, 16>>(); // allocate memory for a 16x16 tile of x
    __syncthreads();

    // setup registers
    rt_bf<16, 16> x_q_r; // 16x16 tile of the x values (seq x d_model chunk)
    rt_bf<16, 16> x_k_r; // 16x16 tile of the x values (seq x d_model chunk)
    rt_bf<16, 16, kittens::ducks::rt_layout::col> W_q_r; // 16x16 tile of the W_q values (d_model chunk x d_k chunk)
    rt_bf<16, 16, kittens::ducks::rt_layout::col> W_k_r; // 16x16 tile of the W_k values (d_model chunk x d_k chunk)

    rt_fl<16, 16> Q_r; // 16x16 accumulator for Q (seq x d_k chunk)
    rt_fl<16, 16> K_r; // 16x16 accumulator for K (seq x d_k chunk)

    using vec_t = typename decltype(Q_r)::col_vec; // use associated col_vec type for row reductions
    vec_t cos_sim; // store the row wise dot products (float for accumulation)
    vec_t q_norm; // store the q row wise norm values (float for accumulation)
    vec_t k_norm; // store the k row wise norm values (float for accumulation)
    vec_t norm; // store the overall norm values (float for computation)
    rv_bf<16> p; // store the row wise p values (can stay bf16 for final result)
    __syncthreads();

    // zero accumulators
    zero(Q_r);
    zero(K_r);
    zero(cos_sim);
    zero(q_norm);
    zero(k_norm);
    zero(norm);
    zero(p);

    // Tiled matrix multiplication for Q = x @ W_q and K = x @ W_k
    // Outer loop over d_k tiles (columns of output)
    for (int out_col = 0; out_col < HEAD_DIM / 16; out_col++) {
        // Reset partial accumulators for this output tile
        zero(Q_r);
        zero(K_r);
        __syncthreads();

        // Inner loop over d_model tiles (for contraction)
        for (int in_col = 0; in_col < HEAD_DIM / 16; in_col++) {
            // Load x tile: seq tile (byte_id) x d_model chunk (in_col)
            load(x_q_s, g.x_q, {0, 0, byte_id, in_col});
            load(x_q_r, x_q_s);
            load(x_k_s, g.x_k, {0, 0, byte_id, in_col});
            load(x_k_r, x_k_s);

            // Load W_q tile: d_model chunk (in_col) x d_k chunk (out_col) - note fixed seq=0 since weights are shared
            load(W_q_s, g.W_q, {0, 0, in_col, out_col});
            load(W_q_r, W_q_s);

            // Load W_k tile: same as above
            load(W_k_s, g.W_k, {0, 0, in_col, out_col});
            load(W_k_r, W_k_s);
            __syncthreads();

            // Accumulate into Q_r and K_r
            mma_AB(Q_r, x_q_r, W_q_r, Q_r);
            mma_AB(K_r, x_k_r, W_k_r, K_r);
            __syncthreads();
        }

        // Now compute partial contributions to cos_sim and norms for this output tile
        rt_fl<16, 16> el_wise_mul;
        __syncthreads();

        // Cosine sim partial: row_sum(Q_r * K_r)
        mul(el_wise_mul, Q_r, K_r);
        row_sum(cos_sim, el_wise_mul, cos_sim);
        __syncthreads();

        // Q norm partial: row_sum(Q_r * Q_r)
        mul(el_wise_mul, Q_r, Q_r);
        row_sum(q_norm, el_wise_mul, q_norm);
        __syncthreads();

        // K norm partial: row_sum(K_r * K_r)
        mul(el_wise_mul, K_r, K_r);
        row_sum(k_norm, el_wise_mul, k_norm);
        __syncthreads();
    }

    // convert to p_t scores
    // p_t = 1/2 (1 -(Q_t x K_(t-1)^T) / (||Q_t|| * ||K_(t-1)||) )
    // p = 1/2 (1 - cos_sim / (k_norm * q_norm))

    // add epsilon to norms to avoid division by zero
    add(q_norm, q_norm, 1e-12f);
    add(k_norm, k_norm, 1e-12f);
    __syncthreads();

    // Then use it with unary_op for vectors
    unary_op<sqrt_op>(q_norm, q_norm);  // sqrt(q_norm) -> q_norm
    unary_op<sqrt_op>(k_norm, k_norm);  // sqrt(k_norm) -> k_norm
    __syncthreads();

    mul(norm, k_norm, q_norm); // k_norm * q_norm
    __syncthreads();

    vec_t p_fl; // temporary vector for p calculation
    //add(norm, norm, 1e-12f); // add epsilon to avoid division by zero
    __syncthreads();

    div(p_fl, cos_sim, norm); // cos_sim / (norm)
    __syncthreads();


    sub(p_fl, p_fl, 1.0f); // p = p - 1
    __syncthreads();

    mul(p_fl, p_fl, -1.0f); // p = -(p - 1) = 1 - p
    __syncthreads();

    mul(p_fl, p_fl, 0.5f); // 0.5 * (1 - p)
    __syncthreads();

    // clamp p_fl to [0.0f, 1.0f]
    unary_op<clamp_op>(p_fl, p_fl); // this is what reduces our numerical error
    __syncthreads();

    copy(p, p_fl); // convert from float to bf16 for output
    __syncthreads();

    store(g.p, p, {0, 0, 0, byte_id});
    __syncthreads();

    rv_bf<16> b_r; // accumulator for boundary token values
    unary_op<b_t_op>(b_r, p); // b_r = p >= 0.5
    store(g.b, b_r, {0, 0, 0, byte_id}); // store the boundary token values

    // update the x values
    // we can't update the x values at this level b/c we've gone through the entire head dim but only have a 16x16 tile right now
    // best to do this on the client side

}

// Launch Kernel
void dispatch_dc(dc_globals g) {
    unsigned long mem_size = 50480; 
    cudaFuncSetAttribute(
        tk_dc,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );
    tk_dc<<<g.grid(), g.block(), mem_size>>>(g);
    cudaDeviceSynchronize();
}

/* OLD DISPATCH FUNCTION */

/* 
void dispatch_dc(float *d_x, float *d_W_q, float *d_W_k, float *d_p, float *d_b) {
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

*/ 
PYBIND11_MODULE(tk_dc, m) {
    m.doc() = "tk_dc python module";
    // For wrapping kernels directly.
    kittens::py::bind_kernel<tk_dc, dc_globals>(m, "tk_dc", &dc_globals::x_q, &dc_globals::x_k, &dc_globals::W_q, &dc_globals::W_k, &dc_globals::p, &dc_globals::b);
    // For host functions that wrap the kernel, this will be called from Python
    kittens::py::bind_function<dispatch_dc, dc_globals>(m, "dispatch_dc", &dc_globals::x_q, &dc_globals::x_k, &dc_globals::W_q, &dc_globals::W_k, &dc_globals::p, &dc_globals::b);
}