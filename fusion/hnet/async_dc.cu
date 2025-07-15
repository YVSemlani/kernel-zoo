/* 

    kernel for the dynamic chunking layer of HNet
*/

#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

constexpr int SEQ_LEN = 8192;
constexpr int HEAD_DIM = 1024;

// Enable Hopper features
#define KITTENS_HOPPER

// Adjust constants for H100
constexpr int NUM_WARPS = 16; // Better occupancy
constexpr int NUM_THREADS = NUM_WARPS * kittens::WARP_THREADS;
constexpr int NUM_BLOCKS = SEQ_LEN / (NUM_WARPS * 64);

constexpr int NUM_COLS = HEAD_DIM / 64;

// define global layouts
using x_gl = gl<bf16, -1, -1, -1, -1, st_bf<64, 64>>;
using weights_gl = gl<bf16, -1, -1, -1, -1, st_bf<64, 64>>;
using p_gl = gl<bf16, -1, -1, -1, -1, sv_bf<64>>;
using b_gl = gl<bf16, -1, -1, -1, -1, sv_bf<64>>;

struct dc_globals {
    x_gl x_q, x_k;
    weights_gl W_q, W_k;
    p_gl p;
    b_gl b;

    dim3 grid() { return dim3(NUM_BLOCKS); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 100000; } // Lower for H100
};

/* CUSTOM OPERATIONS */

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

/* KERNEL */

__global__ __launch_bounds__(NUM_THREADS, 1)
void tk_dc(const __grid_constant__ dc_globals g) {

    // setup shared memory
    extern __shared__ alignment_dummy __shm[]; // allocates a dynamic amount of shared memory
    shared_allocator al((int*)&__shm[0]); // create a shared memory allocator and point it to the starting address of the shared memory

    using load_group = group<4>; // for asynchronous loading with one warpgroup

    int workerid = warpgroup::groupid(); // 0 or 1
    int byte_id = blockIdx.x * 2 + workerid; // unique byte_id per warpgroup

    // allocate shared memory space for our input tiles
    st_bf<64, 64> (&x_q_s)[2] = al.allocate<st_bf<64, 64>, 2>(); // one per warpgroup
    st_bf<64, 64> (&x_k_s)[2] = al.allocate<st_bf<64, 64>, 2>(); // one per warpgroup
    st_bf<64, 64, kittens::ducks::st_layout::col> (&W_q_s) = al.allocate<st_bf<64, 64, kittens::ducks::st_layout::col>>(); // shared single tile
    st_bf<64, 64, kittens::ducks::st_layout::col> (&W_k_s) = al.allocate<st_bf<64, 64, kittens::ducks::st_layout::col>>(); // shared single tile
    st_bf<64, 64> (&Q_r) = al.allocate<st_bf<64, 64>>(); // shared single tile
    st_bf<64, 64> (&K_r) = al.allocate<st_bf<64, 64>>(); // shared single tile

    // Add buffers for pipelined Q/K
    st_fl<64,64> (&Q_buf)[4] = al.allocate<st_fl<64,64>, 4>();
    st_fl<64,64> (&K_buf)[4] = al.allocate<st_fl<64,64>, 4>();

    // Setup 4 semaphores
    __shared__ kittens::semaphore load_barrier[4];
    if(threadIdx.x == 0) {
        for(int i=0; i<4; i++) init_semaphore(load_barrier[i], 0, 1);
    }
    __syncthreads();

    // Pipelined outer loop (depth 4 for async partials)
    for (int out_col = 0; out_col < NUM_COLS; out_col++) {
        zero(Q_r); zero(K_r);
        int tic = out_col % 4;  // 0-3
        int prev_tic = (out_col + 3) % 4;  // For partials on prev stage

        // Inner loop: Loads + Matmul
        for (int in_col = 0; in_col < NUM_COLS; in_col++) {
            // TMA for W (shared)
            if (workerid == 0) {
                tma::expect(load_barrier[tic], sizeof(W_q_s));
                tma::load_async(W_q_s, g.W_q, {0, 0, in_col, out_col}, load_barrier[tic]);
                tma::load_async(W_k_s, g.W_k, {0, 0, in_col, out_col}, load_barrier[tic]);
            }
            wait(load_barrier[tic], 0);
            __syncthreads();

            load(W_q_r, W_q_s);
            load(W_k_r, W_k_s);

            // TMA for x (per-group, pipelined)
            tma::expect(load_barrier[tic], sizeof(x_q_s[0]));
            tma::load_async(x_q_s[workerid], g.x_q, {0, 0, byte_id, in_col}, load_barrier[tic]);
            tma::load_async(x_k_s[workerid], g.x_k, {0, 0, byte_id, in_col}, load_barrier[tic]);
            wait(load_barrier[tic], 0);

            load(x_q_r, x_q_s[workerid]);
            load(x_k_r, x_k_s[workerid]);

            warpgroup::mma_AB(Q_r, x_q_r, W_q_r, Q_r);
            warpgroup::mma_AB(K_r, x_k_r, W_k_r, K_r);
        }

        // After matmul: Buffer Q_r/K_r for this tic (for future partials)
        store(Q_buf[tic], Q_r);
        store(K_buf[tic], K_r);
        __syncthreads();

        // Overlapped partials: Compute on prev_tic's buffered Q/K if ready
        if(out_col > 0) {
            load(temp_Q, Q_buf[prev_tic]);  // Reload prev for partials
            load(temp_K, K_buf[prev_tic]);
            __syncthreads();

            // Trigger partials asynchronously (compute now, overlaps with next loads/matmul)
            rt_fl<64, 64> el_wise_mul;
            // Partial cosine sim
            mul(el_wise_mul, temp_Q, temp_K);
            add(el_wise_mul, el_wise_mul, 1e-6f); // Epsilon for stability
            row_sum(cos_sim, el_wise_mul, cos_sim);
            __syncthreads();

            // Partial q_norm
            mul(el_wise_mul, temp_Q, temp_Q);
            add(el_wise_mul, el_wise_mul, 1e-6f);
            row_sum(q_norm, el_wise_mul, q_norm);
            __syncthreads();

            // Partial k_norm
            mul(el_wise_mul, temp_K, temp_K);
            add(el_wise_mul, el_wise_mul, 1e-6f);
            row_sum(k_norm, el_wise_mul, k_norm);
            __syncthreads();

            // Store partials / Fusion stub here (on prev tile's data)
        }
    }

    // Drain: Process partials for last 3 tics
    for(int drain=0; drain<3; drain++) {
        int tic = (NUM_COLS + drain) % 4;
        load(temp_Q, Q_buf[tic]);
        load(temp_K, K_buf[tic]);
        __syncthreads();

        // Compute partials for this tic
        // Partial cosine sim
        mul(el_wise_mul, temp_Q, temp_K);
        add(el_wise_mul, el_wise_mul, 1e-6f); // Epsilon for stability
        row_sum(cos_sim, el_wise_mul, cos_sim);
        __syncthreads();

        // Partial q_norm
        mul(el_wise_mul, temp_Q, temp_Q);
        add(el_wise_mul, el_wise_mul, 1e-6f);
        row_sum(q_norm, el_wise_mul, q_norm);
        __syncthreads();

        // Partial k_norm
        mul(el_wise_mul, temp_K, temp_K);
        add(el_wise_mul, el_wise_mul, 1e-6f);
        row_sum(k_norm, el_wise_mul, k_norm);
        __syncthreads();

        // Store partials / Fusion stub here (on prev tile's data)
    }

    // Post-drain finals (as before)
    // Post-loop: Add epsilon for stability
    add(q_norm, q_norm, 1e-6f);
    add(k_norm, k_norm, 1e-6f);
    __syncthreads();

    // zero accumulators
    zero(Q_r);
    zero(K_r);
    zero(cos_sim);
    zero(q_norm);
    zero(k_norm);
    zero(norm);

    // Tiled matrix multiplication for Q = x @ W_q and K = x @ W_k
    // Outer loop over d_k tiles (columns of output)
    for (int out_col = 0; out_col < NUM_COLS; out_col++) {
        // Reset partial accumulators for this output tile
        zero(Q_r);
        zero(K_r);
        __syncthreads();

        // Inner loop over d_model tiles (for contraction)
        for (int in_col = 0; in_col < NUM_COLS; in_col++) {
            // Load W_q and W_k asynchronously, shared across warpgroups
            if (workerid == 0) {
                load_group::load_async<1, false>(W_q_s, g.W_q, {0, 0, in_col, out_col});
                load_group::load_async<1, false>(W_k_s, g.W_k, {0, 0, in_col, out_col});
                load_group::load_async_wait();
            }
            __syncthreads();

            // All warpgroups load W from shared to registers
            load(W_q_r, W_q_s);
            load(W_k_r, W_k_s);

            // Load x_q and x_k asynchronously, per warpgroup
            load_group::load_async<1, false>(x_q_s[workerid], g.x_q, {0, 0, byte_id, in_col});
            load_group::load_async<1, false>(x_k_s[workerid], g.x_k, {0, 0, byte_id, in_col});
            load_group::load_async_wait();

            // Load x from shared to registers
            load(x_q_r, x_q_s[workerid]);
            load(x_k_r, x_k_s[workerid]);

            // Accumulate into Q_r and K_r
            warpgroup::mma_AB(Q_r, x_q_r, W_q_r, Q_r);
            warpgroup::mma_AB(K_r, x_k_r, W_k_r, K_r);
            __syncthreads();
        }

        // Compute partials per tile (for fusion)
        rt_fl<64, 64> el_wise_mul;
        __syncthreads();

        // Partial cosine sim
        mul(el_wise_mul, Q_r, K_r);
        add(el_wise_mul, el_wise_mul, 1e-6f); // Epsilon for stability
        row_sum(cos_sim, el_wise_mul, cos_sim);
        __syncthreads();

        // Partial q_norm
        mul(el_wise_mul, Q_r, Q_r);
        add(el_wise_mul, el_wise_mul, 1e-6f);
        row_sum(q_norm, el_wise_mul, q_norm);
        __syncthreads();

        // Partial k_norm
        mul(el_wise_mul, K_r, K_r);
        add(el_wise_mul, el_wise_mul, 1e-6f);
        row_sum(k_norm, el_wise_mul, k_norm);
        __syncthreads();

        // Store partials for fusion (example buffer)
        // st_fl<64, NUM_COLS> partial_norms = al.allocate<st_fl<64, NUM_COLS>>(); // Allocate earlier
        // store(partial_norms[out_col], some_partial); // Example

        // Fused op stub: E.g., attention on this tile's Q_r/K_r
        // warpgroup::mma_AB(fused_output, Q_r, some_attn_matrix, fused_output);

    }

    // Post-loop finals (accumulate partials if needed, then compute p/b as in dc)
    // ... existing finals ...

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