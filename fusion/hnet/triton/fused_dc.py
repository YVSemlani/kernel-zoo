import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

"""

Fused dynamic chunking kernel based off Triton's fused softmax tutorial


"""

@triton.jit
def fused_dc_kernel(Q_ptr, # B x (L - 1) x D
            K_ptr, # B x (L - 1) x D
            p_ptr, # B x L - 1
            b_ptr # B x L - 1
            Q_row_stride,
            K_stride,
            p_stride, # most likely 1
            b_stride, # most likely 1
            # batch_dim, # add back when we use over multiple batches
            seq_len,
            head_dim,
            BLOCK_SIZE: tl.constexpr,
            num_stages: tl.constexpr
            ):

row_start = tl.program_id(0) # assume singular batch dim for now
row_step = tl.num_programs(0)

# instead of load row -> make unit vector -> back to global memory -> load normed row -> dot prod -> back to global memory -> get probabilities -> get boundaries
# we want load row -> make unit vector -> dot prod -> get probabilities -> get boundaries -> back to global memory -> load clientside

for row_start_idx in tl.range(row_start, seq_len, row_step, num_stages=num_stages):
    Q_row_start_ptr = Q_ptr + row_idx * Q_row_stride
    K_row_start_ptr = K_ptr + row_idx * K_row_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)

    Q_input_ptrs = Q_row_start_ptr + col_offsets
    K_input_ptrs = K_row_start_ptr + col_offsets

    mask = col_offsets < head_dim

    Q_row = tl.load(Q_input_ptrs, mask=mask, other=float(0))
    K_row = tl.load(K_input_ptrs, mask=mask, other=float(0))

    # operations

    Q_norm = tl.sqrt(tl.sum(Q_row * Q_row, axis=0)) # ||Q||
    K_norm = tl.sqrt(tl.sum(K_row * K_row, axis=0)) # ||K||

    ele_dot = Q_row * K_row
    norms = Q_norm * K_norm

    cos_sim = ele_dot / norms

    p = tl.clamp((1 - cos_sim) / 2, 0, 1) # clamp should no-op unless we have precision errors

    b = tl.where(x >= 0.5, 1, 0) # piecewise boundary function applied

    # store

    p_out_ptr = p_ptr + row_idx * p_stride # 1-D tensor so our stride to next row is just the next memory address
    b_out_ptr = b_ptr + row_idx * b_stride

    tl.store(p_out_ptr, p, mask=mask)
    tl.store(b_out_ptr, b, mask=mask)

properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}

# we're ignoring batch length right now
def fused_dc(Q, K):
    assert Q.shape == K.shape, "Q and K must have the same shape"

    seq_len, head_dim = Q.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    num_warps = 8 # this can be autotuned

    num_stages = 4 if SIZE_SMEM > 200000 else 2

    p = torch.empty((seq_len))
    b = torch.empty_like(p)

    # pre-compile kernel to get register usage and compute thread occupancy.
    kernel = fused_dc_kernel.warmup(Q, K, p, b Q.stride(0), K.stride(0), p.stride(0), b.stride(0), seq_len, head_dim, BLOCK_SIZE=BLOCK_SIZE,
                                   num_stages=num_stages, num_warps=num_warps, grid=(1, ))

    kernel._init_handles()
    n_regs = kernel.n_regs
    size_smem = kernel.metadata.shared
    if is_hip():
        # NUM_REGS represents the number of regular purpose registers. On CDNA architectures this is half of all registers available.
        # However, this is not always the case. In most cases all registers can be used as regular purpose registers.
        # ISA SECTION (3.6.4 for CDNA3)
        # VGPRs are allocated out of two pools: regular VGPRs and accumulation VGPRs. Accumulation VGPRs are used
        # with matrix VALU instructions, and can also be loaded directly from memory. A wave may have up to 512 total
        # VGPRs, 256 of each type. When a wave has fewer than 512 total VGPRs, the number of each type is flexible - it is
        # not required to be equal numbers of both types.
        NUM_GPRS = NUM_REGS
        if is_cdna():
            NUM_GPRS = NUM_REGS * 2

        # MAX_NUM_THREADS represents maximum number of resident threads per multi-processor.
        # When we divide this number with WARP_SIZE we get maximum number of waves that can
        # execute on a CU (multi-processor)  in parallel.
        MAX_NUM_THREADS = properties["max_threads_per_sm"]
        max_num_waves = MAX_NUM_THREADS // WARP_SIZE
        occupancy = min(NUM_GPRS // WARP_SIZE // n_regs, max_num_waves) // num_warps
    else:
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    occupancy = min(occupancy, SIZE_SMEM // size_smem)
    num_programs = NUM_SM * occupancy

    num_programs = min(num_programs, seq_len)

    # Create a number of persistent programs.
    kernel[(num_programs, 1, 1)](Q, K, p, b, Q.stride(0), K.stride(0) p.stride(0), b.stride(0), seq_len, head_dim, BLOCK_SIZE, num_stages)
    return p, b







