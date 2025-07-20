import torch

import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = driver.active.get_active_torch_device()

"""

Fused dynamic chunking kernel based off Triton's fused softmax tutorial


"""

def get_cuda_autotune_config():
    return [
        triton.Config({}, num_stages=2, num_warps=16),
        triton.Config({}, num_stages=3, num_warps=8),
        triton.Config({}, num_stages=4, num_warps=4),
        triton.Config({}, num_stages=5, num_warps=2),
        triton.Config({}, num_stages=6, num_warps=4),
        triton.Config({}, num_stages=2, num_warps=8),
        triton.Config({}, num_stages=4, num_warps=8),
        triton.Config({}, num_stages=5, num_warps=4)
    ]

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cdna():
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in ('gfx940', 'gfx941', 'gfx942',
                                                                                   'gfx90a', 'gfx908')

@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['batch_dim', 'seq_len', 'head_dim']
)
@triton.jit
def fused_dc_kernel(Q_ptr, # B x (L - 1) x D
            K_ptr, # B x (L - 1) x D
            p_ptr, # B x L - 1
            b_ptr, # B x L - 1
            Q_batch_stride,
            K_batch_stride,
            p_batch_stride,
            b_batch_stride,
            Q_row_stride,
            K_row_stride,
            p_stride, # most likely 1
            b_stride, # most likely 1
            batch_dim, # add back when we use over multiple batches
            seq_len,
            head_dim,
            BLOCK_SIZE: tl.constexpr,
            num_stages: tl.constexpr
            ):

    batch_start = tl.program_id(0) # grid x axis handles a single sequence in the batch
    row_start = tl.program_id(1) # assume singular batch dim for now
    row_step = tl.num_programs(1) # grid y axis handles an entire sequence in the batch

    # get starting pointer for this element in the batch
    Q_batch_ptr = Q_ptr + batch_start * Q_batch_stride
    K_batch_ptr = K_ptr + batch_start * K_batch_stride
    p_batch_ptr = p_ptr + batch_start * p_batch_stride
    b_batch_ptr = b_ptr + batch_start * b_batch_stride

    # BOS token set as mandatory boundary for each sequence in the batch
    if row_start == 0:
        tl.store(p_batch_ptr, 1.0)       
        tl.store(b_batch_ptr, 1)         

    # instead of load row -> make unit vector -> back to global memory -> load normed row -> dot prod -> back to global memory -> get probabilities -> get boundaries
    # we want load row -> make unit vector -> dot prod -> get probabilities -> get boundaries -> back to global memory -> load clientside

    for row_start_idx in tl.range(row_start, seq_len, row_step, num_stages=num_stages):
        Q_row_start_ptr = Q_batch_ptr + row_start_idx * Q_row_stride
        K_row_start_ptr = K_batch_ptr + row_start_idx * K_row_stride

        col_offsets = tl.arange(0, BLOCK_SIZE)

        Q_input_ptrs = Q_row_start_ptr + col_offsets
        K_input_ptrs = K_row_start_ptr + col_offsets

        mask = col_offsets < head_dim

        Q_row = tl.load(Q_input_ptrs, mask=mask, other=float(0))
        K_row = tl.load(K_input_ptrs, mask=mask, other=float(0))

        # operations

        Q_norm = tl.sqrt(tl.sum(Q_row * Q_row, axis=0)) # ||Q||
        K_norm = tl.sqrt(tl.sum(K_row * K_row, axis=0)) # ||K||

        ele_dot = tl.sum(Q_row * K_row)
        norms = Q_norm * K_norm

        cos_sim = ele_dot / norms

        p = tl.clamp((1 - cos_sim) / 2, 0, 1) # clamp should no-op unless we have precision errors

        b = tl.where(p >= 0.5, 1, 0) # piecewise boundary function applied

        # store

        p_out_ptr = p_batch_ptr + row_start_idx * p_stride + 1 # 1-D tensor so our stride to next row is just the next memory address
        b_out_ptr = b_batch_ptr + row_start_idx * b_stride + 1


        tl.store(p_out_ptr, p) # no mask needed because we generate a single value
        tl.store(b_out_ptr, b)
        

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
    assert Q.dim() == 3, "Q and K must be 3D tensors"

    batch_size, seq_len, head_dim = Q.shape
    BLOCK_SIZE = triton.next_power_of_2(head_dim)

    p = torch.empty((batch_size, seq_len + 1), device=DEVICE)
    b = torch.empty_like(p)

    # Create a number of persistent programs.
    fused_dc_kernel[(batch_size, seq_len, 1)](Q, K, p, b, Q.stride(0), K.stride(0), p.stride(0), b.stride(0), Q.stride(1), K.stride(1), p.stride(1), b.stride(1), batch_size, seq_len, head_dim, BLOCK_SIZE)
    return p, b

if __name__ == "__main__":
    torch.manual_seed(0)
    Q = torch.randn(4, 8192, 1024, device=DEVICE)
    K = torch.randn(4, 8192, 1024, device=DEVICE)

    p_triton, b_triton = fused_dc(Q, K)

    # detach
    p = p_triton.detach().cpu()
    b = b_triton.detach().cpu()

    print(p)

    print(b)







