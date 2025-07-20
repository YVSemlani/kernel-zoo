import triton
import triton.language as tl
from triton.runtime import driver

from native_dc import RoutingModule
from fused_dc import fused_dc

import torch
import torch.nn.functional as F

DEVICE = triton.runtime.driver.active.get_active_torch_device()

RUN_NAME = "Short-Sequence"

def run_dc(x, mask, routing_module):
    return routing_module(x, mask=mask)

def run_fused_dc(x, q_proj, k_proj):
    Q = q_proj(x[:, :-1])
    K = k_proj(x[:, 1:])

    p, b = fused_dc(Q, K)

    return p, b

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['SEQ_LEN'],  # argument names to use as an x-axis for the plot
        x_vals=[2 ** i for i in range(2, 14)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['triton', 'torch'],  # possible values for `line_arg``
        line_names=[
            "Fused",
            "Unfused",
        ],  # label name for the lines
        styles=[('blue', '-'), ('green', '-')],  # line styles
        ylabel="time (ms)",  # label name for the y-axis
        plot_name=RUN_NAME,  # name for the plot. Used also as a file name for saving the plot.
        args={'BATCH_SIZE': 4, 'HEAD_DIM': 1024},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(BATCH_SIZE, SEQ_LEN, HEAD_DIM, provider):
    routing_module = RoutingModule(d_model=HEAD_DIM, device=torch.device("cuda"))
    routing_module = torch.compile(routing_module)

    x = torch.randn(BATCH_SIZE, SEQ_LEN, HEAD_DIM, device=torch.device("cuda"))
    mask = torch.ones(BATCH_SIZE, SEQ_LEN, device=torch.device("cuda"), dtype=torch.bool)

    q_proj_layer = routing_module.q_proj_layer
    k_proj_layer = routing_module.k_proj_layer

    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: run_dc(x, mask, routing_module), warmup=10)
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: run_fused_dc(x, q_proj_layer, k_proj_layer), warmup=10)
    return ms


benchmark.run(show_plots=True, print_data=True, save_path="benchmarks",)