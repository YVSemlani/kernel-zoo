# script to benchmark conv kernels vs pytorch implementation

import torch
import torch.nn as nn
import torch.nn.functional as F

import time

def run_benchmark(n_iters, conv_params):
    # create random input tensor
    input_tensor = torch.randn(conv_params["batch_size"], conv_params["in_channels"], conv_params["input_height"], conv_params["input_width"])

    # initialize conv kernel
    conv = nn.Conv2d(conv_params["in_channels"], conv_params["out_channels"], conv_params["kernel_size"], conv_params["stride"], conv_params["padding"])

    # put conv kernel on gpu
    conv.to("cuda")
    input_tensor = input_tensor.to("cuda")

    # warm up convolution
    for _ in range(10):
        _ = conv(input_tensor)

    # Ensure all CUDA operations are finished
    torch.cuda.synchronize()  

    # benchmark convolution
    total_time = 0
    n_iters = 5

    times = []

    for i in range(n_iters):
        # Measure time
        torch.cuda.synchronize()  # Ensure all CUDA operations are finished
        start = time.time()
        _ = conv(input_tensor)
        torch.cuda.synchronize()  # Synchronize again
        end = time.time()
        
        total_time += (end - start) * 1000
        times.append((end - start) * 1000)

    print(f"Convolution Average Computation Time: {(total_time/n_iters):.3f} ms")
    print(f"Times: {times}")

if __name__ == "__main__":
    run_benchmark(5, {"batch_size": 1, "in_channels": 3, "input_height": 1024, "input_width": 1024, "out_channels": 64, "kernel_size": (3, 3), "stride": 1, "padding": 1})