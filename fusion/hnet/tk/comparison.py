import torch
import torch.nn as nn
import torch.nn.functional as F

from native_dc import RoutingModule

import tk_dc
import time


def run_native_dc(routing_module, x, mask, **kwargs):
    routing_module_output = routing_module(x, mask=mask)

    return routing_module_output.boundary_prob, routing_module_output.boundary_mask, routing_module_output.selected_probs

def run_tk_dc(x_q, x_k, W_q, W_k, p, b, **kwargs):

    tk_dc.dispatch_dc(x_q, x_k, W_q, W_k, p, b)

    # Manually set first p to 1.0 and b to 1.0 forcibly making the first token a chunk boundary
    p[:, :, :, 0] = 1.0
    b[:, :, :, 0] = 1.0

    return p, b

def setup(args):
    x = torch.randn(args["batch_size"], args["seq_len"], args["d_model"], dtype=torch.bfloat16, device="cuda")
    mask = torch.ones(args["batch_size"], args["seq_len"], dtype=torch.bool, device="cuda")
    p = torch.zeros(args["batch_size"], args["seq_len"], dtype=torch.bfloat16, device="cuda")
    b = torch.zeros(args["batch_size"], args["seq_len"], dtype=torch.bfloat16, device="cuda")

    routing_module = RoutingModule(d_model=args["d_model"]).to(torch.bfloat16)
    routing_module.to("cuda")

    # get weights from routing module
    W_q = routing_module.q_proj_layer.weight.T.contiguous()
    W_k = routing_module.k_proj_layer.weight.T.contiguous()

    return x, mask, p, b, W_q, W_k, routing_module

args = {
    "d_model": 1024,
    "batch_size": 1,
    "seq_len": 8192,
    "num_warmup": 10,
    "num_timing": 10,
}

if __name__ == "__main__":
    x, mask, p, b, W_q, W_k, routing_module = setup(args)

    # run native dc
    boundary_prob, boundary_mask, selected_probs = run_native_dc(routing_module, x, mask, **args)

    # TIMING FOR NATIVE DC
    # Warm up
    print("Warming up native_dc")
    for _ in range(args["num_warmup"]):
        run_native_dc(routing_module, x, mask, **args)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Measure
    native_times = []
    for _ in range(args["num_timing"]):
        torch.cuda.synchronize()
        start.record()
        run_native_dc(routing_module, x, mask, **args)
        end.record()
        torch.cuda.synchronize()
        native_times.append(start.elapsed_time(end))

    print(f"Native DC average time: {sum(native_times) / args['num_timing']} seconds\n")

    # reshape for tk_dc
    x = x.unsqueeze(1)
    W_q = W_q.view(1, 1, args["d_model"], args["d_model"])
    W_k = W_k.view(1, 1, args["d_model"], args["d_model"])
    p = p.view(1, 1, args["batch_size"], args["seq_len"])
    b = b.view(1, 1, args["batch_size"], args["seq_len"])

    # convert to bfloat16
    #x = x.to(torch.bfloat16)
    #W_q = W_q.to(torch.bfloat16)
    #W_k = W_k.to(torch.bfloat16)

    # we want to shift x_k to the right and zero the initial token
    x_q = x.clone()
    x_k = x.clone()

    x_k = x_k[:, :, :-1, :]
    x_q = x_q[:, :, 1:, :]

    # add padding to the end of both x_q and x_k
    x_q = F.pad(x_q, (0, 0, 1, 0), "constant", 0)
    x_k = F.pad(x_k, (0, 0, 1, 0), "constant", 0)

    print("FLAGGG")
    print(x_q[0, 0, 0, :])
    print(x_k[0, 0, 0, :])

    # run tk dc
    p, b = run_tk_dc(x, x_k, W_q, W_k, p, b, **args)

    # TIMING FOR TK DC
    # Warm up
    print("Warming up tk_dc")
    for _ in range(args["num_warmup"]):
        run_tk_dc(x, x_k, W_q, W_k, p, b, **args)

    # Measure
    tk_times = []
    for _ in range(args["num_timing"]):
        torch.cuda.synchronize()
        start.record()
        run_tk_dc(x, x_k, W_q, W_k, p, b, **args)
        end.record()
        torch.cuda.synchronize()
        tk_times.append(start.elapsed_time(end))

    print(f"TK DC average time: {sum(tk_times) / args['num_timing']} seconds\n")

    # bring back to cpu
    p = p.cpu()[0][0][0]
    b = b.cpu()[0][0][0]
    boundary_prob = boundary_prob.cpu()[0, :, 1]

    print(p)
    print(b)
    print(boundary_prob)

     # convert b into a mask
    mask = b.bool()
    
    # get indices where misses occur for p vs boundary_prob (second column only)
    miss_mask = (p != boundary_prob)
    miss_indices = torch.where(miss_mask)

    print(f"Number of misses: {miss_indices[-1].shape[0]}")

    print(f"Misses: {miss_indices[-1][:10]}")
    print(f"p at the misses: {p[miss_indices[0][:10]]}")
    print(f"boundary_prob at the misses: {boundary_prob[miss_indices[0][:10]]}")

    print(f"\n max difference: {torch.max(torch.abs(p - boundary_prob))}")
    relative_errors = torch.abs((p - boundary_prob) / boundary_prob)  # Already computed, but store it
    max_rel_error = torch.max(relative_errors)  # Find the maximum
    print(f"Max relative error: {max_rel_error.item():.6f} ({max_rel_error.item() * 100:.2f}%)")

    # Check threshold
    threshold = 0.05  # 5%
    if max_rel_error > threshold:
        print(f"WARNING: Max relative error exceeds {threshold*100}% - possible logical mismatch!")
    else:
        print(f"OK: All relative errors <= {threshold*100}% - likely just FP precision noise.")

    # get the max value of p
    max_p = torch.max(p)
    print(f"Max value of p: {max_p.item()}")

    # get the max value of boundary_prob
    max_boundary_prob = torch.max(boundary_prob)
    print(f"Max value of boundary_prob: {max_boundary_prob.item()}")

    # get the max value of p - boundary_prob