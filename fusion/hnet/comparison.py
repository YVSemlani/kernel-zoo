import torch
import torch.nn as nn
import torch.nn.functional as F

from native_dc import RoutingModule

import tk_dc


def run_native_dc(routing_module, x, mask, **kwargs):
    routing_module_output = routing_module(x, mask=mask)

    return routing_module_output.boundary_prob, routing_module_output.boundary_mask, routing_module_output.selected_probs

def run_tk_dc(x_q, W_q, W_k, p, b, **kwargs):
    # we want to shift x_k to the right and zero the initial token
    x_k = x_q.clone()
    x_k[:, :, 0, :] = torch.zeros_like(x_k[:, :, 0, :])
    x_k[:, :, 1:, :] = x_q[:, :, :-1, :]

    tk_dc.dispatch_dc(x_q, x_k, W_q, W_k, p, b)

    # Manually set first p to 1.0 and b to 1.0 forcibly making the first token a chunk boundary
    p[:, :, :, 0] = 1.0
    b[:, :, :, 0] = 1.0

    return p, b

args = {
    "d_model": 1024,
    "batch_size": 1,
    "seq_len": 8192,
}

if __name__ == "__main__":
    x = torch.randn(args["batch_size"], args["seq_len"], args["d_model"], device="cuda")
    mask = torch.ones(args["batch_size"], args["seq_len"], dtype=torch.bool, device="cuda")
    p = torch.zeros(args["batch_size"], args["seq_len"], dtype=torch.bfloat16, device="cuda")
    b = torch.zeros(args["batch_size"], args["seq_len"], dtype=torch.bfloat16, device="cuda")

    routing_module = RoutingModule(d_model=args["d_model"])
    routing_module.to("cuda")

    # get weights from routing module
    W_q = routing_module.q_proj_layer.weight
    W_k = routing_module.k_proj_layer.weight

    # print shapes to verify
    print(f"W_q shape: {W_q.shape}")
    print(f"W_k shape: {W_k.shape}")
    print()

    # run native dc
    boundary_prob, boundary_mask, selected_probs = run_native_dc(routing_module, x, mask, **args)

    # reshape for tk_dc
    x = x.unsqueeze(1)
    W_q = W_q.view(1, 1, args["d_model"], args["d_model"])
    W_k = W_k.view(1, 1, args["d_model"], args["d_model"])
    p = p.view(1, 1, args["batch_size"], args["seq_len"])
    b = b.view(1, 1, args["batch_size"], args["seq_len"])

    # convert to bfloat16
    x = x.to(torch.bfloat16)
    W_q = W_q.to(torch.bfloat16)
    W_k = W_k.to(torch.bfloat16)

    # run tk dc
    p, b = run_tk_dc(x, W_q, W_k, p, b, **args)

    # bring back to cpu
    p = p.cpu()[0][0][0]
    b = b.cpu()[0][0][0]
    boundary_prob = boundary_prob.cpu()[0, :, 1]

    print(p)
    print(b)
    print(boundary_prob)

     # convert b into a mask
    mask = b.bool()
    
    # Reshape p to match boundary_prob for comparison
    p_reshaped = p.squeeze(0).squeeze(0)  # Remove the first two dimensions to get (batch_size, seq_len)
    
    # get indices where misses occur for p vs boundary_prob (second column only)
    miss_mask = (p_reshaped != boundary_prob)
    miss_indices = torch.where(miss_mask)

    print(f"Misses: {miss_indices[-1][:10]}")
    print(f"p at the misses: {p_reshaped[[miss_indices][0][:10]]}")
    print(f"boundary_prob at the misses: {boundary_prob[[miss_indices][0][:10]]}")

    print(f"\n max difference: {torch.max(torch.abs(p_reshaped - boundary_prob))}")
    relative_errors = torch.abs((p_reshaped - boundary_prob) / boundary_prob)  # Already computed, but store it
    max_rel_error = torch.max(relative_errors)  # Find the maximum
    print(f"Max relative error: {max_rel_error.item():.6f} ({max_rel_error.item() * 100:.2f}%)")

    # Check threshold
    threshold = 0.05  # 5%
    if max_rel_error > threshold:
        print(f"WARNING: Max relative error exceeds {threshold*100}% - possible logical mismatch!")
    else:
        print(f"OK: All relative errors <= {threshold*100}% - likely just FP precision noise.")