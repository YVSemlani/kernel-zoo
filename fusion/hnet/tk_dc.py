import torch
import tk_dc  # This imports your compiled module (tk_dc.so)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create input tensors (bfloat16, matching your kernel's x_gl, weights_gl, etc.)
# Shapes from your dc_globals: [1, BATCH_DIM, SEQ_LEN, HEAD_DIM] for x, W_q, W_k
# [1, 1, BATCH_DIM, SEQ_LEN] for p, b (outputs)
batch_dim = 1
seq_len = 8192
d_model = 1024
d_k = d_model

# Random inputs (on GPU, contiguous)
x = torch.randn(1, batch_dim, seq_len, d_model, dtype=torch.bfloat16, device=device).contiguous()
x_q = x.clone()

# we want to shift x_k to the right and zero the initial token
x_k = x.clone()
x_k[:, :, 0, :] = torch.zeros_like(x_k[:, :, 0, :])
x_k[:, :, 1:, :] = x[:, :, :-1, :]

W_q = torch.randn(1, 1, d_model, d_k, dtype=torch.bfloat16, device=device).contiguous()
W_k = torch.randn(1, 1, d_model, d_k, dtype=torch.bfloat16, device=device).contiguous()

# Outputs (initialize as zeros)
p = torch.zeros(1, 1, batch_dim, seq_len, dtype=torch.bfloat16, device=device).contiguous()
b = torch.zeros(1, 1, batch_dim, seq_len, dtype=torch.bfloat16, device=device).contiguous()

# Call the dispatch function (passes tensors directly)
#tk_dc.dispatch_dc(x_q, x_k, W_q, W_k, p, b)

# Manually set first p to 1.0 and b to 1.0 forcibly making the first token a chunk boundary
p[:, :, :, 0] = 1.0
b[:, :, :, 0] = 1.0

# Or call the kernel directly (similar, but dispatch is recommended)
tk_dc.dispatch_dc(x_q, x_k, W_q, W_k, p, b)

# Print results (p and b are now populated by the kernel)
print("Output p:", p)
print("Output b:", b)