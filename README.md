# Kernel Zoo

Collection of CUDA kernels I'm writing or have written for various purposes

## 📁 Project Structure

```
kernel-zoo/
├── 🟢 base-ops/                        # Fundamental Operations                   
├── 🟡 fusion/                          # Kernel Fusions
├── 🔵 benchmarks/                      # Performance comparisons
│   └── baseline/
│       └── torch/                      # PyTorch implementations
└── 🔧 utils/                           # Shared utilities
```
## Development

I've currently implemented these kernels:

- HNet Fused Dynamic Chunking (most recent code available in my [hnet repo](https://github.com/YVSemlani/hnet))
- Attention
- Vector Addition
- 2D Convolution






