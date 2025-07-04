# Kernel Zoo

Collection of CUDA kernels I'm writing or have written for various purposes

## 📁 Project Structure

```
kernel-zoo/
├── README.md
├── pyproject.toml
├── main.py
├── 🟢 base-ops/                        # Fundamental Operations
│   ├── addvec/                     
│   │   ├── addvec.cu              
│   │   └── timed_addvec.cu        
│   └── matvec/                       
│       └── naive_matvec.cu        
├── 🟡 fusion/                          # Kernel Fusions
│   └── conv_groupnorm_act.cu   
├── 🔵 benchmarks/                      # Performance comparisons
│   └── baseline/
│       └── torch/                      # PyTorch implementations
│           └── conv_groupnorm_act.py
└── 🔧 utils/                           # Shared utilities
```

