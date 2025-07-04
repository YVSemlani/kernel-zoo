import torch
import torch.nn as nn
import torch.nn.functional as F

# Baseline PyTorch implementation of model with Conv → GroupNorm → Activation

class ConvGroupNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(ConvGroupNormAct, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.groupnorm = nn.GroupNorm(out_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.groupnorm(x)
        x = self.relu(x)
        return x


# basic shape check
if __name__ == "__main__":
    model = ConvGroupNormAct(3, 3, 3)
    x = torch.randn(1, 3, 224, 224)
    print(model(x).shape)