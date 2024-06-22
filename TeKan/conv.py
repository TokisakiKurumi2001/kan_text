import torch
import torch.nn as nn
import torch.nn.functional as F
from .kan import KANLinear

class KanConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(KanConv1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Linear layer to simulate the convolutional weights
        self.linear = KANLinear(in_channels * kernel_size, out_channels)

    def forward(self, x):
        batch_size, in_channels, width = x.size()

        # Pad the input
        x = F.pad(x, (self.padding, self.padding))

        # Unfold the input to get the sliding windows
        unfolded = x.unfold(2, self.kernel_size, self.stride)

        # Reshape for linear layer: (batch_size, out_channels, new_width)
        new_width = unfolded.size(-1)
        unfolded = unfolded.permute(0, 2, 1, 3).contiguous().view(batch_size, new_width, -1)

        # Apply the linear layer
        out = self.linear(unfolded)

        # Permute back to the shape (batch_size, out_channels, new_width)
        out = out.permute(0, 2, 1)

        return out