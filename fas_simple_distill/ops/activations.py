import torch
import torch.nn as nn
import torch.nn.functional as F


class ReLUClamp(nn.Module):
    def __init__(self, clamp_val=1.0) -> None:
        super().__init__()
        self.clamp_val = clamp_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x).clamp_max(self.clamp_val)
