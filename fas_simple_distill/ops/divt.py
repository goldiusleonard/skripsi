import torch
import torch.nn as nn


class DomainInvariantConcentration(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ["mean", "sum"]:
            raise ValueError(f"{reduction} is not a supported reduction method!")
        self.reduction = reduction

    def forward(self, feat, label):
        loss = torch.inner(label.float(), torch.norm(feat, 1, 1).float())

        if self.reduction == "mean":
            return loss.mean()
        else:
            return loss.sum()
