from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class BCEWithLogitsSmoothLabel(nn.BCEWithLogitsLoss):
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        pos_weight: Optional[Tensor] = None,
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__(weight, size_average, reduce, reduction, pos_weight)

        assert label_smoothing >= 0.0
        self.label_smoothing = label_smoothing

    @torch.no_grad()
    def _smooth_label(self, target):
        target = target * (1.0 - self.label_smoothing) + (0.5 * self.label_smoothing)
        return target

    def forward(self, inp, target):
        target = self._smooth_label(target)
        return super().forward(inp, target)
