import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss_multiclass(
    prediction: torch.Tensor, target: torch.Tensor, gamma: int = 2, reduction="mean",
) -> torch.Tensor:
    """Calculate loss using multi-class implementation of FocalLoss

    Parameters
    ----------
    prediction : torch.Tensor
        Logits predicted by the model.
    target : torch.Tensor
        Target/label tensor.
    gamma : int, optional
        Modulating factor for loss curve, by default 2.

    Returns
    -------
    torch.Tensor
        Loss value(s).
    """
    logp = F.cross_entropy(prediction, target, reduction="none")
    p = torch.exp(-logp)
    loss = (1 - p) ** gamma * logp

    if reduction == "mean":
        return loss.mean()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"reduction must be either 'mean' or 'none', got {reduction}")


class FocalLossMulticlass(nn.Module):
    """Multi-class implementation of FocalLoss"""

    def __init__(self, gamma: int = 2, reduction="mean"):
        """Initialize multi-class FocalLoss callable object.

        Parameters
        ----------
        gamma : int, optional
            Modulating factor for loss curve, by default 2
        """
        super(FocalLossMulticlass, self).__init__()
        self.gamma = gamma

        if reduction != "mean" and reduction != "none":
            raise ValueError(
                f"reduction must be either 'mean' or 'none', got {reduction}"
            )
        self.reduction = reduction

    def forward(self, pred, target):
        return focal_loss_multiclass(pred, target, self.gamma, self.reduction)


def focal_loss_binary(
    pred: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    reduction: str = "mean",
    cast_target: bool = True,
) -> torch.Tensor:
    """Calculate loss using binary implementation of Focal Loss

    Parameters
    ----------
    pred : torch.Tensor
        Logits predicted by the model.
    target : torch.Tensor
        Target/label tensor.
    gamma : int, optional
        Modulating factor for loss curve, by default 2.

    Returns
    -------
    torch.Tensor
        Loss value(s).
    """
    if cast_target:
        target = target.type_as(pred)
    bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    pt = torch.exp(-bce_loss)

    loss = (1 - pt) ** gamma * bce_loss

    if reduction == "mean":
        return loss.mean()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"reduction must be either 'mean' or 'none', got {reduction}")


class FocalLossBinary(nn.Module):
    """Focal Loss for binary classification, supports multi-label classification."""

    def __init__(
        self,
        gamma: float = 2.0,
        reduction: str = "mean",
        reshape_targets: bool = False,
        cast_target: bool = True,
    ):
        assert reduction in ["none", "mean"]

        super(FocalLossBinary, self).__init__()
        self.gamma = gamma

        if reduction != "mean" and reduction != "none":
            raise ValueError(
                f"reduction must be either 'mean' or 'none', got {reduction}"
            )
        self.reduction = reduction

        self.reshape_targets = reshape_targets
        self.cast_target = cast_target

    def forward(self, pred, target):
        if self.reshape_targets:
            target = target.view(-1, 1)

        return focal_loss_binary(
            pred, target, self.gamma, self.reduction, self.cast_target,
        )
