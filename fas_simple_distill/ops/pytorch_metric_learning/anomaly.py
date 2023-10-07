import torch
import torch.nn as nn
from pytorch_metric_learning.losses import CosFaceLoss


class AnomCosFaceLoss(CosFaceLoss):
    def __init__(
        self, num_classes, embedding_size, margin=0.35, scale=64, lam=1.0, **kwargs
    ):
        if num_classes <= 2:
            raise ValueError("num_classes must be above 2")

        # exclude live class (0)
        num_classes = num_classes - 1

        super().__init__(
            num_classes, embedding_size, margin=margin, scale=scale, **kwargs
        )
        self.lam = lam

    # pylint: disable=arguments-differ
    def forward(self, embeddings, labels):
        mask_anom = labels == 0
        embeddings_anom = embeddings[mask_anom]
        loss_anom = torch.sum(1 + self.get_cosine(embeddings_anom), dim=-1)
        loss_anom = loss_anom.mean()

        labels_spoof = labels[~mask_anom] - 1
        loss_spoof = super().forward(
            embeddings[~mask_anom],
            labels_spoof,
        )

        return (loss_anom * self.lam) + loss_spoof
