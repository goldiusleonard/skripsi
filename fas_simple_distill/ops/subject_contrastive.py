import torch
import torch.nn as nn
from pytorch_metric_learning.losses.contrastive_loss import ContrastiveLoss
from typing import Union

class SubjectContrastiveLoss(nn.Module):
    def __init__(self, pos_margin:float = 0.0, neg_margin:float = 1.0) -> None:
        super().__init__()
        self.contrastive_loss = ContrastiveLoss(
            pos_margin=pos_margin,
            neg_margin=neg_margin,
        )

    def forward(self, embeddings: Union[torch.Tensor, torch.tensor], labels: Union[torch.Tensor, torch.tensor], subject_labels: Union[torch.Tensor, torch.tensor]):
        unique_lbls = labels.unique().cpu().numpy()
        loss = None

        for lbl in unique_lbls:
            lbl_idxs = torch.where(labels == lbl)
            
            class_embeddings = embeddings[lbl_idxs]
            class_subject_lbls = subject_labels[lbl_idxs]
            
            if loss is None:
                loss = self.contrastive_loss(class_embeddings, class_subject_lbls)
            else:
                loss += self.contrastive_loss(class_embeddings, class_subject_lbls)

        return loss