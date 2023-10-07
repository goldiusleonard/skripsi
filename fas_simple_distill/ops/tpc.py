import torch
import torch.nn as nn
from pytorch_metric_learning.distances import LpDistance
from typing import Union

# class TotalPairwiseConfusion(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.lp_dist = LpDistance(power=2)
    
#     def forward(self, features: Union[torch.Tensor, torch.tensor], labels: Union[torch.Tensor, torch.tensor]):
#         unique_lbls = labels.unique().cpu().numpy()

#         loss = None

#         for lbl in unique_lbls:
#             lbl_idxs = torch.where(labels == lbl)
            
#             class_features = features[lbl_idxs]
#             batch_size = class_features.size(0)

#             for i in range(batch_size):
#                 for j in range(i+1, batch_size):
#                     x1 = features[i]
#                     x2 = features[j]

#                     if loss is None:
#                         loss = self.lp_dist(x1.unsqueeze(0), x2.unsqueeze(0)).abs()
#                     else:
#                         loss += self.lp_dist(x1.unsqueeze(0), x2.unsqueeze(0)).abs()

#         loss = loss.sum()
#         return loss

class TotalPairwiseConfusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lp_dist = LpDistance(power=2)

    def forward(self, features: Union[torch.Tensor, torch.tensor]):
        batch_size = features.size(0)
        if float(batch_size) % 2 != 0:
            raise Exception('Incorrect batch size provided')
        batch_left = features[:int(0.5*batch_size)]
        batch_right = features[int(0.5*batch_size):]
        loss  = self.lp_dist(batch_left, batch_right).abs().sum()

        return loss