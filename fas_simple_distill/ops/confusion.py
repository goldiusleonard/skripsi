import torch
import torch.nn as nn
# from pytorch_metric_learning.distances.lp_distance import LpDistance

def lp_norm(input, axis:int = 1, p:int = 2, power:int = 1) -> torch.Tensor:
    # torch.pow(input,)
    # norm = torch.norm(input, p, axis, True)
    return torch.pow(input, power)

class ConfusionLoss(nn.Module):
    def __init__(self, num_class:int) -> None:
        super().__init__()
        # self.lp_dist = LpDistance(p=2, power=2)
        self.num_class = num_class

    def forward(self, x):
        loss = lp_norm((x - (1 / float(self.num_class))), axis=1, p=2, power=2).sum()
        # loss = (self.lp_dist((x - (1 / float(self.num_class))))).sum()
        return loss

class TotalPairwiseConfusionLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        batch_size = x.size(0)
        if batch_size == 1:
            raise Exception('Incorrect batch size provided')
        loss = None
        for i in range(batch_size):
            for j in range(i+1, batch_size):
                if loss is None:
                    loss = lp_norm((x[i]-x[j]).abs(), axis=1, p=2, power=2).sum()
                else:
                    loss += lp_norm((x[i]-x[j]).abs(), axis=1, p=2, power=2).sum()
        return loss