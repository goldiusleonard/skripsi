import sys
sys.path.append("..")
from collections import OrderedDict
from termcolor import cprint
from torch.nn import Parameter
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import math
import torch
import torch.nn as nn
import os



class MagLinear(torch.nn.Module):
    """
    Parallel fc for Mag loss
    """

    def __init__(self, in_features, out_features,upper_a=110,lower_a=10,upper_margin=0.8,lower_margin=0.45, scale=64.0, easy_margin=True):
        super(MagLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.scale = scale
        self.easy_margin = easy_margin
        
        self.u_a = upper_a
        self.l_a = lower_a
        self.u_margin = upper_margin
        self.l_margin = lower_margin
        
    def _margin(self, x):
        """generate adaptive margin
        """
        margin = (self.u_margin-self.l_margin) / \
            (self.u_a-self.l_a)*(x-self.l_a) + self.l_margin
        return margin

    def forward(self, x):
        """
        Here m is a function which generate adaptive margin
        """
        x = x.cuda(non_blocking=True)
        #Magnitude of input x 
        x_norm = torch.norm(x, dim=1, keepdim=True).clamp(self.l_a, self.u_a).type(torch.FloatTensor)
        
        # Count the adaptive margin for each magnitude , with an assumption the larger the magnitude the higher the quality , 
        # therefore we penalized more on the larger value
        ada_margin = self._margin(x_norm).type(torch.FloatTensor).cuda(non_blocking=True)
        cos_m, sin_m = torch.cos(ada_margin), torch.sin(ada_margin)

        # norm the weight
        weight_norm = F.normalize(self.weight, dim=0)
#         print("magface")
#         print(x.size())
#         print(weight_norm.size())
        cos_theta = torch.mm(F.normalize(x), weight_norm).type(torch.FloatTensor).cuda(non_blocking=True)
        cos_theta = cos_theta.clamp(-1, 1)
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_theta_m = cos_theta * cos_m - sin_theta * sin_m
        if self.easy_margin:
            cos_theta_m = torch.where(cos_theta > 0, cos_theta_m, cos_theta)
        else:
            mm = torch.sin(math.pi - ada_margin) * ada_margin
            threshold = torch.cos(math.pi - ada_margin)
            cos_theta_m = torch.where(
                cos_theta > threshold, cos_theta_m, cos_theta - mm)
        # multiply the scale in advance
        cos_theta_m = self.scale * cos_theta_m
        cos_theta = self.scale * cos_theta

        return [cos_theta, cos_theta_m], x_norm


class MagLoss(torch.nn.Module):
    """
    MagFace Loss.
    """

    def __init__(self, lower_a, upper_a, lower_margin, upper_margin, scale=64.0):
        super(MagLoss, self).__init__()
        self.l_a = lower_a
        self.u_a = upper_a
        self.scale = scale
        self.cut_off = np.cos(np.pi/2-lower_margin)
        self.large_value = 1 << 10

    def calc_loss_G(self, x_norm):
        g = 1/(self.u_a**2) * x_norm + 1/(x_norm)
        return torch.mean(g)

    def forward(self, input, target, x_norm):
        loss_g = self.calc_loss_G(x_norm)
        target = target.cuda(non_blocking=True)
        
        cos_theta, cos_theta_m = input
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, target.view(-1, 1), 1.0)
        output = one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta
        loss = F.cross_entropy(output, target, reduction='mean')
        return loss.mean(), loss_g, one_hot
