import torch
import torch.nn as nn
import torch.nn.functional as F


def global_min_pool_2d(x: torch.Tensor):
    B, C, H, W = x.shape
    x = x.view(B, C, H * W)
    return torch.max(x, dim=-1)[0].view(B, C, 1, 1)


def global_maxmin_pool2d(x: torch.Tensor):
    x_max = F.adaptive_max_pool2d(x, 1)
    x_min = global_min_pool_2d(x)
    return 0.5 * (x_max + x_min)


def global_catmaxmin_pool2d(x: torch.Tensor):
    x_max = F.adaptive_max_pool2d(x, 1)
    x_min = global_min_pool_2d(x)
    return torch.cat((x_max, x_min), dim=1)


class GlobalMinPool2d(nn.Module):
    def forward(self, x: torch.Tensor):
        return global_min_pool_2d(x)


class GlobalMaxMinPooling2d(nn.Module):
    def forward(self, x):
        return global_maxmin_pool2d(x)


class GlobalCatMaxMinPooling2d(nn.Module):
    def forward(self, x):
        return global_catmaxmin_pool2d(x)
