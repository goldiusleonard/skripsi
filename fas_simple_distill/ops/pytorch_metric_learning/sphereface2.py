import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning.losses import (
    WeightRegularizerMixin,
    CosFaceLoss,
    BaseMetricLossFunction,
)
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.distances import CosineSimilarity

import math


class SphereFace2C(WeightRegularizerMixin, BaseMetricLossFunction):
    def __init__(
        self,
        num_classes,
        embedding_size,
        lambd=0.7,
        r=40.0,
        margin=0.4,
        t=3.0,
        lw=50.0,
        weight_init_func=None,
        **kwargs
    ):
        if weight_init_func is None:
            weight_init_func = c_f.TorchInitWrapper(nn.init.xavier_normal_)

        super().__init__(weight_init_func=weight_init_func, **kwargs)
        c_f.assert_distance_type(self, CosineSimilarity)
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.lambd = lambd
        self.r = r
        self.margin = margin
        self.t = t
        self.lw = lw

        self.W = nn.Parameter(torch.Tensor(embedding_size, num_classes))
        self.weight_init_func(self.W)

        self.bias = nn.Parameter(torch.Tensor(1))
        self.init_bias()
        
    def get_default_distance(self):
        return CosineSimilarity()

    def init_bias(self):
        z = self.lambd / ((1.0 - self.lambd) * (self.num_classes - 1.0))
        ay = self.r * (2.0 * 0.5**self.t - 1.0 - self.margin)
        ai = self.r * (2.0 * 0.5**self.t - 1.0 + self.margin)
        temp = (1.0 - z) ** 2 + 4.0 * z * math.exp(ay - ai)
        b = math.log(2.0 * z) - ai - math.log(1.0 - z + math.sqrt(temp))
        self.bias = nn.Parameter(torch.Tensor(1))
        nn.init.constant_(self.bias, b)

    def get_cosine(self, embeddings):
        return self.distance(embeddings, self.W.t())

    def get_logits(self, embeddings):
        logits = self.r * self.get_cosine(embeddings) + self.bias
        return logits

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        c_f.ref_not_supported(embeddings, labels, ref_emb, ref_labels)
        c_f.indices_tuple_not_supported(indices_tuple)

        cos_theta = self.get_cosine(embeddings)
        one_hot = F.one_hot(labels, num_classes=self.num_classes).type_as(cos_theta)

        with torch.no_grad():
            g_cos_theta = 2.0 * ((cos_theta + 1.0) / 2.0).pow(self.t) - 1.0
            g_cos_theta = g_cos_theta - self.margin * (2.0 * one_hot - 1.0)
            d_theta = g_cos_theta - cos_theta

        logits = self.r * (cos_theta + d_theta) + self.bias
        weight = self.lambd * one_hot + (1.0 - self.lambd) * (1.0 - one_hot)
        weight = self.lw * self.num_classes / self.r * weight
        loss = F.binary_cross_entropy_with_logits(
            logits,
            one_hot,
            weight=weight,
        )

        loss_dict = {
            "loss": {
                "losses": loss,
                "indices": None,
                "reduction_type": "already_reduced",
            }
        }

        self.add_weight_regularization_to_loss_dict(loss_dict, self.W.t())
        return loss_dict


class SphereFace2s(nn.Module):
    def __init__(
        self,
        feat_dim,
        num_class,
        lambd=0.7,
        r=40.0,
        m=0.4,
        t=3.0,
        lw=50.0,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_class = num_class

        self.lambd = lambd
        self.r = r
        self.m = m
        self.t = t
        self.lw = lw

        # init weights
        self.w = nn.Parameter(torch.Tensor(feat_dim, num_class))
        nn.init.xavier_normal_(self.w)

        # bias initialization
        z = lambd / ((1.0 - lambd) * (num_class - 1.0))
        ay = r * (2.0 * 0.5**t - 1.0 - m)
        ai = r * (2.0 * 0.5**t - 1.0 + m)
        temp = (1.0 - z) ** 2 + 4.0 * z * math.exp(ay - ai)
        b = math.log(2.0 * z) - ai - math.log(1.0 - z + math.sqrt(temp))
        self.b = nn.Parameter(torch.Tensor(1))
        nn.init.constant_(self.b, b)

    def forward(self, x, y):
        with torch.no_grad():
            self.w.data = F.normalize(self.w.data, dim=0)

        # delta theta with margin
        cos_theta = F.normalize(x, dim=1).mm(self.w)
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, y.view(-1, 1), 1.0)
        with torch.no_grad():
            g_cos_theta = 2.0 * ((cos_theta + 1.0) / 2.0).pow(self.t) - 1.0
            g_cos_theta = g_cos_theta - self.m * (2.0 * one_hot - 1.0)
            d_theta = g_cos_theta - cos_theta

        logits = self.r * (cos_theta + d_theta) + self.b
        weight = self.lambd * one_hot + (1.0 - self.lambd) * (1.0 - one_hot)
        weight = self.lw * self.num_class / self.r * weight
        loss = F.binary_cross_entropy_with_logits(logits, one_hot, weight=weight)

        return loss
