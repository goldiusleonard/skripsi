from typing import List

import torch

from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.losses import (
    BaseMetricLossFunction,
    WeightRegularizerMixin,
)
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu


class AsymmetricCosFaceLoss(WeightRegularizerMixin, BaseMetricLossFunction):
    def __init__(
        self,
        num_classes: int,
        embedding_size: int,
        margins: List[float],
        scale=64,
        **kwargs
    ):
        super().__init__(**kwargs)
        c_f.assert_distance_type(self, CosineSimilarity)
        self.num_classes = num_classes
        self.scale = scale

        if len(margins) != self.num_classes:
            raise ValueError("Number of margins must be the same as num_classes")
        self.margins = torch.tensor(margins)

        self.W = torch.nn.Parameter(torch.Tensor(embedding_size, num_classes))
        self.weight_init_func(self.W)
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")

    def get_cosine(self, embeddings):
        return self.distance(embeddings, self.W.t())

    def get_angles(self, cosine_of_target_classes):
        angles = torch.acos(torch.clamp(cosine_of_target_classes, -1, 1))
        return angles

    def get_target_mask(self, embeddings, labels):
        batch_size = labels.size(0)
        mask = torch.zeros(
            batch_size,
            self.num_classes,
            dtype=embeddings.dtype,
            device=embeddings.device,
        )
        mask[torch.arange(batch_size), labels] = 1
        return mask

    def get_corresponding_margins(self, labels):
        return self.margins[labels]

    def modify_cosine_of_target_classes(self, cosine_of_target_classes, labels):
        margins = self.get_corresponding_margins(labels)
        return cosine_of_target_classes - margins

    def scale_logits(self, logits, *_):
        return logits * self.scale

    def cast_types(self, dtype, device):
        self.W.data = c_f.to_device(self.W.data, device=device, dtype=dtype)
        self.margins = c_f.to_device(self.margins, device=device, dtype=dtype)

    def compute_loss(self, embeddings, labels, indices_tuple):
        # c_f.ref_not_supported(embeddings, labels, ref_emb, ref_labels)
        dtype, device = embeddings.dtype, embeddings.device
        self.cast_types(dtype, device)
        miner_weights = lmu.convert_to_weights(indices_tuple, labels, dtype=dtype)
        mask = self.get_target_mask(embeddings, labels)
        cosine = self.get_cosine(embeddings)
        cosine_of_target_classes = cosine[mask == 1]
        modified_cosine_of_target_classes = self.modify_cosine_of_target_classes(
            cosine_of_target_classes, labels
        )
        diff = (modified_cosine_of_target_classes - cosine_of_target_classes).unsqueeze(
            1
        )
        logits = cosine + (mask * diff)
        logits = self.scale_logits(logits, embeddings)
        unweighted_loss = self.cross_entropy(logits, labels)
        miner_weighted_loss = unweighted_loss * miner_weights
        loss_dict = {
            "loss": {
                "losses": miner_weighted_loss,
                "indices": c_f.torch_arange_from_size(embeddings),
                "reduction_type": "element",
            }
        }
        self.add_weight_regularization_to_loss_dict(loss_dict, self.W.t())
        return loss_dict

    def get_default_distance(self):
        return CosineSimilarity()

    def get_logits(self, embeddings):
        logits = self.get_cosine(embeddings)
        return self.scale_logits(logits, embeddings)


class AsymmetricArcFaceLoss(AsymmetricCosFaceLoss):
    def modify_cosine_of_target_classes(self, cosine_of_target_classes, labels):
        margins = self.get_corresponding_margins(labels)
        angles = self.get_angles(cosine_of_target_classes)
        return torch.cos(angles + margins)
