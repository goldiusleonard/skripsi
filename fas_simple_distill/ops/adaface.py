import torch

from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.losses import LargeMarginSoftmaxLoss


class AdaFaceLoss(LargeMarginSoftmaxLoss):
    def __init__(
        self,
        num_classes,
        embedding_size,
        margin=0.4,
        scale=64,
        h=0.333,
        t_alpha=0.99,
        **kwargs
    ):
        super().__init__(num_classes, embedding_size, margin, scale, **kwargs)
        self.h = h
        self.t_alpha = t_alpha

        self.batch_mean: torch.Tensor
        self.batch_std: torch.Tensor
        self.register_buffer("batch_mean", torch.ones(1) * (20))
        self.register_buffer("batch_std", torch.ones(1) * 100)

    def init_margin(self):
        pass

    def cast_types(self, dtype, device):
        self.W.data = c_f.to_device(self.W.data, device=device, dtype=dtype)
        self.batch_mean = c_f.to_device(self.batch_mean, device=device, dtype=dtype)
        self.batch_std = c_f.to_device(self.batch_std, device=device, dtype=dtype)

    def modify_cosine_of_target_classes(self, cosine_of_target_classes, margin_scaler):
        angles = self.get_angles(cosine_of_target_classes)

        g_angular = self.margin * margin_scaler * -1
        cosine = torch.cos(angles + g_angular)

        g_add = self.margin + (self.margin * margin_scaler)
        cosine = cosine - g_add

        return cosine

    def scale_logits(self, logits, *_):
        return logits * self.scale

    @torch.no_grad()
    def get_margin_scaler(self, embeddings):
        embedding_norms = self.distance.get_norm(embeddings)
        embedding_norms = torch.clip(embedding_norms, min=0.001, max=100)
        embedding_norms = embedding_norms.clone().detach()

        mean = embedding_norms.mean().detach()
        std = embedding_norms.std().detach()

        if self.training:
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std = std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        margin_scaler = (embedding_norms - self.batch_mean) / (
            self.batch_std + 1e-3
        )
        margin_scaler = margin_scaler * self.h
        margin_scaler = torch.clip(margin_scaler, -1, 1)

        return margin_scaler

    def compute_loss(self, embeddings, labels, indices_tuple):
        dtype, device = embeddings.dtype, embeddings.device
        self.cast_types(dtype, device)
        miner_weights = lmu.convert_to_weights(indices_tuple, labels, dtype=dtype)
        mask = self.get_target_mask(embeddings, labels)

        cosine = self.get_cosine(embeddings)
        margin_scaler = self.get_margin_scaler(embeddings)

        cosine_of_target_classes = cosine[mask == 1]
        modified_cosine_of_target_classes = self.modify_cosine_of_target_classes(
            cosine_of_target_classes, margin_scaler
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
