from collections import OrderedDict

import torch
from torch import Tensor

from pytorch_metric_learning.losses import CosFaceLoss

from .adaface import AdaFaceLoss


class BroadFaceCosFaceLoss(CosFaceLoss):
    def __init__(
        self,
        num_classes,
        embedding_size,
        margin=0.35,
        scale=64,
        queue_size=10000,
        compensate=True,
        _record_angles=False,
        **kwargs,
    ):
        super().__init__(
            num_classes, embedding_size, margin=margin, scale=scale, **kwargs
        )
        self.queue_size = queue_size
        self.compensate = compensate

        self.register_buffer("feature_mb", torch.zeros(0, embedding_size))
        self.register_buffer("label_mb", torch.zeros(0, dtype=torch.int64))
        self.register_buffer("proxy_W", torch.zeros(0, embedding_size))

        self._no_grad_W = False
        self._record_angles = _record_angles
        if self._record_angles:
            print("Recording average target and nontarget angles per iteration")
            self.register_buffer("_median_target_angles", torch.empty([0]))
            self.register_buffer("_mean_target_angles", torch.empty([0]))
            self.register_buffer("_mean_nontarget_angles", torch.empty([0]))

    # pylint: disable=attribute-defined-outside-init
    def load_state_dict(
        self, state_dict: "OrderedDict[str, Tensor]", strict: bool = True
    ):
        for k in list(state_dict.keys()):
            if k.startswith("_"):
                state_dict.pop(k)
                state_dict[k] = torch.empty([0])
        
        if (feature_mb := state_dict.get("feature_mb")) is None:
            raise RuntimeError("Key feature_mb does not exist in state_dict.")
        if (label_mb := state_dict.get("label_mb")) is None:
            raise RuntimeError("Key label_mb does not exist in state_dict.")
        if (proxy_W := state_dict.get("proxy_W")) is None:
            raise RuntimeError("Key proxy_W does not exist in state_dict.")

        if not (feature_mb.shape[0] == label_mb.shape[0] == proxy_W.shape[0]):
            raise RuntimeError(
                f"Expected same length of {feature_mb.shape[0]} for feature_mb, label_mb"
                f" and proxy_W on state_dict, got {feature_mb.shape[0]}, "
                f"{label_mb.shape[0]}, and {proxy_W.shape[0]}"
            )

        self.feature_mb = torch.zeros_like(feature_mb)
        self.label_mb = torch.zeros_like(label_mb)
        self.proxy_W = torch.zeros_like(proxy_W)

        return super().load_state_dict(state_dict, strict)

    @torch.no_grad()
    def update_queue(self, embeddings, labels):
        self.feature_mb = torch.cat([self.feature_mb, embeddings.detach()], dim=0)
        self.label_mb = torch.cat([self.label_mb, labels.detach()], dim=0)
        self.proxy_W = torch.cat(
            [self.proxy_W, self.W.detach().T[labels].clone()], dim=0
        )

        over_size = self.feature_mb.shape[0] - self.queue_size
        if over_size > 0:
            self.feature_mb = self.feature_mb[over_size:]
            self.label_mb = self.label_mb[over_size:]
            self.proxy_W = self.proxy_W[over_size:]

        assert (
            self.feature_mb.shape[0] == self.label_mb.shape[0] == self.proxy_W.shape[0]
        )

    # pylint: enable=attribute-defined-outside-init
    def get_queued_embeddings(self):
        if not self.compensate:
            return self.feature_mb

        curr_W = self.W.detach().T[self.label_mb]
        delta_W = curr_W - self.proxy_W

        update_feature_mb = (
            self.feature_mb
            + (
                self.feature_mb.norm(p=2, dim=1, keepdim=True)
                / self.proxy_W.norm(p=2, dim=1, keepdim=True)
            )
            * delta_W
        )

        return update_feature_mb

    def get_cosine(self, embeddings):
        if self._no_grad_W:
            return self.distance(embeddings, self.W.t().detach())
        return super().get_cosine(embeddings)

    @torch.no_grad()
    def _record_angles_to_statedict(self, embeddings, labels):
        if not self._record_angles:
            raise RuntimeError("_recod_angles must be True to use this method")
        cosine = self.get_cosine(embeddings)
        angles = torch.acos(torch.clamp(cosine, -1, 1))

        labels_onehot = torch.nn.functional.one_hot(labels).bool()

        angles_of_target_classes = angles[labels_onehot]
        self._mean_target_angles = torch.cat(
            (self._mean_target_angles, angles_of_target_classes.mean().unsqueeze(0))
        )
        self._median_target_angles = torch.cat(
            (self._median_target_angles, angles_of_target_classes.median().unsqueeze(0))
        )

        mean_angle_of_nontarget_classes = torch.mean(angles[~labels_onehot])
        self._mean_nontarget_angles = torch.cat(
            (self._mean_nontarget_angles, mean_angle_of_nontarget_classes.unsqueeze(0))
        )

    def forward(self, embeddings, labels, *args, **kwargs):
        if self._record_angles:
            self._record_angles_to_statedict(embeddings, labels)

        # minibatch loss should only update encoder parameters.
        # this prevents calculating W gradients, therefore preventing update on W
        self._no_grad_W = True
        batch_loss = super().forward(embeddings, labels, *args, **kwargs)

        # broad loss should only update W parameters, so embeddings should be detached
        self._no_grad_W = False
        broad_embeddings = torch.cat(
            [self.get_queued_embeddings(), embeddings.detach()], dim=0
        )
        broad_labels = torch.cat([self.label_mb, labels], dim=0)
        broad_loss = super().forward(broad_embeddings, broad_labels)
        self.update_queue(embeddings.detach(), labels)

        if isinstance(batch_loss, torch.Tensor) and batch_loss.numel() > 1:
            return batch_loss, broad_loss

        return batch_loss + broad_loss


class BroadFaceAdaFaceLoss(AdaFaceLoss):
    def __init__(
        self,
        num_classes,
        embedding_size,
        margin=0.4,
        scale=64,
        h=0.333,
        t_alpha=0.99,
        queue_size=10000,
        compensate=True,
        **kwargs,
    ):
        super().__init__(
            num_classes, embedding_size, margin, scale, h, t_alpha, **kwargs
        )

        self.queue_size = queue_size
        self.compensate = compensate

        self.register_buffer("feature_mb", torch.zeros(0, embedding_size))
        self.register_buffer("label_mb", torch.zeros(0, dtype=torch.int64))
        self.register_buffer("proxy_W", torch.zeros(0, embedding_size))

        self._no_grad_W = False

    # pylint: disable=attribute-defined-outside-init
    def load_state_dict(
        self, state_dict: "OrderedDict[str, Tensor]", strict: bool = True
    ):
        if (feature_mb := state_dict.get("feature_mb")) is None:
            raise RuntimeError("Key feature_mb does not exist in state_dict.")
        if (label_mb := state_dict.get("label_mb")) is None:
            raise RuntimeError("Key label_mb does not exist in state_dict.")
        if (proxy_W := state_dict.get("proxy_W")) is None:
            raise RuntimeError("Key proxy_W does not exist in state_dict.")

        if not (feature_mb.shape[0] == label_mb.shape[0] == proxy_W.shape[0]):
            raise RuntimeError(
                f"Expected same length of {feature_mb.shape[0]} for feature_mb, label_mb"
                f" and proxy_W on state_dict, got {feature_mb.shape[0]}, "
                f"{label_mb.shape[0]}, and {proxy_W.shape[0]}"
            )

        self.feature_mb = torch.zeros_like(feature_mb)
        self.label_mb = torch.zeros_like(label_mb)
        self.proxy_W = torch.zeros_like(proxy_W)

        return super().load_state_dict(state_dict, strict)

    @torch.no_grad()
    def update_queue(self, embeddings, labels):
        self.feature_mb = torch.cat([self.feature_mb, embeddings.detach()], dim=0)
        self.label_mb = torch.cat([self.label_mb, labels.detach()], dim=0)
        self.proxy_W = torch.cat(
            [self.proxy_W, self.W.detach().T[labels].clone()], dim=0
        )

        over_size = self.feature_mb.shape[0] - self.queue_size
        if over_size > 0:
            self.feature_mb = self.feature_mb[over_size:]
            self.label_mb = self.label_mb[over_size:]
            self.proxy_W = self.proxy_W[over_size:]

        assert (
            self.feature_mb.shape[0] == self.label_mb.shape[0] == self.proxy_W.shape[0]
        )

    # pylint: enable=attribute-defined-outside-init
    def get_queued_embeddings(self):
        if not self.compensate:
            return self.feature_mb

        curr_W = self.W.detach().T[self.label_mb]
        delta_W = curr_W - self.proxy_W

        update_feature_mb = (
            self.feature_mb
            + (
                self.feature_mb.norm(p=2, dim=1, keepdim=True)
                / self.proxy_W.norm(p=2, dim=1, keepdim=True)
            )
            * delta_W
        )

        return update_feature_mb

    def get_cosine(self, embeddings):
        if self._no_grad_W:
            return self.distance(embeddings, self.W.t().detach())
        return super().get_cosine(embeddings)

    def forward(self, embeddings, labels, *args, **kwargs):
        # minibatch loss should only update encoder parameters.
        # this prevents calculating W gradients, therefore preventing update on W
        self._no_grad_W = True
        batch_loss = super().forward(embeddings, labels, *args, **kwargs)

        # broad loss should only update W parameters, so embeddings should be detached
        self._no_grad_W = False
        broad_embeddings = torch.cat(
            [self.get_queued_embeddings(), embeddings.detach()], dim=0
        )
        broad_labels = torch.cat([self.label_mb, labels], dim=0)
        broad_loss = super().forward(broad_embeddings, broad_labels)
        self.update_queue(embeddings.detach(), labels)

        if isinstance(batch_loss, torch.Tensor) and batch_loss.numel() > 1:
            return batch_loss, broad_loss

        return batch_loss + broad_loss
