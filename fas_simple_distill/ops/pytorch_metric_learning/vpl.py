import torch
from torch import Tensor
from torch.nn import functional as F

from pytorch_metric_learning.losses import CosFaceLoss


class VPLCosFaceLoss(CosFaceLoss):
    def __init__(
        self,
        num_classes: int,
        embedding_size: int,
        start_iter: int,
        delta: int = 200,
        lambda_weight: float = 0.15,
        margin: float = 0.35,
        scale: float = 64,
        **kwargs
    ):
        super().__init__(
            num_classes, embedding_size, margin=margin, scale=scale, **kwargs
        )

        self.register_buffer("queue", F.normalize(torch.randn_like(self.W.T)))
        self.register_buffer(
            "queue_iters", torch.zeros((num_classes), dtype=torch.long)
        )
        self.register_buffer(
            "queue_lambda", torch.zeros((num_classes), dtype=torch.float32)
        )
        self.register_buffer("curr_iter", torch.tensor(0))
        self.curr_iter: Tensor

        self.start_iter = start_iter
        self.delta = delta
        self.lambda_weight = lambda_weight

        self._active_vpl = False

    @torch.no_grad()
    def prepare_queue_lambda(self):
        if self.curr_iter.item() >= self.start_iter:
            past_iters = self.curr_iter - self.queue_iters
            idx = torch.where(past_iters <= self.delta)[0]
            self.queue_lambda[idx] = self.lambda_weight

            self._active_vpl = True

    @torch.no_grad()
    def set_queue(self, embeddings, labels):
        normed_embeddings = F.normalize(embeddings)
        self.queue[labels] = normed_embeddings
        self.queue_iters[labels] = self.curr_iter.item()

    def get_cosine(self, embeddings):
        # workaround to allow correct results on get_logits without VPL interference
        if self.training and self._active_vpl:
            self._active_vpl = False

            lambda_ = self.queue_lambda.unsqueeze(0)
            injected_weight = self.W * (1.0 - lambda_) + self.queue.T * lambda_
            injected_weight = F.normalize(injected_weight)
            return self.distance(embeddings, injected_weight.t())

        return super().get_cosine(embeddings)

    def forward(self, embeddings, labels, *args, **kwargs):
        self.curr_iter += 1

        if self.training:
            self.prepare_queue_lambda()

        loss = super().forward(embeddings, labels, *args, **kwargs)
        self.set_queue(embeddings.detach(), labels)

        return loss
