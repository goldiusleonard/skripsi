from pytorch_metric_learning.losses.cosface_loss import CosFaceLoss
from pytorch_metric_learning.utils import (
    loss_and_miner_utils as lmu,
    common_functions as c_f,
)


class cosface_loss(CosFaceLoss):
    def __init__(self, *args, margin=0.35, scale=64, **kwargs):
        super().__init__(*args, margin=margin, scale=scale, **kwargs)

    # def compute_loss(self, embeddings, labels, indices_tuple):
    #     dtype, device = embeddings.dtype, embeddings.device
    #     self.cast_types(dtype, device)
    #     miner_weights = lmu.convert_to_weights(indices_tuple, labels, dtype=dtype)
    #     mask = self.get_target_mask(embeddings, labels)
    #     cosine = self.get_cosine(embeddings)
    #     cosine_of_target_classes = cosine[mask == 1]
    #     modified_cosine_of_target_classes = self.modify_cosine_of_target_classes(
    #         cosine_of_target_classes
    #     )
    #     diff = (modified_cosine_of_target_classes - cosine_of_target_classes).unsqueeze(
    #         1
    #     )
    #     logits = cosine + (mask * diff)
    #     logits = self.scale_logits(logits, embeddings)
    #     unweighted_loss = self.cross_entropy(logits, labels)
    #     miner_weighted_loss = unweighted_loss * miner_weights
    #     loss_dict = {
    #         "loss": {
    #             "losses": miner_weighted_loss,
    #             "indices": c_f.torch_arange_from_size(embeddings),
    #             "reduction_type": "element",
    #         }
    #     }
    #     self.add_weight_regularization_to_loss_dict(loss_dict, self.W.t())
    #     return loss_dict, logits

    def get_logits(self, embeddings):
        logits = self.get_cosine(embeddings)
        logits = self.scale_logits(logits, embeddings)
        return logits

    # def forward(self, embeddings, labels, indices_tuple=None):
    #     """
    #     Args:
    #         embeddings: tensor of size (batch_size, embedding_size)
    #         labels: tensor of size (batch_size)
    #         indices_tuple: tuple of size 3 for triplets (anchors, positives, negatives)
    #                         or size 4 for pairs (anchor1, postives, anchor2, negatives)
    #                         Can also be left as None
    #     Returns: the loss
    #     """
    #     self.reset_stats()
    #     c_f.check_shapes(embeddings, labels)
    #     labels = c_f.to_device(labels, embeddings)
    #     loss_dict, logits = self.compute_loss(embeddings, labels, indices_tuple)
    #     self.add_embedding_regularization_to_loss_dict(loss_dict, embeddings)
    #     return self.reducer(loss_dict, embeddings, labels), logits
