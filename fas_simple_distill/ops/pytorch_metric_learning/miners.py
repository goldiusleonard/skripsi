from pytorch_metric_learning.miners import (
    BatchHardMiner,
    TripletMarginMiner,
)


class SpecificAnchorBatchHardMiner(BatchHardMiner):
    def __init__(self, anchor_label, **kwargs):
        super().__init__(**kwargs)
        self.anchor_label = anchor_label

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        a, p, n = super().mine(embeddings, labels, ref_emb, ref_labels)
        anchor_labels = labels == self.anchor_label
        valid_idxs = anchor_labels[a].nonzero()

        return a[valid_idxs], p[valid_idxs], n[valid_idxs]


class SpecificAnchorTripletMarginMiner(TripletMarginMiner):
    def __init__(self, anchor_label, **kwargs):
        super().__init__(**kwargs)
        self.anchor_label = anchor_label

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        a, p, n = super().mine(embeddings, labels, ref_emb, ref_labels)
        anchor_labels = labels == self.anchor_label
        valid_idxs = anchor_labels[a].nonzero()

        return a[valid_idxs], p[valid_idxs], n[valid_idxs]
