"""PyTorch reimplementation from https://github.com/kynkaat/improved-precision-and-recall-metric"""

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .Metric import Metric
else:
    from fld.metrics.Metric import Metric

# Batch implementation for memory issues (equivalent)
BATCH_SIZE = 10000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PrecisionRecall(Metric):
    # We use 4 to ignore distance to self
    def __init__(self, mode, num_neighbors=4):
        super().__init__()

        self.name = mode  # One of ("Precision", "Recall")
        self.num_neighbors = num_neighbors

    def get_nn_dists(self, feat):
        dists = torch.zeros(feat.shape[0]).to(DEVICE)
        for i in range(math.ceil(feat.shape[0] / BATCH_SIZE)):
            start, end = i * BATCH_SIZE, (i + 1) * BATCH_SIZE
            curr_dists = torch.cdist(feat[start:end], feat)
            curr_dists = curr_dists.topk(self.num_neighbors, dim=1, largest=False).values
            dists[start:end] = curr_dists[:, -1]
        return dists

    def pct_in_manifold(self, evaluated_feat, manifold_feat):
        total_in_manifold = 0
        nn_dists = self.get_nn_dists(manifold_feat)

        for i in range(math.ceil(evaluated_feat.shape[0] / BATCH_SIZE)):
            start, end = i * BATCH_SIZE, (i + 1) * BATCH_SIZE
            pairwise_dists = torch.cdist(evaluated_feat[start:end], manifold_feat)
            comparison_tensor = nn_dists.unsqueeze(0).repeat(pairwise_dists.shape[0], 1)

            num_in_manifold = (pairwise_dists < comparison_tensor).sum(dim=1)
            num_in_manifold = (num_in_manifold > 0).sum()
            total_in_manifold += num_in_manifold

        return total_in_manifold / evaluated_feat.shape[0]

    def compute_metric(
        self,
        train_feat,
        test_feat,  # Test samples not used by Precision/Recall
        gen_feat,
    ):
        train_feat = train_feat.to(DEVICE)
        gen_feat = gen_feat.to(DEVICE)

        if self.name == "Precision":
            return self.pct_in_manifold(gen_feat, train_feat).item()
        elif self.name == "Recall":
            return self.pct_in_manifold(train_feat, gen_feat).item()
        else:
            raise NotImplementedError
