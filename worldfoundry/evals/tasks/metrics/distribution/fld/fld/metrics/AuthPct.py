# Faster implementation of Authenticity metric defined here https://arxiv.org/abs/2102.08921

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..utils import shuffle
    from .Metric import Metric
else:
    from fld.metrics.Metric import Metric
    from fld.utils import shuffle

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Of samples to take from the compared train/gen sets (full sets aren't used for memory reasons)
SIZE = 10000


class AuthPct(Metric):
    """
    Computes the % of samples where the distance to the sample's nearest neighbor in the train set
    is smaller than the distance between that train sample and its nearest train sample
    """

    def __init__(self):
        super().__init__()
        self.name = "AuthPct"

    def compute_metric(
        self,
        train_feat,
        test_feat,  # Test samples not used by AuthPct
        gen_feat,
    ):
        train_feat, gen_feat = train_feat.to(DEVICE), gen_feat.to(DEVICE)
        train_feat, gen_feat = shuffle(train_feat, SIZE), shuffle(gen_feat, SIZE)
        real_dists = torch.cdist(train_feat, train_feat)

        # Hacky way to get it to ignore distance to self in nearest neighbor calculation
        real_dists.fill_diagonal_(float("inf"))
        gen_dists = torch.cdist(train_feat, gen_feat)

        real_min_dists = real_dists.min(axis=0)
        gen_min_dists = gen_dists.min(dim=0)

        authen = real_min_dists.values[gen_min_dists.indices] < gen_min_dists.values
        return (100 * torch.sum(authen) / len(authen)).item() - 50
