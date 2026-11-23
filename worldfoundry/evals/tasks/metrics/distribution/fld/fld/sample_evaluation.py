from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .MoG import MoG, compute_dists, preprocess_feat
    from .utils import shuffle
else:
    from fld.MoG import MoG, compute_dists, preprocess_feat
    from fld.utils import shuffle

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 10000


def sample_quality_scores(train_feat, test_feat, gen_feat, batch_size=BATCH_SIZE):
    """
    Gets quality score for each sample. Quality score is the dimension-adjusted LL of a generated sample from MoG centered at test samples fit to the train samples.

    A higher score (i.e. LL) would indicate a higher quality sample
    """
    train_feat, test_feat, gen_feat = preprocess_feat(train_feat, test_feat, gen_feat)
    train_feat = shuffle(train_feat)
    test_feat = shuffle(test_feat)

    mog_test = MoG(test_feat)
    mog_test.fit(train_feat)

    scores = []
    for batch in gen_feat.split(batch_size):
        lls = mog_test.lls(batch)
        scores.append(lls)

    scores = torch.cat(scores)
    return scores / train_feat.shape[1]


def sample_memorization_scores(train_feat, test_feat, gen_feat, batch_size=BATCH_SIZE):
    """
    Gets memorizations score for each sample.

    Memorization score for a generated sample is the max LL of a train sample for a Gaussian centered at the generated sample fit to the train samples.

    A higher score (i.e. LL) indicates a more memorized sample
    """
    train_feat, test_feat, gen_feat = preprocess_feat(train_feat, test_feat, gen_feat)
    mog_gen = MoG(gen_feat)
    mog_gen.fit(train_feat)

    scores = float("-inf") * torch.ones((len(gen_feat),), device=DEVICE)
    for batch in train_feat.split(batch_size):
        dists = mog_gen.dists(batch)
        pairwise_lls = mog_gen.pairwise_lls_from_dists(dists, mog_gen.log_sigmas)
        scores = torch.maximum(scores, pairwise_lls.max(dim=0).values)

    return scores / train_feat.shape[1]


def nn(base_feat, other_feat, batch_size=BATCH_SIZE):
    """
    Returns idxs, dists of nearest neighbor of base_feat in other_feat
    """
    base_feat, other_feat = base_feat.to(DEVICE), other_feat.to(DEVICE)
    min_idxs = torch.zeros((len(base_feat),), dtype=torch.long).to(DEVICE)
    min_dists = float("inf") * torch.ones((len(base_feat),)).to(DEVICE)

    for i, batch in enumerate(other_feat.split(batch_size)):
        dists = compute_dists(batch, base_feat)
        curr_min_dists = dists.min(dim=0)

        smaller = curr_min_dists.values < min_dists
        min_idxs[smaller] = curr_min_dists.indices[smaller] + i * batch_size
        min_dists[smaller] = curr_min_dists.values[smaller]

    return min_idxs, min_dists
