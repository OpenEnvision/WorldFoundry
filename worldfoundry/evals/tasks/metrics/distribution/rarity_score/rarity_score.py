from typing import cast

import numpy as np
import torch

from worldfoundry.evals.tasks.metrics.distribution.improved_precision_recall.improved_precision_recall import (
    compute_pairwise_distances,
)


class MANIFOLD:
    @torch.no_grad()
    def __init__(self, real_features, fake_features, metric="euclidian", device="cpu"):
        self.metric = metric
        self.device = device

        self.real_features = real_features
        self.fake_features = fake_features

        self.num_reals = real_features.shape[0]
        self.num_fakes = fake_features.shape[0]

        print("Pre-processing pairwise diatances ...")
        self.real2real_distances = compute_pairwise_distances(real_features, metric=self.metric, device=self.device)

        # self.real2real_sorted = np.sort(self.real2real_distances, axis = 1)
        # self.real2real_sorted_ids = self.real2real_distances.argsort(axis = 1)
        self.real2real_sorted = self.real2real_distances.sort(dim=1)[0].detach().cpu().numpy()
        self.real2real_sorted_ids = self.real2real_distances.argsort(dim=1).detach().cpu().numpy()

        torch.cuda.empty_cache()

        # for clustering
        self.num_cluster = 0
        self.modes = None
        self.sample_mode_ids = None

        print("Initialization is DONE !")

    def is_in_ball(self, k=3, samples=None, cluster=False):
        """Compute the differences between radii of kNN balls and distances
        for judging whether they are in each kNN ball or not.

        args:
                k (int): real ball's size is distance between reference real sample
                                 and k th nearest real sample.
                samples (np.array, num_samples * embed_dim): embedded generation samples
        return:
                dist_radi (np.array, num_reals * num_samples): each element means
                                 (radii of a ball - (distance between center of the ball and a sample)).
                                 if it is larger than 0, it is out of the ball.
                r (np.array, num_reals * 1): radii of each kNN real balls
                out_ball_ids (np.array, num_out_ball_samples): indices of samples outside of balls.
        """
        if not cluster:
            real2samples_distances = compute_pairwise_distances(
                self.real_features, samples, metric=self.metric, device=self.device
            )
        else:
            real2samples_distances = compute_pairwise_distances(
                self.modes, samples, metric=self.metric, device=self.device
            )
        real2samples_distances = real2samples_distances.detach().cpu().numpy()

        # The upstream cluster path reads a mode-distance cache it does not initialize.
        r = self.real2real_sorted[:, k] if not cluster else self.mode2mode_sorted[:, k]  # ty: ignore[unresolved-attribute]

        dist_radi = r[:, None].repeat(cast(np.ndarray, samples).shape[0], axis=1) - real2samples_distances
        out_ball_ids = np.where(np.logical_not((dist_radi > 0).any(axis=0)))[0]

        return dist_radi, r, out_ball_ids

    def rarity(self, k=3, samples=None):
        """The larger the real ball's size, the rare the real sample would be.

        args:
                k (int): real ball's size is distance between reference real sample
                                 and k th nearest real sample.
                samples (np.array, N * embed_dim): embedded generation samples
        return:
                scores (np.array, num_samples): scores of each samples which are not sorted.
                scores_ids (np.array, num_samples_in_valid_ball): for samples in valid real balls,
                          sorted indices in decreasing order.
        """

        samples = self.fake_features if samples is None else samples
        in_ball_dist, r, out_ball_ids = self.is_in_ball(samples=samples, k=k)

        num_out_ball = len(out_ball_ids)
        valid_real_balls = in_ball_dist > 0

        scores = np.zeros(samples.shape[0])

        for i in range(samples.shape[0]):
            if i not in out_ball_ids:
                scores[i] = r[valid_real_balls[:, i]].min()

        scores_ids = (-scores).argsort()[: samples.shape[0] - num_out_ball]

        return scores, scores_ids
