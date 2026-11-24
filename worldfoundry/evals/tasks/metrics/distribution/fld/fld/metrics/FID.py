# Taken from https://github.com/mseitzer/pytorch-fid
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..utils import shuffle
    from .Metric import Metric
else:
    from fld.metrics.Metric import Metric
    from fld.utils import shuffle
from scipy import linalg


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = f"fid calculation produces singular product; adding {eps} to diagonal of cov estimates"
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def get_activation_statistics(act):
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


class FID(Metric):
    def __init__(self, ref_feat="train", ref_size=None):
        super().__init__()

        self.ref_feat = ref_feat  # One of ("train", "test")
        self.ref_size = ref_size

        if ref_size is None:
            self.name = f"FID {ref_feat.title()}"
        else:
            self.name = f"FID {ref_feat.title()} - {ref_size // 1000}k"

    def compute_metric(
        self,
        train_feat,
        test_feat,
        gen_feat,
    ):
        if self.ref_feat == "train":
            ref_feat = shuffle(train_feat, self.ref_size).cpu()
        elif self.ref_feat == "test":
            ref_feat = shuffle(test_feat, self.ref_size).cpu()
        else:
            raise ValueError("ref_feat must be one of 'train' or 'test'")

        gen_feat = shuffle(gen_feat, 50_000)
        mu1, sigma1 = get_activation_statistics(gen_feat.cpu().numpy())
        mu2, sigma2 = get_activation_statistics(ref_feat.cpu().numpy())
        return calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
