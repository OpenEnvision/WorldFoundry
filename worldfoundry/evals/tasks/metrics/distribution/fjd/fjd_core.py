# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
from typing import ClassVar

import numpy as np
import torch
from scipy import linalg
from tqdm import tqdm


class FJDMetric:
    """Helper function for calculating FJD metric.

    Args:
        gan: Model that takes in a conditioning tensor and yields image samples.
        reference_loader: DataLoader that yields (images, conditioning) pairs
            to be used as the reference distribution.
        condition_loader: Dataloader that yields (image, conditioning) pairs.
            Images are ignored, and conditions are fed to the GAN.
        image_embedding: Function that takes in 4D [B, 3, H, W] image tensor
            and yields 2D [B, D] embedding vectors.
        condition_embedding: Function that takes in conditioning from
            condition_loader and yields 2D [B, D] embedding vectors.
        reference_stats_path: File path to save precomputed statistics of
            reference distribution. Default: current directory.
        save_reference_stats: Boolean indicating whether statistics of
            reference distribution should be saved. Default: False.
        samples_per_condition: Integer indicating the number of samples to
            generate for each condition from the condition_loader. Default: 1.
        cuda: Boolean indicating whether to use GPU accelerated FJD or not.
              Default: False.
        eps: Float value which is added to diagonals of covariance matrices
             to improve computational stability. Default: 1e-6.
    """

    def __init__(
        self,
        gan,
        reference_loader,
        condition_loader,
        image_embedding,
        condition_embedding,
        reference_stats_path="",
        save_reference_stats=False,
        samples_per_condition=1,
        cuda=False,
        eps=1e-6,
    ):
        self.gan = gan
        self.reference_loader = reference_loader
        self.condition_loader = condition_loader
        self.image_embedding = image_embedding
        self.condition_embedding = condition_embedding
        self.reference_stats_path = reference_stats_path
        self.save_reference_stats = save_reference_stats
        self.samples_per_condition = samples_per_condition
        self.cuda = cuda
        self.eps = eps

        self.mu_fake, self.sigma_fake = None, None
        self.mu_real, self.sigma_real = None, None

    def _get_joint_statistics(self, image_embed, cond_embed):
        if self.cuda:
            joint_embed = torch.cat([image_embed, cond_embed], dim=1)
        else:
            joint_embed = np.concatenate([image_embed, cond_embed], axis=1)
        mu, sigma = get_embedding_statistics(joint_embed, cuda=self.cuda)
        return mu, sigma

    def _calculate_alpha(self, image_embed, cond_embed):
        self.alpha = calculate_alpha(image_embed, cond_embed, cuda=self.cuda)
        return self.alpha

    def _get_generated_distribution(self):
        image_embed = []
        cond_embed = []

        for _i, data in tqdm(
            enumerate(self.condition_loader), desc="Computing generated distribution", total=len(self.condition_loader)
        ):
            _, condition = data  # it is assumed data contains (image, condition)
            condition = condition.cuda()

            with torch.no_grad():
                for _n in range(self.samples_per_condition):
                    image = self.gan(condition)

                    img_e = self.image_embedding(image)
                    cond_e = self.condition_embedding(condition)

                    if self.cuda:
                        image_embed.append(img_e)
                        cond_embed.append(cond_e)
                    else:
                        image_embed.append(img_e.cpu().numpy())
                        cond_embed.append(cond_e.cpu().numpy())

        if self.cuda:
            image_embed = torch.cat(image_embed, dim=0)
            cond_embed = torch.cat(cond_embed, dim=0)
        else:
            image_embed = np.concatenate(image_embed, axis=0)
            cond_embed = np.concatenate(cond_embed, axis=0)

        mu_fake, sigma_fake = self._get_joint_statistics(image_embed, cond_embed)
        del image_embed
        del cond_embed

        self.mu_fake, self.sigma_fake = mu_fake, sigma_fake
        return mu_fake, sigma_fake

    def _get_reference_distribution(self):
        if self.reference_stats_path:
            if os.path.isfile(self.reference_stats_path):
                stats = self._get_statistics_from_file(self.reference_stats_path)
                mu_real, sigma_real, alpha = stats
                self.alpha = alpha
            else:
                mu_real, sigma_real = self._compute_reference_distribution()
                if self.save_reference_stats:
                    self._save_activation_statistics(mu_real, sigma_real, self.alpha)
        else:
            mu_real, sigma_real = self._compute_reference_distribution()

        self.mu_real, self.sigma_real = mu_real, sigma_real
        return mu_real, sigma_real

    def _compute_reference_distribution(self):
        image_embed = []
        cond_embed = []

        for data in tqdm(self.reference_loader, desc="Computing reference distribution"):
            image, condition = data
            image = image.cuda()
            condition = condition.cuda()

            with torch.no_grad():
                image = self.image_embedding(image)
                condition = self.condition_embedding(condition)

                if self.cuda:
                    image_embed.append(image)
                    cond_embed.append(condition)
                else:
                    image_embed.append(image.cpu().numpy())
                    cond_embed.append(condition.cpu().numpy())

        if self.cuda:
            image_embed = torch.cat(image_embed, dim=0)
            cond_embed = torch.cat(cond_embed, dim=0)
        else:
            image_embed = np.concatenate(image_embed, axis=0)
            cond_embed = np.concatenate(cond_embed, axis=0)

        self._calculate_alpha(image_embed, cond_embed)
        mu_real, sigma_real = self._get_joint_statistics(image_embed, cond_embed)
        del image_embed
        del cond_embed

        return mu_real, sigma_real

    def _save_activation_statistics(self, mu, sigma, alpha):
        if self.cuda:
            mu = mu.cpu().numpy()
            sigma = sigma.cpu().numpy()

        np.savez(self.reference_stats_path, mu=mu, sigma=sigma, alpha=alpha)

    def _get_statistics_from_file(self, path):
        print(f"Loading reference statistics from {path}")
        assert path.endswith(".npz"), f'Invalid filepath "{path}". Should be .npz'

        f = np.load(path)
        mu, sigma, alpha = f["mu"][:], f["sigma"][:], f["alpha"]
        f.close()

        if self.cuda:
            mu = torch.tensor(mu).cuda()
            sigma = torch.tensor(sigma).cuda()
            alpha = torch.tensor(alpha).cuda()

        return mu, sigma, alpha

    def _scale_statistics(self, mu1, sigma1, mu2, sigma2, alpha):
        # Perform scaling operations directly on the precomputed mean and
        # covariance matrices, rather than scaling the conditioning embeddings
        # and recomputing mu and sigma

        if self.cuda:
            mu1, mu2 = mu1.clone(), mu2.clone()
            sigma1, sigma2 = sigma1.clone(), sigma2.clone()
        else:
            mu1, mu2 = np.copy(mu1), np.copy(mu2)
            sigma1, sigma2 = np.copy(sigma1), np.copy(sigma2)

        mu1[2048:] = mu1[2048:] * alpha
        mu2[2048:] = mu2[2048:] * alpha

        sigma1[2048:, 2048:] = sigma1[2048:, 2048:] * alpha**2
        sigma1[2048:, :2048] = sigma1[2048:, :2048] * alpha
        sigma1[:2048, 2048:] = sigma1[:2048, 2048:] * alpha

        sigma2[2048:, 2048:] = sigma2[2048:, 2048:] * alpha**2
        sigma2[2048:, :2048] = sigma2[2048:, :2048] * alpha
        sigma2[:2048, 2048:] = sigma2[:2048, 2048:] * alpha

        return mu1, sigma1, mu2, sigma2

    def get_fjd(self, alpha=None, resample=True):
        """Calculate FJD.

        Args:
            alpha (float): Scaling factor for the conditioning embedding. If
                None, alpha is set to be the ratio between the average norm of
                the image embedding and conditioning embedding. Default: None.
            resample (bool): If True, draws new samples from GAN and recomputes
                generated distribution statistics. Default: True.

        Returns:
            FJD value.
        """

        if self.mu_real is None:
            self._get_reference_distribution()
            self._get_generated_distribution()
        elif resample:
            self._get_generated_distribution()

        if alpha is None:
            alpha = self.alpha

        m1, s1, m2, s2 = self._scale_statistics(self.mu_real, self.sigma_real, self.mu_fake, self.sigma_fake, alpha)
        fjd = calculate_fd(m1, s1, m2, s2, cuda=self.cuda, eps=self.eps)
        return fjd

    def get_fid(self, resample=True):
        """Calculate FID (equivalent to FJD at alpha = 0).

        Args:
            resample (bool): If True, draws new samples from GAN and recomputes
                generated distribution statistics. Default: True.

        Returns:
            FID value.
        """
        fid = self.get_fjd(alpha=0.0, resample=resample)
        return fid

    _DEFAULT_ALPHAS: ClassVar[list[float]] = []

    def sweep_alpha(self, alphas=_DEFAULT_ALPHAS):
        """Calculate FJD at a range of alpha values.

        Args:
            alphas (list of floats): Values of alpha with which FJD will be
                calculated.

        Returns:
            List of FJD values.
        """

        fjds = []

        for i, alpha in enumerate(alphas):
            fjd = self.get_fjd(alpha, resample=(i == 0))
            fjds.append(fjd)

        return fjds


def get_embedding_statistics(embeddings, cuda=False):
    if cuda:
        embeddings = embeddings.double()  # More precision = more stable
        mu = torch.mean(embeddings, 0)
        sigma = torch_cov(embeddings, rowvar=False)
    else:
        mu = np.mean(embeddings, axis=0)
        sigma = np.cov(embeddings, rowvar=False)
    return mu, sigma


def calculate_alpha(image_embed, cond_embed, cuda=False):
    if cuda:
        image_norm = torch.mean(torch.norm(image_embed, dim=1))
        cond_norm = torch.mean(torch.norm(cond_embed, dim=1))
        alpha = (image_norm / cond_norm).item()
    else:
        image_norm = np.mean(linalg.norm(image_embed, axis=1))
        cond_norm = np.mean(linalg.norm(cond_embed, axis=1))
        alpha = image_norm / cond_norm
    return alpha


def calculate_fd(mu1, sigma1, mu2, sigma2, cuda=False, eps=1e-6):
    if cuda:
        fid = torch_calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=eps)
        fid = fid.cpu().numpy()
    else:
        fid = numpy_calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=eps)
    return fid


# A pytorch implementation of cov, from Modar M. Alfadly
# https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/2
def torch_cov(m, rowvar=False):
    """Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
            m: A 1-D or 2-D array containing multiple variables and observations.
                    Each row of `m` represents a variable, and each column a single
                    observation of all those variables.
            rowvar: If `rowvar` is True, then each row represents a
                    variable, with observations in the columns. Otherwise, the
                    relationship is transposed: each column represents a variable,
                    while the rows contain observations.

    Returns:
            The covariance matrix of the variables.
    """
    if m.dim() > 2:
        raise ValueError("m has more than 2 dimensions")
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()


# Pytorch implementation of matrix sqrt, from Tsung-Yu Lin, and Subhransu Maji
# https://github.com/msubhransu/matrix-sqrt
def sqrt_newton_schulz(A, numIters, dtype=None):
    with torch.no_grad():
        if dtype is None:
            dtype = A.type()
        batchSize = A.shape[0]
        dim = A.shape[1]
        normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt()
        Y = A.div(normA.view(batchSize, 1, 1).expand_as(A))
        identity = torch.eye(dim, dim).view(1, dim, dim).repeat(batchSize, 1, 1).type(dtype)
        Z = torch.eye(dim, dim).view(1, dim, dim).repeat(batchSize, 1, 1).type(dtype)
        for _i in range(numIters):
            T = 0.5 * (3.0 * identity - Z.bmm(Y))
            Y = Y.bmm(T)
            Z = T.bmm(Z)
        sA = Y * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
    return sA


def numpy_calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    Taken from https://github.com/bioinf-jku/TTUR
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
                    d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
                         inception net (like returned by the function 'get_predictions')
                         for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
                         representive data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
                         representive data set.
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
            # raise ValueError('Imaginary component {}'.format(m))
            print(f"Imaginary component of {m}, may affect results")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    out = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return out


# PyTorch implementation of Frechet distance, from Andrew Brock (modified slightly)
# https://github.com/ajbrock/BigGAN-PyTorch/blob/master/inception_utils.py
def torch_calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Pytorch implementation of the Frechet Distance.
    Taken from https://github.com/bioinf-jku/TTUR
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
          d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
             inception net (like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
             representive data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
             representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    # Using double precision instead of float seems to make the GPU FD more stable
    mu1, mu2 = mu1.double(), mu2.double()
    sigma1, sigma2 = sigma1.double(), sigma2.double()

    # Add a tiny offset to the covariance matrices to make covmean estimate more stable
    # Will change the output by a couple decimal places compared to not doing this
    offset = torch.eye(sigma1.size(0)).cuda().double() * eps
    sigma1, sigma2 = sigma1 + offset, sigma2 + offset

    diff = mu1 - mu2

    # Run 50 itrs of newton-schulz to get the matrix sqrt of sigma1 dot sigma2
    covmean = sqrt_newton_schulz(sigma1.mm(sigma2).unsqueeze(0), 50).squeeze()
    tr_covmean = torch.trace(covmean)

    out = diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean
    return out
