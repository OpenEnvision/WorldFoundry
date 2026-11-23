"""
mid.metric
Copyright (c) 2022-present NAVER Corp.
Apache-2.0
"""

from typing import *  # noqa: F403 - Preserve upstream module exports.

import torch
from torch import Tensor


def _rank_zero_info(msg: str) -> None:
    pass


def log_det(X):
    eigenvalues = X.svd()[1]
    return eigenvalues.log().sum()


def robust_inv(x, eps=0):
    Id = torch.eye(x.shape[0]).to(x.device)
    return (x + eps * Id).inverse()


def exp_smd(a, b, reduction=True):
    a_inv = robust_inv(a)
    if reduction:
        assert b.shape[0] == b.shape[1]
        return (a_inv @ b).trace()
    else:
        return (b @ a_inv @ b.t()).diag()


def _compute_pmi(
    x: Tensor, y: Tensor, x0: Tensor, limit: int = 30000, reduction: bool = True, full: bool = False
) -> Tensor:
    r"""
    A numerical stable version of the MID score.

    Args:
        x (Tensor): features for real samples
        y (Tensor): features for text samples
        x0 (Tensor): features for fake samples
        limit (int): limit the number of samples
        reduction (bool): returns the expectation of PMI if true else sample-wise results
        full (bool): use full samples from real images

    Returns:
        Scalar value of the mutual information divergence between the sets.
    """
    N = x.shape[0]
    excess = N - limit
    if excess > 0:
        if not full:
            x = x[:-excess]
            y = y[:-excess]
        x0 = x0[:-excess]
    N = x.shape[0]
    M = x0.shape[0]

    assert x.shape[1] <= N, "not full rank for matrix inversion!"
    if x.shape[0] < 30000:
        _rank_zero_info("if it underperforms, please consider to use the epsilon of 5e-4 or something else.")

    z = torch.cat([x, y], dim=-1)
    z0 = torch.cat([x0, y[: x0.shape[0]]], dim=-1)
    x_mean = x.mean(dim=0, keepdim=True)
    y_mean = y.mean(dim=0, keepdim=True)
    z_mean = torch.cat([x_mean, y_mean], dim=-1)
    _x0_mean = x0.mean(dim=0, keepdim=True)
    _z0_mean = z0.mean(dim=0, keepdim=True)

    X = (x - x_mean).t() @ (x - x_mean) / (N - 1)
    Y = (y - y_mean).t() @ (y - y_mean) / (N - 1)
    Z = (z - z_mean).t() @ (z - z_mean) / (N - 1)
    X0 = (x0 - x_mean).t() @ (x0 - x_mean) / (M - 1)  # use the reference mean
    Z0 = (z0 - z_mean).t() @ (z0 - z_mean) / (M - 1)  # use the reference mean

    alternative_comp = False
    # notice that it may have numerical unstability. we don't use this.
    if alternative_comp:

        def factorized_cov(x, m):
            N = x.shape[0]
            return (x.t() @ x - N * m.t() @ m) / (N - 1)

        X0 = factorized_cov(x0, x_mean)
        Z0 = factorized_cov(z0, z_mean)

    # assert double precision
    for _ in [X, Y, Z, X0, Z0]:
        assert _.dtype == torch.float64

    # Expectation of PMI
    mi = (log_det(X) + log_det(Y) - log_det(Z)) / 2
    _rank_zero_info(f"MI of real images: {mi:.4f}")

    # Squared Mahalanobis Distance terms
    if reduction:
        smd = (exp_smd(X, X0) + exp_smd(Y, Y) - exp_smd(Z, Z0)) / 2
    else:
        smd = (exp_smd(X, x0 - x_mean, False) + exp_smd(Y, y - y_mean, False) - exp_smd(Z, z0 - z_mean, False)) / 2
        mi = mi.unsqueeze(0)  # for broadcasting

    return mi + smd
