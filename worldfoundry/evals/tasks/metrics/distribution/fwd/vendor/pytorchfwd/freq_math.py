"""Wavelet utils."""

from itertools import product

import numpy as np
import ptwt
import pywt
import torch
from scipy import linalg


def get_freq_order(level: int):
    """Get the frequency order for a given packet decomposition level.

    Adapted from:
    https://github.com/PyWavelets/pywt/blob/master/pywt/_wavelet_packets.py
    The code elements denote the filter application order. The filters
    are named following the pywt convention as:
    a - LL, low-low coefficients
    h - LH, low-high coefficients
    v - HL, high-low coefficients
    d - HH, high-high coefficients
    """
    wp_natural_path = list(product(["a", "h", "v", "d"], repeat=level))

    def _get_graycode_order(level, x="a", y="d"):
        graycode_order = [x, y]
        for _ in range(level - 1):
            graycode_order = [x + path for path in graycode_order] + [y + path for path in graycode_order[::-1]]
        return graycode_order

    def _expand_2d_path(path):
        expanded_paths = {"d": "hh", "h": "hl", "v": "lh", "a": "ll"}
        return (
            "".join([expanded_paths[p][0] for p in path]),
            "".join([expanded_paths[p][1] for p in path]),
        )

    nodes: dict = {}
    for (row_path, col_path), node in [(_expand_2d_path(node), node) for node in wp_natural_path]:
        nodes.setdefault(row_path, {})[col_path] = node
    graycode_order = _get_graycode_order(level, x="l", y="h")
    nodes_list: list = [nodes[path] for path in graycode_order if path in nodes]
    wp_frequency_path = []
    for row in nodes_list:
        wp_frequency_path.append([row[path] for path in graycode_order if path in row])
    return wp_frequency_path, wp_natural_path


def forward_wavelet_packet_transform(
    tensor: torch.Tensor,
    wavelet: str,
    max_level: int,
    log_scale: bool,
) -> torch.Tensor:
    """Compute wavelet packet transform.

    Args:
        tensor (torch.Tensor): Input torch tensor
        wavelet (str): Choice of wavelet
        max_level (int): Level of decomposition
        log_scale (bool): Log scale boolean

    Returns:
        torch.Tensor: Packets
    """
    # ideally the output dtype should depend in the input.
    # tensor = tensor.type(torch.FloatTensor)
    packets = ptwt.WaveletPacket2D(tensor, pywt.Wavelet(wavelet), maxlevel=max_level)
    packet_list = [packets[node] for node in packets.get_natural_order(max_level)]

    # for node in packets.get_natural_order(max_level):
    # packet = torch.squeeze(packets[node], dim=1)
    # packet_list.append(packets[node])
    wp_pt_rs = torch.stack(packet_list, dim=1)
    if log_scale:
        wp_pt_rs = torch.log(torch.abs(wp_pt_rs) + 1e-6)

    return wp_pt_rs


def generate_frequency_packet_image(packet_array: np.ndarray, degree: int):
    """Create a ready-to-polt image with frequency-order packages.

       Given a packet array in natural order, creat an image which is
       ready to plot in frequency order.
    Args:
        packet_array (np.ndarray): [packet_no, packet_height, packet_width]
            in natural order.
        degree (int): The degree of the packet decomposition.
    Returns:
        [np.ndarray]: The image of shape [original_height, original_width]
    """
    wp_freq_path, wp_natural_path = get_freq_order(degree)

    image = []
    # go through the rows.
    for row_paths in wp_freq_path:
        row = []
        for row_path in row_paths:
            index = wp_natural_path.index(row_path)
            packet = packet_array[:, index]
            row.append(packet)
        image.append(np.concatenate(row, -1))
    return np.concatenate(image, 2)


def compute_kl_divergence(output: torch.Tensor, target: torch.Tensor, eps: float | None = 1e-30) -> torch.Tensor:
    """
    Args:
        output (torch.Tensor): Output Images
        target (torch.Tensor): Target Images
        eps (float, optional): Epsilon. Defaults to 1e-12.

    Returns:
        torch.Tensor: KL Divergence value
    """
    # Tried with eps 1e-30 and this improves the precision by a small margin but overall ranking remains the same
    return target * torch.log((target / (output + eps)) + eps)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Frechet Distance Implementation from https://github.com/bioinf-jku/TTUR/blob/master/fid.py.

    Numpy implementation of the Frechet Distance.
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

    Raises:
        ValueError: Value error if imaginary component has large value.

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
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
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
