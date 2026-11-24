"""Vendored linear separability helpers from NVlabs/stylegan."""

import numpy as np


def prob_normalize(p):
    p = np.asarray(p).astype(np.float32)
    assert len(p.shape) == 2
    return p / np.sum(p)


def mutual_information(p):
    p = prob_normalize(p)
    px = np.sum(p, axis=1)
    py = np.sum(p, axis=0)
    result = 0.0
    for x in range(p.shape[0]):
        p_x = px[x]
        for y in range(p.shape[1]):
            p_xy = p[x][y]
            p_y = py[y]
            if p_xy > 0.0:
                result += p_xy * np.log2(p_xy / (p_x * p_y))  # get bits as output
    return result


def entropy(p):
    p = prob_normalize(p)
    result = 0.0
    for x in range(p.shape[0]):
        for y in range(p.shape[1]):
            p_xy = p[x][y]
            if p_xy > 0.0:
                result -= p_xy * np.log2(p_xy)
    return result


def conditional_entropy(p):
    # H(Y|X) where X corresponds to axis 0, Y to axis 1
    # i.e., How many bits of additional information are needed to where we are on axis 1 if we know where we are on axis 0?
    p = prob_normalize(p)
    y = np.sum(p, axis=0, keepdims=True)  # marginalize to calculate H(Y)
    return max(0.0, entropy(y) - mutual_information(p))  # can slip just below 0 due to FP inaccuracies, clean those up.


def compute_linear_separability(confusion_matrix: np.ndarray) -> float:
    """Compute StyleGAN linear separability score from a 2x2 confusion matrix."""
    ce = conditional_entropy(confusion_matrix)
    return float(2**ce)
