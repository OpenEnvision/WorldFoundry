import numpy as np
import scipy
import scipy.linalg

preprocessing = __import__("sklearn", fromlist=["preprocessing"]).preprocessing


def weight_K(K, p=None):
    if p is None:
        return K / K.shape[0]
    else:
        return K * np.outer(np.sqrt(p), np.sqrt(p))


def normalize_K(K):
    d = np.sqrt(np.diagonal(K))
    return K / np.outer(d, d)


def entropy_q(p, q=1):
    p_ = p[p > 0]
    if q == 1:
        return -(p_ * np.log(p_)).sum()
    if q == "inf":
        return -np.log(np.max(p))
    return np.log((p_**q).sum()) / (1 - q)


def score_K(K, q=1, p=None, normalize=False):
    if normalize:
        K = normalize_K(K)
    K_ = weight_K(K, p)
    if type(K_) is scipy.sparse.csr.csr_matrix:
        w, _ = scipy.sparse.linalg.eigsh(K_)
    else:
        w = scipy.linalg.eigvalsh(K_)
    return np.exp(entropy_q(w, q=q))


def score_X(X, q=1, p=None, normalize=True):
    if normalize:
        X = preprocessing.normalize(X, axis=1)
    K = X @ X.T
    return score_K(K, q=1, p=p)


def score_dual(X, q=1, normalize=True):
    if normalize:
        X = preprocessing.normalize(X, axis=1)
    n = X.shape[0]
    S = X.T @ X
    w = scipy.linalg.eigvalsh(S / n)
    _m = w > 0
    return np.exp(entropy_q(w, q=q))


def score(samples, k, q=1, p=None, normalize=False):
    n = len(samples)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            K[i, j] = K[j, i] = k(samples[i], samples[j])
    return score_K(K, p=p, q=q, normalize=normalize)


def intdiv_K(K, q=1, p=None):
    K_ = K**q
    if p is None:
        p = np.ones(K.shape[0]) / K.shape[0]
    return 1 - np.sum(K_ * np.outer(p, p))


def intdiv_X(X, q=1, p=None, normalize=True):
    if normalize:
        X = preprocessing.normalize(X, axis=1)
    K = X @ X.T
    return intdiv_K(K, q=q, p=p)


def intdiv(samples, k, q=1, p=None):
    n = len(samples)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            K[i, j] = K[j, i] = k(samples[i], samples[j])
    return intdiv_K(K, q=q, p=p)
