import numpy as np
from scipy.spatial.distance import cdist


def compute_mle(dists):
    epsilon = 1.0e-10
    dists = dists + epsilon
    k = len(dists)
    log_vals = np.log(dists / (dists[-1] + epsilon))
    log_sum = np.sum(log_vals)
    lid = -1.0 * k / (log_sum)
    return lid


def compute_crosslid_batch(X_ref, X, k, batch_size):
    batch_selection = np.random.choice(X_ref.shape[0], batch_size, False)
    X_ref_batch = X_ref[batch_selection]
    dist = cdist(X, X_ref_batch)
    sorted = np.sort(dist, axis=1)
    least_k = sorted[:, 0 : k - 1]
    lid_batch = np.apply_along_axis(func1d=compute_mle, axis=1, arr=least_k)
    return lid_batch


def compute_crosslid(X, Y, k, batch_size):
    X = X.reshape(X.shape[0], -1)
    Y = Y.reshape(Y.shape[0], -1)
    n_batches = int(np.ceil(Y.shape[0] / batch_size))
    lid_vals = []
    for i in range(n_batches):
        start = i * batch_size
        end = np.min([(i + 1) * batch_size, Y.shape[0]])
        Y_batch = Y[start:end]
        lid_batch = compute_crosslid_batch(X, Y_batch, k, batch_size)
        lid_vals.extend(lid_batch)
    return np.mean(np.array(lid_vals))


def scale_value(X, to_min_max):
    from_min_max = get_min_max(X)
    X = X - from_min_max[0]
    X = X / (from_min_max[1] - from_min_max[0])
    X = X * (to_min_max[1] - to_min_max[0])
    X = X + to_min_max[0]
    return X


def get_min_max(inp):
    minv = np.min(inp, axis=0)
    minv = np.min(minv)
    minv = np.min(minv)
    minv = np.min(minv)
    maxv = np.max(inp, axis=0)
    maxv = np.max(maxv)
    maxv = np.max(maxv)
    maxv = np.max(maxv)
    return [minv, maxv]
