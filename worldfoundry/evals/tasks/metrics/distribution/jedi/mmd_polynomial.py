# The code is adapted from https://github.com/jindongwang/transferlearning/blob/master/code/distance/mmd_numpy_sklearn.py
# Compute MMD (maximum mean discrepancy) using numpy and scikit-learn.
import numpy as np

metrics = __import__("sklearn", fromlist=["metrics"]).metrics


def mmd_poly(X, Y, degree=2, gamma=None, coef0=0):
    """MMD using polynomial kernel (i.e., k(x,y) = (gamma <X, Y> + coef0)^degree)

    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    Keyword Arguments:
        degree {int} -- [degree] (default: {2})
        gamma {int} -- [gamma] (default: {1})
        coef0 {int} -- [constant item] (default: {0})

    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.polynomial_kernel(X, X, degree, gamma, coef0)
    YY = metrics.pairwise.polynomial_kernel(Y, Y, degree, gamma, coef0)
    XY = metrics.pairwise.polynomial_kernel(X, Y, degree, gamma, coef0)
    return XX.mean() + YY.mean() - 2 * XY.mean()


if __name__ == "__main__":
    a = np.arange(1, 10).reshape(3, 3)
    b = [[7, 6, 5], [4, 3, 2], [1, 1, 8], [0, 2, 5]]
    b = np.array(b)
    print(a)
    print(b)
    print(mmd_poly(a, b, degree=3, coef0=1))  # 2436.5
    print(mmd_poly(a, b, degree=2, coef0=0))  # 270.7
