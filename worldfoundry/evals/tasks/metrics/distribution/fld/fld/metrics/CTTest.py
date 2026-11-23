# Taken from https://github.com/casey-meehan/data-copying
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import mannwhitneyu

if TYPE_CHECKING:
    from .Metric import Metric
else:
    from fld.metrics.Metric import Metric

KMeans = __import__("sklearn.cluster", fromlist=["KMeans"]).KMeans
PCA = __import__("sklearn.decomposition", fromlist=["PCA"]).PCA
NN = __import__("sklearn.neighbors", fromlist=["NearestNeighbors"]).NearestNeighbors


def Zu(Pn, Qm, T):
    """Extracts distances to training nearest neighbor
    L(P_n), L(Q_m), and runs Z-scored Mann Whitney U-test.
    For the global test, this is used on the samples within each cell.

    Inputs:
        Pn: (n X d) np array representing test sample of
            length n (with dimension d)

        Qm: (m X d) np array representing generated sample of
            length n (with dimension d)

        T: (l X d) np array representing training sample of
            length l (with dimension d)

    Ouptuts:
        Zu: Z-scored U value. A large value >>0 indicates
            underfitting by Qm. A small value <<0 indicates.
    """
    m = Qm.shape[0]
    n = Pn.shape[0]

    # fit NN model to training sample to get distances to test and generated samples
    T_NN = NN(n_neighbors=1).fit(T)
    LQm, _ = T_NN.kneighbors(X=Qm, n_neighbors=1)
    LPn, _ = T_NN.kneighbors(X=Pn, n_neighbors=1)

    # Get Mann-Whitney U score and manually Z-score it using the conditions of null hypothesis H_0
    u, _ = mannwhitneyu(LQm, LPn, alternative="less")
    mean = (n * m / 2) - 0.5  # 0.5 is continuity correction
    std = np.sqrt(n * m * (n + m + 1) / 12)
    Z_u = (u - mean) / std
    return Z_u


def Zu_cells(Pn, Pn_cells, Qm, Qm_cells, T, T_cells):
    """Collects the Zu statistic in each of k cells.
    There should be >0 test (Pn) and train (T) samples in each of the cells.

    Inputs:
        Pn: (n X d) np array representing test sample of length
            n (with dimension d)

        Pn_cells: (1 X n) np array of integers indicating which
            of the k cells each sample belongs to

        Qm: (m X d) np array representing generated sample of
            length n (with dimension d)

        Qm_cells: (1 X m) np array of integers indicating which of the
            k cells each sample belongs to

        T: (l X d) np array representing training sample of
            length l (with dimension d)

        T_cells: (1 X l) np array of integers indicating which of the
            k cells each sample belongs to

    Outputs:
        Zus: length k np array, where entry i indicates the Zu score for cell i
    """
    # assume cells are labeled 0 to k-1
    k = len(np.unique(Pn_cells))
    Zu_cells = np.zeros(k)

    # get samples in each cell and collect Zu
    for i in range(k):
        Pn_cell_i = Pn[Pn_cells == i]
        Qm_cell_i = Qm[Qm_cells == i]
        T_cell_i = T[T_cells == i]
        # check that the cell has test and training samples present
        if len(Pn_cell_i) * len(T_cell_i) == 0:
            raise ValueError(
                f"Cell {i:n} lacks test samples and/or training samples. Consider reducing the number of cells in partition."
            )

        # if there are no generated samples present, add a 0 for Zu. This cell will be excluded in \Pi_\tau
        if len(Qm_cell_i) > 0:
            Zu_cells[i] = Zu(Pn_cell_i, Qm_cell_i, T_cell_i)
        else:
            Zu_cells[i] = 0
            print(f"cell {i:n} unrepresented by Qm")

    return Zu_cells


def C_T(Pn, Pn_cells, Qm, Qm_cells, T, T_cells, tau):
    """Runs C_T test given samples and their respective cell labels.
    The C_T statistic is a weighted average of the in-cell Zu statistics, weighted
    by the share of test samples (Pn) in each cell. Cells with an insufficient number
    of generated samples (Qm) are not included in the statistic.

    Inputs:
        Pn: (n X d) np array representing test sample of length
            n (with dimension d)

        Pn_cells: (1 X n) np array of integers indicating which
            of the k cells each sample belongs to

        Qm: (m X d) np array representing generated sample of
            length n (with dimension d)

        Qm_cells: (1 X m) np array of integers indicating which of the
            k cells each sample belongs to

        T: (l X d) np array representing training sample of
            length l (with dimension d)

        T_cells: (1 X l) np array of integers indicating which of the
            k cells each sample belongs to

        tau: (scalar between 0 and 1) fraction of Qm samples that a
            cell needs to be included in C_T statistic.

    Outputs:
        C_T: The C_T statistic for the three samples Pn, Qm, T
    """

    m = Qm.shape[0]
    n = Pn.shape[0]
    k = np.max(np.unique(T_cells)) + 1  # number of cells

    # First, determine which of the cells have sufficient generated samples (Qm(pi) > tau)
    labels, cts = np.unique(Qm_cells, return_counts=True)
    Qm_cts = np.zeros(k)
    Qm_cts[labels.astype(int)] = cts  # put in order of cell label
    Qm_of_pi = Qm_cts / m
    Pi_tau = Qm_of_pi > tau  # binary array selecting which cells have sufficient samples

    # Get the fraction of test samples in each cell Pn(pi)
    labels, cts = np.unique(Pn_cells, return_counts=True)
    Pn_cts = np.zeros(k)
    Pn_cts[labels.astype(int)] = cts  # put in order of cell label
    Pn_of_pi = Pn_cts / n

    # Now get the in-cell Zu scores
    Zu_scores = Zu_cells(Pn, Pn_cells, Qm, Qm_cells, T, T_cells)

    # compute C_T:
    C_T = Pn_of_pi[Pi_tau].dot(Zu_scores[Pi_tau]) / np.sum(Pn_of_pi[Pi_tau])

    return C_T


class CTTest(Metric):
    def __init__(self):
        super().__init__()
        self.name = r"$C_T$"

    def compute_metric(
        self,
        train_feat,
        test_feat,
        gen_feat,
    ):
        if train_feat.shape[1] > 64:
            pca_xf = PCA(n_components=64).fit(train_feat)

            train_feat = pca_xf.transform(train_feat)
            test_feat = pca_xf.transform(test_feat)
            gen_feat = pca_xf.transform(gen_feat)

        km_clf = KMeans(n_clusters=3, n_init=10).fit(train_feat)

        T_labels = km_clf.predict(train_feat)
        Pn_labels = km_clf.predict(test_feat)
        Qm_labels = km_clf.predict(gen_feat)
        C_T_val = C_T(
            test_feat,
            Pn_labels,
            gen_feat,
            Qm_labels,
            train_feat,
            T_labels,
            tau=20 / len(gen_feat),
        )

        return C_T_val
