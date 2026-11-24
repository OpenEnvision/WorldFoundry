import os

import numpy as np
import torchvision.transforms as transforms
from scipy import special
from scipy.optimize import Bounds, minimize
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm, trange

from worldfoundry.evals.tasks.metrics.distribution.trend.trend_util import *  # noqa: F403 - Preserve upstream module exports.
from worldfoundry.evals.tasks.metrics.distribution.trend.trend_util import (
    get_activations_loader,
    get_init,
    get_pdf,
    jsd,
)

InceptionV3 = __import__("pytorch_fid.inception", fromlist=["InceptionV3"]).InceptionV3


class TrendEstimator:
    def __init__(self, x):
        self.x = x

    def llfunc(self, params):
        mu, var, gamma = params
        x = self.x
        n = len(x)

        t1 = n * (np.log(gamma) - np.log(var))

        t2 = np.sum((np.abs(x - mu) / var) ** gamma)

        ul = np.abs((-mu) / var) ** gamma
        t31 = special.gamma(1 / gamma)
        t32 = special.gammainc(1 / gamma, ul) * special.gamma(1 / gamma)
        t3 = n * np.log(np.sum(t31 + t32))

        ll = t1 - t2 - t3
        return -ll


def extract_embeddings(dir_sub, transform=None, batch_size=None, cuda=True, n_images=50000):
    if batch_size is None:
        batch_size = 50

    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])

    if transform is None:
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    testset = datasets.ImageFolder(dir_sub, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    activations = get_activations_loader(testloader, model, cuda=cuda, batch_size=batch_size, n_images=n_images)

    del model, testloader, testset

    return activations


def estimate_params(activations):  # [n * 2048] dimension
    if isinstance(activations, str) and os.path.isfile(activations):
        with open(activations, "rb") as f:
            return np.load(f)

    bounds = Bounds([-np.inf, 0, 0], [np.inf, np.inf, np.inf])

    acts = np.transpose(activations)  # covert into [2048 * n]
    n = acts.shape[0]
    params = np.empty([n, 3])
    for i in tqdm(range(n)):
        act = acts[i]
        act = act[act > 0]

        e = TrendEstimator(act)
        param_init = get_init(act)
        param_est = minimize(e.llfunc, param_init, method="trust-constr", bounds=bounds).x
        params[i] = param_est

        del e

    return params


def compute_jsd(params_r, params_g):
    if isinstance(params_r, str):
        with open(params_r, "rb") as f:
            params_r = np.load(f)

    if isinstance(params_g, str):
        with open(params_g, "rb") as f:
            params_g = np.load(f)

    n = params_r.shape[0]
    jsd_all = np.empty(n)

    for i in trange(n):
        param_r = params_r[i]
        param_g = params_g[i]

        p_r, p_g = get_pdf(param_r), get_pdf(param_g)
        jsd_all[i] = jsd(p_r, p_g)

    return jsd_all
