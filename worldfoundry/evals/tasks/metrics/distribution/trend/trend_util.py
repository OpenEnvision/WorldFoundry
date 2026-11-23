import numpy as np
from scipy import integrate, stats
from scipy import special as special
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm
from tqdm import trange as trange


def get_activations_loader(loader, model, batch_size=50, dims=2048, cuda=True, n_images=None):
    model.eval()

    if n_images is None:
        n_images = len(loader) * batch_size
    if batch_size > n_images:
        print("Warning: batch size is bigger than the data size. Setting batch size to data size")
        batch_size = n_images

    pred_arr = np.empty((n_images, dims))

    for i, data in enumerate(tqdm(loader)):
        batch, _ = data

        start = i * batch_size
        end = start + batch.shape[0]

        if start > n_images:
            break
        if end > n_images:
            end = n_images

        if cuda:
            batch = batch.cuda()

        pred = model(batch)[0]
        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr[start:end] = pred.cpu().data.numpy().reshape(pred.size(0), -1)

    return pred_arr[:end]


def get_init(x):
    # set heuristic initial params
    hist, bins = np.histogram(x, bins="auto")
    pp = np.argmax(hist)
    mu = (bins[pp] + bins[pp + 1]) / 2  # 0.98 corr

    var = 0.44 * np.std(x)  # c1=0.44 with 0.15 corr // old; c1=1.5 with 0.18 corr

    gamma = 0.67  # mean of gammas // c2*kurtosis(x) <<< no corr

    return [mu, var, gamma]


def get_pdf(par_est):
    def pdf(x):
        return stats.gennorm.pdf(x, beta=par_est[2], loc=par_est[0], scale=par_est[1]) / stats.gennorm.sf(
            0, beta=par_est[2], loc=par_est[0], scale=par_est[1]
        )

    return pdf


def kld_integrand(x, p, q):
    if p(x) == 0 or q(x) == 0:
        return 0
    return p(x) * np.log2(p(x) / q(x))


def kld(p, q):
    d, _ = integrate.quad(kld_integrand, 0, np.inf, args=(p, q))
    return d


def jsd(p, q):
    def m(x):
        return (p(x) + q(x)) / 2

    return (kld(p, m) + kld(q, m)) / 2
