"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the last layer of the SwAV Resnet50 for generated samples and real world
samples respectively.

See --help to see further details.
"""
import os
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from multiprocessing import cpu_count

import numpy as np
import torch
import torchvision.transforms as TF
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from torchvision.transforms import Compose, Resize, Normalize, ToTensor

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

from worldfoundry.evaluation.tasks.metrics.fid.vendor.swav.resnet50 import resnet50

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('--max-size', type=str, default='all',
                    help=('Number of images to calculate the SwAV FID'))
parser.add_argument('--device', type=str, default=None,
                    help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--dims', type=int, default=2048,
                    choices=[2048],
                    help=('Dimensionality of SwAV Resnet50 features to use. '
                          'Now 2048 only supported'))
parser.add_argument('path', type=str, nargs=2,
                    help=('Paths to the generated images or '
                          'to .npz statistic files'))

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img


def get_activations(files, model, batch_size=50, dims=2048, device='cpu'):
    """Calculates the activations of the last layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of SwAV Resnet50 model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by SwAV Resnet50 (now 2048 only supported)
    -- device      : Device to run calculations

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding SwAV Resnet50 with the
       query tensor.
    """
    model.eval()

    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    dataset = ImagePathDataset(files, transforms=Compose([Resize(224),
                                                          ToTensor(),
                                                          Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])
                                                         ]))
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=cpu_count())

    pred_arr = np.empty((len(files), dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               SwAV Resnet50 net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(files, model, batch_size=50, dims=2048,
                                    device='cpu'):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of SwAV Resnet50 model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by SwAV Resnet50 (now 2048 only supported)
    -- device      : Device to run calculations

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the SwAV Resnet50 model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the SwAV Resnet50 model.
    """
    act = get_activations(files, model, batch_size, dims, device)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def compute_statistics_of_path(path, model, batch_size, max_size, dims, device):
    if isinstance(path, (str, os.PathLike)):
        path = os.fspath(path)
    if isinstance(path, str) and path.endswith('.npz'):
        with np.load(path) as f:
            m, s = f['mu'][:], f['sigma'][:]
    else:
        if isinstance(path, str):
            path = pathlib.Path(path)
            files = sorted([file for ext in IMAGE_EXTENSIONS
                           for file in path.glob('*.{}'.format(ext))])
        else:
            files = list(path)
        if max_size.isdigit():
            files = files[:int(max_size)]
            print('Using {} images from {}'.format(max_size, path))
        elif max_size.lower() == 'all':
            print('Using all images from {}'.format(path))
        else:
            raise RuntimeError('Invalid max size {}'.format(max_size))

        m, s = calculate_activation_statistics(files, model, batch_size,
                                               dims, device)

    return m, s


def calculate_fid_given_paths(paths, batch_size, max_size, device, dims):
    """Calculates the FID of two directories, statistics files, or image lists."""
    for source in paths:
        for p in (source,) if isinstance(source, (str, os.PathLike)) else source:
            if not os.path.exists(p):
                raise RuntimeError('Invalid path: %s' % p)

    model = resnet50().to(device)

    m1, s1 = compute_statistics_of_path(paths[0], model, batch_size, max_size,
                                        dims, device)
    m2, s2 = compute_statistics_of_path(paths[1], model, batch_size, max_size,
                                        dims, device)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


def main():
    args = parser.parse_args()
    assert args.dims in [2048], "Only [2048] dimensions are supported"

    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)

    fid_value = calculate_fid_given_paths(args.path,
                                          args.batch_size,
                                          args.max_size,
                                          device,
                                          args.dims)
    print('FID: ', fid_value)


if __name__ == '__main__':
    main()
