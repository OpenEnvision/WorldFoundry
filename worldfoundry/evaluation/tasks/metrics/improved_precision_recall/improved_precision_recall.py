"""
MIT License

Copyright (c) 2019 Youngjung

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

#!/usr/bin/env python3
import os
from functools import partial
from collections import namedtuple
from glob import glob
import numpy as np
from PIL import Image
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

try:
    from tqdm import tqdm, trange
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x, desc=''):
        if len(desc) > 0:
            print(desc)
        return x

    def trange(x, desc=''):
        if len(desc) > 0:
            print(desc)
        return range(x)

import torch
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

Manifold = namedtuple('Manifold', ['features', 'radii'])
PrecisionAndRecall = namedtuple('PrecisinoAndRecall', ['precision', 'recall'])


class IPR():
    def __init__(self, batch_size=50, k=3, num_samples=10000, model=None, device=None):
        self.manifold_ref = None
        self.batch_size = batch_size
        self.k = k
        self.num_samples = num_samples
        self.model = model
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if model is None:
            print('loading vgg16 for improved precision and recall...', end='', flush=True)
            self.vgg16 = models.vgg16(pretrained=True).to(self.device).eval()
            print('done')
        else:
            self.vgg16 = model

        

    def __call__(self, subject):
        return self.precision_and_recall(subject)

    def precision_and_recall(self, subject):
        '''
        Compute precision and recall for given subject
        reference should be precomputed by IPR.compute_manifold_ref()
        args:
            subject: path or images
                path: a directory containing images or precalculated .npz file
                images: torch.Tensor of N x C x H x W
        returns:
            PrecisionAndRecall
        '''
        assert self.manifold_ref is not None, "call IPR.compute_manifold_ref() first"

        manifold_subject = self.compute_manifold(subject)
        precision = compute_metric(self.manifold_ref, manifold_subject.features, 'computing precision...')
        recall = compute_metric(manifold_subject, self.manifold_ref.features, 'computing recall...')
        return PrecisionAndRecall(precision, recall)

    def compute_manifold_ref(self, path):
        self.manifold_ref = self.compute_manifold(path)

    def realism(self, image):
        '''
        args:
            image: torch.Tensor of 1 x C x H x W
        '''
        feat = self.extract_features(image)
        return realism(self.manifold_ref, feat)

    def compute_manifold(self, input):
        '''
        Compute manifold of given input
        args:
            input: path or images, same as above
        returns:
            Manifold(features, radii)
        '''
        # features
        if isinstance(input, str):
            if input.endswith('.npz'):  # input is precalculated file
                print('loading', input)
                f = np.load(input)
                feats = f['feature']
                radii = f['radii']
                f.close()
                return Manifold(feats, radii)
            else:  # input is dir
                feats = self.extract_features_from_files(input)
        elif isinstance(input, torch.Tensor):
            feats = self.extract_features(input)
        elif isinstance(input, np.ndarray):
            input = torch.Tensor(input)
            feats = self.extract_features(input)
        elif isinstance(input, list):
            if isinstance(input[0], torch.Tensor):
                input = torch.cat(input, dim=0)
                feats = self.extract_features(input)
            elif isinstance(input[0], np.ndarray):
                input = np.concatenate(input, axis=0)
                input = torch.Tensor(input)
                feats = self.extract_features(input)
            elif isinstance(input[0], str):  # input is list of fnames
                feats = self.extract_features_from_files(input)
            else:
                raise TypeError
        else:
            print(type(input))
            raise TypeError

        # radii
        distances = compute_pairwise_distances(feats)
        radii = distances2radii(distances, k=self.k)
        return Manifold(feats, radii)

    def extract_features(self, images):
        """
        Extract features of vgg16-fc2 for all images
        params:
            images: torch.Tensors of size N x C x H x W
        returns:
            A numpy array of dimension (num images, dims)
        """
        desc = 'extracting features of %d images' % images.size(0)
        num_batches = int(np.ceil(images.size(0) / self.batch_size))
        _, _, height, width = images.shape
        if height != 224 or width != 224:
            print('IPR: resizing %s to (224, 224)' % str((height, width)))
            resize = partial(F.interpolate, size=(224, 224))
        else:
            def resize(x): return x

        features = []
        for bi in trange(num_batches, desc=desc):
            start = bi * self.batch_size
            end = start + self.batch_size
            batch = images[start:end]
            batch = resize(batch)
            if self.model is None:
                before_fc = self.vgg16.features(batch.to(self.device))
                before_fc = before_fc.view(-1, 7 * 7 * 512)
                feature = self.vgg16.classifier[:5](before_fc)
            else:
                feature = self.vgg16(batch.to(self.device))
            features.append(feature.cpu().data.numpy())

        return np.concatenate(features, axis=0)

    def extract_features_from_files(self, path_or_fnames):
        """
        Extract features of vgg16-fc2 for all images in path
        params:
            path_or_fnames: dir containing images or list of fnames(str)
        returns:
            A numpy array of dimension (num images, dims)
        """

        dataloader = get_custom_loader(path_or_fnames, batch_size=self.batch_size, num_samples=self.num_samples)
        num_found_images = len(dataloader.dataset)
        desc = 'extracting features of %d images' % num_found_images    
        if num_found_images < self.num_samples:
            print('WARNING: num_found_images(%d) < num_samples(%d)' % (num_found_images, self.num_samples))

        features = []
        for batch in tqdm(dataloader, desc=desc):
            if self.model is None:
                before_fc = self.vgg16.features(batch.to(self.device))
                before_fc = before_fc.view(-1, 7 * 7 * 512)
                feature = self.vgg16.classifier[:5](before_fc)
            else:
                feature = self.vgg16(batch.to(self.device)).view(self.batch_size, -1)
            features.append(feature.cpu().data.numpy())

        return np.concatenate(features, axis=0)

    def save_ref(self, fname):
        print('saving manifold to', fname, '...')
        np.savez_compressed(fname,
                            feature=self.manifold_ref.features,
                            radii=self.manifold_ref.radii)






# def compute_pairwise_distances(X, Y=None, metric = 'euclidian', device = 'cpu'):
#     '''
#     args:
#         X: np.array of shape N x dim
#         Y: np.array of shape N x dim
#     returns:
#         N x N symmetric np.array
#     '''
#     num_X = X.shape[0]
#     if Y is None:
#         num_Y = num_X
#     else:
#         num_Y = Y.shape[0]
#     X = X.astype(np.float64)  # to prevent underflow
#     X_norm_square = np.sum(X**2, axis=1, keepdims=True)
#     if Y is None:
#         Y_norm_square = X_norm_square
#     else:
#         Y_norm_square = np.sum(Y**2, axis=1, keepdims=True)
#     X_square = np.repeat(X_norm_square, num_Y, axis=1)
#     Y_square = np.repeat(Y_norm_square.T, num_X, axis=0)
#     if Y is None:
#         Y = X
#     XY = np.dot(X, Y.T)
#     diff_square = X_square - 2*XY + Y_square

#     # check negative distance
#     min_diff_square = diff_square.min()
#     if min_diff_square < 0:
#         idx = diff_square < 0
#         diff_square[idx] = 0
#         print('WARNING: %d negative diff_squares found and set to zero, min_diff_square=' % idx.sum(),
#               min_diff_square)

#     distances = np.sqrt(diff_square)
#     return distances
def cuda_empty(func):
    def wrapper(*args, **kwargs):
        ret = func(*args, **kwargs)
        torch.cuda.empty_cache()
        return ret
    return wrapper


@torch.no_grad()
@cuda_empty
def compute_pairwise_distances(X, Y=None, metric = 'euclidian', device = 'cpu'):

    X = torch.from_numpy(X).to(device)

    if Y is not None:
        Y = torch.from_numpy(Y).to(device).type(torch.float64)

    if metric == 'euclidian':
        num_X = X.shape[0]
        if Y is None:
            num_Y = num_X
        else:
            num_Y = Y.shape[0]
        X = X.type(torch.float64)  # to prevent underflow
        X_norm_square = torch.sum(X**2, dim=1, keepdims=True)
        if Y is None:
            Y_norm_square = X_norm_square
        else:
            Y_norm_square = torch.sum(Y**2, dim=1, keepdims=True)

        torch.cuda.empty_cache()
        # X_square = torch.repeat(X_norm_square, num_Y, dim=1)
        X_square = X_norm_square.expand(-1,num_Y)
        # Y_square = torch.repeat(Y_norm_square.T, num_X, dim=0)
        Y_square = Y_norm_square.T.expand(num_X, -1)
        torch.cuda.empty_cache()

        if Y is None:
            Y = X
        XY = torch.matmul(X, Y.T)
        diff_square = X_square - 2*XY + Y_square

        # check negative distance
        min_diff_square = diff_square.min()
        torch.cuda.empty_cache()
        if min_diff_square < 0:
            idx = diff_square < 0
            diff_square[idx] = 0
            print('WARNING: %d negative diff_squares found and set to zero, min_diff_square=' % idx.sum(),
                  min_diff_square)

        distances = torch.sqrt(diff_square).detach().cpu()

    elif metric == 'cossim':
        X = X.type(torch.float64)
        X = (X.T / torch.linalg.norm(X, dim = 1)).T

        if Y is None:
            Y = X.T
        else:
            Y = Y.T / torch.linalg.norm(Y, dim = 1)

        distances = (-(1 + X@Y)).detach().cpu()

    return distances

def distances2radii(distances, k=3):
    num_features = distances.shape[0]
    radii = np.zeros(num_features)
    for i in range(num_features):
        radii[i] = get_kth_value(distances[i], k=k)
    return radii


def get_kth_value(np_array, k):
    kprime = k+1  # kth NN should be (k+1)th because closest one is itself
    idx = np.argpartition(np_array, kprime)
    k_smallests = np_array[idx[:kprime]]
    kth_value = k_smallests.max()
    return kth_value


def compute_metric(manifold_ref, feats_subject, desc=''):
    num_subjects = feats_subject.shape[0]
    count = 0
    dist = compute_pairwise_distances(manifold_ref.features, feats_subject)
    for i in trange(num_subjects, desc=desc):
        count += (dist[:, i] < manifold_ref.radii).any()
    return count / num_subjects


def is_in_ball(center, radius, subject):
    return distance(center, subject) < radius


def distance(feat1, feat2):
    return np.linalg.norm(feat1 - feat2)


def realism(manifold_real, feat_subject):
    feats_real = manifold_real.features
    radii_real = manifold_real.radii
    diff = feats_real - feat_subject
    dists = np.linalg.norm(diff, axis=1)
    eps = 1e-6
    ratios = radii_real / (dists + eps)
    max_realism = float(ratios.max())
    return max_realism


def _load_dataset_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224), Image.BICUBIC)
    return transform(image) if transform is not None else image


class ImageFolder(Dataset):
    def __init__(self, root, transform=None):
        # self.fnames = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.fnames = glob(os.path.join(root, '**', '*.jpg'), recursive=True) + \
            glob(os.path.join(root, '**', '*.png'), recursive=True)

        self.transform = transform

    def __getitem__(self, index):
        return _load_dataset_image(self.fnames[index], self.transform)

    def __len__(self):
        return len(self.fnames)


class FileNames(Dataset):
    def __init__(self, fnames, transform=None):
        self.fnames = fnames
        self.transform = transform

    def __getitem__(self, index):
        return _load_dataset_image(self.fnames[index], self.transform)

    def __len__(self):
        return len(self.fnames)


def get_custom_loader(image_dir_or_fnames, image_size=224, batch_size=50, num_workers=4, num_samples=-1):
    transform = []
    # transform.append(transforms.Resize([image_size, image_size]))
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225]))

    transform = transforms.Compose(transform)

    if isinstance(image_dir_or_fnames, list):
        dataset = FileNames(image_dir_or_fnames, transform)
    elif isinstance(image_dir_or_fnames, str):
        dataset = ImageFolder(image_dir_or_fnames, transform=transform)
    else:
        raise TypeError

    if num_samples > 0:
        dataset.fnames = dataset.fnames[:num_samples]
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True)
    return data_loader


def toy():
    offset = 2
    feats_real = np.random.rand(10).reshape(-1, 1)
    feats_fake = np.random.rand(10).reshape(-1, 1) + offset
    feats_real[0] = offset
    feats_fake[0] = 1
    print('real:', feats_real)
    print('fake:', feats_fake)

    print('computing pairwise distances...')
    distances_real = compute_pairwise_distances(feats_real)
    print('distances to radii...')
    radii_real = distances2radii(distances_real)
    manifold_real = Manifold(feats_real, radii_real)

    print('computing pairwise distances...')
    distances_fake = compute_pairwise_distances(feats_fake)
    print('distances to radii...')
    radii_fake = distances2radii(distances_fake)
    manifold_fake = Manifold(feats_fake, radii_fake)

    precision = compute_metric(manifold_real, feats_fake)
    recall = compute_metric(manifold_fake, feats_real)
    print('precision:', precision)
    print('recall:', recall)


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('path_real', type=str, help='Path to the real images')
    parser.add_argument('path_fake', type=str, help='Path to the fake images')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size to use')
    parser.add_argument('--k', type=int, default=3, help='Batch size to use')
    parser.add_argument('--num_samples', type=int, default=5000, help='number of samples to use')
    parser.add_argument('--toy', action='store_true')
    parser.add_argument('--fname_precalc', type=str, default='', help='fname for precalculating manifold')
    args = parser.parse_args()

    # toy problem
    if args.toy:
        print('running toy example...')
        toy()
        exit()

    # Example usage: with real and fake paths
    # python improved_precision_recall.py [path_real] [path_fake]
    ipr = IPR(args.batch_size, args.k, args.num_samples)
    with torch.no_grad():
        # real
        ipr.compute_manifold_ref(args.path_real)

        # save and exit for precalc
        # python improved_precision_recall.py [path_real] [dummy_str] --fname_precalc [filename]
        if len(args.fname_precalc) > 0:
            ipr.save_ref(args.fname_precalc)
            print('path_fake (%s) is ignored for precalc' % args.path_fake)
            exit()

        # fake
        precision, recall = ipr.precision_and_recall(args.path_fake)

    print('precision:', precision)
    print('recall:', recall)

    # Example usage: realism of a real image
    if args.path_real.endswith('.npz'):
        print('skip realism score for real image because [path_real] is .npz file')
    else:
        dataloader = get_custom_loader(args.path_real, batch_size=args.batch_size, num_samples=1)
        desc = 'found %d images in ' % len(dataloader.dataset) + args.path_real
        print(desc)
        first_image = iter(dataloader).next()
        realism_score = ipr.realism(first_image)
        print('realism of first image in real:', realism_score)

    # Example usage: realism of a fake image
    dataloader = get_custom_loader(args.path_fake, batch_size=args.batch_size, num_samples=1)
    desc = 'found %d images in ' % len(dataloader.dataset) + args.path_fake
    print(desc)
    first_image = iter(dataloader).next()
    realism_score = ipr.realism(first_image)
    print('realism of first image in fake:', realism_score)

    # Example usage: on-memory case
    # dataloader = get_custom_loader(args.path_fake,
    #                                batch_size=args.batch_size,
    #                                num_samples=args.num_samples)
    # desc = 'found %d images in ' % len(dataloader.dataset) + args.path_fake
    # images = []
    # for batch in tqdm(dataloader, desc=desc):
    #     images.append(batch)
    # images = torch.cat(images, dim=0)
    # print(ipr.precision_and_recall(images))
