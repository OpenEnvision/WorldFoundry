from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
import torchvision
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
import torchvision.transforms.functional as TF


from vendi_score import data_utils, vendi
from vendi_score.data_utils import Example, Group


def get_cifar100(root, split="test"):
    dataset = torchvision.datasets.CIFAR100(
        root=root, train=split == "train", download=True
    )
    examples = []
    for x, y in dataset:
        features = {"pixels": np.array(x).flatten()}
        labels = {"y": dataset.classes[y]}
        examples.append(Example(x=x, features=features, labels=labels))
    return examples


def get_cifar10(root, split="test"):
    dataset = torchvision.datasets.CIFAR10(
        root=root, train=split == "train", download=True
    )
    examples = []
    for x, y in dataset:
        features = {"pixels": np.array(x).flatten()}
        labels = {"y": dataset.classes[y]}
        examples.append(Example(x=x, features=features, labels=labels))
    return examples


def get_mnist(root, split="train", transform=None):
    dataset = torchvision.datasets.MNIST(
        root, train=split == "train", download=True, transform=transform
    )
    examples = []
    for x, y in dataset:
        features = {"pixels": np.array(x).flatten()}
        labels = {"y": y}
        examples.append(Example(x=x, features=features, labels=labels))
    return examples


def get_inception(pretrained=True, pool=True):
    if pretrained:
        weights = Inception_V3_Weights.DEFAULT
    else:
        weights = None
    model = inception_v3(
        weights=weights, transform_input=True
    ).eval()
    if pool:
        model.fc = nn.Identity()
    return model


def inception_transforms():
    return transforms.Compose(
        [
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.expand(3, -1, -1)),
        ]
    )


def get_embeddings(
    images,
    model=None,
    transform=None,
    batch_size=64,
    device=torch.device("cpu"),
):
    if type(device) == str:
        device = torch.device(device)
    if model is None:
        model = get_inception(pretrained=True, pool=True).to(device)
        transform = inception_transforms()
    if transform is None:
        transform = transforms.ToTensor()
    uids = []
    embeddings = []
    for batch in data_utils.to_batches(images, batch_size):
        x = torch.stack([transform(img) for img in batch], 0).to(device)
        with torch.no_grad():
            output = model(x)
        if type(output) == list:
            output = output[0]
        output_arr = output.squeeze().cpu().numpy()
        if output_arr.ndim==1:
            output_arr = output_arr.reshape(1, output_arr.size)
        embeddings.append(output_arr)
    return np.concatenate(embeddings, 0)


def get_pixel_vectors(images, resize=32):
    if resize:
        images = [img.resize((resize, resize)) for img in images]
    return np.stack([np.array(img).flatten() for img in images], 0)


def get_inception_embeddings(images, batch_size=64, device="cpu"):
    if type(device) == str:
        device = torch.device(device)
    model = get_inception(pretrained=True, pool=True).to(device)
    transform = inception_transforms()
    return get_embeddings(
        images,
        batch_size=batch_size,
        device=device,
        model=model,
        transform=transform,
    )


def pixel_vendi_score(images, resize=32):
    X = get_pixel_vectors(images)
    n, d = X.shape
    if n < d:
        return vendi.score_X(X)
    return vendi.score_dual(X)


def embedding_vendi_score(
    images, batch_size=64, device="cpu", model=None, transform=None
):
    X = get_embeddings(
        images,
        batch_size=batch_size,
        device=device,
        model=model,
        transform=transform,
    )
    n, d = X.shape
    if n < d:
        return vendi.score_X(X)
    return vendi.score_dual(X)


def plot_images(images, cols=None, ax=None):
    if cols is None:
        cols = len(images)
    if ax is None:
        fig, ax = plt.subplots()
    rows = data_utils.to_batches([np.array(x) for x in images], cols)
    shape = rows[0][0].shape
    while len(rows[-1]) < cols:
        rows[-1].append(np.zeros(shape))
    rows = [np.concatenate(row, 1) for row in rows]
    ax.imshow(np.concatenate(rows, 0))
    ax.set_xticks([])
    ax.set_yticks([])
