import torch
import numpy as np
import random
from matplotlib import pyplot as plt
import warnings
# lazy import pytorchltr
try:
    from pytorchltr.evaluation import arp, ndcg
except ImportError:
    warnings.warn("pytorchltr is not installed. `compute_metrics` during training will not work. ")
    arp = None
    ndcg = None


def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def show_img(tensor_image):
    plt.imshow(tensor_image.permute(1, 2, 0))


def count_parameters(model):
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_p = sum(p.numel() for p in model.parameters())
    print(f'trainable parameters: {trainable_p}')
    print(f'all parameters: {all_p}')


def compute_metrics(preds, labels, paintings):
    # input shape: tensor of size (batch_size, list_size)
    ndcg_score = ndcg(preds, labels, torch.tensor([int(preds.size()[1]) for _ in range(preds.size()[0])]), k=15).mean()
    apr_score = arp(preds, labels, torch.tensor([int(preds.size()[1]) for _ in range(preds.size()[0])])).mean()
    per_rank_rank, per_rank_score, painting_score = prediction_per_rank(preds, labels, paintings)
    return {
        'ndcg_score': ndcg_score,
        'apr_score': apr_score,
        'painting_score': painting_score,
        'per_rank_rank': per_rank_rank,
        'per_rank_score': per_rank_score,
    }


def prediction_per_rank(preds, labels, paintings):
    # calculate per rank score
    label_indexes = torch.argsort(labels)
    sorted_all_preds = []
    for pred, index in zip(preds, label_indexes):
        sorted_all_preds.append(pred[index])
    sorted_all_preds = torch.stack(sorted_all_preds)
    pred_mean_score = torch.mean(sorted_all_preds, axis=0)
    # cal rank
    index = torch.argsort(sorted_all_preds)
    pred_rank = torch.argsort(index)
    pred_mean_rank = torch.mean(pred_rank.float(), axis=0)
    try:
        painting_score = torch.sum(sorted_all_preds[:, -1]*paintings)/(torch.sum(paintings))
    except Exception:
        painting_score = 0.0
    return pred_mean_rank, pred_mean_score, painting_score
