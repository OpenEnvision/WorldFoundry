from itertools import combinations

import numpy as np
import torch
from scipy.stats import gaussian_kde
from tqdm import tqdm


def kde_2d(attr1, attr2):
    kde_A = gaussian_kde(attr1)
    kde_B = gaussian_kde(attr2)
    temp = np.vstack((attr1, attr2))
    kde_A_and_B = gaussian_kde(temp)

    return kde_A, kde_B, kde_A_and_B


def into_grid_prob_2d(pdf_real, pdf_gen, n_points, pooling):
    n_bins = int(np.sqrt(n_points))
    x = np.linspace(-35, 35, n_bins + 1)
    y = np.linspace(-35, 35, n_bins + 1)
    X, Y = np.meshgrid(x, y)

    sliced_array = np.array([X.flatten(), Y.flatten()])
    points_real = pdf_real(sliced_array)
    reshaped_point_real = points_real.reshape(
        n_bins + 1, n_bins + 1
    )  # input = 101,101,2 ,x axis to y axis.. #output = 100,100

    points_gen = pdf_gen(sliced_array)
    reshaped_point_gen = points_gen.reshape(
        n_bins + 1, n_bins + 1
    )  # input = 101,101,2 ,x axis to y axis.. #output = 100,100

    reshaped_point_real_cuda = torch.Tensor(reshaped_point_real).to("cuda")
    reshaped_point_gen_cuda = torch.Tensor(reshaped_point_gen).to("cuda")

    prob_ori = pooling(reshaped_point_real_cuda.unsqueeze(0)) * ((70 / n_bins) * (70 / n_bins))
    prob_input = pooling(reshaped_point_gen_cuda.unsqueeze(0)) * ((70 / n_bins) * (70 / n_bins))

    return prob_ori.squeeze(), prob_input.squeeze()


def get_PaD(HCS_real, HCS_gen, text_list, n_points):
    print("calculating stats to get PaD")
    pooling = torch.nn.AvgPool2d(kernel_size=2, stride=1)
    corr_total_num = len(list(combinations(np.arange(len(text_list)), 2)))
    pad_results = torch.zeros((corr_total_num, 2))
    eps = 1e-10

    for corr_index, pair in tqdm(enumerate(combinations(np.arange(len(text_list)), 2)), total=corr_total_num):
        i, j = pair[0], pair[1]

        attr1_real = HCS_real[:, i].detach().cpu().numpy()
        attr2_real = HCS_real[:, j].detach().cpu().numpy()

        attr1_gen = HCS_gen[:, i].detach().cpu().numpy()
        attr2_gen = HCS_gen[:, j].detach().cpu().numpy()

        pdf_A, pdf_B, pdf_A_and_B_real = kde_2d(attr1_real, attr2_real)
        pdf_A_gen, pdf_B_gen, pdf_A_and_B_gen = kde_2d(attr1_gen, attr2_gen)

        prob_real_2d, prob_gen_2d = into_grid_prob_2d(pdf_A_and_B_real, pdf_A_and_B_gen, n_points, pooling)
        prob_real_2d, prob_gen_2d = prob_real_2d + eps, prob_gen_2d + eps

        CE_P = -prob_real_2d * torch.log(prob_real_2d)
        CE_Q = -prob_gen_2d * torch.log(prob_gen_2d)
        CE_P_Q = -prob_real_2d * torch.log(prob_gen_2d)
        CE_Q_P = -prob_gen_2d * torch.log(prob_real_2d)

        KL_P_Q = torch.mean(CE_P_Q - CE_P)  # H(P,Q) - H(P)
        if KL_P_Q < 0:
            KL_P_Q = 0
        KL_Q_P = torch.mean(CE_Q_P - CE_Q)  # H(P,Q) - H(Q)
        if KL_Q_P < 0:
            KL_Q_P = 0

        JSD = (KL_P_Q + KL_Q_P) / 2
        pad_results[corr_index] = torch.Tensor([KL_P_Q, JSD])  # KL(P,Q) JSD

    return pad_results
