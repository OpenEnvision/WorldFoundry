import numpy as np
from scipy.stats import gaussian_kde
from tqdm import tqdm


def kde_1d(attr1, attr2):
    kde_A = gaussian_kde(attr1)
    kde_B = gaussian_kde(attr2)
    return kde_A, kde_B


def into_grid_prob_1d(pdf_ori_1d, pdf_input_1d, n_points):
    sliced_array = np.linspace(-35, 35, n_points)
    points_ori = pdf_ori_1d(sliced_array) * (70 / n_points)
    points_input = pdf_input_1d(sliced_array) * (70 / n_points)

    return points_ori, points_input


def get_SaD(HCS_real, HCS_gen, text_list, n_points):
    print("calculating stats to get SaD")
    total_num = len(text_list)
    sad_result = np.zeros((total_num, 3))  # KL(P,Q) JSD, MI(P,Q)
    eps = 1e-10

    for index, _text in tqdm(enumerate(text_list), total=total_num):
        attr_real = HCS_real[:, index].detach().cpu().numpy()
        attr_gen = HCS_gen[:, index].detach().cpu().numpy()
        pdf_real, pdf_gen = kde_1d(attr_real, attr_gen)

        prob_real_1d, prob_gen_1d = into_grid_prob_1d(pdf_real, pdf_gen, n_points)
        prob_real_1d, prob_gen_1d = prob_real_1d + eps, prob_gen_1d + eps
        difference_1d = np.mean(attr_gen) - np.mean(attr_real)

        CE_P = -prob_real_1d * np.log(prob_real_1d)
        CE_Q = -prob_gen_1d * np.log(prob_gen_1d)
        CE_P_Q = -prob_real_1d * np.log(prob_gen_1d)
        CE_Q_P = -prob_gen_1d * np.log(prob_real_1d)

        KL_P_Q = np.mean(CE_P_Q - CE_P)  # H(P,Q) - H(P)
        if KL_P_Q < 0:
            KL_P_Q = 0
        KL_Q_P = np.mean(CE_Q_P - CE_Q)  # H(P,Q) - H(Q)
        if KL_Q_P < 0:
            KL_Q_P = 0
        JSD = (KL_P_Q + KL_Q_P) / 2

        sad_result[index] = [KL_P_Q, difference_1d, JSD]  # KL(P,Q) mean difference, JSD(P,Q)
    return sad_result
