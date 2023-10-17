import numpy as np


def woodbury_lemma(K_UU_inv_K_UX, K_UX, K_XU, sigma_n_neg2):
    out = sigma_n_neg2 - sigma_n_neg2 @ K_XU @ (
            K_UU_inv_K_UX + K_UX @ sigma_n_neg2 @ K_XU @ K_UX) @ sigma_n_neg2
    return out