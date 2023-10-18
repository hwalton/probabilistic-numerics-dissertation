import numpy as np
from utils import debug_print


def compute_fast_det(U, V_T, D):
    n = U.shape[0]
    m = U.shape[1]
    diag_D = np.diag(D)
    D_inv = np.diag(np.reciprocal(diag_D))  # O(n)
    det_D = np.prod(diag_D)  # O(n)
    det_D = np.clip(det_D, 1E-6, 1E6)
    det_in = np.eye(m) + V_T @ D_inv @ U
    debug_print(f"min det_d: {np.min(det_D)}, max: {np.max(det_D)}")
    debug_print(f"min det_in: {np.min(det_in)}, max: {np.max(det_in)}")
    fast_det = det_D * np.linalg.det(det_in)
    return fast_det