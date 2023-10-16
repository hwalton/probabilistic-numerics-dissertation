import numpy as np



def compute_fast_det(U, V_T, D):
    n = U.shape[0]
    m = U.shape[1]
    diag_D = np.diag(D)
    D_inv = np.diag(np.reciprocal(diag_D))  # O(n)
    det_D = np.prod(diag_D)  # O(n)
    fast_det = det_D * np.linalg.det(np.eye(m) + V_T @ D_inv @ U)
    return fast_det