import gp_model
from utils import debug_print
import numpy as np
import scipy
from fast_det import compute_fast_det
class GP_NLL_FITC_18_134:
    def __init__(self,X, y, U, gp_kernel, hyperparameters_obj):
        self.hyperparameters_obj = hyperparameters_obj
        self.X = X
        self.y = y
        self.U = U
        self.gp_kernel = gp_kernel

    def compute(self):
        # self.hyperparameters_obj.update(hyperparameters)
        # K = self.gp_kernel.compute_kernel(self.X, self.X)
        # K += np.repeat(
        #    np.array(np.eye(len(self.X)) * 1e-3)[:, :, np.newaxis],
        #   self.X.shape[1], axis=2)
        n = len(self.y)
        #    L = scipy.linalg.cholesky(K[:, :, 0], lower=True)
        y_adj = np.squeeze(self.y - self.hyperparameters_obj.dict()[
            'mean_func_c'])
        K_XX_FITC, K_XU, K_UX, K_UU, K_XX, Q_XX, K_UU_inv_K_UX = self.K_XX_FITC()
        # beta = self.hyperparameters_obj.dict()['noise_level'] ** -1
        K_sigma_inv = self.K_sigma_inv()
        # eigs = np.linalg.eigvalsh(K_sigma_inv)
        # debug_print(f"cov eigenvalues: {eigs}")
        n = K_XX.shape[0]
        # var645 = self.hyperparameters_obj.dict()['noise_level'] ** 2 * np.eye(n)
        big_lambda = np.diag(np.diag(K_XX - Q_XX)) + \
                     self.hyperparameters_obj.dict()[
                         'noise_level'] ** 2 * np.eye(n)
        # eigs_big_lambda = np.linalg.eigvalsh(big_lambda)
        # debug_print(f"big lambda eigenvalues: {eigs_big_lambda}")
        in53 = Q_XX + big_lambda + 1E-3 * np.eye(n)
        cond_in53 = np.linalg.cond(in53)
        debug_print(f"cond_in53: {cond_in53}")
        fast_det = compute_fast_det(K_XU, K_UU_inv_K_UX, big_lambda)
        debug_print(f"compute_fast_det: {fast_det}")
        debug_print(f"K_sigma_inv: {K_sigma_inv}")
        term_1_f = 0.5 * np.log(fast_det)
        term_2_f = 0.5 * y_adj.T @ K_sigma_inv @ y_adj
        term_3_f = 0.5 * n * np.log(2 * np.pi)
        nll = term_1_f + term_2_f + term_3_f
        nll = np.array(-nll).item()
        out_f = {
            'nll': nll,
            'term_1': term_1_f,
            'term_2': term_2_f,
            'term_3': term_3_f
        }
        return out_f

    def K_sigma_inv(self, method='woodbury'):
        if method == 'woodbury':
            K_XX_FITC, K_XU, K_UX, K_UU, K_XX, Q_XX, K_UU_inv_K_UX = self.K_XX_FITC()
            sigma_n_neg2 = np.multiply(
                self.hyperparameters_obj.dict()['noise_level'] ** -2,
                np.eye(len(self.X)))
            # sigma_n_neg2 = np.multiply(1, np.eye(len(self.X)))

            # var237 = np.linalg.solve(K_UU, K_UX)
            var = sigma_n_neg2 @ K_XU @ (
                        K_UU_inv_K_UX + K_UX @ sigma_n_neg2 @ K_XU @ K_UX) @ sigma_n_neg2
            # var2 = sigma_n_neg2 @ K_XU @ (K_UU_inv_K_UX + np.array(K_UX @ sigma_n_neg2 @ K_XU @ K_UX)) @ sigma_n_neg2
            # debug_print(f"var == var2: {np.allclose(var,var2, atol = 1E-3)}")
            out = sigma_n_neg2 - var

        else:
            raise ValueError("Invalid inducing method")
        return out


    def K_XX_FITC(self):
        X = np.squeeze(self.X)
        U = np.squeeze(self.U)

        K_XU = np.squeeze(self.gp_kernel.compute_kernel(X, U))
        K_UX = np.squeeze(self.gp_kernel.compute_kernel(U, X))
        K_UU = np.squeeze(self.gp_kernel.compute_kernel(U, U))
        K_XX = np.squeeze(self.gp_kernel.compute_kernel(X, X))

        K_UU_stable = K_UU + 1e-6 * np.eye(U.shape[0])

        L_UU = scipy.linalg.cholesky(K_UU_stable, lower=True)
        K_UU_inv_KUX = scipy.linalg.cho_solve((L_UU, True), K_UX)
        Q_XX = K_XU @ K_UU_inv_KUX
        rank_Q_XX = np.linalg.matrix_rank(Q_XX)
        debug_print(f"Q_XX Rank: {rank_Q_XX}")
        # K_XX_FITC = K_XX + Q_XX - K_XU @ var4
        K_XX_FITC = K_XU @ K_UU_inv_KUX

        # out = {
        #     'K_XX_FITC': K_XX_FITC,
        #     'K_XU': K_XU,
        #     'K_UU': K_UU,
        #     'K_XX': K_XX,
        # }
        return K_XX_FITC, K_XU, K_UX, K_UU, K_XX, Q_XX, K_UU_inv_KUX