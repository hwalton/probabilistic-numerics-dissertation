import gp_model
from utils import debug_print
import numpy as np
import scipy
from fast_det import compute_fast_det
from woodbury_lemma import woodbury_lemma
class GP_NLL_FITC_18_134:
    def __init__(self,X, y, y_mean, U, gp_kernel, hyperparameters_obj):
        self.hyperparameters_obj = hyperparameters_obj
        self.X = X
        self.y = y
        self.U = U
        self.gp_kernel = gp_kernel
        self.y_mean = y_mean

    def compute(self):

        y_adj = np.squeeze(self.y - self.y_mean)

        K_XX_FITC, K_XU, K_UX, K_UU, K_XX, Q_XX, K_UU_inv_K_UX = self.K_XX_FITC()

        K_sigma_inv = self.K_sigma_inv()

        n = K_XX.shape[0]

        big_lambda = self._compute_big_lambda(K_XX, Q_XX, n)

        term_1_f = self._compute_term_1_f(K_UU_inv_K_UX, K_XU, big_lambda)
        term_2_f = self._compute_term_2_f(K_sigma_inv, y_adj)
        term_3_f = self._compute_term_3_f(n)
        nll = term_1_f + term_2_f + term_3_f
        nll = np.array(nll).item()
        out_f = {
            'nll': nll,
            'term_1': term_1_f,
            'term_2': term_2_f,
            'term_3': term_3_f
        }
        return out_f

    def _compute_term_3_f(self, n):
        return 0.5 * n * np.log(2 * np.pi)

    def _compute_term_2_f(self, K_sigma_inv, y_adj):
        return 0.5 * y_adj.T @ K_sigma_inv @ y_adj

    def _compute_term_1_f(self, K_UU_inv_K_UX, K_XU, big_lambda):
        fast_det = compute_fast_det(K_XU, K_UU_inv_K_UX, big_lambda)
        term_1_f = 0.5 * np.log(fast_det)
        return term_1_f

    def _compute_big_lambda(self, K_XX, Q_XX, n):
        big_lambda = np.diag(np.diag(K_XX - Q_XX)) + \
                     self.hyperparameters_obj.dict()[
                         'noise_level'] ** 2 * np.eye(n)
        return big_lambda

    def K_sigma_inv(self, method='woodbury'):
        if method == 'woodbury':
            K_XX_FITC, K_XU, K_UX, K_UU, K_XX, Q_XX, K_UU_inv_K_UX = self.K_XX_FITC()
            sigma_n_neg2 = np.multiply(
                self.hyperparameters_obj.dict()['noise_level'] ** -2,
                np.eye(len(self.X)))
            # sigma_n_neg2 = np.multiply(1, np.eye(len(self.X)))


            # var2 = sigma_n_neg2 @ K_XU @ (K_UU_inv_K_UX + np.array(K_UX @ sigma_n_neg2 @ K_XU @ K_UX)) @ sigma_n_neg2
            # debug_print(f"var == var2: {np.allclose(var,var2, atol = 1E-3)}")

            out = woodbury_lemma(K_UU_inv_K_UX, K_UX, K_XU, sigma_n_neg2)

        else:
            raise ValueError("Invalid inducing method")
        return out


    def K_XX_FITC(self):
        K_UU, K_UX, K_XU, K_XX, U = self._compute_kernels()

        K_UU_inv_KUX, Q_XX = self._calculate_inputs_to_K_XX_FITC_calc(K_UU,
                                                                      K_UX,
                                                                      K_XU, U)

        K_XX_FITC = self._compute_K_XX_FITC(K_UU_inv_KUX, K_XU)

        return K_XX_FITC, K_XU, K_UX, K_UU, K_XX, Q_XX, K_UU_inv_KUX

    def _compute_K_XX_FITC(self, K_UU_inv_KUX, K_XU):
        K_XX_FITC = K_XU @ K_UU_inv_KUX
        return K_XX_FITC

    def _calculate_inputs_to_K_XX_FITC_calc(self, K_UU, K_UX, K_XU, U):
        K_UU_stable = K_UU + 1e-6 * np.eye(U.shape[0])
        L_UU = scipy.linalg.cholesky(K_UU_stable, lower=True)
        K_UU_inv_KUX = scipy.linalg.cho_solve((L_UU, True), K_UX)
        Q_XX = K_XU @ K_UU_inv_KUX
        return K_UU_inv_KUX, Q_XX

    def _compute_kernels(self):
        X = np.squeeze(self.X)
        U = np.squeeze(self.U)
        K_XU = np.squeeze(self.gp_kernel.compute_kernel(X, U))
        K_UX = np.squeeze(self.gp_kernel.compute_kernel(U, X))
        K_UU = np.squeeze(self.gp_kernel.compute_kernel(U, U))
        K_XX = np.squeeze(self.gp_kernel.compute_kernel(X, X))
        return K_UU, K_UX, K_XU, K_XX, U