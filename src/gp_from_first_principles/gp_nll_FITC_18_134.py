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

        K_ff = np.squeeze(self.gp_kernel.compute_kernel(self.X, self.X))
        K_fU = np.squeeze(self.gp_kernel.compute_kernel(self.X, self.U))
        K_UU = np.squeeze(self.gp_kernel.compute_kernel(self.U, self.U))

        L_UU = scipy.linalg.cholesky(K_UU, lower=True)
        L_inv = self._inverse_lower_triangular(L_UU)
        L_inv_T = L_inv.T

        #Q_ff = (K_fU @ L_inv_T) @ (K_fU @ L_inv_T).T
        Q_ff = K_fU @ scipy.linalg.cho_solve((L_UU, True), K_fU.T)
        #Q_ff = K_fU @ np.linalg.inv(K_UU) @ K_fU.T
        n = np.shape(K_ff)[0]

        big_lambda = self.hyperparameters_obj.dict()['noise_level'] ** 2 * np.eye(n) + K_ff - Q_ff

        big_lambda_inv = np.diag(np.reciprocal(np.diag(big_lambda)))

        det_big_lambda = np.prod(np.diag(big_lambda))
        det_big_lambda = np.clip(det_big_lambda, 1E-12, 1E12)

        K_tilde = K_UU + K_fU.T @ big_lambda_inv @ K_fU

        L = scipy.linalg.cholesky(K_tilde, lower= True)


        A = 2 * np.sum(np.log(np.diag(L)))
        B = np.log(np.reciprocal(np.linalg.det(K_UU)))
        C = np.log(det_big_lambda)
        E = K_fU.T @ big_lambda_inv
        D = scipy.linalg.cho_solve((L,True), E)

        term_1_f = -0.5 *(A + B + C)

        term_2_f = -0.5 * y_adj.T @ (big_lambda_inv - D.T @ D) @ y_adj

        term_3_f = n/2.0 * np.log(2 * np.pi)


        nll = term_1_f + term_2_f + term_3_f
        nll = np.array(nll).item()
        out_f = {
            'nll': nll,
            'term_1': term_1_f,
            'term_2': term_2_f,
            'term_3': term_3_f
        }
        debug_print(f"out = {out_f}")
        return out_f

    def _inverse_lower_triangular(self, matrix):
        # Convert the input matrix to a NumPy array for easier manipulation
        A = np.array(matrix, dtype=float)

        # Check if the matrix is square
        rows, cols = A.shape
        if rows != cols:
            raise ValueError("Input matrix is not square")

        # Check if the matrix is lower triangular
        if not np.allclose(np.triu(A, k=1), 0):
            raise ValueError("Input matrix is not lower triangular")

        # Check for zeros on the main diagonal
        if any(np.isclose(np.diag(A), 0)):
            return None  # Matrix is singular and has no inverse

        # Create an identity matrix of the same size
        I = np.identity(rows)

        # Perform forward elimination to transform A into an identity matrix
        for i in range(rows):
            # Make the diagonal element of the current row equal to 1
            A[i] /= A[i, i]
            I[i] /= A[i, i]
            for j in range(i + 1, rows):
                # Eliminate non-zero elements above the diagonal
                factor = A[j, i]
                A[j] -= factor * A[i]
                I[j] -= factor * I[i]

        # At this point, A should be an identity matrix, and I will be the inverse
        return np.array(I.tolist())
    # def _inverse_lower_triangular(self, L):
    #         n = L.shape[0]
    #         L_inv = np.zeros((n, n), dtype=float)  # Initialize an n x n matrix filled with zeros
    #
    #         for j in range(n):
    #             L_inv[j][j] = 1.0 / L[j][j]  # Set the diagonal element
    #
    #             for i in range(j + 1, n):
    #                 L_inv[i][j] = -np.dot(L[i][j:i], L_inv[j:i, j]) / L[j][j]
    #
    #         return L_inv


# class GP_NLL_FITC_18_134: # old from 24-10-23
#     def __init__(self, X, y, y_mean, U, gp_kernel, hyperparameters_obj):
#         self.hyperparameters_obj = hyperparameters_obj
#         self.X = X
#         self.y = y
#         self.U = U
#         self.gp_kernel = gp_kernel
#         self.y_mean = y_mean
#
#     def compute(self):
#
#         y_adj = np.squeeze(self.y - self.y_mean)
#
#         K_XX_FITC, K_XU, K_UX, K_UU, K_XX, Q_XX, K_UU_inv_K_UX = self.K_XX_FITC()
#
#         K_sigma_inv = self.K_sigma_inv()
#
#         n = K_XX.shape[0]
#
#         big_lambda = self._compute_big_lambda(K_XX, Q_XX, n)
#
#         term_1_f = self._compute_term_1_f(K_UU_inv_K_UX, K_XU, big_lambda)
#         term_2_f = self._compute_term_2_f(K_sigma_inv, y_adj)
#         term_3_f = self._compute_term_3_f(n)
#         nll = term_1_f + term_2_f + term_3_f
#         nll = np.array(nll).item()
#         out_f = {
#             'nll': nll,
#             'term_1': term_1_f,
#             'term_2': term_2_f,
#             'term_3': term_3_f
#         }
#         debug_print(f"out = {out_f}")
#         return out_f
#
#     def _compute_term_3_f(self, n):
#         return 0.5 * n * np.log(2 * np.pi)
#
#     def _compute_term_2_f(self, K_sigma_inv, y_adj):
#         return 0.5 * y_adj.T @ K_sigma_inv @ y_adj
#
#     def _compute_term_1_f(self, K_UU_inv_K_UX, K_XU, big_lambda):
#         fast_det = compute_fast_det(K_XU, K_UU_inv_K_UX, big_lambda)
#         fast_det = self.clip_array(fast_det, 1E-30, 1E30)
#         term_1_f = 0.5 * np.log(fast_det)
#         return term_1_f
#
#     def _compute_big_lambda(self, K_XX, Q_XX, n):
#         big_lambda = self.hyperparameters_obj.dict()['noise_level'] ** 2 * np.eye(n) + np.diag(
#             np.diag(K_XX - Q_XX))
#         return big_lambda
#
#     def K_sigma_inv(self, method='woodbury'):
#         if method == 'woodbury':
#             K_XX_FITC, K_XU, K_UX, K_UU, K_XX, Q_XX, K_UU_inv_K_UX = self.K_XX_FITC()
#
#             lower = 1E-24
#             upper = 1E24
#
#             self.clip_array(K_XX_FITC, lower, upper)
#             self.clip_array(K_XU, lower, upper)
#             self.clip_array(K_UX, lower, upper)
#             self.clip_array(K_UU, lower, upper)
#             self.clip_array(K_XX, lower, upper)
#             self.clip_array(Q_XX, lower, upper)
#             self.clip_array(K_UU_inv_K_UX, lower, upper)
#
#             sigma_n_neg2 = np.multiply(
#                 self.hyperparameters_obj.dict()['noise_level'] ** -2,
#                 np.eye(len(self.X)))
#             # sigma_n_neg2 = np.multiply(1, np.eye(len(self.X)))
#
#             # var2 = sigma_n_neg2 @ K_XU @ (K_UU_inv_K_UX + np.array(K_UX @ sigma_n_neg2 @ K_XU @ K_UX)) @ sigma_n_neg2
#             # debug_print(f"var == var2: {np.allclose(var,var2, atol = 1E-3)}")
#
#             K_UU_stable = K_UU + np.eye(K_UU.shape[0]) * 1E-5
#             L = scipy.linalg.cholesky(K_UU_stable, lower=True)
#             L_inv_T = self._inverse_lower_triangular(L.T)
#             out = sigma_n_neg2 - sigma_n_neg2 @ (
#                         K_XU @ K_UX @ sigma_n_neg2 @ K_XU @ K_UX + (K_XU @ L_inv_T) @ (
#                             K_XU @ L_inv_T).T) @ sigma_n_neg2
#             out = self.clip_array(out, -1E24, -1E-24)
#
#         else:
#             raise ValueError("Invalid inducing method")
#         return out
#
#     def _inverse_lower_triangular(self, L):
#         n = L.shape[0]
#         L_inv = np.zeros((n, n), dtype=float)  # Initialize an n x n matrix filled with zeros
#
#         for j in range(n):
#             L_inv[j][j] = 1.0 / L[j][j]  # Set the diagonal element
#
#             for i in range(j + 1, n):
#                 L_inv[i][j] = -np.dot(L[i][j:i], L_inv[j:i, j]) / L[j][j]
#
#         return L_inv
#
#     def clip_array(self, array, lower=1E-24, upper=1E24):
#         array = np.clip(array, lower, upper)
#         return array
#
#     def K_XX_FITC(self):
#         K_UU, K_UX, K_XU, K_XX, U = self._compute_kernels()
#
#         K_UU = self.clip_array(K_UU)
#         K_UX = self.clip_array(K_UX)
#         K_XU = self.clip_array(K_XU)
#         K_XX = self.clip_array(K_XX)
#
#         K_UU_inv_KUX, Q_XX = self._calculate_inputs_to_K_XX_FITC_calc(K_UU,
#                                                                       K_UX,
#                                                                       K_XU, U)
#
#         K_UU_inv_KUX = self.clip_array(K_UU_inv_KUX)
#         Q_XX = self.clip_array(Q_XX)
#
#         # K_XX_FITC = self.clip_array(self._compute_K_XX_FITC(K_UU_inv_KUX, K_XU), 1E-6, 1E6)
#         K_XX_FITC = Q_XX
#
#         return K_XX_FITC, K_XU, K_UX, K_UU, K_XX, Q_XX, K_UU_inv_KUX
#
#     def _compute_K_XX_FITC(self, K_UU_inv_KUX, K_XU):
#         K_XX_FITC = K_XU @ K_UU_inv_KUX
#         return K_XX_FITC
#
#     def _calculate_inputs_to_K_XX_FITC_calc(self, K_UU, K_UX, K_XU, U):
#         K_UU_stable = K_UU + 1e-6 * np.eye(U.shape[0])
#         K_UU_inv_KUX = np.linalg.pinv(K_UU, rcond=1E-6) @ K_UX
#
#         L_UU = scipy.linalg.cholesky(K_UU_stable, lower=True)
#         L_inv_T = self._inverse_lower_triangular(L_UU.T)
#
#         Q_XX = (K_XU @ L_inv_T) @ (K_XU @ L_inv_T).T
#         return K_UU_inv_KUX, Q_XX
#
#     def _compute_kernels(self):
#         X = np.squeeze(self.X)
#         U = np.squeeze(self.U)
#         K_XU = np.squeeze(self.gp_kernel.compute_kernel(X, U))
#         K_UX = np.squeeze(self.gp_kernel.compute_kernel(U, X))
#         K_UU = np.squeeze(self.gp_kernel.compute_kernel(U, U))
#         K_XX = np.squeeze(self.gp_kernel.compute_kernel(X, X))
#         return K_UU, K_UX, K_XU, K_XX, U