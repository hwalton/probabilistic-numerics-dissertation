import gp_model
from utils import debug_print
import numpy as np
import scipy
from fast_det import compute_fast_det
from woodbury_lemma import woodbury_lemma
class GP_NLL_FITC:
    def __init__(self,X, y, y_mean, U, gp_kernel, hyperparameters_obj):
        self.stdv = None
        self.K_y_hat_U_R = None
        self.K_y_hat_U = None
        self.K_y_hat_U_T = None
        self.y_hat = None
        self.R = None
        self.K_tilde_Uf = None
        self.big_lambda = None
        self.Q_ff = None
        self.L_UU = None
        self.K_UU = None
        self.K_fU = None
        self.K_ff = None
        self.n_u = None
        self.n_f = None
        self.y_adj = None
        self.hyperparameters_obj = hyperparameters_obj
        self.X = X
        self.y = y
        self.U = U
        self.gp_kernel = gp_kernel
        self.y_mean = y_mean

    def compute_nll(self):

        self.y_adj = np.squeeze(self.y - self.y_mean)
        self.n_f = np.shape(self.X)[0]
        self.n_u = np.shape(self.U)[0]

        jitter = 1E-5

        self.K_ff = np.squeeze(self.gp_kernel.compute_kernel(self.X, self.X))
        self.K_fU = np.squeeze(self.gp_kernel.compute_kernel(self.X, self.U))
        self.K_UU = np.squeeze(self.gp_kernel.compute_kernel(self.U, self.U)) + np.eye(self.n_u) * jitter

        self.L_UU = scipy.linalg.cholesky(self.K_UU, lower=True)

        self.Q_ff = self.K_fU @ scipy.linalg.cho_solve((self.L_UU, True), self.K_fU.T)

        self.big_lambda = self.hyperparameters_obj.dict()['noise_level'] ** 2 * np.eye(self.n_f) + self.K_ff - self.Q_ff

        self.K_tilde_Uf = self.K_fU.T * np.reciprocal(np.sqrt(np.diag(self.big_lambda)[None,:]))

        QR = np.transpose(np.concatenate((self.L_UU,self.K_tilde_Uf), axis=1))
        self.R = np.abs(np.linalg.qr(QR, mode='r'))

        self.y_hat = np.reciprocal(np.diag(self.big_lambda)) * self.y_adj

        self.K_y_hat_U_T = self.y_hat.T @ self.K_fU

        self.K_y_hat_U = self.K_y_hat_U_T.T

        self.R_inv = self._inverse_upper_triangular(self.R)

        self.K_y_hat_U_R = self.K_y_hat_U @ self.R_inv

        term_1_f = (self.n_f / 2.0) * np.log(2 * np.pi)

        term_2_f =  0.5 * np.sum(np.log((np.diag(self.big_lambda)))) \
                    - np.sum(np.log(np.diag(self.L_UU))) \
                    + np.sum(np.log(np.diag(self.R)))

        term_3_f = np.inner(self.y_adj.T, self.y_hat) - (self.K_y_hat_U_R ** 2).sum()


        nll = term_1_f + term_2_f + term_3_f
        nll = np.array(nll).item()
        out_f = {
            'nll': nll,
            'term_1': term_1_f,
            'term_2': term_2_f,
            'term_3': term_3_f,
        }
        debug_print(f"out = {out_f}")
        return out_f

    def predict(self, X_test):
        if any(getattr(self, attr) is None for attr in [
            'K_y_hat_U_R', 'K_y_hat_U', 'K_y_hat_U_T', 'y_hat', 'R', 'K_tilde_Uf',
            'big_lambda', 'Q_ff', 'L_UU', 'K_UU', 'K_fU', 'K_ff', 'n_u', 'n_f', 'y_adj']):
            self.compute_nll()

        K_star_U = self.gp_kernel.compute_kernel(X_test, self.U)[:,:,0]

        big_sigma = self.R_inv @ self.R_inv.T

        self.mu = K_star_U @ big_sigma @ self.K_fU.T @ self.y_hat # from quinonero-candela eq. 24b

        K_star_star = self.gp_kernel.compute_kernel(X_test, X_test)[:,:,0]

        K_tilde_star_U = K_star_U @ self.R_inv

        Q_star_star = K_star_star - K_star_U @ scipy.linalg.cho_solve((self.L_UU, True), K_star_U.T)

        s2 = K_star_star - Q_star_star + K_tilde_star_U @ K_tilde_star_U.T # from quinonero-candela eq. 24b

        self.stdv = np.sqrt(np.diag(s2))

        return self.mu, self.stdv

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

        # Compute the inverse
        for i in range(rows):
            I[i, i] = 1 / A[i, i]
            for j in range(i):
                I[i, j] = -sum(A[i, k] * I[k, j] for k in range(j, i)) / A[i, i]

        return np.array(I.tolist())

    def _inverse_upper_triangular(self, matrix):
        # Convert the input matrix to a NumPy array for easier manipulation
        A = np.array(matrix, dtype=float)

        # Check if the matrix is square
        rows, cols = A.shape
        if rows != cols:
            raise ValueError("Input matrix is not square")

        # Check if the matrix is upper triangular
        if not np.allclose(np.tril(A, k=-1), 0):
            raise ValueError("Input matrix is not upper triangular")

        # Check for zeros on the main diagonal
        if any(np.isclose(np.diag(A), 0)):
            raise ValueError(
                "Matrix is singular and has no inverse")  # Changed from return None for consistency

        # Create an identity matrix of the same size
        I = np.identity(rows)

        # Compute the inverse
        for i in reversed(range(rows)):
            I[i, i] = 1 / A[i, i]
            for j in reversed(range(i + 1, rows)):
                I[i, j] = -sum(A[i, k] * I[k, j] for k in range(i + 1, j + 1)) / A[i, i]

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

# before supervisor meeting 08-11-23
# def compute_nll(self):
    #
    #     y_adj = np.squeeze(self.y - self.y_mean)
    #     n_f = np.shape(self.X)[0]
    #     n_u = np.shape(self.U)[0]
    #
    #     K_UU_jitter = 1E-6
    #     K_tilde_jitter = 1E-6
    #
    #     K_ff = np.squeeze(self.gp_kernel.compute_kernel(self.X, self.X))
    #     K_fU = np.squeeze(self.gp_kernel.compute_kernel(self.X, self.U))
    #     K_UU = np.squeeze(self.gp_kernel.compute_kernel(self.U, self.U)) + np.eye(n_u) * K_UU_jitter
    #
    #     L_UU = scipy.linalg.cholesky(K_UU, lower=True)
    #     L_inv = self._inverse_lower_triangular(L_UU)
    #     L_inv_T = L_inv.T
    #
    #     #Q_ff = (K_fU @ L_inv_T) @ (K_fU @ L_inv_T).T
    #     Q_ff = K_fU @ scipy.linalg.cho_solve((L_UU, True), K_fU.T)
    #     #Q_ff = K_fU @ np.linalg.inv(K_UU) @ K_fU.T
    #
    #     big_lambda = self.hyperparameters_obj.dict()['noise_level'] ** 2 * np.eye(n_f) + K_ff - Q_ff
    #
    #     big_lambda_inv = np.diag(np.reciprocal(np.diag(big_lambda)))
    #
    #     det_big_lambda = np.prod(np.diag(big_lambda))
    #     det_big_lambda = np.clip(det_big_lambda, 1E-15, 1E15)
    #
    #     K_tilde = (K_UU + K_fU.T @ big_lambda_inv @ K_fU) + np.eye(n_u) * K_tilde_jitter
    #
    #     L = scipy.linalg.cholesky(K_tilde, lower= True)
    #
    #
    #     A = 2 * np.sum(np.log(np.diag(L)))
    #     B = np.log(np.linalg.det(K_UU) ** -1)
    #     C = np.log(det_big_lambda)
    #     E = K_fU.T @ big_lambda_inv
    #     D = scipy.linalg.cho_solve((L,True), E)
    #     #D2 = np.linalg.solve(L.T,np.linalg.solve(L,E))
    #
    #     term_1_f = 0.5 * (A + B + C)
    #
    #     term_2_f = 0.5 * y_adj.T @ (big_lambda_inv - D.T @ D) @ y_adj
    #
    #     term_3_f = n_f/2.0 * np.log(2 * np.pi)
    #     l2_regularization = 1E15
    #
    #     # Compute L2 regularization term
    #     l2_term = 0.5 * l2_regularization * np.sum((self.gp_kernel.hyperparameters_obj.array())**2 / (self.gp_kernel.hyperparameters_obj.array(attribute = 'initial') ** 2))
    #
    #     nll = term_1_f + term_2_f + term_3_f       # + l2_term
    #     nll = np.array(nll).item()
    #     out_f = {
    #         'nll': nll,
    #         'term_1': term_1_f,
    #         'term_2': term_2_f,
    #         'term_3': term_3_f,
    #         'l2_term': l2_term
    #     }
    #     debug_print(f"out = {out_f}")
    #     return out_f
