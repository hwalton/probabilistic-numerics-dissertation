import gp_model
from utils import debug_print
import numpy as np
import scipy
from fast_det import compute_fast_det
from woodbury_lemma import woodbury_lemma
class GP_NLL_FITC:
    def __init__(self,X, y, y_mean, U, gp_kernel, hyperparameters_obj):
        self.RSig = None
        self.big_lambda_reciprocal = None
        self.stdv = None
        self.K_y_hat_U_R = None
        self.K_y_hat_U_T = None
        self.y_hat_adj = None
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
        self.X = np.squeeze(X)
        self.y = y
        self.U = U
        self.gp_kernel = gp_kernel
        self.y_mean = y_mean

    def kernel_diag(self, X):
        if X.ndim > 1:
            np.squeeze(X)

        list_ = []
        for i in range(len(X)):
            list_.append(np.squeeze(self.gp_kernel.compute_kernel(X[i], X[i])))
        return np.array(list_)

    # def compute_nll(self):    # as per sparse_gp_1_code
    #
    #     self.y_adj = np.squeeze(self.y - self.y_mean)
    #     self.n_f = np.shape(self.X)[0]
    #     self.n_u = np.shape(self.U)[0]
    #
    #     jitter = 1E-6
    #
    #     self.K_ff = np.squeeze(self.gp_kernel.compute_kernel(self.X, self.X))
    #     self.K_fU = np.squeeze(self.gp_kernel.compute_kernel(self.X, self.U))
    #     self.K_UU = np.squeeze(self.gp_kernel.compute_kernel(self.U, self.U)) + np.eye(self.n_u) * jitter
    #
    #     self.L_UU = scipy.linalg.cholesky(self.K_UU + np.eye(self.n_u) * jitter, lower=True)
    #
    #     #code from supervisor example
    #
    #     from scipy.linalg import solve_triangular
    #     from jax import numpy as jnp
    #
    #     Lfu = solve_triangular(self.L_UU, self.K_fU.T, lower=True).T
    #     self.Qff_diag = (Lfu ** 2).sum(-1)
    #
    #     self.kernel_diag_XX = self.kernel_diag(self.X)
    #
    #     self.lam =  self.kernel_diag_XX -  self.Qff_diag + self.hyperparameters_obj.dict()['noise_level'] ** 2
    #
    #     self.RSig = jnp.linalg.qr(
    #         jnp.vstack([self.L_UU.T, (self.lam[:, None] ** -0.5) * self.K_fU]), mode="r"
    #     )
    #
    #     self.Rhat = solve_triangular(self.RSig, self.K_fU.T, trans="T", lower=False)
    #
    #     term_1_f = 0.5 * (((self.y / self.lam**0.5) ** 2).sum()- ((self.Rhat.dot(self.y / self.lam)) ** 2).sum())
    #
    #     term_2_f = 0.5 * jnp.log(self.lam).sum() - jnp.log(jnp.diag(self.L_UU)).sum() + jnp.log(jnp.abs(jnp.diag(self.RSig))).sum()
    #
    #     term_3_f = (self.n_f / 2.0) * np.log(2.0 * np.pi)
    #
    #
    #     #end of code from supervisor example
    #
    #     nll = term_1_f + term_2_f + term_3_f
    #     nll = np.array(nll).item()
    #     out_f = {
    #         'nll': nll,
    #         'term_1': term_1_f,
    #         'term_2': term_2_f,
    #         'term_3': term_3_f,
    #     }
    #     debug_print(f"out_f = {out_f}")
    #     return out_f
    #
    # def predict(self, X_test):    # as per sparse_gp_1_code
    #     from scipy.linalg import solve_triangular
    #     from jax import numpy as jnp
    #
    #     if self.L_UU is None:
    #         self.compute_nll()
    #
    #     K_star_U = np.squeeze(self.gp_kernel.compute_kernel(X_test, self.U))
    #     Lsu = solve_triangular(self.RSig, K_star_U.T, trans="T", lower=False).T
    #     self.mu = Lsu.dot(self.Rhat).dot(self.y[:, 0] / self.lam)
    #
    #     K_star_star = np.squeeze(self.gp_kernel.compute_kernel(X_test, X_test))
    #     Lsu = solve_triangular(self.RSig, K_star_U.T, trans="T", lower=False).T
    #     Lss = solve_triangular(self.L_UU, K_star_U.T, lower=True).T
    #     #self.stdv = self.kernel_diag(X_test) - (Lss**2).sum(-1) + (Lsu**2).sum(-1)
    #     self.stdv = K_star_star - Lss.dot(Lss.T) + Lsu.dot(Lsu.T)
    #
    #     return self.mu, self.stdv
    #
    #
    # def compute_nll(self):     # as per supervisor meeting 8/11/23 maths whiteboard
    #
    #     self.y_adj = np.squeeze(self.y)# - self.y_mean)
    #     self.n_f = np.shape(self.X)[0]
    #     self.n_u = np.shape(self.U)[0]
    #
    #     jitter = 1E-8
    #
    #     self.K_ff = np.squeeze(self.gp_kernel.compute_kernel(self.X, self.X))
    #     self.K_fU = np.squeeze(self.gp_kernel.compute_kernel(self.X, self.U))
    #     self.K_UU = np.squeeze(self.gp_kernel.compute_kernel(self.U, self.U)) + np.eye(self.n_u) * jitter
    #
    #     self.L_UU = scipy.linalg.cholesky(self.K_UU + np.eye(self.n_u) * jitter, lower=True)
    #
    #     self.Q_ff = self.K_fU @ scipy.linalg.cho_solve((self.L_UU, True), self.K_fU.T)
    #
    #     self.big_lambda = self.hyperparameters_obj.dict()['noise_level'] ** 2 * np.eye(self.n_f) + np.diag(self.K_ff - self.Q_ff) * np.eye(self.n_f)
    #     # Modify this to only compute this diagonal
    #
    #     self.K_tilde_Uf = self.K_fU.T * np.reciprocal(np.sqrt(np.diag(self.big_lambda)[None,:]))
    #
    #     QR = np.transpose(np.concatenate((self.L_UU,self.K_tilde_Uf), axis=1))
    #     debug_T = np.transpose(np.concatenate((self.L_UU,self.K_tilde_Uf), axis=1))
    #     debug_U = np.vstack([self.L_UU.T, (np.diag(self.big_lambda)[:, None] ** -0.5) * self.K_fU])
    #
    #     assert np.isclose(debug_T, debug_U).all()
    #     self.R = np.linalg.qr(QR, mode='r')
    #
    #
    #
    #     self.big_lambda_reciprocal = np.reciprocal(np.diag(self.big_lambda))
    #
    #     self.y_hat_adj = self.big_lambda_reciprocal * self.y_adj
    #
    #
    #
    #     self.K_y_hat_U_T = self.y_hat_adj.T @ self.K_fU
    #
    #     debug_v = self.y_hat_adj.T @ self.K_fU
    #     debug_w = np.transpose(self.y_hat_adj) @ self.K_fU
    #
    #     assert np.allclose(debug_v, debug_w)
    #
    #     self.R_inv = self._inverse_upper_triangular(self.R)
    #
    #     debug_x = self._inverse_upper_triangular(self.R)
    #     debug_y = np.linalg.inv(self.R)
    #     difference = debug_x- debug_y
    #
    #     assert np.allclose(debug_x, debug_y)
    #
    #     self.K_y_hat_U_R = self.K_y_hat_U_T @ self.R_inv
    #
    #     a_debug = np.inner(self.y_adj.T, self.y_hat_adj)
    #     b_debug = self.y_adj.T @ self.y_hat_adj
    #
    #     assert np.isclose(a_debug, b_debug)
    #
    #     c_debug = (self.K_y_hat_U_R ** 2).sum()
    #     d_Debug = self.K_y_hat_U_R @ self.K_y_hat_U_R.T
    #
    #     assert np.isclose(c_debug, d_Debug)
    #
    #     term_1_f = 0.5*(np.inner(self.y_adj.T, self.y_hat_adj) - (self.K_y_hat_U_R ** 2).sum())
    #
    #     term_2_f =  0.5 * np.sum(np.log((np.diag(self.big_lambda)))) \
    #                 - np.sum(np.log(np.diag(self.L_UU))) \
    #                 + np.sum(np.log(np.abs(np.diag(self.R))))
    #
    #     term_3_f = (self.n_f / 2.0) * np.log(2.0 * np.pi)
    #
    #     nll = term_1_f + term_2_f + term_3_f
    #     nll = np.array(nll).item()
    #     out_f = {
    #         'nll': nll,
    #         'term_1': term_1_f,
    #         'term_2': term_2_f,
    #         'term_3': term_3_f,
    #     }
    #     debug_print(f"out_f = {out_f}")
    #     return out_f
    #
    # def predict(self, X_test):     # as per supervisor meeting 8/11/23 maths whiteboard
    #     if any(getattr(self, attr) is None for attr in [
    #         'K_y_hat_U_R', 'K_y_hat_U_T', 'y_hat_adj', 'R', 'K_tilde_Uf',
    #         'big_lambda', 'Q_ff', 'L_UU', 'K_UU', 'K_fU', 'K_ff', 'n_u', 'n_f', 'y_adj']):
    #         self.compute_nll()
    #
    #     K_star_U = np.squeeze(self.gp_kernel.compute_kernel(X_test, self.U))
    #
    #     big_sigma = self.R_inv @ self.R_inv.T
    #
    #     y_hat = self.big_lambda_reciprocal * np.squeeze(self.y)
    #
    #     self.mu = K_star_U @ big_sigma @ self.K_fU.T @ y_hat # from quinonero-candela eq. 24b
    #
    #     K_star_star = np.squeeze(self.gp_kernel.compute_kernel(X_test, X_test))
    #
    #     K_tilde_star_U = K_star_U @ self.R_inv
    #
    #     Q_star_star = K_star_U @ scipy.linalg.cho_solve((self.L_UU, True), K_star_U.T)
    #
    #     s2 = K_star_star - Q_star_star + K_tilde_star_U @ K_tilde_star_U.T # from quinonero-candela eq. 24b
    #
    #     self.stdv = np.sqrt(np.diag(s2))
    #
    #     return self.mu, self.stdv

    def compute_nll(self):     # as per max_C_FITC.py

        self.y_adj = np.squeeze(self.y) # - self.y_mean)
        self.n_f = np.shape(self.X)[0]
        self.n_u = np.shape(self.U)[0]

        jitter = 1E-6

        # self.K_ff = np.squeeze(self.gp_kernel.compute_kernel(self.X, self.X))
        self.K_UU = np.squeeze(self.gp_kernel.compute_kernel(self.U, self.U))
        self.K_fU = np.squeeze(self.gp_kernel.compute_kernel(self.X, self.U))

        self.L_UU = np.linalg.cholesky(self.K_UU + np.eye(self.n_u) * jitter)
        self.L_ff = scipy.linalg.solve_triangular(self.L_UU, self.K_fU.T, lower=True).T
        self.Lam_vec = self.hyperparameters_obj.dict()['sigma'] - (self.L_ff ** 2).sum(1) + self.hyperparameters_obj.dict()['noise_level']
        self.LL = np.hstack([self.L_UU, self.K_fU.T * np.sqrt(1 / self.Lam_vec[None, :])])
        self.R = np.linalg.qr(self.LL.T, mode='r')
        self.RI = self._inverse_upper_triangular(self.R)
        self.alpha = (self.RI @ self.RI.T) @ self.K_fU.T @ (1 / self.Lam_vec * np.squeeze(self.y_adj))
        self.yh = (self.y_adj.T * self.Lam_vec[None] ** 0.5).T

        term_1_f = 0.5 * (self.y_adj.T @ self.yh - ((self.yh.T @ self.K_fU @ self.RI) ** 2).sum())

        term_2_f = 0.5 * np.sum(np.log(self.Lam_vec)) - np.sum(np.log(np.diag(self.L_UU))) + np.sum(np.log(np.abs(np.diag(self.R))))

        term_3_f = (self.n_f / 2.0) * np.log(2.0 * np.pi)

        nll = term_1_f + term_2_f + term_3_f
        nll = np.array(nll).item()
        out_f = {
            'nll': nll,
            'term_1': term_1_f,
            'term_2': term_2_f,
            'term_3': term_3_f,
        }
        debug_print(f"out_f = {out_f}")
        return out_f

    def predict(self, X_test):# as per max_C_FITC.py
        if any(getattr(self, attr) is None for attr in [
            'K_y_hat_U_R', 'K_y_hat_U_T', 'y_hat_adj', 'R', 'K_tilde_Uf',
            'big_lambda', 'Q_ff', 'L_UU', 'K_UU', 'K_fU', 'K_ff', 'n_u', 'n_f', 'y_adj']):
            self.compute_nll()

        K_star_U = np.squeeze(self.gp_kernel.compute_kernel(X_test, self.U))

        self.mu = K_star_U @ self.alpha

        K_star_star = np.squeeze(self.gp_kernel.compute_kernel(X_test, X_test))

        Q_star_star = K_star_U @ scipy.linalg.cho_solve((self.L_UU, True), K_star_U.T)

        K_tilde_star_U = K_star_U @ self.RI

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