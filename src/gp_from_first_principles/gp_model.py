import numpy as np
import scipy
#import freelunch
from numpy import linalg as npla
from gp_nll_FITC_18_134 import GP_NLL_FITC_18_134
from hyperparameters import Hyperparameters
from gp_kernel import GaussianProcessKernel
from iterative_search import iterative_search_solve
from metropolis_hastings import metropolis_hastings_solve
from adam import adam_optimize
from utils import debug_print
from sklearn.cluster import KMeans


class GPModel:
    def __init__(self, kernel_type, X, y, solver_type = 'iterative_search', n_iter=10, gp_algo ='cholesky', M_one_in = 1):
        self.hyperparameters_obj = Hyperparameters(kernel_type)
        self.initial_hyperparameters = self.hyperparameters_obj._initial_hyperparameters.copy()
        self.hyperparameter_bounds = self.hyperparameters_obj._hyperparameter_bounds.copy()
        self.X = X
        self.y = y
        self.n_iter = n_iter
        self.template = self.hyperparameters_obj._initial_hyperparameters.copy()
        self.solver_type = solver_type
        self.gp_kernel = GaussianProcessKernel(self.hyperparameters_obj)
        self.U = self.U_induced(M_one_in)
        self.gp_algo = gp_algo
        self.y_mean = np.mean(y)
        self.gp_nll_algo_obj = GP_NLL_FITC_18_134(self.X, self.y, self.y_mean, self.U,
                                                  self.gp_kernel,
                                                  self.hyperparameters_obj)
        self.M_one_in = M_one_in

    def fit_model(self):
        self.gp_kernel.set_params(self.hyperparameters_obj)
        optimal_hyperparameters = self.get_optimal_hyperparameters()
        self.hyperparameters_obj.update(optimal_hyperparameters)
        debug_print(optimal_hyperparameters)
        debug_print(self.hyperparameters_obj.array())
        nll = self.compute_nll(self.hyperparameters_obj)['nll']
        debug_print(nll)
        return(nll)

    def solve(self, solver_type):
        if solver_type == 'iterative_search':
            return iterative_search_solve(self.hyperparameters_obj.array(), self.hyperparameters_obj.array(attribute='bounds'), self.compute_nll, n_iter=self.n_iter)
        elif solver_type == 'metropolis_hastings':
            #debug = self.hyperparameters_obj.array()
            return metropolis_hastings_solve(self.hyperparameters_obj.array(), self.hyperparameters_obj.array(attribute='bounds'), self.compute_nll, n_iter=self.n_iter, sample_length = len(self.X))
        elif solver_type == 'free_lunch':
            self.solver = freelunch.DE(self.compute_nll, bounds = self.hyperparameters_obj.array(attribute='bounds'))
            return self.solver()
        elif solver_type == 'adam':
            return adam_optimize(self.compute_nll,self.X, self.y, self.hyperparameters_obj.array(), self.gp_kernel, self.hyperparameters_obj.reconstruct_params)
        else:
            assert False, "invalid solver_type"

    def get_optimal_hyperparameters(self):
        debug_print("solving")
        optimal_hyperparameters = self.solve(self.solver_type).T
        debug_print("solved")
        return self.hyperparameters_obj.reconstruct_params(optimal_hyperparameters)

    import numpy as np
    import scipy.linalg

    from sklearn.cluster import KMeans
    import numpy as np

    def U_induced(self, M_one_in = 1, method='k_means'):
        M = len(self.X) // M_one_in

        if method == 'k_means':
            # Perform k-means clustering on self.X
            kmeans = KMeans(n_clusters=M, n_init=3).fit(self.X)

            # Extract the cluster centers for self.X
            U_X = kmeans.cluster_centers_

            # Determine the corresponding self.y values for the inducing points
            U_y = np.zeros((M, self.y.shape[1] if self.y.ndim > 1 else 1))
            for i in range(M):
                # Find the indices of the points assigned to cluster i
                idx = np.where(kmeans.labels_ == i)[0]

                # Assign the mean of the self.y values of the points in cluster i
                # to the corresponding inducing output
                U_y[i] = np.mean(self.y[idx], axis=0)
        else:
            raise ValueError("Invalid inducing method")
        self.U_X = U_X
        self.U_y = U_y
        #self.X = U_X
        #self.y = U_y
        return U_X

    # def K_XX_FITC(self):
    #
    #     X = np.squeeze(self.X)
    #     U = np.squeeze(self.U)
    #
    #     K_XU = np.squeeze(self.gp_kernel.compute_kernel(X, U))
    #     K_UX = np.squeeze(self.gp_kernel.compute_kernel(U, X))
    #     K_UU = np.squeeze(self.gp_kernel.compute_kernel(U, U))
    #     K_XX = np.squeeze(self.gp_kernel.compute_kernel(X, X))
    #
    #     K_UU_stable = K_UU + 1e-6 * np.eye(U.shape[0])
    #
    #     L_UU = scipy.linalg.cholesky(K_UU_stable, lower=True)
    #     K_UU_inv_KUX = scipy.linalg.cho_solve((L_UU, True), K_UX)
    #     Q_XX = K_XU @ K_UU_inv_KUX
    #     rank_Q_XX = np.linalg.matrix_rank(Q_XX)
    #     debug_print(f"Q_XX Rank: {rank_Q_XX}")
    #     #K_XX_FITC = K_XX + Q_XX - K_XU @ var4
    #     K_XX_FITC = K_XU @ K_UU_inv_KUX
    #
    #
    #     # out = {
    #     #     'K_XX_FITC': K_XX_FITC,
    #     #     'K_XU': K_XU,
    #     #     'K_UU': K_UU,
    #     #     'K_XX': K_XX,
    #     # }
    #     return K_XX_FITC, K_XU, K_UX, K_UU, K_XX, Q_XX, K_UU_inv_KUX

    # def K_sigma_inv(self, method = 'woodbury'):
    #     if method == 'woodbury':
    #         K_XX_FITC, K_XU, K_UX, K_UU, K_XX = self.K_XX_FITC()
    #         sigma_n = np.multiply(self.hyperparameters_obj.dict()['noise_level'] ** -2, np.eye(len(self.X)))
    #         var = (sigma_n @ K_XU @ (np.linalg.inv(K_UU) + np.array(K_UX @ sigma_n @ K_XU)) @ K_UX @ sigma_n)
    #         out = sigma_n - var
    #
    #     else:
    #         raise ValueError("Invalid inducing method")
    #     return out
    # def K_sigma_inv(self, method = 'woodbury'):
    #     if method == 'woodbury':
    #         K_XX_FITC, K_XU, K_UX, K_UU, K_XX, Q_XX, K_UU_inv_K_UX= self.gp_nll_algo.K_XX_FITC()
    #         sigma_n_neg2 = np.multiply(self.hyperparameters_obj.dict()['noise_level'] ** -2, np.eye(len(self.X)))
    #         #sigma_n_neg2 = np.multiply(1, np.eye(len(self.X)))
    #
    #         #var237 = np.linalg.solve(K_UU, K_UX)
    #         var = sigma_n_neg2 @ K_XU @ (K_UU_inv_K_UX + K_UX @ sigma_n_neg2 @ K_XU @ K_UX) @ sigma_n_neg2
    #         #var2 = sigma_n_neg2 @ K_XU @ (K_UU_inv_K_UX + np.array(K_UX @ sigma_n_neg2 @ K_XU @ K_UX)) @ sigma_n_neg2
    #         #debug_print(f"var == var2: {np.allclose(var,var2, atol = 1E-3)}")
    #         out = sigma_n_neg2 - var
    #
    #     else:
    #         raise ValueError("Invalid inducing method")
    #     return out
    # def K_sigma_inv(self, method = 'woodbury'):
    #     if method == 'woodbury':
    #         K_XX_FITC, K_XU, K_UX, K_UU, K_XX = self.K_XX_FITC()
    #         sigma_n = np.multiply(self.hyperparameters_obj.dict()['noise_level'] ** -2, np.eye(len(self.X)))
    #         var = (sigma_n @ K_XU @ (np.linalg.inv(K_UU) + np.array(K_UX @ sigma_n @ K_XU)) @ K_UX @ sigma_n)
    #         out = sigma_n - var
    #
    #     else:
    #         raise ValueError("Invalid inducing method")
    #     return out


    def predict(self, X_star, method = 'FITC'):
        if method == 'FITC':
            K_X_X = self.gp_kernel.compute_kernel(self.X, self.X) + np.array(self.hyperparameters_obj.dict()['noise_level'] ** 2 * np.eye(len(self.X)))[:,:,None]
            K_star_X = self.gp_kernel.compute_kernel(self.X, X_star)
            K_star_star = self.gp_kernel.compute_kernel(X_star, X_star)
            K_sigma_inv = self.gp_nll_algo_obj.K_sigma_inv()
            L = np.zeros_like(K_X_X)
            self.mu = np.zeros((K_star_X.shape[1], K_X_X.shape[2]))
            self.s2 = np.zeros((K_star_X.shape[1], K_X_X.shape[2]))
            for i in range(K_X_X.shape[2]):
                L[:, :, i] = npla.cholesky(K_X_X[:, :, i] + 1e-10 * np.eye(K_X_X.shape[0]))
                Lk = np.squeeze(npla.solve(L[:, :, i], K_star_X[:,:,i]))
                alpha = K_sigma_inv @ self.y
                self.mu[:, i] = self.y_mean + np.array(K_star_X.T @ alpha).flatten()
                self.s2[:, i] = np.diag(K_star_star[:, :, i]) - np.sum(Lk ** 2, axis=0)
                self.stdv = np.sqrt(self.s2)
        elif method == 'cholesky':
            K_X_X = self.gp_kernel.compute_kernel(self.X,
                                                  self.X) + np.array(
                self.hyperparameters_obj.dict()[
                    'noise_level'] ** 2 * np.eye(len(self.X)))[:, :, None]
            K_star_X = self.gp_kernel.compute_kernel(self.X, X_star)
            K_star_star = self.gp_kernel.compute_kernel(X_star, X_star)
            L = np.zeros_like(K_X_X)
            self.mu = np.zeros((K_star_X.shape[1], K_X_X.shape[2]))
            self.s2 = np.zeros((K_star_X.shape[1], K_X_X.shape[2]))
            for i in range(K_X_X.shape[2]):
                L[:, :, i] = npla.cholesky( K_X_X[:, :, i] + 1e-10 * np.eye(K_X_X.shape[0]))
                Lk = np.squeeze(npla.solve(L[:, :, i], K_star_X[:, :, i]))
                self.mu[:, i] = self.y_mean + np.dot(Lk.T,npla.solve(
                                                                L[:, :, i],
                                                                self.y -
                                                                self.y_mean).flatten())
                self.s2[:, i] = np.diag(K_star_star[:, :, i]) - np.sum(
                    Lk ** 2, axis=0)
                self.stdv = np.sqrt(self.s2)
        return self.mu, self.stdv

    def is_positive_definite(self, K):
        if K.ndim > 3:
            raise ValueError("Input must be at most 3D.")
        if K.ndim == 2:
            K = K[:, :, np.newaxis]
        if K.shape[0] != K.shape[1]:
            raise ValueError("The first two dimensions must be equal.")
        for i in range(K.shape[2]):
            if not np.all(np.linalg.eigvals(K[:, :, i]) > 0):
                return False
        return True





    def compute_nll(self, hyperparameters):
        self.update_hyperparameters_and_debug(hyperparameters)
        self.reshape_X_and_y()



        if self.gp_algo == 'cholesky':
            #self.hyperparameters_obj.update(hyperparameters)
            K = self.gp_kernel.compute_kernel(self.X, self.X)
            K += np.repeat(np.array(np.eye(len(self.X)) * 1e-3)[:,:, np.newaxis], self.X.shape[1], axis=2)
            debug_K = np.squeeze(K)
            L = scipy.linalg.cholesky(K[:, :, 0], lower=True)
            n = len(self.y)
            one_vector = np.ones(n)
            y_adj = self.y - self.y_mean
            #debug_print(f"y_adj: {y_adj}")

            alpha = scipy.linalg.cho_solve((L, True), y_adj)

            # term_1_c_array = np.array()
            # term_2_c_array = np.array()
            # term_3_c_array = np.array()


            term_1_c = (0.5 * y_adj.T @ alpha).item()
            term_2_c = np.sum(np.log(np.diag(L)))
            term_3_c = 0.5 * n * np.log(2 * np.pi)

            # # DEBUG
            #
            # term_1_c_array.append(term_1_c_array)
            # term_2_c_array.append(term_2_c_array)
            # term_3_c_array.append(term_3_c_array)
            #
            # # /DEBUG

            nll = term_1_c + term_2_c + term_3_c

            out_c = {
                'nll': nll,
                'term_1': term_1_c,
                'term_2': term_2_c,
                'term_3': term_3_c
            }
            return out_c

        elif self.gp_algo == 'FITC_18_134':
            #out_f = self.run_FITC_18_134()

            out_f = self.gp_nll_algo_obj.compute()
            return out_f

        else:
            raise ValueError("Invalid compute_nll method")


    def reshape_X_and_y(self):
        if self.X.ndim == 1: self.X = self.X.reshape(-1, 1)
        if self.y.ndim == 1: self.y = self.y.reshape(-1, 1)

    def update_hyperparameters_and_debug(self, hyperparameters):
        if type(hyperparameters) == dict or type(
                hyperparameters) == Hyperparameters or type(
                hyperparameters) == np.ndarray:
            self.hyperparameters_obj.update(hyperparameters)
        else:
            raise ValueError(
                "Incorrect hyperparameter type: must be 'dict' or 'ndarray'")
        debug_var = self.hyperparameters_obj.array()
        debug_print(
            f"compute_NLL {self.gp_algo} hyperparameters: {debug_var}")

    # def flatten_params(self, params):
    #     flat_params = []
    #     for key, value in params.items():
    #         if isinstance(value, str):
    #             continue
    #         if isinstance(value, list):
    #             for item in value:
    #                 flat_params.extend(self.flatten_params(item))
    #         elif isinstance(value, dict):
    #             flat_params.extend(self.flatten_params(value))
    #         else:
    #             flat_params.append(value)
    #     return flat_params
    #
    # def reconstruct_params_implementation(self, flat_params, template):
    #     reconstructed_params = {}
    #     index = 0
    #     for key, value in template.items():
    #         if isinstance(value, str):
    #             reconstructed_params[key] = value
    #             continue
    #         if isinstance(value, list):
    #             reconstructed_params[key] = []
    #             for item in value:
    #                 reconstructed_item, item_length = self.reconstruct_params_implementation(flat_params[index:], item)
    #                 index += item_length
    #                 reconstructed_params[key].append(reconstructed_item)
    #         elif isinstance(value, dict):
    #             reconstructed_params[key], item_length = self.reconstruct_params_implementation(flat_params[index:], value)
    #             index += item_length
    #         else:
    #             reconstructed_params[key] = flat_params[index]
    #             index += 1
    #     return reconstructed_params, index
    #
    # def reconstruct_params(self, flat_params):
    #     reconstructed_params, index = self.reconstruct_params_implementation(flat_params,self.template)
    #     return reconstructed_params
