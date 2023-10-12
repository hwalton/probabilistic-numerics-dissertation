import numpy as np
import scipy
import freelunch
from numpy import linalg as npla

from hyperparameters import Hyperparameters
from gaussian_process_kernel import GaussianProcessKernel
from iterative_search import iterative_search_solve
from metropolis_hastings import metropolis_hastings_solve
from adam import adam_optimize
from utils import debug_print
from sklearn.cluster import KMeans


class GPModel:
    def __init__(self, kernel_type, X, y, solver_type = 'iterative_search', n_iter=10):
        self.hyperparameters_obj = Hyperparameters(kernel_type)
        self.initial_hyperparameters = self.hyperparameters_obj._initial_hyperparameters.copy()
        self.hyperparameter_bounds = self.hyperparameters_obj._hyperparameter_bounds.copy()
        self.X = X
        self.y = y
        self.n_iter = n_iter
        self.template = self.hyperparameters_obj._initial_hyperparameters.copy()
        self.solver_type = solver_type
        self.gp_kernel = GaussianProcessKernel(self.hyperparameters_obj)
        self.U = self.U_induced()

    def fit_model(self):
        self.gp_kernel.set_params(self.hyperparameters_obj)
        optimal_hyperparameters = self.get_optimal_hyperparameters()
        self.hyperparameters_obj.update(optimal_hyperparameters)
        debug_print(optimal_hyperparameters)
        nll = self.compute_nll(self.hyperparameters_obj)
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

    def U_induced(self, M_one_in = 1, method ='k_means'):
        M = len(self.X) // M_one_in
        if method == 'k_means':
            kmeans = KMeans(n_clusters=M, n_init = 3).fit(self.X)
            U = kmeans.cluster_centers_
        else:
            raise ValueError("Invalid inducing method")
        return U

    def K_XX_FITC(self):

        X = np.squeeze(self.X)
        U = np.squeeze(self.U)

        K_XU = np.squeeze(self.gp_kernel.compute_kernel(X, U))
        K_UX = np.squeeze(self.gp_kernel.compute_kernel(U, X))
        K_UU = np.squeeze(self.gp_kernel.compute_kernel(U, U))
        K_XX = np.squeeze(self.gp_kernel.compute_kernel(X, X))

        K_UU_stable = K_UU + 1e-6 * np.eye(U.shape[0])

        X = np.linalg.solve(K_UU_stable, K_XU.T)
        Q_XX = K_XU @ X

        K_XX_FITC = K_XX + Q_XX - K_XU @ X


        # out = {
        #     'K_XX_FITC': K_XX_FITC,
        #     'K_XU': K_XU,
        #     'K_UU': K_UU,
        #     'K_XX': K_XX,
        # }
        return K_XX_FITC, K_XU, K_UX, K_UU, K_XX, Q_XX

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
    def K_sigma_inv(self, method = 'woodbury'):
        if method == 'woodbury':
            K_XX_FITC, K_XU, K_UX, K_UU, K_XX, Q_XX = self.K_XX_FITC()
            sigma_n_neg2 = np.multiply(self.hyperparameters_obj.dict()['noise_level'] ** -2, np.eye(len(self.X)))

            X = np.linalg.solve(K_UU, K_XU)
            var = sigma_n_neg2 @ K_XU @ (X + K_UX @ sigma_n_neg2 @ K_XU @ K_UX) @ sigma_n_neg2
            #var2 = sigma_n @ K_XU @ (np.linalg.inv(K_UU) + np.array(K_UX @ sigma_n @ K_XU)) @ K_UX @ sigma_n
            #debug_print(f"var == var2: {np.allclose(var,var2, atol = 1E-3)}")
            out = sigma_n_neg2 - var

        else:
            raise ValueError("Invalid inducing method")
        return out
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
            K_sigma_inv = self.K_sigma_inv()
            L = np.zeros_like(K_X_X)
            self.mu = np.zeros((K_star_X.shape[1], K_X_X.shape[2]))
            self.s2 = np.zeros((K_star_X.shape[1], K_X_X.shape[2]))
            for i in range(K_X_X.shape[2]):
                L[:, :, i] = npla.cholesky(K_X_X[:, :, i] + 1e-10 * np.eye(K_X_X.shape[0]))
                Lk = np.squeeze(npla.solve(L[:, :, i], K_star_X[:,:,i]))
                alpha = K_sigma_inv @ self.y
                self.mu[:, i] = self.hyperparameters_obj.dict()['mean_func_c'] + np.array(K_star_X.T @ alpha).flatten()
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
                L[:, :, i] = npla.cholesky(
                    K_X_X[:, :, i] + 1e-10 * np.eye(K_X_X.shape[0]))
                Lk = np.squeeze(npla.solve(L[:, :, i], K_star_X[:, :, i]))
                self.mu[:, i] = self.hyperparameters_obj.dict()[
                                    'mean_func_c'] + np.dot(Lk.T,
                                                            npla.solve(
                                                                L[:, :, i],
                                                                self.y -
                                                                self.hyperparameters_obj.dict()[
                                                                    'mean_func_c'])).flatten()
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

    def compute_nll(self, hyperparameters, method = 'FITC2'):
        if method == 'cholesky':
            if type(hyperparameters) == dict:
                self.hyperparameters_obj.update(hyperparameters)
            if type(hyperparameters) == Hyperparameters:
                self.hyperparameters_obj.update(hyperparameters)
            elif type(hyperparameters) == np.ndarray:
                hyperparameters = self.hyperparameters_obj.reconstruct_params(hyperparameters)
            else:
                raise ValueError("Incorrect hyperparameter type: must be 'dict' or 'ndarray'")

            if self.X.ndim == 1: self.X = self.X.reshape(-1, 1)
            if self.y.ndim == 1: self.y = self.y.reshape(-1, 1)
            self.hyperparameters_obj.update(hyperparameters)
            K = self.gp_kernel.compute_kernel(self.X, self.X)
            K += np.repeat(np.array(np.eye(len(self.X)) * 1e-3)[:,:, np.newaxis], self.X.shape[1], axis=2)
            for i in range(K.shape[2]):
                L = scipy.linalg.cholesky(K[:, :, i], lower=True)
                n = len(self.y)
                one_vector = np.ones(n)
                y_adj = self.y - self.hyperparameters_obj.dict()['mean_func_c']
                alpha = scipy.linalg.cho_solve((L, True), y_adj)
                nll = 0.5 * y_adj.T @ alpha + np.sum(np.log(np.diag(L))) + 0.5 * n * np.log(2 * np.pi)
        if method == 'FITC':
            if type(hyperparameters) == dict:
                self.hyperparameters_obj.update(hyperparameters)
            if type(hyperparameters) == Hyperparameters:
                self.hyperparameters_obj.update(hyperparameters)
            elif type(hyperparameters) == np.ndarray:
                hyperparameters = self.hyperparameters_obj.reconstruct_params(hyperparameters)
            else:
                raise ValueError("Incorrect hyperparameter type: must be 'dict' or 'ndarray'")

            if self.X.ndim == 1: self.X = self.X.reshape(-1, 1)
            if self.y.ndim == 1: self.y = self.y.reshape(-1, 1)
            self.hyperparameters_obj.update(hyperparameters)
            K = self.gp_kernel.compute_kernel(self.X, self.X)
            K += np.repeat(np.array(np.eye(len(self.X)) * 1e-3)[:,:, np.newaxis], self.X.shape[1], axis=2)
            for i in range(K.shape[2]):
                n = len(self.y)
                L = scipy.linalg.cholesky(K[:, :, i], lower=True)
                y_adj = self.y - self.hyperparameters_obj.dict()['mean_func_c']
                K_sigma_inv = self.K_sigma_inv()
                nll = -0.5 * n * np.log(2*np.pi) - np.sum(np.log(np.diag(L))) - 0.5 * y_adj.T @ K_sigma_inv @ y_adj
        if method == 'FITC2':
            if type(hyperparameters) == dict:
                self.hyperparameters_obj.update(hyperparameters)
            if type(hyperparameters) == Hyperparameters:
                self.hyperparameters_obj.update(hyperparameters)
            elif type(hyperparameters) == np.ndarray:
                hyperparameters = self.hyperparameters_obj.reconstruct_params(
                    hyperparameters)
            else:
                raise ValueError(
                    "Incorrect hyperparameter type: must be 'dict' or 'ndarray'")

            if self.X.ndim == 1: self.X = self.X.reshape(-1, 1)
            if self.y.ndim == 1: self.y = self.y.reshape(-1, 1)
            self.hyperparameters_obj.update(hyperparameters)
            #K = self.gp_kernel.compute_kernel(self.X, self.X)
            #K += np.repeat(
            #    np.array(np.eye(len(self.X)) * 1e-3)[:, :, np.newaxis],
            #   self.X.shape[1], axis=2)
            for i in range(1):
                n = len(self.y)
            #    L = scipy.linalg.cholesky(K[:, :, i], lower=True)
                y_adj = np.squeeze(self.y - self.hyperparameters_obj.dict()[
                    'mean_func_c'])
                K_XX_FITC, K_XU, K_UX, K_UU, K_XX, Q_XX = self.K_XX_FITC()
                beta = self.hyperparameters_obj.dict()['noise_level'] ** -1
                cov = Q_XX + beta ** -1 * np.eye(len(Q_XX))
                log_likelihood = scipy.stats.multivariate_normal.logpdf(y_adj,
                                                            mean=np.zeros(
                                                                n),
                                                            cov=cov, allow_singular= True)

                # Compute the second term of the lower bound
                trace_term = 0.5 * beta * np.trace(K_XX - Q_XX)

                # Compute the lower bound on the log marginal likelihood
                nll = np.array(-log_likelihood - trace_term)
        else:
            raise ValueError("Invalid compute_nll method")
        return nll.item()

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
