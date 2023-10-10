import numpy as np
import scipy
import freelunch
from numpy import linalg as npla

from gaussian_process_kernel import GaussianProcessKernel
from iterative_search import iterative_search_solve
from metropolis_hastings import metropolis_hastings_solve
from adam import adam_optimize
from utils import debug_print


class GPModel:
    def __init__(self, initial_hyperparameters, hyperparameter_bounds, X, y, solver_type = 'iterative_search', n_iter=10):
        self.initial_hyperparameters = initial_hyperparameters
        self.hyperparameter_bounds = hyperparameter_bounds
        self.X = X
        self.y = y
        self.n_iter = n_iter
        self.template = initial_hyperparameters
        self.solver_type = solver_type

    def fit_model(self):
        self.gp_kernel = GaussianProcessKernel(**self.initial_hyperparameters)
        self.gp_kernel.set_params(self.initial_hyperparameters)
        self.optimal_hyperparameters = self.get_optimal_hyperparameters()
        debug_print(self.optimal_hyperparameters)
        nll = self.compute_nll(self.optimal_hyperparameters)
        debug_print(nll)
        return(nll)

    def solve(self, solver_type, initial_hyperparameters_array, bounds_array):
        if solver_type == 'iterative_search':
            return iterative_search_solve(initial_hyperparameters_array, bounds_array, self.compute_nll, n_iter=self.n_iter)
        elif solver_type == 'metropolis_hastings':
            return metropolis_hastings_solve(initial_hyperparameters_array, bounds_array, self.compute_nll, n_iter=self.n_iter, sample_length = len(self.X))
        elif solver_type == 'free_lunch':
            self.solver = freelunch.DE(self.compute_nll, bounds = bounds_array)
            return self.solver()
        elif solver_type == 'adam':
            return adam_optimize(self.compute_nll,initial_hyperparameters_array)
        else:
            assert False, "invalid solver_type"

    def get_optimal_hyperparameters(self):
        initial_hyperparameters_array = np.array(self.flatten_params(self.initial_hyperparameters))
        bounds_array = self.flatten_params(self.hyperparameter_bounds)
        debug_print("solving")
        optimal_hyperparameters = self.solve(self.solver_type, initial_hyperparameters_array, bounds_array).T
        debug_print("solved")
        return self.reconstruct_params(optimal_hyperparameters)


    def predict(self, X_star):
        K_X_X = self.gp_kernel.compute_kernel(self.X, self.X) + np.array(self.optimal_hyperparameters['noise_level'] ** 2 * np.eye(len(self.X)))[:,:,None]
        K_star_X = self.gp_kernel.compute_kernel(self.X, X_star)
        K_star_star = self.gp_kernel.compute_kernel(X_star, X_star)
        L = np.zeros_like(K_X_X)
        self.mu = np.zeros((K_star_X.shape[1], K_X_X.shape[2]))
        self.s2 = np.zeros((K_star_X.shape[1], K_X_X.shape[2]))
        for i in range(K_X_X.shape[2]):
            L[:, :, i] = npla.cholesky(K_X_X[:, :, i] + 1e-10 * np.eye(K_X_X.shape[0]))
            Lk = np.squeeze(npla.solve(L[:, :, i], K_star_X[:,:,i]))
            self.mu[:, i] = self.optimal_hyperparameters['mean_func_c'] + np.dot(Lk.T, npla.solve(L[:, :, i], self.y - self.optimal_hyperparameters['mean_func_c'])).flatten()
            self.s2[:, i] = np.diag(K_star_star[:, :, i]) - np.sum(Lk ** 2, axis=0)
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
        if type(hyperparameters) == dict:
            pass
        elif type(hyperparameters) == np.ndarray:
            hyperparameters = self.reconstruct_params(hyperparameters)
        else:
            raise ValueError("Incorrect hyperparameter type: must be 'dict' or 'ndarray'")

        if self.X.ndim == 1: self.X = self.X.reshape(-1, 1)
        if self.y.ndim == 1: self.y = self.y.reshape(-1, 1)
        self.gp_kernel.set_params(hyperparameters)
        K = self.gp_kernel.compute_kernel(self.X, self.X)
        K += np.repeat(np.array(np.eye(len(self.X)) * 1e-3)[:,:, np.newaxis], self.X.shape[1], axis=2)
        for i in range(K.shape[2]):
            L = scipy.linalg.cholesky(K[:, :, i], lower=True)
            n = len(self.y)
            one_vector = np.ones(n)
            y_adj = self.y - hyperparameters['mean_func_c']
            alpha = scipy.linalg.cho_solve((L, True), y_adj)
            nll = 0.5 * y_adj.T @ alpha + np.sum(np.log(np.diag(L))) + 0.5 * n * np.log(2 * np.pi)

        return nll.item()

    def flatten_params(self, params):
        flat_params = []
        for key, value in params.items():
            if isinstance(value, str):
                continue
            if isinstance(value, list):
                for item in value:
                    flat_params.extend(self.flatten_params(item))
            elif isinstance(value, dict):
                flat_params.extend(self.flatten_params(value))
            else:
                flat_params.append(value)
        return flat_params

    def reconstruct_params_implementation(self, flat_params, template):
        reconstructed_params = {}
        index = 0
        for key, value in template.items():
            if isinstance(value, str):
                reconstructed_params[key] = value
                continue
            if isinstance(value, list):
                reconstructed_params[key] = []
                for item in value:
                    reconstructed_item, item_length = self.reconstruct_params_implementation(flat_params[index:], item)
                    index += item_length
                    reconstructed_params[key].append(reconstructed_item)
            elif isinstance(value, dict):
                reconstructed_params[key], item_length = self.reconstruct_params_implementation(flat_params[index:], value)
                index += item_length
            else:
                reconstructed_params[key] = flat_params[index]
                index += 1
        return reconstructed_params, index

    def reconstruct_params(self, flat_params):
        reconstructed_params, index = self.reconstruct_params_implementation(flat_params,self.template)
        return reconstructed_params
