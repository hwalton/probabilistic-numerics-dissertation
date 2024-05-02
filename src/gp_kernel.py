import numpy as np
from scipy.special import kv
from utils import debug_print

from hyperparameters import Hyperparameters

class GaussianProcessKernel:
    def __init__(self, hyperparameters_obj):
        self.hyperparameters_obj = hyperparameters_obj
        self.kernel_type = self.hyperparameters_obj.dict()['kernel_type']

    def set_params(self, hyperparameters):
        if type(hyperparameters) == Hyperparameters:
            self.hyperparameters_obj = hyperparameters
        if type(hyperparameters) == dict:
            self.hyperparameters_obj = Hyperparameters(hyperparameters['kernel_type'])

    def compute_kernel_SE_fourier(self, xi_1, xi_2 = 0):
        delta = xi_1 - xi_2
        return self.hyperparameters_obj.dict()['sigma'] ** 2 * np.exp( - 0.5 * self.hyperparameters_obj.dict()['l'] ** 2 * delta ** 2) * self.hyperparameters_obj.dict()['l'] / np.sqrt(2 * np.pi)

    def compute_kernel_SE_exponent(self, xi):
        out = -0.5 * xi ** 2 / (self.hyperparameters_obj.dict()['l'] ** 2)
        return out

    def compute_kernel_fourier_SE_squared(self, xi):
        out = self.hyperparameters_obj.dict()['sigma'] ** 4 * np.exp(- 0.25 * self.hyperparameters_obj.dict()['l'] ** 2 * xi ** 2) * self.hyperparameters_obj.dict()['l'] / ( np.sqrt(2) * np.sqrt(2 * np.pi))
        return out

    def compute_kernel(self, X1, X2):
        # Check if X1 is an integer and convert it to an ndarray of shape (-1, 1) if so
        if isinstance(X1, int):
            X1 = np.array([X1]).reshape(-1, 1)
        elif X1.ndim == 1:
            X1 = X1.reshape(-1, 1)

        # Check if X2 is an integer and convert it to an ndarray of shape (-1, 1) if so
        if isinstance(X2, int):
            X2 = np.array([X2]).reshape(-1, 1)
        elif X2.ndim == 1:
            X2 = X2.reshape(-1, 1)

        kernel_methods = {
            'linear': self.linear_kernel,
            'periodic': self.periodic_kernel,
            'squared_exponential': self.squared_exponential_kernel,
            'matern': self.matern_kernel,
            'rational_quadratic': self.rational_quadratic_kernel,
            'exponential': self.exponential_kernel,
            'cosine': self.cosine_kernel,
            'white_noise': self.white_noise_kernel,
            'polynomial': self.polynomial_kernel,
            'p_se_composite': self.p_se_composite_kernel,
            'wn_se_composite': self.wn_se_composite_kernel,
            'cosine_composite': self.cosine_composite_kernel
            # Add more kernel types as needed
        }

        try:
            kernel_function = kernel_methods[self.hyperparameters_obj.dict()['kernel_type']]
        except KeyError:
            raise ValueError(f"Unknown kernel type: {self.hyperparameters_obj.dict()['kernel_type']}")

        return kernel_function(X1, X2, **self.hyperparameters_obj.dict())

    def p_se_composite_kernel(self, X1, X2, **params):
        #test3 = X1.shape
        periodic_sum = np.zeros((X1.shape[0], X2.shape[0], X1.shape[1]))
        for param in params['periodic_params']:
            periodic_sum += self.periodic_kernel(X1, X2, **param)

        se_kernel = self.squared_exponential_kernel(X1, X2, **params['se_params'])

        composite_kernel = np.multiply(periodic_sum, se_kernel)

        return composite_kernel

    def wn_se_composite_kernel(self, X1, X2, **params):
        white_noise_kernel = self.white_noise_kernel(X1, X2, **params['wn_params'])
        squared_exponential_kernel = self.squared_exponential_kernel(X1, X2, **params['se_params'])

        composite_kernel = white_noise_kernel + squared_exponential_kernel

        return composite_kernel

    def linear_kernel(self, X1, X2):
        return np.dot(X1, X2.T)

    import numpy as np

    def periodic_kernel(self, X1, X2, **params):
        if np.isscalar(X1):
            X1 = np.array([X1])
        if np.isscalar(X2):
            X2 = np.array([X2])

        if X1.ndim == 1:
            X1 = X1[:, None]
        if X2.ndim == 1:
            X2 = X2[:, None]

        delta_X = X1[:, None, :] - X2[None, :, :]

        out = params['sigma'] ** 2 * np.exp(
            -2 * np.sin(np.pi * np.abs(delta_X) / params['p']) ** 2 / params['l'] ** 2)
        out = np.clip(out, 1E-6, 1E6) + 1E-6

        return out

    def squared_exponential_kernel(self, X1, X2, **params):
        if np.isscalar(X1):
            X1 = np.array([X1])
        if np.isscalar(X2):
            X2 = np.array([X2])

        if X1.ndim == 1:
            X1 = X1[:, None]
        if X2.ndim == 1:
            X2 = X2[:, None]

        delta_X = X1[:, None, :] - X2[None, :, :]
        out = params['sigma'] ** 2 * np.exp(
            -0.5 * (delta_X ** 2) / params['l'] ** 2)
        return out

    def matern_kernel(self, X1, X2, sigma, nu, l):
        delta_X = np.sqrt(
            np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=-1))
        const_term = (2 ** (1 - nu)) / np.math.gamma(nu)
        exp_term = (-np.sqrt(2 * nu) * delta_X) / l
        bessel_term = kv(nu, np.sqrt(2 * nu) * delta_X / l)
        return sigma ** 2 * const_term * (
                    delta_X / l) ** nu * bessel_term * np.exp(exp_term)

    def rational_quadratic_kernel(self, X1, X2, sigma, alpha, l):
        delta_X = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=-1)
        return sigma ** 2 * (1 + delta_X / (2 * alpha * l ** 2)) ** (-alpha)

    def exponential_kernel(self, X1, X2, sigma, l):
        delta_X = np.sqrt(
            np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=-1))
        return sigma ** 2 * np.exp(-delta_X / l)

    def cosine_kernel(self, X1, X2, **params):
        delta_X = X1[:, None, :] - X2[None, :, :]
        return params['sigma'] ** 2 * np.cos(2 * np.pi * delta_X / params['p'])+1E-6

    def cosine_composite_kernel(self, X1, X2, **params):
        #test3 = X1.shape
        cosine_sum = np.zeros((X1.shape[0], X2.shape[0], X1.shape[1]))
        for param in params['cosine_params']:
            cosine_sum += self.cosine_kernel(X1, X2, **param)

        return cosine_sum

    def white_noise_kernel(self, X1, X2, **params):
        if np.isscalar(X1):
            X1 = np.array([X1])
        if np.isscalar(X2):
            X2 = np.array([X2])

        if X1.ndim == 1:
            X1 = X1[:, None]
        if X2.ndim == 1:
            X2 = X2[:, None]

        delta_X = X1[:, None, :] - X2[None, :, :]
        return params['sigma'] ** 2 * np.where(delta_X == 0, 1, 0)

    def polynomial_kernel(self, X1, X2, alpha, beta, d):
        return (alpha + beta * np.dot(X1, X2.T)) ** d

    def compute_kernel_derivative(self, X1, X2, j):
        if X1.ndim == 1:
            X1 = X1.reshape(-1, 1)
        if X2.ndim == 1:
            X2 = X2.reshape(-1, 1)

        kernel_derivative_methods = {
            #'linear': self.linear_kernel_derivative,
            'periodic': self.periodic_kernel_derivative,
            # 'squared_exponential': self.squared_exponential_kernel_derivative,
            # 'matern': self.matern_kernel_derivative,
            # 'rational_quadratic': self.rational_quadratic_kernel_derivative,
            # 'exponential': self.exponential_kernel_derivative,
            # 'cosine': self.cosine_kernel_derivative,
            # 'white_noise': self.white_noise_kernel_derivative,
            # 'polynomial': self.polynomial_kernel_derivative,
            # 'p_se_composite': self.p_se_composite_kernel_derivative,
            # 'wn_se_composite': self.wn_se_composite_kernel_derivative,
            # Add more kernel types as needed
        }

        try:
            kernel_derivative_function = kernel_derivative_methods[
                self.kernel_type]
        except KeyError:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

        return kernel_derivative_function(X1, X2, j)

    def periodic_kernel_derivative(self,X1, X2, j):
        def periodic_kernel_derivative_sigma(theta):
            delta_X = X1[:, None, :] - X2[None, :, :]
            base_kernel = np.exp(
                -2 * np.sin(np.pi * np.abs(delta_X) / theta[2]) ** 2 /
                theta[1] ** 2)
            derivative = 2 * theta[0] * base_kernel
            return np.squeeze(derivative)

        import numpy as np

        def periodic_kernel_derivative_p(theta):
            delta_X = X1[:, None, :] - X2[None, :, :]
            base_kernel = np.exp(
                -2 * np.sin(np.pi * np.abs(delta_X) / theta[2]) ** 2 /
                theta[1] ** 2)
            derivative_p = (
                    theta[0] ** 2
                    * base_kernel
                    * 2
                    * (np.sin(np.pi * np.abs(delta_X) / theta[2]) ** 2)
                    * (np.pi * np.abs(delta_X) / theta[2] ** 2)
            )
            return np.squeeze(derivative_p)

        def periodic_kernel_derivative_l(theta):
            delta_X = X1[:, None, :] - X2[None, :, :]
            base_kernel = np.exp(
                -2 * np.sin(np.pi * np.abs(delta_X) / theta[2]) ** 2 /
                theta[1] ** 2)
            derivative_l = (
                    theta[0] ** 2
                    * base_kernel
                    * 4
                    * (np.sin(np.pi * np.abs(delta_X) / theta[2]) ** 2)
                    / (theta[1] ** 3)
            )
            return np.squeeze(derivative_l)

        func_array = [
            periodic_kernel_derivative_sigma,
            periodic_kernel_derivative_p,
            periodic_kernel_derivative_l,
            lambda theta: 0,  # Always returns 0
            lambda theta: 0  # Always returns 0
        ]

        return func_array[j]
