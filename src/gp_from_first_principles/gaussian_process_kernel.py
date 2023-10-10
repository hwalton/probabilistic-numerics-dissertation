import numpy as np
from scipy.special import kv
from utils import debug_print

class GaussianProcessKernel:
    def __init__(self, **kwargs):
        self.params = kwargs

    def set_params(self, hyperparameters):
        self.params = hyperparameters
        self.kernel_type = self.params['kernel_type']

    def compute_kernel(self, X1, X2):
        if X1.ndim == 1:
            X1 = X1.reshape(-1, 1)
        if X2.ndim == 1:
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
            # Add more kernel types as needed
        }

        try:
            kernel_function = kernel_methods[self.kernel_type]
        except KeyError:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

        return kernel_function(X1, X2, **self.params)

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

    def periodic_kernel(self, X1, X2, **params):
        delta_X = X1[:, None, :] - X2[None, :, :]
        out = params['sigma'] ** 2 * np.exp(
            -2 * np.sin(np.pi * np.abs(delta_X) / params['p']) ** 2 / params['l'] ** 2)
        return out

    def squared_exponential_kernel(self, X1, X2, **params):
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

    def cosine_kernel(self, X1, X2, sigma, p):
        delta_X = np.sum(X1[:, None, :] - X2[None, :, :], axis=-1)
        return sigma ** 2 * np.cos(2 * np.pi * delta_X / p)

    def white_noise_kernel(self, X1, X2, **params):
        delta_X = X1[:, None, :] - X2[None, :, :]
        return params['sigma'] ** 2 * np.where(delta_X == 0, 1, 0)

    def polynomial_kernel(self, X1, X2, alpha, beta, d):
        return (alpha + beta * np.dot(X1, X2.T)) ** d
