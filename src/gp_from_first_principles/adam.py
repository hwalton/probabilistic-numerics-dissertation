import numpy as np
from utils import debug_print

def adam_optimize(objective_function, params, lr=0.001, beta1=0.9, beta2=0.999,
                  epsilon=1e-4, epochs=1000):

    m = np.zeros_like(params)
    v = np.zeros_like(params)

    for epoch in range(epochs):
        # Compute the gradient of the objective function
        grad = compute_gradient(objective_function, params)

        # Update biased first and second moment estimates
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2

        # Compute bias-corrected moment estimates
        m_hat = m / (1 - beta1 ** (epoch + 1))
        v_hat = v / (1 - beta2 ** (epoch + 1))

        # Update parameters
        params = params - lr * m_hat / (np.sqrt(v_hat) + epsilon)

    return params


import numpy as np


# def compute_gradient(X, y, theta, j):
#     # Define a kernel function and its derivative w.r.t. theta_j
#     def kernel_function(x, y, theta):
#         # Implement your kernel function here using theta
#         pass
#
#     def dkernel_dtheta_j(x, y, theta, j):
#         # Implement the derivative of your kernel function w.r.t. theta_j here using theta
#         pass
#
#     # Compute the covariance matrix K
#     K = np.array([[kernel_function(xi, xj, theta) for xj in X] for xi in X])
#
#     # Access mean_func_c as the second to last item in theta
#     mean_func_c = theta[-2]
#
#     # Compute alpha = K^(-1)(y - mean_func_c)
#     alpha = np.linalg.solve(K, y - mean_func_c)
#
#     # Compute the derivative of K w.r.t. theta_j
#     dK_dthetaj = np.array(
#         [[dkernel_dtheta_j(xi, xj, theta, j) for xj in X] for xi in X])
#
#     # Compute the gradient w.r.t. the covariance function hyperparameters
#     gradient_cov = 0.5 * np.trace(
#         (np.outer(alpha, alpha) - np.linalg.inv(K)) @ dK_dthetaj)
#
#     # If j corresponds to mean_func_c, compute its gradient
#     if j == len(theta) - 2:  # Check if j corresponds to mean_func_c
#         gradient_mean = -np.sum(alpha)
#     else:
#         gradient_mean = 0
#
#     # Total gradient is the sum of the two components
#     gradient = gradient_cov + gradient_mean
#
#     return gradient

def compute_gradient(f, params, epsilon=1e-5):
    """
    Compute the numerical gradient of the function `f` at `params`.

    :param f: The function for which to compute the gradient.
    :param params: The point at which to evaluate the gradient.
    :param epsilon: A small scalar to use for finite difference.
    :return: The gradient of `f` at `params`.
    """
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_up = params.copy()
        params_down = params.copy()
        params_up[i] += epsilon
        params_down[i] -= epsilon
        grad[i] = (f(params_up) - f(params_down)) / (2 * epsilon)
    return grad

