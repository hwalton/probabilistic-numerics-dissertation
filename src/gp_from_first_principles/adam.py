import numpy as np
from utils import debug_print

import numpy as np
from utils import debug_print


def adam_optimize(objective_function, X, y, params, kernel, reconstruct_params, initial_lr=0.06,
                  beta1=0.9, beta2=0.999, epsilon=1e-4, epochs=5, lr_decay_rate=0.9):
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    lr = initial_lr  # Initialize the learning rate

    for epoch in range(epochs):
        debug_print(f"Epoch: {epoch}/{epochs}")

        # Compute all gradients at once
        grads = compute_all_gradients(objective_function, X, y, params, kernel, reconstruct_params)

        for j, grad_j in enumerate(grads):
            # Update biased first and second moment estimates for j-th parameter
            m[j] = beta1 * m[j] + (1 - beta1) * grad_j
            v[j] = beta2 * v[j] + (1 - beta2) * grad_j ** 2

            # Compute bias-corrected moment estimates for j-th parameter
            m_hat_j = m[j] / (1 - beta1 ** (epoch + 1))
            v_hat_j = v[j] / (1 - beta2 ** (epoch + 1))

            # Update the j-th parameter with the scheduled learning rate
            params[j] = params[j] - lr * m_hat_j / (np.sqrt(v_hat_j) + epsilon)

        # Update the learning rate using exponential decay
        lr *= lr_decay_rate

    debug_print(f"params out {params}")
    return params


import numpy as np

import numpy as np


def compute_all_gradients(objective_function,X, y, params, kernel, reconstruct_params):
    # Extract X, y, and theta from params or modify as per your use case
    # X, y, theta = ...

    # Define a kernel function and its derivative w.r.t. theta_j
    def kernel_function(x, y, theta, kernel):
        kernel.hyperparameters_obj.update(theta)
        out = np.squeeze(kernel.compute_kernel(x,y))
        return out

    def dkernel_dtheta_j(x, y, theta, j, kernel):
        derivative = kernel.compute_kernel_derivative(x, y, j)
        #debug_print(theta)
        out = derivative(theta)
        #debug_print(out)
        return out

    # Compute the covariance matrix K
    K = np.array([[kernel_function(xi, xj, params, kernel) for xj in X] for xi in X])

    # Access mean_func_c as the second to last item in params
    mean_func_c = params[-2]

    # Compute alpha = K^(-1)(y - mean_func_c)
    alpha = np.linalg.solve(K, y - mean_func_c)

    # Initialize an array to store the gradients
    gradients = np.zeros_like(params)

    # Compute the gradient for each parameter
    for j in range(len(params)):
        # Compute the derivative of K w.r.t. theta_j
        dK_dthetaj = np.array(
            [[dkernel_dtheta_j(xi, xj, params, j, kernel) for xj in X] for xi in X])

        # Compute the gradient w.r.t. the covariance function hyperparameters
        gradient_cov = 0.5 * np.trace(
            (np.outer(alpha, alpha) - np.linalg.inv(K)) @ dK_dthetaj)

        # If j corresponds to mean_func_c, compute its gradient
        if j == len(params) - 2:  # Check if j corresponds to mean_func_c
            gradient_mean = -np.sum(alpha)
        else:
            gradient_mean = 0

        # Total gradient is the sum of the two components
        gradients[j] = gradient_cov + gradient_mean

    return gradients

# def compute_gradient(f, params, epsilon=1e-5):
#     """
#     Compute the numerical gradient of the function `f` at `params`.
#
#     :param f: The function for which to compute the gradient.
#     :param params: The point at which to evaluate the gradient.
#     :param epsilon: A small scalar to use for finite difference.
#     :return: The gradient of `f` at `params`.
#     """
#     grad = np.zeros_like(params)
#     for i in range(len(params)):
#         params_up = params.copy()
#         params_down = params.copy()
#         params_up[i] += epsilon
#         params_down[i] -= epsilon
#         grad[i] = (f(params_up) - f(params_down)) / (2 * epsilon)
#     return grad

