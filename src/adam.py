import numpy as np
from utils import debug_print

import numpy as np
from utils import debug_print


def adam_optimize(objective_function, X, y, inducing_points, params, kernel, reconstruct_params, initial_lr=0.06,
                  beta1=0.9, beta2=0.999, epsilon=1e-4, epochs=5, lr_decay_rate=0.9):
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    lr = initial_lr  # Initialize the learning rate

    for epoch in range(epochs):
        debug_print(f"Epoch: {epoch}/{epochs}")

        # Compute all gradients at once
        grads = compute_all_gradients(objective_function, X, y, inducing_points, params, kernel, reconstruct_params)

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


def compute_all_gradients(objective_function, X, y, inducing_points, params, kernel, reconstruct_params):
    # Define a kernel function and its derivative w.r.t. theta_j
    def kernel_function(x, y, theta, kernel):
        kernel.hyperparameters_obj.update(theta)
        out = np.squeeze(kernel.compute_kernel(x,y))
        return out

    def dkernel_dtheta_j(x, y, theta, j, kernel):
        derivative = kernel.compute_kernel_derivative(x, y, j)
        out = derivative(theta)
        return out

    # Compute the covariance matrices
    K_ff = np.array([[kernel_function(xi, xj, params, kernel) for xj in X] for xi in X])
    K_uu = np.array([[kernel_function(zi, zj, params, kernel) for zj in inducing_points] for zi in inducing_points])
    K_uf = np.array([[kernel_function(zi, xj, params, kernel) for xj in X] for zi in inducing_points])
    K_fu = K_uf.T

    # Compute the FITC approximated covariance matrix
    Q_ff = K_fu @ np.linalg.solve(K_uu, K_uf)
    K_tilde = np.diag(K_ff - np.diag(Q_ff))

    # Compute the FITC approximated alpha
    jitter = 1e-2
    L = np.linalg.cholesky(Q_ff + np.diag(K_tilde) + jitter * np.eye(Q_ff.shape[0]))

    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))

    # Initialize an array to store the gradients
    gradients = np.zeros_like(params)

    # Compute the gradient for each hyperparameter
    for j in range(len(params)):
        dK_uu_dtheta = np.array([[dkernel_dtheta_j(zi, zj, params, j, kernel) for zj in inducing_points] for zi in inducing_points])
        dK_uf_dtheta = np.array([[dkernel_dtheta_j(zi, xj, params, j, kernel) for xj in X] for zi in inducing_points])
        dK_fu_dtheta = dK_uf_dtheta.T

        # Compute the gradient of the FITC approximated likelihood
        dQ_ff_dtheta = dK_fu_dtheta @ np.linalg.solve(K_uu, K_uf) + K_fu @ np.linalg.solve(K_uu, dK_uf_dtheta)
        dK_tilde_dtheta = np.diag(dK_fu_dtheta @ np.linalg.solve(K_uu, K_uf) - dQ_ff_dtheta)

        gradient = -0.5 * np.trace(np.linalg.solve(L, dQ_ff_dtheta + np.diag(dK_tilde_dtheta)) @ np.linalg.solve(L.T, alpha @ alpha.T - np.linalg.inv(L)))
        gradients[j] = gradient

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

