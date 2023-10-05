from scipy.io import loadmat
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import kv
import numpy.linalg as npla
import time
from scipy.optimize import minimize

developer = False


def load_data(start = 0, length = 65536):

    assert length <= 65536, "Length must be less than or equal to 65536"

    #data collected during MEC326
    input = np.loadtxt('datasets/input.csv', delimiter=',')
    output = np.loadtxt('datasets/output.csv', delimiter=',')
    time = np.loadtxt('datasets/time.csv', delimiter=',')

    input= input[start:start+length]
    output = output[start:start+length]
    time = time[start:start+length]

    return input, output, time

def plot_data(force_input, force_response, prediction, time, time_test):
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)  # 2 rows, 1 column, plot 1
    plt.scatter(time, force_input, label='Input', color='blue')
    plt.xlabel('Time')
    plt.ylabel('Input')
    plt.title('Input over Time')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)  # 2 rows, 1 column, plot 2
    plt.scatter(time, force_response, label='Force Response', color='green')
    plt.scatter(time_test, prediction[0], label='Predicted Mean', color='red')
    plt.scatter(time_test, prediction[1], label='Predicted Std Dev', color='blue')
    plt.xlabel('Time')
    plt.ylabel('Force Response')
    plt.title('Force Response over Time')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()  # Adjusts subplot params for better layout
    plt.show()

def format_data(X):
    if X.ndim == 1: X = X.reshape(-1,1)
    return X

class GaussianProcessKernel:
    def __init__(self, **kwargs):
        self.params = kwargs

    def set_params(self, hyperparameters):
        self.params = hyperparameters

    def compute_kernel(self, X1, X2):
        if developer == True: X1_shape_before = X1.shape
        if X1.ndim == 1: X1 = X1.reshape(-1,1)
        if developer == True: X1_shape_after = X1.shape
        if developer == True: X2_shape_before = X2.shape
        if X2.ndim == 1: X2 = X2.reshape(-1,1)
        if developer == True: X2_shape_after = X2.shape
        if self.params['kernel_type'] == 'linear':
            return self.linear_kernel(X1, X2)
        elif self.params['kernel_type'] == 'periodic':
            return self.periodic_kernel(X1, X2, **self.params)
        elif self.params['kernel_type'] == 'squared_exponential':
            return self.squared_exponential_kernel(X1, X2, **self.params)
        elif self.params['kernel_type'] == 'matern':
            return self.matern_kernel(X1, X2, **self.params)
        elif self.params['kernel_type'] == 'rational_quadratic':
            return self.rational_quadratic_kernel(X1, X2, **self.params)
        elif self.params['kernel_type'] == 'exponential':
            return self.exponential_kernel(X1, X2, **self.params)
        elif self.params['kernel_type'] == 'cosine':
            return self.cosine_kernel(X1, X2, **self.params)
        elif self.params['kernel_type'] == 'white_noise':
            return self.white_noise_kernel(X1, X2, **self.params)
        elif self.params['kernel_type'] == 'polynomial':
            return self.polynomial_kernel(X1, X2, **self.params)
        elif self.params['kernel_type'] == 'composite':
            return self.composite_kernel(X1, X2, **self.params)
        # Add more kernel types as needed
        else:
            raise ValueError(f"Unknown kernel type: {self.params['kernel_type']}")

    def composite_kernel(self, X1, X2, **hyperparameters):

        # Sum of periodic kernels
        test3 = X1.shape
        periodic_sum = np.zeros((X1.shape[0], X2.shape[0], X1.shape[1]))
        for params in hyperparameters['periodic_params']:
            periodic_sum += self.periodic_kernel(X1, X2, **params)

        # Squared exponential kernel
        se_kernel = self.squared_exponential_kernel(X1, X2, **hyperparameters['se_params'])

        # Multiplying the sum of periodic kernels with the squared exponential kernel
        composite_kernel = np.multiply(periodic_sum, se_kernel)

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

    def white_noise_kernel(self, X1, X2, sigma):
        delta_X = np.sum(X1[:, None, :] - X2[None, :, :], axis=-1)
        return sigma ** 2 * np.where(delta_X == 0, 1, 0)

    def polynomial_kernel(self, X1, X2, alpha, beta, d):
        return (alpha + beta * np.dot(X1, X2.T)) ** d

def gp_predict(X_train, y_train, X_test, kernel_func, **hyperparameters):

    # Kernel matrix for training data plus noise term
    K_X_X = kernel_func(X_train, X_train) + np.array(hyperparameters['noise_level'] ** 2 * np.eye(len(X_train)))[:,:,None]
    #assert is_positive_definite(K_X_X), "Warning: K_X_X is not positive definite!"

    # Kernel matrix between test and training data
    K_star_X = kernel_func(X_train, X_test)
   # assert is_positive_definite(K_star_X), "Warning: K_star_X is not positive definite!"


    # Kernel matrix for test data
    K_star_star = kernel_func(X_test, X_test)
    #assert is_positive_definite(K_star_star), "Warning: K_star_X is not positive definite!"

    # Initialize arrays to store the Cholesky decompositions, means, and variances
    L = np.zeros_like(K_X_X)
    mu = np.zeros((K_star_X.shape[1], K_X_X.shape[2]))
    s2 = np.zeros((K_star_X.shape[1], K_X_X.shape[2]))

    # Loop through the third dimension and compute the Cholesky decomposition for each 2D slice
    for i in range(K_X_X.shape[2]):
        L[:, :, i] = npla.cholesky(
            K_X_X[:, :, i] + 1e-10 * np.eye(K_X_X.shape[0]))

        # Compute the mean at our test points.

        if developer == True: start_time = time.time()

        Lk = np.squeeze(npla.solve(L[:, :, i], K_star_X[:,:,i]))
        mu[:, i] = np.dot(Lk.T, npla.solve(L[:, :, i], y_train)).flatten()

       #L_inv = npla.inv(L[:,:,i])
       #mu[:,i] = np.dot(np.dot(np.transpose(K_star_X[:,:,i]),L_inv.T), np.dot(L_inv,y_train)) #alternate function (slower?)

        if developer == True:
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"The code ran in {elapsed_time} seconds")

        # Compute the standard deviation
        s2[:, i] = np.diag(K_star_star[:, :, i]) - np.sum(Lk ** 2, axis=0)
        stdv = np.sqrt(s2)

    return mu, stdv


def is_positive_definite(K):
    # Ensure K is at most 3D
    if K.ndim > 3:
        raise ValueError("Input must be at most 3D.")

    # If K is 2D, add an extra dimension to make the logic below work for both 2D and 3D cases
    if K.ndim == 2:
        K = K[:, :, np.newaxis]

    # Check if the 2D slices are square
    if K.shape[0] != K.shape[1]:
        raise ValueError("The first two dimensions must be equal.")

    # Check each 2D slice for positive definiteness
    for i in range(K.shape[2]):
        if not np.all(np.linalg.eigvals(K[:, :, i]) > 0):
            return False

    return True


def compute_nll(X, y, kernel, hyperparameters):
    if X.ndim == 1: X = X.reshape(-1, 1)
    if y.ndim == 1: y = y.reshape(-1, 1)
    # Update kernel with new hyperparameters
    kernel.set_params(hyperparameters)

    # Compute covariance matrix
    K = kernel.compute_kernel(X, X)

    # Add noise term
    test = X.shape[1]
    testk = K
    K += np.repeat(np.array(np.eye(len(X)) * 1e-3)[:,:, np.newaxis], X.shape[1], axis=2)

    for i in range(K.shape[2]):
        # Compute negative log marginal likelihood
        nll = 0.5 * y.T @ np.linalg.inv(K[:,:,i]) @ y + 0.5 * np.log(np.linalg.det(K[:,:,i])) + 0.5 * len(X) * np.log(2 * np.pi)

    return nll.item()


# Function to flatten hyperparameters and bounds
def flatten_params(params):
    flat_params = []
    for key, value in params.items():
        # Ignore string values
        if isinstance(value, str):
            continue

        if isinstance(value, list):
            for item in value:
                flat_params.extend(flatten_params(item))
        elif isinstance(value, dict):
            flat_params.extend(flatten_params(value))
        else:
            flat_params.append(value)
    return flat_params


# Function to reconstruct nested hyperparameters from flat array
def reconstruct_params(flat_params, template):
    reconstructed_params = {}
    index = 0
    for key, value in template.items():
        # Ignore string values
        if isinstance(value, str):
            reconstructed_params[key] = value
            continue

        if isinstance(value, list):
            reconstructed_params[key] = []
            for item in value:
                reconstructed_item, item_length = reconstruct_params(
                    flat_params[index:], item)
                index += item_length
                reconstructed_params[key].append(reconstructed_item)
        elif isinstance(value, dict):
            reconstructed_params[key], item_length = reconstruct_params(
                flat_params[index:], value)
            index += item_length
        else:
            reconstructed_params[key] = flat_params[index]
            index += 1
    return reconstructed_params, index

def get_optimal_hyperparameters(hyperparameters, hyperparameter_bounds, y, kernel):
    # Flatten hyperparameters and bounds
    initial_hyperparameters_array = np.array(flatten_params(hyperparameters))
    bounds_array = flatten_params(hyperparameter_bounds)

    # Optimization
    result = minimize(
        fun=compute_nll,  # Function to minimize
        x0=initial_hyperparameters_array,  # Initial guess
        args=(y, kernel, hyperparameters),
        bounds=bounds_array,  # Bounds for hyperparameters
        method='L-BFGS-B'  # Optimization algorithm
    )

    # Extract optimal hyperparameters
    optimal_hyperparameters_array = result.x

    # Reconstruct optimal hyperparameters
    optimal_hyperparameters, _ = reconstruct_params(
        optimal_hyperparameters_array, hyperparameters)

def main():
    sample_start_index = 10000
    sample_length = 100

    force_input, force_response, time = load_data(sample_start_index, sample_length)
    time_test = np.linspace(time[0],time[-1], num=250, endpoint = True)

    force_input = format_data(force_input)
    time = format_data(time)
    force_response = format_data(force_response)
    time_test = format_data(time_test)

    if developer == True:
        print(force_input)
        print(force_response)
        print(time)


    hyperparameters = {

    # #periodic_hyperparameters
    #     'kernel_type': 'periodic',
    #     'sigma': 1.0,
    #     'l': 0.3,
    #     'p': 1E-2,
    #     'noise_level': 0.1
    #     }

    #composite_hyperparameters
        'kernel_type': 'composite',
        'periodic_params': [
        {'sigma': 0.1, 'l': 0.01, 'p': 1E-3},
        {'sigma': 0.1, 'l': 0.02, 'p': 1E-3}
        ],
        'se_params': {'sigma': 0.1, 'l': 0.01},
        'noise_level': 0.001
        }

    hyperparameter_bounds = {
    # #periodic_hyperparameter_bounds
    #     'kernel_type': 'periodic',
    #     'sigma': (0.001,10),
    #     'l': (0.01,10),
    #     'p': (0.0001,1),
    #     'noise_level': (0.0001,0.5)
    #     }

    #composite_hyperparameter_bounds
        'kernel_type': 'composite',
        'periodic_param_bounds': [
        {'sigma': (0.001,10), 'l': (0.01,10), 'p': (0.0001,1)},
        {'sigma': (0.001,10), 'l': (0.01,10), 'p': (0.0001,1)}
        ],
        'se_param_bounds': {'sigma': (0.001,10), 'l': (0.01,10)},
        'noise_level': (0.0001,1)
    }


    gp_kernel = GaussianProcessKernel(**hyperparameters)
    gp_kernel.set_params(hyperparameters)

    optimal_hyperparameters = get_optimal_hyperparameters(hyperparameters,hyperparameter_bounds,force_response,gp_kernel)

    prediction = gp_predict(time, force_response, time_test, gp_kernel.compute_kernel, **optimal_hyperparameters)
    plot_data(force_input,force_response, prediction, time, time_test)



if __name__ == "__main__":
    main()