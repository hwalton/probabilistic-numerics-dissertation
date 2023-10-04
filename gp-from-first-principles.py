from scipy.io import loadmat
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import kv

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

def plot_data(input, output, time):
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)  # 2 rows, 1 column, plot 1
    plt.scatter(time, input, label='Input', color='blue')
    plt.xlabel('Time')
    plt.ylabel('Input')
    plt.title('Input over Time')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)  # 2 rows, 1 column, plot 2
    plt.scatter(time, output, label='Output', color='green')
    plt.xlabel('Time')
    plt.ylabel('Output')
    plt.title('Output over Time')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()  # Adjusts subplot params for better layout
    plt.show()



class GaussianProcessKernel:
    def __init__(self, kernel_type='linear', **kwargs):
        self.kernel_type = kernel_type
        self.params = kwargs

    def compute_kernel(self, X1, X2):
        if developer == True: X1_shape_before = X1.shape
        X1 = X1.reshape(-1,1)
        if developer == True: X1_shape_after = X1.shape
        print(X1.shape)
        X2 = X2.reshape(-1,1)
        if self.kernel_type == 'linear':
            return self.linear_kernel(X1, X2)
        elif self.kernel_type == 'periodic':
            return self.periodic_kernel(X1, X2, **self.params)
        elif self.kernel_type == 'squared_exponential':
            return self.squared_exponential_kernel(X1, X2, **self.params)
        elif self.kernel_type == 'matern':
            return self.matern_kernel(X1, X2, **self.params)
        elif self.kernel_type == 'rational_quadratic':
            return self.rational_quadratic_kernel(X1, X2, **self.params)
        elif self.kernel_type == 'exponential':
            return self.exponential_kernel(X1, X2, **self.params)
        elif self.kernel_type == 'cosine':
            return self.cosine_kernel(X1, X2, **self.params)
        elif self.kernel_type == 'white_noise':
            return self.white_noise_kernel(X1, X2, **self.params)
        elif self.kernel_type == 'polynomial':
            return self.polynomial_kernel(X1, X2, **self.params)
        # Add more kernel types as needed
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

    def linear_kernel(self, X1, X2):
        return np.dot(X1, X2.T)

    def periodic_kernel(self, X1, X2, sigma, l, p):
        delta_X = X1[:, None, :] - X2[None, :, :]
        return sigma ** 2 * np.exp(
            -2 * np.sin(np.pi * np.abs(delta_X) / p) ** 2 / l ** 2)

    def squared_exponential_kernel(self, X1, X2, sigma, l):
        delta_X = X1[:, None, :] - X2[None, :, :]
        return sigma ** 2 * np.exp(
            -0.5 * np.sum(delta_X ** 2, axis=-1) / l ** 2)

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

def main():
    force_input, force_response, time = load_data(1000, 1000)
    if developer == True:
        print(force_input)
        print(force_response)
        print(time)

    plot_data(force_input,force_response,time)

    gp_kernel_periodic = GaussianProcessKernel(kernel_type='periodic',
                                               sigma=10, l=8E-4, p=8E-4)


    out = gp_kernel_periodic.compute_kernel(force_input,force_input)
    print(out)


if __name__ == "__main__":
    main()