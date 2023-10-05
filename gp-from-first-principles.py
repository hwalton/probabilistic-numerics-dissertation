import scipy
from scipy.special import kv
import time as timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import numpy.linalg as npla

developer = True


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



def plot_data(force_input, force_response, force_input_prediction, force_response_prediction, time, time_test):
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)  # 2 rows, 1 column, plot 1
    plt.scatter(time, force_input, label='Force input', color='green')
    plt.scatter(time_test, force_input_prediction[0], label='Predicted Mean', color='red')

    upper_bound = force_input_prediction[0] + force_input_prediction[1]
    lower_bound = force_input_prediction[0] - force_input_prediction[1]

    plt.fill_between(np.squeeze(time_test), np.squeeze(lower_bound), np.squeeze(upper_bound), color='blue',
                     alpha=0.2, label='Std Dev')

    plt.xlabel('Time')
    plt.ylabel('Force Input')
    plt.title('Force Input over Time')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)  # 2 rows, 1 column, plot 2
    plt.scatter(time, force_response, label='Force Response', color='green')
    plt.scatter(time_test, force_response_prediction[0], label='Predicted Mean', color='red')

    # Assuming prediction[1] is the standard deviation
    upper_bound = force_response_prediction[0] + force_response_prediction[1]
    lower_bound = force_response_prediction[0] - force_response_prediction[1]

    plt.fill_between(np.squeeze(time_test), np.squeeze(lower_bound), np.squeeze(upper_bound), color='blue',
                     alpha=0.2, label='Std Dev')

    plt.xlabel('Time')
    plt.ylabel('Force Response')
    plt.title('Force Response over Time')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
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
        if X1.ndim == 1: X1 = X1.reshape(-1,1)
        if X2.ndim == 1: X2 = X2.reshape(-1,1)
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
        #test3 = X1.shape
        periodic_sum = np.zeros((X1.shape[0], X2.shape[0], X1.shape[1]))
        for params in hyperparameters['periodic_params']:
            periodic_sum += self.periodic_kernel(X1, X2, **params)

        se_kernel = self.squared_exponential_kernel(X1, X2, **hyperparameters['se_params'])

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


class GP_model:
    def __init__(self, initial_hyperparameters, hyperparameter_bounds, X, y, n_iter=10):
        self.initial_hyperparameters = initial_hyperparameters
        self.hyperparameter_bounds = hyperparameter_bounds
        self.X = X
        self.y = y
        self.n_iter = n_iter
        self.template = initial_hyperparameters

    def fit_model(self):
        self.gp_kernel = GaussianProcessKernel(**self.initial_hyperparameters)
        self.gp_kernel.set_params(self.initial_hyperparameters)
        self.optimal_hyperparameters = self.get_optimal_hyperparameters()
        if developer:
            print(self.optimal_hyperparameters)
            print(self.compute_nll( self.optimal_hyperparameters))

    def get_optimal_hyperparameters(self):
        optimal_hyperparameters = self.iterative_search()
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
            self.mu[:, i] = np.dot(Lk.T, npla.solve(L[:, :, i], self.y)).flatten()
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
        if self.X.ndim == 1: self.X = self.X.reshape(-1, 1)
        if self.y.ndim == 1: self.y = self.y.reshape(-1, 1)
        self.gp_kernel.set_params(hyperparameters)
        K = self.gp_kernel.compute_kernel(self.X, self.X)
        K += np.repeat(np.array(np.eye(len(self.X)) * 1e-3)[:,:, np.newaxis], self.X.shape[1], axis=2)
        for i in range(K.shape[2]):
            L = scipy.linalg.cholesky(K[:, :, i], lower=True)
            alpha = scipy.linalg.cho_solve((L, True), self.y)
            nll = 0.5 * self.y.T @ alpha + np.sum(np.log(np.diag(L))) + 0.5 * len(self.X) * np.log(2 * np.pi)
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

    def iterative_search(self):
        initial_hyperparameters_array = np.array(self.flatten_params(self.initial_hyperparameters))
        bounds_array = self.flatten_params(self.hyperparameter_bounds)

        best_hyperparameters = initial_hyperparameters_array
        reconstruct_params_test = self.reconstruct_params(initial_hyperparameters_array)
        best_nll = self.compute_nll(reconstruct_params_test)
        for j in range(self.n_iter):
            if developer:
                print(f"Search Iteration: {j+1}/{self.n_iter}")
            for i, (lower, upper) in enumerate(bounds_array):
                for modifier in [0.75, 2]:
                    new_hyperparameters = best_hyperparameters.copy()
                    new_hyperparameters[i] = np.clip(new_hyperparameters[i] * modifier, lower, upper)
                    new_nll = self.compute_nll(self.reconstruct_params(new_hyperparameters))
                    if new_nll < best_nll:
                        best_nll = new_nll
                        best_hyperparameters = new_hyperparameters
        return best_hyperparameters




def main():
    if developer == True: start_time = timer.time()

    sample_start_index = 2000
    sample_length = 100
    num_predictions = 100
    kernel_type = ('periodic')
    n_iter = 10


    force_input, force_response, time = load_data(sample_start_index, sample_length)
    lower = time[0]-0.25*(time[-1]-time[0])
    upper = time[-1]+0.25*(time[-1]-time[0])
    time_test = np.linspace(lower,upper, num=num_predictions, endpoint = True)

    force_input = format_data(force_input)
    time = format_data(time)
    force_response = format_data(force_response)
    time_test = format_data(time_test)

    if developer == True:
        print(force_input)
        print(force_response)
        print(time)

    if kernel_type == 'periodic':
        initial_hyperparameters = {
            'kernel_type': 'periodic',
            'sigma': 1,
            'l': 0.1,
            'p': 0.08,
            'noise_level': 0.1
            }

        hyperparameter_bounds = {
            'kernel_type': 'periodic',
            'sigma': (0.001, 100),
            'l': (0.001, 10),
            'p': (0.0001, 1),
            'noise_level': (0.0001, 0.25)
        }

    if kernel_type == 'composite':
        initial_hyperparameters = {
            'kernel_type': 'composite',
            'periodic_params': [
            {'sigma': 0.1, 'l': 0.01, 'p': 1E-3},
            {'sigma': 0.1, 'l': 0.02, 'p': 1E-3}
            ],
            'se_params': {'sigma': 0.1, 'l': 0.01},
            'noise_level': 0.001
            }


        hyperparameter_bounds = {
            'kernel_type': 'composite',
            'periodic_param_bounds': [
            {'sigma': (0.001,10), 'l': (0.001,10), 'p': (0.0001,1)},
            {'sigma': (0.001,10), 'l': (0.001,10), 'p': (0.0001,1)}
            ],
            'se_param_bounds': {'sigma': (0.001,10), 'l': (0.001,10)},
            'noise_level': (0.0001,1)
            }

    force_response_model = GP_model(initial_hyperparameters, hyperparameter_bounds, time, force_response, n_iter = n_iter)
    force_response_model.fit_model()
    force_response_prediction = force_response_model.predict(time_test)

    force_input_model = GP_model(initial_hyperparameters, hyperparameter_bounds, time, force_input, n_iter = n_iter)
    force_input_model.fit_model()
    force_input_prediction = force_input_model.predict((time_test))

    plot_data(force_input,force_response, force_input_prediction, force_response_prediction, time, time_test)

    if developer == True:
        end_time = timer.time()
        elapsed_time = end_time - start_time
        print(f"The code ran in {elapsed_time} seconds")


if __name__ == "__main__":
    main()