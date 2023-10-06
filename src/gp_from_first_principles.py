import scipy
import time as timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import numpy.linalg as npla

from src.gaussian_process_kernel import GaussianProcessKernel
from src.iterative_search import IterativeSearch
from utils import debug_print

developer = True


def load_data(start = 0, length = 65536):

    assert length <= 65536, "Length must be less than or equal to 65536"

    #data collected during MEC326
    input = np.loadtxt('../datasets/input.csv', delimiter=',')
    output = np.loadtxt('../datasets/output.csv', delimiter=',')
    time = np.loadtxt('../datasets/time.csv', delimiter=',')

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


class MetropolisHastings:
    def __init__(self,initial_hyperparameters_array, bounds_array, compute_nll):
        self.initial_hyperparameters_array = initial_hyperparameters_array
        self.bounds_array = bounds_array
        self.compute_nll = compute_nll

    def solve(self, n_iter = '10'):
        return
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
        debug_print(self.compute_nll( self.optimal_hyperparameters))

    def construct_solver(self,solver_type, initial_hyperparameters_array, bounds_array):
        if solver_type == 'iterative_search':
            self.solver = IterativeSearch(initial_hyperparameters_array, bounds_array, self.compute_nll)
        elif solver_type == 'metropolis_hastings':
            self.solver = MetropolisHastings()

    def get_optimal_hyperparameters(self):
        initial_hyperparameters_array = np.array(self.flatten_params(self.initial_hyperparameters))
        bounds_array = self.flatten_params(self.hyperparameter_bounds)
        self.construct_solver(self.solver_type, initial_hyperparameters_array, bounds_array)
        optimal_hyperparameters = self.solver.solve(self.n_iter)
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

def get_kernel_hyperparameters(kernel_type):
    if kernel_type == 'periodic':
        initial_hyperparameters = {
            'kernel_type': 'periodic',
            'sigma': 1,
            'l': 0.1,
            'p': 0.08,
            'mean_func_c': 1.0,
            'noise_level': 0.1
            }

        hyperparameter_bounds = {
            'kernel_type': 'periodic',
            'sigma': (0.001, 100),
            'l': (0.001, 10),
            'p': (0.0001, 1),
            'mean_func_c': (-1000,1000),
            'noise_level': (0.0001, 0.25)
        }

    if kernel_type == 'p_se_composite':
        initial_hyperparameters = {
            'kernel_type': 'p_se_composite',
            'periodic_params': [
            {'sigma': 0.1, 'l': 0.01, 'p': 1E-3},
            {'sigma': 0.1, 'l': 0.02, 'p': 1E-3}
            ],
            'se_params': {'sigma': 0.1, 'l': 0.01},
            'mean_func_c': 1.0,
            'noise_level': 0.001
            }

        hyperparameter_bounds = {
            'kernel_type': 'p_se_composite',
            'periodic_param_bounds': [
            {'sigma': (0.001,10), 'l': (0.001,10), 'p': (0.0001,1)},
            {'sigma': (0.001,10), 'l': (0.001,10), 'p': (0.0001,1)}
            ],
            'se_param_bounds': {'sigma': (0.001,10), 'l': (0.001,10)},
            'mean_func_c': (-1000,1000),
            'noise_level': (0.0001,1)
            }

    if kernel_type == 'white_noise':
        initial_hyperparameters = {
            'kernel_type': 'white_noise',
            'sigma': 1,
            'mean_func_c': 1.0,
            'noise_level': 0.1
        }

        hyperparameter_bounds = {
            'kernel_type': 'white_noise',
            'sigma': (0.001, 100),
            'mean_func_c': (-1000,1000),
            'noise_level': (0.0001, 0.25)
        }

    if kernel_type == 'wn_se_composite':
        initial_hyperparameters = {
            'kernel_type': 'wn_se_composite',
            'wn_params': {'sigma': 0.1},
            'se_params': {'sigma': 0.1, 'l': 0.01},
            'mean_func_c': 1.0,
            'noise_level': 0.001
            }

        hyperparameter_bounds = {
            'kernel_type': 'wn_se_composite',
            'periodic_param_bounds': {'sigma': (0.001,10)},
            'se_param_bounds': {'sigma': (0.001,10), 'l': (0.001,10)},
            'mean_func_c': (-1000,1000),
            'noise_level': (0.0001,1)
            }

    return initial_hyperparameters, hyperparameter_bounds

def main():
    if developer == True: start_time = timer.time()

    sample_start_index = 15000
    sample_length = 100
    num_predictions = 50
    force_input_kernel_type = 'wn_se_composite'
    force_input_solver_type = 'iterative_search'
    force_response_kernel_type = 'p_se_composite'
    force_response_solver_type = 'iterative_search'
    n_iter = 15

    force_input, force_response, time = load_data(sample_start_index, sample_length)
    lower = time[0]-0.25*(time[-1]-time[0])
    upper = time[-1]+0.25*(time[-1]-time[0])
    time_test = np.linspace(lower,upper, num=num_predictions, endpoint = True)

    force_input = format_data(force_input)
    time = format_data(time)
    force_response = format_data(force_response)
    time_test = format_data(time_test)

    debug_print(force_input)
    debug_print(force_response)
    debug_print(time)

    force_input_initial_hyperparameters, force_input_hyperparameter_bounds = get_kernel_hyperparameters(force_input_kernel_type)
    force_response_initial_hyperparameters, force_response_hyperparameter_bounds = get_kernel_hyperparameters(force_response_kernel_type)

    force_input_model = GPModel(force_input_initial_hyperparameters, force_input_hyperparameter_bounds, time, force_input, solver_type = force_input_solver_type, n_iter = n_iter)
    force_input_model.fit_model()
    force_input_prediction = force_input_model.predict((time_test))

    force_response_model = GPModel(force_response_initial_hyperparameters, force_response_hyperparameter_bounds, time, force_response, solver_type = force_response_solver_type, n_iter = n_iter)
    force_response_model.fit_model()
    force_response_prediction = force_response_model.predict(time_test)

    plot_data(force_input,force_response, force_input_prediction, force_response_prediction, time, time_test)

    if developer == True:
        end_time = timer.time()
        elapsed_time = end_time - start_time
        debug_print(f"The code ran in {elapsed_time} seconds")

if __name__ == "__main__":
    main()