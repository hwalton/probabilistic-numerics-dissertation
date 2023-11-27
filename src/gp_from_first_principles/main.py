import time as timer
import matplotlib.pyplot as plt
import numpy as np


from gp_model import GPModel
from utils import debug_print

def load_data(start = 0, length = 65536):

    assert length <= 65536, "Length must be less than or equal to 65536"

    #data collected during MEC326
    input = np.loadtxt('../../datasets/input.csv', delimiter=',')
    output = np.loadtxt('../../datasets/output.csv', delimiter=',')
    time = np.loadtxt('../../datasets/time.csv', delimiter=',')

    input= input[start:start+length]
    output = output[start:start+length]
    time = time[start:start+length]

    return input, output, time

def plot_data(force_input, force_response, force_input_prediction, force_response_prediction, time, time_test):
    plt.figure(figsize=(12, 9))
    plt.rcParams.update({'font.size': 14})

    plt.subplot(2, 1, 1)  # 2 rows, 1 column, plot 1
    plt.scatter(time, force_input, label='Force input', color='green')
    plt.scatter(time_test, force_input_prediction[0], label='Predicted Mean', color='red')

    upper_bound = force_input_prediction[0] + force_input_prediction[1]
    lower_bound = force_input_prediction[0] - force_input_prediction[1]

    plt.fill_between(np.squeeze(time_test), np.squeeze(lower_bound), np.squeeze(upper_bound), color='blue',
                     alpha=0.2, label='Std Dev')

    plt.xlabel('Time [s]')
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

    plt.xlabel('Time [s]')
    plt.ylabel('Force Response')
    plt.title('Force Response over Time')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def format_data(X):
    if X.ndim == 1: X = X.reshape(-1,1)
    return X

# def get_kernel_hyperparameters(kernel_type):
#     if kernel_type == 'squared_exponential':
#         initial_hyperparameters = {
#             'kernel_type': 'squared_exponential',
#             'sigma': 0.1,
#             'l': 0.01,
#             'p': 1E-3,
#             'mean_func_c': 1.0,
#             'noise_level': 0.001
#             }
#
#         hyperparameter_bounds = {
#             'kernel_type': 'squared_exponential',
#             'sigma': (0.001,10),
#             'l': (0.001,10),
#             'p': (0.0001,1),
#             'mean_func_c': (-1000,1000),
#             'noise_level': (0.0001,1)
#             }
#     elif kernel_type == 'periodic':
#         initial_hyperparameters = {
#             'kernel_type': 'periodic',
#             'sigma': 1,
#             'l': 0.1,
#             'p': 0.08,
#             'mean_func_c': 1.0,
#             'noise_level': 0.1
#             }
#
#         hyperparameter_bounds = {
#             'kernel_type': 'periodic',
#             'sigma': (0.001, 100),
#             'l': (0.001, 10),
#             'p': (0.0001, 1),
#             'mean_func_c': (-1000,1000),
#             'noise_level': (0.0001, 0.25)
#         }
#     elif kernel_type == 'p_se_composite':
#         initial_hyperparameters = {
#             'kernel_type': 'p_se_composite',
#             'periodic_params': [
#             {'sigma': 0.1, 'l': 0.01, 'p': 1E-3},
#             {'sigma': 0.1, 'l': 0.02, 'p': 1E-3}
#             ],
#             'se_params': {'sigma': 0.1, 'l': 0.01},
#             'mean_func_c': 1.0,
#             'noise_level': 0.001
#             }
#
#         hyperparameter_bounds = {
#             'kernel_type': 'p_se_composite',
#             'periodic_param_bounds': [
#             {'sigma': (0.0001,100), 'l': (0.0001,100), 'p': (0.0001,10)},
#             {'sigma': (0.0001,100), 'l': (0.0001,100), 'p': (0.0001,10)}
#             ],
#             'se_param_bounds': {'sigma': (0.0001,100), 'l': (0.0001,100)},
#             'mean_func_c': (-1000,1000),
#             'noise_level': (0.0001,1)
#             }
#     elif kernel_type == 'white_noise':
#         initial_hyperparameters = {
#             'kernel_type': 'white_noise',
#             'sigma': 1,
#             'mean_func_c': 1.0,
#             'noise_level': 0.1
#         }
#
#         hyperparameter_bounds = {
#             'kernel_type': 'white_noise',
#             'sigma': (0.001, 100),
#             'mean_func_c': (-1000,1000),
#             'noise_level': (0.0001, 0.25)
#         }
#     elif kernel_type == 'wn_se_composite':
#         initial_hyperparameters = {
#             'kernel_type': 'wn_se_composite',
#             'wn_params': {'sigma': 0.1},
#             'se_params': {'sigma': 0.1, 'l': 0.01},
#             'mean_func_c': 1.0,
#             'noise_level': 0.001
#             }
#
#         hyperparameter_bounds = {
#             'kernel_type': 'wn_se_composite',
#             'periodic_param_bounds': {'sigma': (0.001,10)},
#             'se_param_bounds': {'sigma': (0.001,10), 'l': (0.001,10)},
#             'mean_func_c': (-1000,1000),
#             'noise_level': (0.0001,1)
#             }
#     else:
#         assert False, "Invalid kernel_type"
#
#     return initial_hyperparameters, hyperparameter_bounds

def execute_gp_model():
    sample_start_index = 5000
    sample_length = 100
    num_predictions = 60
    force_input_kernel_type = 'white_noise'                    #'squared_exponential', 'periodic', 'p_se_composite', 'white_noise', or 'wn_se_composite
    force_input_solver_type = 'metropolis_hastings'            #'iterative_search', 'metropolis_hastings', 'adam', or 'free_lunch'
    force_response_kernel_type = 'periodic'                    #'squared_exponential', 'periodic', 'p_se_composite', 'white_noise', or 'wn_se_composite
    force_response_solver_type = 'metropolis_hastings'         #'iterative_search' or 'metropolis_hastings
    n_iter = 100
    force_input, force_response, time = load_data(sample_start_index,
                                                  sample_length)
    lower = time[0] - 0 * (time[-1] - time[0])
    upper = time[-1] + 0 * (time[-1] - time[0])
    time_test = np.linspace(lower, upper, num=num_predictions, endpoint=True)
    force_input = format_data(force_input)
    time = format_data(time)
    force_response = format_data(force_response)
    time_test = format_data(time_test)


    force_input_model = GPModel(force_input_kernel_type,
                                time,
                                force_input,
                                solver_type=force_input_solver_type,
                                n_iter=n_iter)
    model_1_nll = force_input_model.fit_model()
    force_input_prediction = force_input_model.predict((time_test))

    force_response_model = GPModel(force_response_kernel_type,
                                   time,
                                   force_response,
                                   solver_type=force_response_solver_type,
                                   n_iter=n_iter)
    model_2_nll = force_response_model.fit_model()
    force_response_prediction = force_response_model.predict(time_test)

    plot_data(force_input, force_response, force_input_prediction,
              force_response_prediction, time, time_test)

    return model_1_nll, model_2_nll
def main():
    start_time = timer.time()

    model_1_nll, model_2_nll = execute_gp_model()

    print(model_1_nll)
    print(model_2_nll)

    end_time = timer.time()
    elapsed_time = end_time - start_time
    print(f"The code ran in {elapsed_time} seconds")

if __name__ == "__main__":
    main()