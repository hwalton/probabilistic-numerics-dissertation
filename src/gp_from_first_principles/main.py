import time as timer
import matplotlib.pyplot as plt
import numpy as np
from load_data import load_data

from gp_model import GPModel


def plot_data(force_response, force_response_prediction, time, time_test, force_response_model):
    plt.figure(figsize=(12, 4.5))
    plt.rcParams.update({'font.size': 16})

    if force_response_prediction[1].ndim > 1:
        force_response_prediction_diag = np.diag(force_response_prediction[1])


    plt.subplot(1, 1, 1)  # 2 rows, 1 column, plot 2
    plt.scatter(time, force_response, label='Force Response', color='green')
    plt.scatter(time_test, force_response_prediction[0], label='Prediction Mean', color='red')
    plt.scatter(force_response_model.U_X, force_response_model.U_y, label='Inducing Points', color='purple')

    # Assuming prediction[1] is the standard deviation
    upper_bound = force_response_prediction[0] + force_response_prediction_diag
    lower_bound = force_response_prediction[0] - force_response_prediction_diag

    plt.fill_between(np.squeeze(time_test), np.squeeze(lower_bound), np.squeeze(upper_bound), color='blue',
                     alpha=0.2, label='Prediction Std Dev')

    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [ms$^{-2}$]')
    plt.title('Force Response Over Time')

    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def format_data(X):
    if X.ndim == 1: X = X.reshape(-1,1)
    return X


def execute_gp_model():

    force_response_kernel_type = ['squared_exponential', 'p_se_composite', 'white_noise', 'wn_se_composite', 'periodic', 'cosine', 'cosine_composite'][0]
    force_response_solver_type = ['metropolis_hastings', 'iterative_search', 'adam', 'free_lunch'][0]
    force_response_predict_type = ['cholesky', 'FITC'][0]
    force_response_nll_method = ['cholesky', 'FITC_18_134'][0]
    force_response_U_induced_method = ['k_means', 'even'][1]
    force_response_n_iter = 0

    M_one_in = 1

    force_response, time = load_data()

    num_predictions = time.size * 4 // 5
    lower = time[0] - 0 * (time[-1] - time[0])
    upper = time[-1] + 0 * (time[-1] - time[0])
    time_test = np.linspace(lower, upper, num=num_predictions, endpoint=True)
    time = format_data(time)
    force_response = format_data(force_response)
    time_test = format_data(time_test)

    force_response_model = GPModel(force_response_kernel_type,
                                   time,
                                   force_response,
                                   solver_type=force_response_solver_type,
                                   n_iter=force_response_n_iter, gp_algo=force_response_nll_method,
                                   U_induced_method=force_response_U_induced_method,
                                   M_one_in=M_one_in)
    model_2_nll = force_response_model.fit_model()
    force_response_prediction = force_response_model.predict(time_test,
                                                             method=force_response_predict_type)

    plot_data(force_response,
              force_response_prediction, time, time_test, force_response_model)

    return model_2_nll
def main():
    start_time = timer.time()

    model_2_nll = execute_gp_model()

    print(f"Force Response final NLL: {model_2_nll}")

    end_time = timer.time()
    elapsed_time = end_time - start_time
    print(f"The code ran in {elapsed_time} seconds")

if __name__ == "__main__":
    main()