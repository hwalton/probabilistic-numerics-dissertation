import time as timer
import matplotlib.pyplot as plt
import numpy as np
from load_data import load_data

from gp_model import GPModel


def plot_data(force_response, force_response_prediction, time, time_test, force_response_model):
    plt.figure(figsize=(12, 10.5))
    plt.rcParams.update({'font.size': 16})

    if force_response_prediction[1].ndim > 1:
        force_response_prediction_diag = np.diag(force_response_prediction[1])
    else:
        force_response_prediction_diag = force_response_prediction[1]


    plt.subplot(3, 1, 1)  # a rows, b columns, plot c
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


    upper_bound_abs = np.abs(force_response_model.mu_fourier) + np.abs(force_response_model.stdv_fourier)
    lower_bound_abs = np.abs(force_response_model.mu_fourier) - np.abs(force_response_model.stdv_fourier)

    plt.subplot(3, 1, 2)  # a rows, b columns, plot c
    (plt.plot(force_response_model.xi, np.abs(force_response_model.mu_fourier)))
    plt.fill_between(np.squeeze(force_response_model.xi), lower_bound_abs, upper_bound_abs, color='blue',
                     alpha=0.2, label='Std Dev')
    plt.xlabel('Freq [Rad/s]')
    plt.ylabel('Magnitude of Fourier Transform')

    upper_bound_angle = np.angle(force_response_model.mu_fourier) + np.angle(force_response_model.stdv_fourier)
    lower_bound_angle = np.angle(force_response_model.mu_fourier) - np.angle(force_response_model.stdv_fourier)

    plt.subplot(3, 1, 3)  # a rows, b columns, plot c
    plt.plot(force_response_model.xi, np.angle(force_response_model.mu_fourier))
    plt.fill_between(np.squeeze(force_response_model.xi), lower_bound_angle, upper_bound_angle, color='blue',
                     alpha=0.2, label='Std Dev')
    plt.xlabel('Freq [Rad/s]')
    plt.ylabel('Phase of Fourier Transform')

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
    force_response_fourier_type = ['GP', 'DFT'][1]
    force_response_n_iter = 50
    M_one_in = 1

    force_response, time = load_data()

    num_predictions = time.size * 2//3
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
    force_response_fourier_prediction = force_response_model.predict_fourier(time_test, method=force_response_fourier_type)

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