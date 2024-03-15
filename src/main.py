import time as timer
import numpy as np
from load_data import load_data
import pandas as pd
from gp_model import GPModel

from plot_data import plot_data



def format_data(X):
    if X.ndim == 1: X = X.reshape(-1,1)
    return X


def execute_gp_model():

    force_response_kernel_type = ['squared_exponential', 'p_se_composite', 'white_noise', 'wn_se_composite', 'periodic', 'cosine', 'cosine_composite'][0]
    force_response_solver_type = ['metropolis_hastings', 'iterative_search', 'adam', 'free_lunch'][0]
    force_response_predict_type = ['cholesky', 'FITC'][0]
    force_response_nll_method = ['cholesky', 'FITC_18_134'][0]
    force_response_U_induced_method = ['k_means', 'even'][1]
    force_response_fourier_type = ['GP', 'GP_2', 'GP_3', 'GP_4', 'GP_5', 'DFT', 'set'][6]
    force_response_n_iter = 0
    M_one_in = 1

    force_response, time = load_data()

    num_predictions = time.size
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
    xi, GP_FT_mu, GP_FT_stdv = force_response_model.predict_fourier(time_test, method=force_response_fourier_type)

    _, DFT, _ = force_response_model.predict_fourier(time_test, method='DFT')

    _, analytical_FT, _ = force_response_model.predict_fourier(time_test, method='set')

    plot_df = pd.DataFrame({
        'time': np.squeeze(time),
        'force_response': np.squeeze(force_response),
        'time_test': np.squeeze(time_test),
        'force_response_prediction_t': np.squeeze(force_response_prediction[0]),
        'force_response_prediction_f': np.squeeze(force_response_prediction[1]),
        'xi': np.squeeze(xi),
        'analytical_FT': np.squeeze(analytical_FT),
        'DFT': np.squeeze(DFT),
        'GP_FT_mu': np.squeeze(GP_FT_mu),
        'GP_FT_stdv': np.squeeze(GP_FT_stdv)
    })


    plot_data(plot_df)

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