from datetime import datetime
import time as timer
import numpy as np
from load_data import load_data
import pandas as pd
from gp_model import GPModel

from plot_data import plot_data
from save_data import save_data

import os
import dotenv
dotenv.load_dotenv()

import copy



def format_data(X):
    if X.ndim == 1: X = X.reshape(-1,1)
    return X


def get_xi(X_star, mode='uniform', peak=(10, 2, 100)):
    X_star = np.squeeze(np.asarray(X_star))
    N = len(X_star)
    delta_t = (X_star[-1] - X_star[0]) / (N - 1)
    fs = 1 / delta_t
    delta_f = fs / N
    xi = np.linspace(-(fs / 2 - delta_f), fs / 2, N)

    if mode == 'nyquist_limit':
        N = 2 * len(X_star)
        xi = np.linspace(-(fs - delta_f), fs, N)
    xi *= 2 * np.pi  # convert from Hz to rad/s
    if mode == 'cluster_peak':
        cluster = np.linspace(peak[0] - peak[1], peak[0] + peak[1], peak[2])
        debug = np.concatenate((xi, cluster))
        xi = np.concatenate((xi, cluster))
        xi.sort()
    return xi

def DFT_hw(y):
    DFT = np.fft.fft(np.squeeze(y) * np.hanning(len(np.squeeze(y)))) / (len(np.squeeze(y)) / 2)
    N = len(DFT)
    DFT = np.concatenate((DFT[(N // 2 + 1):], DFT[:(N // 2 + 1)]))
    return DFT

def get_analytical_FT(xi, xi_mode):
    m = float(os.getenv('M'))  # Mass
    c = float(os.getenv('C'))  # Damping coefficient
    k = float(os.getenv('K'))  # Stiffness
    A = float(os.getenv('A'))  # Amplitude
    ft = np.squeeze(A / ((k - m * xi ** 2) + 1j * c * xi))

    if xi_mode == 'nyquist_limit':
        m_2 = float(os.getenv('M_2'))  # Mass
        c_2 = float(os.getenv('C_2'))
        k_2 = float(os.getenv('K_2'))
        A_2 = float(os.getenv('A_2'))

        ft += np.squeeze(A_2 / ((k_2 - m_2 * xi ** 2) + 1j * c_2 * xi))
    return ft


def execute_gp_model(date_time_formatted,
                     initial_hyps,
                     force_response_n_iter,
                     xi_mode,
                     length,
                     dataset,
                     sample_rate,
                     input_noise_stdv,
                     response_noise_stdv,
                     suptitle,
                     peak):

    print("\n\n")
    print("=" * 100)
    print(f"Running {suptitle}...")
    print("=" * 100)

    force_response_kernel_type = ['squared_exponential', 'p_se_composite', 'white_noise', 'wn_se_composite', 'periodic', 'cosine', 'cosine_composite'][0]
    force_response_solver_type = ['metropolis_hastings', 'iterative_search', 'adam', 'free_lunch'][0]
    force_response_predict_type = ['cholesky', 'FITC'][0]
    force_response_nll_method = ['cholesky', 'FITC_18_134'][0]
    force_response_U_induced_method = ['k_means', 'even'][1]
    force_response_fourier_type = ['GP_NULL', 'GP', 'GP_2', 'GP_3', 'GP_4', 'GP_5', 'GP_6', 'DFT', 'set'][6]
    M_one_in = 1


    save_data(sample_rate, length, dataset, input_noise_stdv, response_noise_stdv)
    force_response, time_nonuniform_input = load_data()

    time_test = np.linspace(0, (length - 1) / sample_rate, length)[:, None]

    # if xi_mode == 'nyquist_limit':
    #     time_test = np.linspace(0, (length - 1) / sample_rate, 2 * length)[:, None]

    time_nonuniform_input = format_data(time_nonuniform_input)
    force_response = format_data(force_response)
    time_test = format_data(time_test)

    force_response_model = GPModel(force_response_kernel_type,
                                   time_nonuniform_input,
                                   force_response,
                                   solver_type=force_response_solver_type,
                                   n_iter=force_response_n_iter, gp_algo=force_response_nll_method,
                                   U_induced_method=force_response_U_induced_method,
                                   M_one_in=M_one_in)

    force_response_model.hyperparameters_obj.update(initial_hyps)

    start_time_gp = timer.time()

    model_2_nll = force_response_model.fit_model()

    end_time_gp_hyp_fit = timer.time()
    elapsed_time = end_time_gp_hyp_fit - start_time_gp
    print(f"The GP hyps were fit at {elapsed_time} seconds")

    force_response_prediction = force_response_model.predict(time_nonuniform_input,
                                                             method=force_response_predict_type)

    end_time_gp_time_predict = timer.time()
    elapsed_time = end_time_gp_time_predict - start_time_gp
    print(f"The GP time domain prediction was calculated at {elapsed_time} seconds")

    xi_cont = get_xi(time_nonuniform_input, mode=xi_mode, peak=peak)
    GP_FT_mu, GP_FT_stdv = force_response_model.predict_fourier(xi_cont, method=force_response_fourier_type)

    end_time_gp_time_predict = timer.time()
    elapsed_time = end_time_gp_time_predict - start_time_gp
    print(f"The GP FT was calculated at {elapsed_time} seconds")

    xi_disc = get_xi(time_test, mode='uniform')
    response_interp = np.interp(np.squeeze(time_test), np.squeeze(time_nonuniform_input), np.squeeze(force_response))

    start_time_dft = timer.time()

    DFT = DFT_hw(response_interp)

    end_time_dft = timer.time()
    elapsed_time = end_time_dft - start_time_dft
    print(f"The DFT was calculated in {elapsed_time} seconds")

    analytical_FT = get_analytical_FT(xi_cont, xi_mode)

    plot_df_dict = {
        'time': np.squeeze(time_nonuniform_input),
        'force_response': np.squeeze(force_response),
        'time_test': np.squeeze(time_test),
        'force_response_prediction_t': np.squeeze(force_response_prediction[0]),
        'force_response_prediction_f': np.squeeze(force_response_prediction[1]),
        'xi_cont': np.squeeze(xi_cont),
        'xi_disc': np.squeeze(xi_disc),
        'analytical_FT': np.squeeze(analytical_FT),
        'DFT': np.squeeze(DFT),
        'GP_FT_mu': np.squeeze(GP_FT_mu),
        'GP_FT_stdv': np.squeeze(GP_FT_stdv),
        'xi_mode': xi_mode,
        'suptitle': suptitle
    }

    # Find the maximum length among all columns
    max_length = max(len(column) for column in plot_df_dict.values() if isinstance(column, np.ndarray))

    # Iterate through each column, extending shorter ones with None
    for key, column in plot_df_dict.items():
        if isinstance(column, np.ndarray):  # Ensure it's an array we're working with
            deficit = max_length - len(column)  # Calculate how much shorter this column is
            if deficit > 0:  # If the column is shorter, extend it with None
                plot_df_dict[key] = np.concatenate([column, [None] * deficit])

    # Now, create the DataFrame with columns adjusted to the same length
    plot_df = pd.DataFrame(plot_df_dict)




    data_dir = f'../output_data/plot_df_{date_time_formatted}_{suptitle}.csv'
    plot_df.to_csv(data_dir, index=False)


    plot_data(data_dir)

    return model_2_nll

def main():
    start_time = timer.time()
    date_time_formatted = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    params_basic = {
        'date_time_formatted': date_time_formatted,
        'initial_hyps': {
            'kernel_type': 'squared_exponential',
            'sigma': 0.56,
            'l': 0.015,
            'noise_level': 0.01
        },
        'force_response_n_iter': 0,
        'xi_mode': 'uniform',
        'length': 256,
        'dataset': 4,
        'sample_rate': 32,
        'input_noise_stdv': 0.0,
        'response_noise_stdv': 0.0,
        'suptitle': 'Basic Setup',
        'peak': (10, 0.5, 100)
    }

    params = copy.deepcopy(params_basic)
    _ = execute_gp_model(**params)

    # params = copy.deepcopy(params_basic)
    # params['suptitle'] = 'With Response Noise'
    # params['response_noise_stdv'] = 0.25
    # # params['initial_hyps']['noise_level'] = 0.25
    # _ = execute_gp_model(**params)
    #
    # params = copy.deepcopy(params_basic)
    # params['suptitle'] = 'With Input Noise'
    # params['input_noise_stdv'] = 0.25
    # # params['response_noise_stdv'] = 0.25
    # # params['initial_hyps']['noise_level'] = 0.25
    # _ = execute_gp_model(**params)
    #
    # params = copy.deepcopy(params_basic)
    # params['suptitle'] = 'With Cluster Peak'
    # params['xi_mode'] = 'cluster_peak'
    # _ = execute_gp_model(**params)
    #
    # params = copy.deepcopy(params_basic)
    # N = 64
    # params['suptitle'] = f'With Cluster Peak, Short: N = {N}'
    # params['xi_mode'] = 'cluster_peak'
    # params['length'] = N
    # _ = execute_gp_model(**params)
    #
    # params = copy.deepcopy(params_basic)
    # params['suptitle'] = 'With Response Noise, Input Noise and Cluster Peak'
    # params['response_noise_stdv'] = 0.25
    # params['input_noise_stdv'] = 0.25
    # # params['initial_hyps']['noise_level'] = 0.25
    # params['xi_mode'] = 'cluster_peak'
    # _ = execute_gp_model(**params)
    #
    # # Nyquist Limit Test
    # params = copy.deepcopy(params_basic)
    # params['suptitle'] = 'Nyquist Limit Test'
    # params['xi_mode'] = 'nyquist_limit'
    # params['dataset'] = 5
    # _ = execute_gp_model(**params)

    end_time = timer.time()
    elapsed_time = end_time - start_time
    print(f"The code ran in {elapsed_time} seconds")

# def main_single():
#     start_time = timer.time()
#     date_time_formatted = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#
#     params_basic = {
#         'date_time_formatted': date_time_formatted,
#         'initial_hyps': {
#             'kernel_type': 'squared_exponential',
#             'sigma': 0.56,
#             'l': 0.015,
#             'noise_level': 0.01
#         },
#         'force_response_n_iter': 50,
#         'xi_mode': 'uniform',
#         'length': 512,
#         'sample_rate': 32,
#         'input_noise_stdv': 0.,
#         'response_noise_stdv': 0.,
#         'suptitle': 'Basic Setup',
#         'peak': (10, 2, 100)
#     }
#
#     params = copy.deepcopy(params_basic)
#     _ = execute_gp_model(**params)
#
#     end_time = timer.time()
#     elapsed_time = end_time - start_time
#     print(f"The code ran in {elapsed_time} seconds")

if __name__ == "__main__":
    main()
