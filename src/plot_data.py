import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
from dotenv import load_dotenv
load_dotenv()

def print_MSE(analytical_FT, DFT, GP_FT_mu):
    mse_DFT_mag = np.mean((np.abs(analytical_FT) - np.abs(DFT)) ** 2)
    mse_GP_FT_mag = np.mean((np.abs(analytical_FT) - np.abs(GP_FT_mu)) ** 2)

    mse_DFT_phase = np.mean((np.angle(analytical_FT) - np.angle(DFT)) ** 2)
    mse_GP_FT_phase = np.mean((np.angle(analytical_FT) - np.angle(GP_FT_mu)) ** 2)

    mse_DFT_real = np.mean((np.real(analytical_FT) - np.real(DFT)) ** 2)
    mse_GP_FT_real = np.mean((np.real(analytical_FT) - np.real(GP_FT_mu)) ** 2)

    mse_DFT_imag = np.mean((np.imag(analytical_FT) - np.imag(DFT)) ** 2)
    mse_GP_FT_imag = np.mean((np.imag(analytical_FT) - np.imag(GP_FT_mu)) ** 2)

    print(f"\nMSE DFT Magnitude: {mse_DFT_mag}")
    print(f"MSE GP FT Magnitude: {mse_GP_FT_mag}\n")

    print(f"MSE DFT Phase: {mse_DFT_phase}")
    print(f"MSE GP FT Phase: {mse_GP_FT_phase}\n")

    print(f"MSE DFT Real: {mse_DFT_real}")
    print(f"MSE GP FT Real: {mse_GP_FT_real}\n")

    print(f"MSE DFT Imaginary: {mse_DFT_imag}")
    print(f"MSE GP FT Imaginary: {mse_GP_FT_imag}\n")


def print_wd(analytical_FT, DFT, GP_FT_mu, xi_cont, xi_disc, xi_mode):

    if xi_mode == 'nyquist_limit':
        m = float(os.getenv('M_2'))
        c = float(os.getenv('C_2'))
        k = float(os.getenv('K_2'))
    else:
        m = float(os.getenv('M')) #=1.0
        c = float(os.getenv('C')) #=0.1
        k = float(os.getenv('K')) #=100.0

    wn = np.sqrt(k/m)
    zeta = c / (2 * np.sqrt(m * k))
    debug = np.sqrt(1 - zeta**2)
    wd_analytical_FT =  wn * np.sqrt(1 - zeta**2)

    i = np.squeeze(np.argwhere(np.abs(DFT) == max(np.abs(DFT)))[-1])
    wd_DFT = np.squeeze(np.abs(xi_disc[i]))

    j = np.squeeze(np.argwhere(np.abs(GP_FT_mu) == max(np.abs(GP_FT_mu)))[-1])
    wd_GP_FT = np.squeeze(xi_cont[j])

    peak_analytical_FT = np.abs(np.squeeze(1 / ((k - m * (wd_analytical_FT) ** 2) + 1j * c * (wd_analytical_FT))))

    print("~" * 100)
    print(f"\u03C9_d analytical FT: {wd_analytical_FT}")
    print(f"\u03C9_d DFT: {wd_DFT}")
    print(f"\u03C9_d GP FT: {wd_GP_FT}\n")
    print(f"Error in \u03C9_d DFT: {np.abs(wd_analytical_FT - wd_DFT)}")
    print(f"Error in \u03C9_d GP FT: {np.abs(wd_analytical_FT - wd_GP_FT)}")
    print("~" * 100)
    print(f"Peak analytical FT: {peak_analytical_FT}")
    print(f"Peak DFT: {max(np.abs(DFT))}")
    print(f"Peak GP FT: {max(np.abs(GP_FT_mu))}\n")
    print(f"Error in peak DFT: {peak_analytical_FT - max(np.abs(DFT))}")
    print(f"Error in peak GP FT: {peak_analytical_FT - max(np.abs(GP_FT_mu))}\n")


def plot_data(data_dir):
    plot_df = pd.read_csv(data_dir)


    fig = plt.figure(figsize=(33.1, 23.4), dpi=150)
    plt.rcParams.update({'font.size': 16})

    arrays = {col: np.array(plot_df[col].dropna()) for col in plot_df.columns}

    time = arrays['time']
    time_test = arrays['time_test']
    force_response = arrays['force_response']
    force_response_prediction_t = arrays['force_response_prediction_t']
    force_response_prediction_f = arrays['force_response_prediction_f']
    xi_cont = arrays['xi_cont']
    xi_disc = arrays['xi_disc']
    GP_FT_mu = arrays['GP_FT_mu']
    GP_FT_stdv = arrays['GP_FT_stdv']
    analytical_FT = arrays['analytical_FT']
    DFT = arrays['DFT']
    xi_mode = arrays['xi_mode'][0]
    suptitle = arrays['suptitle'][0]



    def convert_to_complex(s):
        return complex(s)

    # Vectorize this function so it can be applied over each element of the array
    vectorized_convert = np.vectorize(convert_to_complex)

    # Apply the conversion to the entire array
    GP_FT_mu = vectorized_convert(GP_FT_mu)
    GP_FT_stdv = vectorized_convert(GP_FT_stdv)
    analytical_FT = vectorized_convert(analytical_FT)
    DFT = vectorized_convert(DFT)


    # # Split the complex numbers into real and imaginary parts for saving
    # plot_df['RealPart'] = plot_df['ComplexData'].apply(lambda x: x.real)
    # plot_df['ImagPart'] = plot_df['ComplexData'].apply(lambda x: x.imag)
    #
    # # Save the DataFrame with real and imaginary parts to a CSV file
    # plot_df[['RealPart', 'ImagPart']].to_csv('complex_data.csv', index=False)



    if force_response_prediction_f.ndim > 1:
        assert 0, "Not yet implemented"
        #force_response_prediction_diag = np.array(np.diag(force_response_prediction_f))
    else:
        force_response_prediction_diag = np.array(force_response_prediction_f)

    # plt.subplot(3, 1, 1)  # a rows, b columns, plot c
    ax1 = plt.subplot2grid((5, 3), (0, 1), colspan=2)
    ax1.plot(time, force_response_prediction_t, label='Prediction Mean', color='red', marker='o', zorder=1)
    ax1.scatter(time, force_response, label='Training Data', color='green', zorder=2)

    # Assuming prediction_f is the standard deviation
    upper_bound = force_response_prediction_t + force_response_prediction_diag
    lower_bound = force_response_prediction_t - force_response_prediction_diag

    debug_types_time_test = {type(item) for item in time_test}
    debug_types_upper_bound = {type(item) for item in upper_bound}
    debug_types_lower_bound = {type(item) for item in lower_bound}

    plt.fill_between(np.squeeze(time), np.squeeze(lower_bound), np.squeeze(upper_bound),
                     alpha=0.2, label='Prediction Std Dev', color='blue')

    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Acceleration [ms$^{-2}$]')
    ax1.set_title('Simulated SDOF System')

    ax1.legend()
    ax1.grid(True)

    upper_bound_abs = np.abs(GP_FT_mu) + np.abs(GP_FT_stdv)
    lower_bound_abs = np.abs(GP_FT_mu) - np.abs(GP_FT_stdv)


    ax = plt.subplot2grid((5, 3), (1, 2))
    ax.plot(xi_cont, np.abs(GP_FT_mu), label='Prediction Mean', color='red', marker='o')
    # ax.fill_between(np.squeeze(xi_cont), lower_bound_abs, upper_bound_abs, alpha=0.2, label='Prediction Std Dev')
    ax.set_xlabel('Frequency [rad s$^{-1}$]')
    ax.set_ylabel('Magnitude [ms$^{-2}$]')
    ax.set_title('GP Fourier Transform')
    upper_bound_angle = np.angle(GP_FT_mu) + np.angle(GP_FT_stdv)
    lower_bound_angle = np.angle(GP_FT_mu) - np.angle(GP_FT_stdv)
    max_mag = 1.1 * max(np.max(np.abs(analytical_FT)), np.max(np.abs(DFT)), np.max(np.abs(GP_FT_mu)), np.max(np.abs(GP_FT_stdv)), np.max(np.abs(upper_bound_abs)), np.max(np.abs(lower_bound_abs)))
    min_mag = - 1.1 * max(np.abs(GP_FT_stdv))
    ax.set_ylim(min_mag, max_mag)
    ax.legend()


    ax = plt.subplot2grid((5, 3), (2, 2))
    ax.plot(xi_cont, np.angle(GP_FT_mu), label='Prediction Mean', color='red', marker='o')
    # ax.fill_between(np.squeeze(xi_cont), lower_bound_angle, upper_bound_angle, color='blue', alpha=0.2, label='Prediction Std Dev')
    ax.set_xlabel('Frequency [rad s$^{-1}$]')
    ax.set_ylabel('Phase [rad]')
    ax.set_title('GP Fourier Transform')
    max_phase = (1.1 * np.pi)
    ax.set_ylim(-max_phase, max_phase)
    ax.legend()

    ax = plt.subplot2grid((5, 3), (3, 2))
    ax.plot(xi_cont, np.real(GP_FT_mu), label='Prediction Mean', color='red', marker='o')
    ax.set_xlabel('Frequency [rad s$^{-1}$]')
    ax.set_ylabel('Real Part [ms$^{-2}$]')
    ax.set_title('GP Fourier Transform')
    max_re = 1.1 * max(np.max(np.abs(np.real(analytical_FT))), np.max(np.abs(np.real(DFT))), np.max(np.abs(np.real(GP_FT_mu))))
    ax.set_ylim(-max_re, max_re)
    ax.legend()

    ax = plt.subplot2grid((5, 3), (4, 2))
    ax.plot(xi_cont, np.imag(GP_FT_mu), label='Prediction Mean', color='red', marker='o')
    ax.set_xlabel('Frequency [rad s$^{-1}$]')
    ax.set_ylabel('Imaginary Part [ms$^{-2}$]')
    ax.set_title('GP Fourier Transform')
    max_imag = 1.1 * max(np.max(np.abs(np.imag(analytical_FT))), np.max(np.abs(np.imag(DFT))), np.max(np.abs(np.imag(GP_FT_mu))))
    ax.set_ylim(-max_imag, max_imag)
    ax.legend()


    ax = plt.subplot2grid((5, 3), (1, 1))
    ax.scatter(xi_disc, np.abs(DFT), color='red')
    ax.set_xlabel('Frequency [rad s$^{-1}$]')
    ax.set_ylabel('Magnitude [ms$^{-2}$]')
    ax.set_title('Discrete Fourier Transform')
    ax.set_ylim(min_mag, max_mag)

    # plt.subplot(3, 1, 3)  # a rows, b columns, plot c
    ax = plt.subplot2grid((5, 3), (2, 1))
    ax.scatter(xi_disc, np.angle(DFT), color='red')
    ax.set_xlabel('Frequency [rad s$^{-1}$]')
    ax.set_ylabel('Phase [rad]')
    ax.set_title('Discrete Fourier Transform')
    ax.set_ylim(-max_phase, max_phase)

    # plt.subplot(3, 1, 3)  # a rows, b columns, plot c
    ax = plt.subplot2grid((5, 3), (3, 1))
    ax.scatter(xi_disc, np.real(DFT), color='red')
    ax.set_xlabel('Frequency [rad s$^{-1}$]')
    ax.set_ylabel('Real Part [ms$^{-2}$]')
    ax.set_title('Discrete Fourier Transform')
    ax.set_ylim(-max_re, max_re)

    # plt.subplot(3, 1, 3)  # a rows, b columns, plot c
    ax = plt.subplot2grid((5, 3), (4, 1))
    ax.scatter(xi_disc, np.imag(DFT), color='red')
    ax.set_xlabel('Frequency [rad s$^{-1}$]')
    ax.set_ylabel('Imaginary Part [ms$^{-2}$]')
    ax.set_title('Discrete Fourier Transform')
    ax.set_ylim(-max_imag, max_imag)


    ax = plt.subplot2grid((5, 3), (1, 0))
    ax.plot(xi_cont, np.abs(analytical_FT), color='green', marker='o')
    ax.set_xlabel('Frequency [rad s$^{-1}$]')
    ax.set_ylabel('Magnitude [ms$^{-2}$]')
    ax.set_title('Analytical Fourier Transform')
    ax.set_ylim(min_mag, max_mag)

    # plt.subplot(3, 1, 3)  # a rows, b columns, plot c
    ax = plt.subplot2grid((5, 3), (2, 0))
    ax.plot(xi_cont, np.angle(analytical_FT), color='green', marker='o')
    ax.set_xlabel('Frequency [rad s$^{-1}$]')
    ax.set_ylabel('Phase [rad]')
    ax.set_title('Analytical Fourier Transform')
    ax.set_ylim(-max_phase, max_phase)

    # plt.subplot(3, 1, 3)  # a rows, b columns, plot c
    ax = plt.subplot2grid((5, 3), (3, 0))
    ax.plot(xi_cont, np.real(analytical_FT), color='green', marker='o')
    ax.set_xlabel('Frequency [rad s$^{-1}$]')
    ax.set_ylabel('Real Part [ms$^{-2}$]')
    ax.set_title('Analytical Fourier Transform')
    ax.set_ylim(-max_re, max_re)

    # plt.subplot(3, 1, 3)  # a rows, b columns, plot c
    ax = plt.subplot2grid((5, 3), (4, 0))
    ax.plot(xi_cont, np.imag(analytical_FT), color='green', marker='o')
    ax.set_xlabel('Frequency [rad s$^{-1}$]')
    ax.set_ylabel('Imaginary Part [ms$^{-2}$]')
    ax.set_title('Analytical Fourier Transform')
    ax.set_ylim(-max_imag, max_imag)

    if xi_mode == 'uniform':
        print_MSE(analytical_FT, DFT, GP_FT_mu)

    print_wd(analytical_FT, DFT, GP_FT_mu, xi_cont, xi_disc, xi_mode)

    fig.suptitle(suptitle, fontsize=24, ha='left', x=0.1)
    plt.tight_layout()
    plt.show()

    fig.savefig(data_dir.replace(".csv", ".png"), dpi=150, bbox_inches='tight', transparent=False)



if __name__ == "__main__":
    plot_data(f"../output_data/plot_df_2024-03-17_17-38-27_With Cluster Peak.csv")