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

    print(f"MSE DFT Magnitude: {mse_DFT_mag}")
    print(f"MSE GP FT Magnitude: {mse_GP_FT_mag}\n")

    print(f"MSE DFT Phase: {mse_DFT_phase}")
    print(f"MSE GP FT Phase: {mse_GP_FT_phase}\n")

    print(f"MSE DFT Real: {mse_DFT_real}")
    print(f"MSE GP FT Real: {mse_GP_FT_real}\n")

    print(f"MSE DFT Imaginary: {mse_DFT_imag}")
    print(f"MSE GP FT Imaginary: {mse_GP_FT_imag}\n")


def print_wd(analytical_FT, DFT, GP_FT_mu, xi):

    m = float(os.getenv('M'))
    c = float(os.getenv('C'))
    k = float(os.getenv('K'))

    wn = np.sqrt(k/m)
    zeta = c / (2 * np.sqrt(m * k))
    wd_analytical_FT =  wn * np.sqrt(1 - zeta**2)

    i = np.squeeze(np.argwhere(np.abs(DFT) == max(np.abs(DFT)))[-1])
    wd_DFT = np.squeeze(np.abs(xi[i]))

    j = np.squeeze(np.argwhere(np.abs(GP_FT_mu) == max(np.abs(GP_FT_mu)))[-1])
    wd_GP_FT = np.squeeze(xi[j])

    print(f"\u03C9_d analytical FT: {wd_analytical_FT}")
    print(f"\u03C9_d DFT: {wd_DFT}")
    print(f"\u03C9_d GP FT: {wd_GP_FT}\n")

def plot_data(plot_df):
    plt.figure(figsize=(33.1 , 23.4))
    plt.rcParams.update({'font.size': 16})

    arrays = {col: np.array(plot_df[col]) for col in plot_df.columns}

    time = arrays['time']
    time_test = arrays['time_test']
    force_response = arrays['force_response']
    force_response_prediction_t = arrays['force_response_prediction_t']
    force_response_prediction_f = arrays['force_response_prediction_f']
    xi = arrays['xi']
    GP_FT_mu = arrays['GP_FT_mu']
    GP_FT_stdv = arrays['GP_FT_stdv']
    analytical_FT = arrays['analytical_FT']
    DFT = arrays['DFT']



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
    ax1 = plt.subplot2grid((5, 3), (0, 0), colspan=3)
    ax1.scatter(time, force_response, label='Force Response', color='green')
    ax1.plot(time_test, force_response_prediction_t, label='Prediction Mean', color='red')

    # Assuming prediction_f is the standard deviation
    upper_bound = force_response_prediction_t + force_response_prediction_diag
    lower_bound = force_response_prediction_t - force_response_prediction_diag

    plt.fill_between(np.squeeze(time_test), np.squeeze(lower_bound), np.squeeze(upper_bound), color='blue',
                     alpha=0.2, label='Prediction Std Dev')

    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Acceleration [ms$^{-2}$]')
    ax1.set_title('Force Response Over Time')

    ax1.legend()
    ax1.grid(True)

    upper_bound_abs = np.abs(GP_FT_mu) + np.abs(GP_FT_stdv)
    lower_bound_abs = np.abs(GP_FT_mu) - np.abs(GP_FT_stdv)


    ax = plt.subplot2grid((5, 3), (1, 2))
    ax.scatter(xi, np.abs(GP_FT_mu))
    ax.fill_between(np.squeeze(xi), lower_bound_abs, upper_bound_abs, color='blue',
                     alpha=0.2, label='Std Dev')
    ax.set_xlabel('Freq [Rad/s]')
    ax.set_ylabel('Magnitude [ms$^{-2}$]')
    ax.set_title('GP Fourier Transform')
    upper_bound_angle = np.angle(GP_FT_mu) + np.angle(GP_FT_stdv)
    lower_bound_angle = np.angle(GP_FT_mu) - np.angle(GP_FT_stdv)
    max_mag = 1.1 * max(np.max(np.abs(analytical_FT)), np.max(np.abs(DFT)), np.max(np.abs(GP_FT_mu)))
    ax.set_ylim(-0.5 * max_mag, 1.5 * max_mag)


    ax = plt.subplot2grid((5, 3), (2, 2))
    ax.scatter(xi, np.angle(GP_FT_mu))
    ax.fill_between(np.squeeze(xi), lower_bound_angle, upper_bound_angle, color='blue',
                     alpha=0.2, label='Std Dev')
    ax.set_xlabel('Freq [Rad/s]')
    ax.set_ylabel('Phase [Rad]')
    max_phase = (1.1 * np.pi)
    ax.set_ylim(-max_phase, max_phase)

    ax = plt.subplot2grid((5, 3), (3, 2))
    ax.scatter(xi, np.real(GP_FT_mu))
    ax.set_xlabel('Freq [Rad/s]')
    ax.set_ylabel('Real Part [ms$^{-2}$]')
    max_re = 1.1 * max(np.max(np.abs(np.real(analytical_FT))), np.max(np.abs(np.real(DFT))), np.max(np.abs(np.real(GP_FT_mu))))
    ax.set_ylim(-max_re, max_re)

    ax = plt.subplot2grid((5, 3), (4, 2))
    ax.scatter(xi, np.imag(GP_FT_mu))
    ax.set_xlabel('Freq [Rad/s]')
    ax.set_ylabel('Imaginary Part [ms$^{-2}$]')
    max_imag = 1.1 * max(np.max(np.abs(np.imag(analytical_FT))), np.max(np.abs(np.imag(DFT))), np.max(np.abs(np.imag(GP_FT_mu))))
    ax.set_ylim(-max_imag, max_imag)


    ax = plt.subplot2grid((5, 3), (1, 1))
    ax.scatter(xi, np.abs(DFT))
    ax.set_xlabel('Freq [Rad/s]')
    ax.set_ylabel('Magnitude [ms$^{-2}$]')
    ax.set_title('Discrete Fourier Transform')
    ax.set_ylim(-0.5 * max_mag, 1.5 * max_mag)

    # plt.subplot(3, 1, 3)  # a rows, b columns, plot c
    ax = plt.subplot2grid((5, 3), (2, 1))
    ax.scatter(xi, np.angle(DFT))
    ax.set_xlabel('Freq [Rad/s]')
    ax.set_ylabel('Phase [Rad]')
    ax.set_ylim(-max_phase, max_phase)

    # plt.subplot(3, 1, 3)  # a rows, b columns, plot c
    ax = plt.subplot2grid((5, 3), (3, 1))
    ax.scatter(xi, np.real(DFT))
    ax.set_xlabel('Freq [Rad/s]')
    ax.set_ylabel('Real Part [ms$^{-2}$]')
    ax.set_ylim(-max_re, max_re)

    # plt.subplot(3, 1, 3)  # a rows, b columns, plot c
    ax = plt.subplot2grid((5, 3), (4, 1))
    ax.scatter(xi, np.imag(DFT))
    ax.set_xlabel('Freq [Rad/s]')
    ax.set_ylabel('Imaginary Part [ms$^{-2}$]')
    ax.set_ylim(-max_imag, max_imag)


    ax = plt.subplot2grid((5, 3), (1, 0))
    ax.scatter(xi, np.abs(analytical_FT))
    ax.set_xlabel('Freq [Rad/s]')
    ax.set_ylabel('Magnitude [ms$^{-2}$]')
    ax.set_title('Analytical Fourier Transform')
    ax.set_ylim(-0.5 * max_mag, 1.5 * max_mag)

    # plt.subplot(3, 1, 3)  # a rows, b columns, plot c
    ax = plt.subplot2grid((5, 3), (2, 0))
    ax.scatter(xi, np.angle(analytical_FT))
    ax.set_xlabel('Freq [Rad/s]')
    ax.set_ylabel('Phase [Rad]')
    ax.set_ylim(-max_phase, max_phase)

    # plt.subplot(3, 1, 3)  # a rows, b columns, plot c
    ax = plt.subplot2grid((5, 3), (3, 0))
    ax.scatter(xi, np.real(analytical_FT))
    ax.set_xlabel('Freq [Rad/s]')
    ax.set_ylabel('Real Part [ms$^{-2}$]')
    ax.set_ylim(-max_re, max_re)

    # plt.subplot(3, 1, 3)  # a rows, b columns, plot c
    ax = plt.subplot2grid((5, 3), (4, 0))
    ax.scatter(xi, np.imag(analytical_FT))
    ax.set_xlabel('Freq [Rad/s]')
    ax.set_ylabel('Imaginary Part [ms$^{-2}$]')
    ax.set_ylim(-max_imag, max_imag)

    print_MSE(analytical_FT, DFT, GP_FT_mu)
    print_wd(analytical_FT, DFT, GP_FT_mu, xi)


    plt.tight_layout()
    plt.show()





def main(csv_file):
    plot_df = pd.read_csv(f'../output_data/{csv_file}.csv')
    plot_data(plot_df)



if __name__ == "__main__":
    main("plot_df_2024-03-15_14-58-18_noisy")