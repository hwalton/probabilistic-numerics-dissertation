import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_data(plot_df):
    plt.figure(figsize=(12, 10.5))
    plt.rcParams.update({'font.size': 16})

    time = plot_df.time

    if plot_df.force_response_prediction_f.ndim > 1:
        force_response_prediction_diag = np.diag(plot_df.force_response_prediction_f)
    else:
        force_response_prediction_diag = plot_df.force_response_prediction_f


    plt.subplot(3, 1, 1)  # a rows, b columns, plot c
    plt.scatter(time, plot_df.force_response, label='Force Response', color='green')
    plt.plot(plot_df.time_test, plot_df.force_response_prediction_t, label='Prediction Mean', color='red')

    # Assuming prediction_f is the standard deviation
    upper_bound = plot_df.force_response_prediction_t + force_response_prediction_diag
    lower_bound = plot_df.force_response_prediction_t - force_response_prediction_diag

    plt.fill_between(np.squeeze(plot_df.time_test), np.squeeze(lower_bound), np.squeeze(upper_bound), color='blue',
                     alpha=0.2, label='Prediction Std Dev')

    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [ms$^{-2}$]')
    plt.title('Force Response Over Time')

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    debug_abs = np.abs(plot_df.GP_FT_stdv)
    upper_bound_abs = np.abs(plot_df.GP_FT_mu) + np.abs(plot_df.GP_FT_stdv)
    lower_bound_abs = np.abs(plot_df.GP_FT_mu) - np.abs(plot_df.GP_FT_stdv)


    plt.subplot(3, 1, 2)  # a rows, b columns, plot c
    (plt.scatter(plot_df.xi, np.abs(plot_df.GP_FT_mu)))
    plt.fill_between(np.squeeze(plot_df.xi), lower_bound_abs, upper_bound_abs, color='blue',
                     alpha=0.2, label='Std Dev')
    plt.xlabel('Freq [Rad/s]')
    plt.ylabel('Magnitude of Fourier Transform')

    debug_angle = np.angle(plot_df.GP_FT_stdv)

    upper_bound_angle = np.angle(plot_df.GP_FT_mu) + np.angle(plot_df.GP_FT_stdv)
    lower_bound_angle = np.angle(plot_df.GP_FT_mu) - np.angle(plot_df.GP_FT_stdv)

    plt.ylim(-0.5 * np.max(np.abs(plot_df.GP_FT_mu)), 1.5 * np.max(np.abs(plot_df.GP_FT_mu)))


    plt.subplot(3, 1, 3)  # a rows, b columns, plot c
    plt.scatter(plot_df.xi, np.angle(plot_df.GP_FT_mu))
    plt.fill_between(np.squeeze(plot_df.xi), lower_bound_angle, upper_bound_angle, color='blue',
                     alpha=0.2, label='Std Dev')
    plt.xlabel('Freq [Rad/s]')
    plt.ylabel('Phase of Fourier Transform')
    plt.show()