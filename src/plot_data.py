import matplotlib.pyplot as plt
import numpy as np


def plot_data(force_response, force_response_prediction, time, time_test, xi, analytical_FT, DFT, GP_FT_mu, GP_FT_stdv):
    plt.figure(figsize=(12, 10.5))
    plt.rcParams.update({'font.size': 16})

    if force_response_prediction[1].ndim > 1:
        force_response_prediction_diag = np.diag(force_response_prediction[1])
    else:
        force_response_prediction_diag = force_response_prediction[1]


    plt.subplot(3, 1, 1)  # a rows, b columns, plot c
    plt.scatter(time, force_response, label='Force Response', color='green')
    plt.plot(time_test, force_response_prediction[0], label='Prediction Mean', color='red')

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

    debug_abs = np.abs(GP_FT_stdv)
    upper_bound_abs = np.abs(GP_FT_mu) + np.abs(GP_FT_stdv)
    lower_bound_abs = np.abs(GP_FT_mu) - np.abs(GP_FT_stdv)


    plt.subplot(3, 1, 2)  # a rows, b columns, plot c
    (plt.scatter(xi, np.abs(GP_FT_mu)))
    plt.fill_between(np.squeeze(xi), lower_bound_abs, upper_bound_abs, color='blue',
                     alpha=0.2, label='Std Dev')
    plt.xlabel('Freq [Rad/s]')
    plt.ylabel('Magnitude of Fourier Transform')

    debug_angle = np.angle(GP_FT_stdv)

    upper_bound_angle = np.angle(GP_FT_mu) + np.angle(GP_FT_stdv)
    lower_bound_angle = np.angle(GP_FT_mu) - np.angle(GP_FT_stdv)

    plt.ylim(-0.5 * np.max(np.abs(GP_FT_mu)), 1.5 * np.max(np.abs(GP_FT_mu)))


    plt.subplot(3, 1, 3)  # a rows, b columns, plot c
    plt.scatter(xi, np.angle(GP_FT_mu))
    plt.fill_between(np.squeeze(xi), lower_bound_angle, upper_bound_angle, color='blue',
                     alpha=0.2, label='Std Dev')
    plt.xlabel('Freq [Rad/s]')
    plt.ylabel('Phase of Fourier Transform')
    plt.show()