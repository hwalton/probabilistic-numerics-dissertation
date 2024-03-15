import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_data(plot_df):
    plt.figure(figsize=(16.5, 11.7))
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

    if force_response_prediction_f.ndim > 1:
        assert 0, "Not yet implemented"
        #force_response_prediction_diag = np.array(np.diag(force_response_prediction_f))
    else:
        force_response_prediction_diag = np.array(force_response_prediction_f)


    plt.subplot(3, 1, 1)  # a rows, b columns, plot c
    plt.scatter(time, force_response, label='Force Response', color='green')
    plt.plot(time_test, force_response_prediction_t, label='Prediction Mean', color='red')

    # Assuming prediction_f is the standard deviation
    upper_bound = force_response_prediction_t + force_response_prediction_diag
    lower_bound = force_response_prediction_t - force_response_prediction_diag

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


def main(csv_file):
    plot_df = pd.read_csv(f'../output_data/{csv_file}.csv')
    plot_data(plot_df)

if __name__ == "__main__":
    main("plot_df_2024-03-15_10-35-40")