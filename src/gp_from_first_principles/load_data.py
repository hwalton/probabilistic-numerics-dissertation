import numpy as np
import os

def load_data():
    base_path = '/home/harvey/Git/probabilistic-numerics-dissertation/datasets'

    force_response_path = f'{base_path}/force_response.csv'
    time_path = f'{base_path}/time_truncated.csv'

    force_response = np.loadtxt(force_response_path, delimiter=',')
    time_truncated = np.loadtxt(time_path, delimiter=',')

    assert force_response.size == time_truncated.size, "Force response and time arrays must have the same size"

    return force_response, time_truncated