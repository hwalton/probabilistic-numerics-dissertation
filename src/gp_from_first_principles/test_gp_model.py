# test_my_module.py
import math

from gp_model import GPModel
import unittest
import utils
from main import format_data, load_data
import numpy as np

class TestGPModel(unittest.TestCase):
    def test_fast_det(self):
        sample_start_index = 1000
        sample_length = 100
        num_predictions = 40
        force_input_kernel_type = \
        ['squared_exponential', 'p_se_composite', 'white_noise',
         'wn_se_composite'][2]
        force_input_solver_type = \
        ['metropolis_hastings', 'iterative_search', 'adam', 'free_lunch'][0]
        force_input_predict_type = ['cholesky', 'FITC'][0]
        force_input_n_iter = 50
        force_response_kernel_type = \
        ['squared_exponential', 'p_se_composite', 'white_noise',
         'wn_se_composite'][1]
        force_response_solver_type = \
        ['metropolis_hastings', 'iterative_search', 'adam', 'free_lunch'][0]
        force_response_predict_type = ['cholesky', 'FITC'][0]
        force_response_n_iter = 50
        force_input, force_response, time = load_data(sample_start_index,
                                                      sample_length)
        lower = time[0] - 0 * (time[-1] - time[0])
        upper = time[-1] + 0 * (time[-1] - time[0])
        time_test = np.linspace(lower, upper, num=num_predictions,
                                endpoint=True)
        force_input = format_data(force_input)
        time = format_data(time)
        force_response = format_data(force_response)
        time_test = format_data(time_test)

        force_input_model = GPModel(force_input_kernel_type,
                                    time,
                                    force_input,
                                    solver_type=force_input_solver_type,
                                    n_iter=force_input_n_iter)

        U = np.array([[1, 2],
                     [3, 4],
                     [5, 6]])
        V_T = U.T
        D = 3 * np.eye(U.shape[0])

        fast_det = force_input_model.fast_det(U,V_T,D)
        det = np.linalg.det(D+U @ V_T)
        equal = math.isclose(fast_det, det, rel_tol=1E-9, abs_tol=1E-9)
        assert equal, "Incorrect Fast_det"




if __name__ == '__main__':
    unittest.main()
