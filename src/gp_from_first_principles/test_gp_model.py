# test_my_module.py
import math
import time as timer
from gp_model import GPModel
import unittest
import utils
from main import format_data, load_data
import numpy as np

class TestGPModel(unittest.TestCase):

    def test_fast_det(self):
        force_input_model = self.setup_object()

        U = np.ones((4000,2))
        V_T = U.T
        D = 1.001 * np.eye(U.shape[0])

        start_time = timer.time()

        fast_det = force_input_model.fast_det(U,V_T,D)

        end_time = timer.time()
        elapsed_time = end_time - start_time
        print(f"fast_det ran in {elapsed_time} seconds")

        start_time = timer.time()

        det = np.linalg.det(D+U @ V_T)
        equal = math.isclose(fast_det, det, rel_tol=1E-9, abs_tol=1E-9)
        assert equal, "Incorrect Fast_det"

        end_time = timer.time()
        elapsed_time = end_time - start_time
        print(f"det ran in {elapsed_time} seconds")

    def test_K_XX_FITC(self):
        gp = self.setup_object(0)
        gp.X = np.array([2,4,6,8])
        gp.U = np.array([3,7])

        gp.hyperparameters_obj.update(np.array([0.1,0.01,1E-3,0.001,0.1]))

        K_XX_FITC, K_XU, K_UX, K_UU, K_XX, Q_XX, K_UU_inv_KUX = gp.K_XX_FITC()

        result = np.array([K_XX_FITC, K_XU, K_UX, K_UU, K_XX, Q_XX, K_UU_inv_KUX])

        correct = np.array([

        ])


        assert np.K_XX_FITC ==



    def setup_object(self, force_input_kernel_index = 2):
        sample_start_index = 1000
        sample_length = 100
        num_predictions = 40
        force_input_kernel_type = \
            ['squared_exponential', 'p_se_composite', 'white_noise',
             'wn_se_composite'][force_input_kernel_index]
        force_input_solver_type = \
            ['metropolis_hastings', 'iterative_search', 'adam', 'free_lunch'][
                0]
        force_input_predict_type = ['cholesky', 'FITC'][0]
        force_input_n_iter = 50
        force_response_kernel_type = \
            ['squared_exponential', 'p_se_composite', 'white_noise',
             'wn_se_composite'][1]
        force_response_solver_type = \
            ['metropolis_hastings', 'iterative_search', 'adam', 'free_lunch'][
                0]
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
        return force_input_model


if __name__ == '__main__':
    unittest.main()
