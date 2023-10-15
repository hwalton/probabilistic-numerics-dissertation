# test_my_module.py
import math
import time as timer
from gp_model import GPModel
import unittest
import utils
from main import format_data, load_data
import numpy as np

class TestGPModel(unittest.TestCase):
    def setup_object(self, force_input_kernel_index=2, return_model='input', nll_method = 'cholesky'):
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
                                    n_iter=force_input_n_iter, nll_method = nll_method)

        force_response_model = GPModel(force_response_kernel_type,
                                       time,
                                       force_response,
                                       solver_type=force_response_solver_type,
                                       n_iter=force_response_n_iter)

        if return_model == 'input':
            return force_input_model
        if return_model == 'response':
            return force_response_model
    def test_fast_det(self):
        force_input_model = self.setup_object()

        U = np.ones((1000,2))
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

        gp.hyperparameters_obj.update(np.array([0.1,1.,0.001,0.1]))

        K_XX_FITC, K_XU, K_UX, K_UU, K_XX, Q_XX, K_UU_inv_KUX = gp.K_XX_FITC()

        result = [K_XX_FITC, K_XU, K_UX, K_UU, K_XX, Q_XX, K_UU_inv_KUX]

        K_XX_FITC_C = np.array([[1.00E-02, 1.35E-03, 3.35E-06, 1.52E-10],
            [1.35E-03, 1.00E-02, 1.35E-03, 3.35E-06],
            [3.35E-06, 1.35E-03, 1.00E-02, 1.35E-03],
            [1.52E-10, 3.35E-06, 1.35E-03, 1.00E-02]])

        K_XU_C = np.array([[6.07E-03, 3.73E-08],
                 [6.07E-03, 1.11E-04],
                 [1.11E-04, 6.07E-03],
                 [3.73E-08, 6.07E-03]])

        K_UX_C = K_XU_C.T

        K_UU_C = np.array([[1.00E-02,3.35E-06],
                    [3.35E-06,1.00E-02]])

        K_XX_C = K_XX_FITC_C

        Q_XX_C = K_XU_C @ np.linalg.inv(K_UU_C) @ K_UX_C

        K_UU_inv_KUX_C = np.linalg.inv(K_UU_C) @ K_UX_C



        correct = [ K_XX_FITC_C, K_XU_C, K_UX_C, K_XX_C, Q_XX_C, K_UU_inv_KUX]

        for i in correct:
            assert np.allclose(K_XX_FITC, K_XX_FITC_C, atol = 1E-2), "incorrect K_XX_FITC"
            assert np.allclose(K_XU, K_XU_C, atol=1E-2), "incorrect K_XU"
            assert np.allclose(K_UX, K_UX_C, atol=1E-2), "incorrect K_UX"
            assert np.allclose(K_UU, K_UU_C, atol=1E-2), "incorrect K_UU"
            assert np.allclose(K_XX, K_XX_C, atol=1E-2), "incorrect K_XX"
            assert np.allclose(Q_XX, Q_XX_C, atol=1E-2), "incorrect Q_XX"
            assert np.allclose(K_UU_inv_KUX, K_UU_inv_KUX_C, atol=1E-2), "incorrect K_UU_inv_KUX"

    def test_K_sigma_inv(self):
        gp = self.setup_object(2, return_model='response')
        # gp.hyperparameters_obj.update(np.array([0.1, 1., 1E-3, 0.1, 1., 1E-3, 0.1, 0.01, 0.001, 1.]))

        gp.hyperparameters_obj.update(np.array(
            [1.55651623e+00, 1.29376154e+00, 1.93658826e-03, 1.00000000e-04,
             4.34121551e-01, 3.50302199e-01, 1.03262772e+01, 1.20501525e-04,
             3.68744450e-13, 5.43714621e-01]))
        #gp.X = np.array([2,4,6,8])
        #gp.U = np.array([3,7])

        gp.hyperparameters_obj.update(np.array([0.1, 1., 1E-3, 0.1, 1., 1E-3, 0.1, 0.01, 0.001, 1.]))

        result = gp.K_sigma_inv()

        correct = np.linalg.inv(np.squeeze(gp.gp_kernel.compute_kernel(gp.X, gp.X)) + np.multiply(gp.hyperparameters_obj.dict()['noise_level'] ** 2, np.eye(gp.X.shape[0])))

        assert np.allclose(result, correct, atol=1E-3, rtol=2E-2), "incorrect K_sigma_inv"

    # def test_def_compute_nll_input(self):
    #     gp = self.setup_object(2, return_model='input')
    #     # gp.hyperparameters_obj.update(np.array([0.1, 1., 1E-3, 0.1, 1., 1E-3, 0.1, 0.01, 0.001, 1.]))
    #
    #     gp.hyperparameters_obj.update(np.array([6.91127308e-02, 3.04299891e-11, 1.00000000e+00]))
    #
    #     hyp_array = gp.hyperparameters_obj.array()
    #     print(f"test hyperparameters updated to: {hyp_array}")
    #
    #     result = gp.compute_nll(gp.hyperparameters_obj,
    #                             method='FITC_18_134')
    #
    #     K_XX_FITC, K_XU, K_UX, K_UU, K_XX, Q_XX, K_UU_inv_KUX = gp.K_XX_FITC()
    #     n = K_XX.shape[0]
    #     y_adj = np.squeeze(
    #         gp.y - gp.hyperparameters_obj.dict()['mean_func_c'])
    #     big_lambda = np.diag(np.diag(K_XX - Q_XX)) + \
    #                  gp.hyperparameters_obj.dict()[
    #                      'noise_level'] ** 2 * np.eye(n)
    #     det = np.linalg.det(Q_XX + big_lambda)
    #     lml = 0.5 * np.log(
    #         det) + 0.5 * y_adj.T @ gp.K_sigma_inv() @ y_adj + 0.5 * n * np.log(
    #         2 * np.pi)
    #     nlml = np.array(-lml)
    #
    #     cholesky = gp.compute_nll(gp.hyperparameters_obj,
    #                               method='cholesky')
    #     correct = nlml
    #     assert correct < 1000, "nll too large"
    #     assert np.allclose(result, cholesky, atol=1E-3,
    #                        rtol=15E-2), "nll mismatch with cholesky"
    #     assert np.allclose(result, correct, atol=1E-3,
    #                        rtol=2E-2), "nll mismatch with correct"

    def test_def_compute_nll_response(self):

        gp_cholesky = self.setup_object(2, return_model='response', nll_method = 'cholesky')
        gp_FITC = self.setup_object(2, return_model='response', nll_method = 'FITC_18_134')
        #gp.hyperparameters_obj.update(np.array([0.1, 1., 1E-3, 0.1, 1., 1E-3, 0.1, 0.01, 0.001, 1.]))


        optimal_hyperparameters = np.array([1.55651623e+00, 1.29376154e+00, 1.93658826e-03, 1.00000000e-04,
 4.34121551e-01, 3.50302199e-01, 1.03262772e+01, 1.20501525e-04,
 3.68744450e-13, 5.43714621e-01])

        gp_cholesky.hyperparameters_obj.update(optimal_hyperparameters)

        gp_FITC.hyperparameters_obj.update(optimal_hyperparameters)

        hyp_array = gp_cholesky.hyperparameters_obj.array()
        print(f"test hyperparameters updated to: {hyp_array}")

        result_cholesky = gp_cholesky.compute_nll(gp_cholesky.hyperparameters_obj)
        result_FITC = gp_FITC.compute_nll(gp_FITC.hyperparameters_obj)



        # K_XX_FITC, K_XU, K_UX, K_UU, K_XX, Q_XX, K_UU_inv_KUX = gp.K_XX_FITC()
        # n = K_XX.shape[0]
        # y_adj = np.squeeze(gp.y - gp.hyperparameters_obj.dict()['mean_func_c'])
        # big_lambda = np.diag(np.diag(K_XX-Q_XX)) + gp.hyperparameters_obj.dict()['noise_level'] ** 2 * np.eye(n)
        # det = np.linalg.det(Q_XX + big_lambda)
        # K_sigma_inv = np.linalg.inv(K_XX + gp.hyperparameters_obj.dict()['noise_level'] ** 2 * np.eye(n))
        # lml = 0.5 * np.log(det) + 0.5 * y_adj.T @ K_sigma_inv @ y_adj + 0.5 * n * np.log(2 * np.pi)
        # nlml = np.array(-lml)
        #
        # cholesky = gp.compute_nll(gp.hyperparameters_obj, method = 'cholesky')
        # correct = nlml
        # assert result < 1000, "nll too large"
        # assert np.allclose(result, cholesky, atol=1E-3,
        #                    rtol=15E-2), "nll mismatch with cholesky"
        # assert np.allclose(result, correct, atol=1E-3,
        #                    rtol=2E-2), "nll mismatch with correct"






if __name__ == '__main__':
    unittest.main()
