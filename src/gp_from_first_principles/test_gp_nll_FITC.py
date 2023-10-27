import scipy
# test_my_module.py
import math
import time as timer
from gp_model import GPModel
import unittest
from utils import debug_print
from main import format_data, load_data
import numpy as np
from hyperparameters import Hyperparameters
from gp_kernel import GaussianProcessKernel
from gp_nll_FITC import GP_NLL_FITC

class TestGP_NLL_FITC_18_134(unittest.TestCase):
    def test_compute(self):
        pass

    def test_K_XX_FITC(self):
        hyperparameters_obj = Hyperparameters('periodic')
        kernel = GaussianProcessKernel(hyperparameters_obj)
        dummy_X = np.array([0.001, 0.00102, 0.004, 0.00402]).reshape(-1,1)
        dummy_y = np.array([0.002, 0.004, 0.006, 0.008]).reshape(-1,1)
        dummy_U = np.array([0.00101, 0.00401]).reshape(-1,1)
        mean_dummy_y = np.mean(dummy_y)
        gp_nll_fitc_18_134 = GP_NLL_FITC(dummy_X,
                                         dummy_y,
                                         mean_dummy_y,
                                         dummy_U,
                                         kernel,
                                         hyperparameters_obj)

        K_XX_FITC_result, K_XU_result, K_UX_result, K_UU_result, K_XX_result, Q_XX_result, K_UU_inv_KUX_result = gp_nll_fitc_18_134.K_XX_FITC()

        K_UX_correct = np.array([[95.23616994, 95.23616994, 6.0173E-188, 2.0587E-190],
                                [2.0587E-190, 6.0173E-188, 95.23616994, 95.23616994]])
        K_XU_correct = K_UX_correct.T
        K_UU_correct = np.array([[6.38E+01, 2.02E-200],
                                 [3.14E-178, 6.38E+01]])
        K_XX_correct = np.array([[9.57E+01, 9.38E+01, 3.53E-189, 1.19E-191],
                                [9.38E+01, 9.57E+01, 1.02E-186, 3.53E-189],
                                [3.53E-189, 1.02E-186, 9.57E+01, 9.38E+01],
                                [1.19E-191, 3.53E-189, 9.38E+01, 9.57E+01]])


        K_UU_stable_correct = K_UU_correct + 1e-6 * np.eye(K_UU_correct.shape[0])
        L_UU_correct = scipy.linalg.cholesky(K_UU_stable_correct, lower=True)
        K_UU_inv_KUX_correct = scipy.linalg.cho_solve((L_UU_correct, True), K_UX_correct)
        Q_XX_correct = K_XU_correct @ K_UU_inv_KUX_correct

        atol_ = 1E-150
        rtol_ = 0.6

        assert np.allclose(K_XX_FITC_result, K_XX_result, atol=atol_,rtol=rtol_), "incorrect K_XX_FITC"
        assert np.allclose(K_XU_result, K_XU_correct, atol=atol_, rtol=rtol_), "incorrect K_XU"
        assert np.allclose(K_UX_result, K_UX_correct, atol=atol_, rtol=rtol_), "incorrect K_UX"
        assert np.allclose(K_UU_result, K_UU_correct, atol=atol_, rtol=rtol_), "incorrect K_UU"
        assert np.allclose(K_XX_result, K_XX_correct, atol=atol_, rtol=rtol_), "incorrect K_XX"
        assert np.allclose(Q_XX_result, Q_XX_correct, atol=atol_, rtol=rtol_), "incorrect Q_XX"
        assert np.allclose(K_UU_inv_KUX_result, K_UU_inv_KUX_correct, atol=atol_, rtol=rtol_), "incorrect K_UU_inv_KUX"

        # assert result is what_we_think
        # assert k_XU is what_we_think
        # assert k_UX_result is what_we_think
        # assert k_UU_result is what_we_think





        pass