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
from gp_nll_FITC_18_134 import GP_NLL_FITC_18_134

class TestGP_NLL_FITC_18_134(unittest.TestCase):
    def test_compute(self):
        pass

    def test_K_XX_FITC(self):
        hyperparameters_obj = Hyperparameters('periodic')
        kernel = GaussianProcessKernel(hyperparameters_obj)
        dummy_X = np.array([0.001, 0.002, 0.003, 0.004]).reshape(-1,1)
        dummy_y = np.array([0.002, 0.004, 0.006, 0.008]).reshape(-1,1)
        dummy_U = np.array([0.0015, 0.0035]).reshape(-1,1)
        mean_dummy_y = np.mean(dummy_y)
        gp_nll_fitc_18_134 = GP_NLL_FITC_18_134(dummy_X,
                                                dummy_y,
                                                mean_dummy_y,
                                                dummy_U,
                                                kernel,
                                                hyperparameters_obj)

        result, K_XU_result, K_UX_result, K_UU_result, K_XX_result, Q_XX_result, K_UU_inv_KUX_result = gp_nll_fitc_18_134.K_XX_FITC()

        K_UX_correct = np.array([[0.000347591, 0.000347591, 2.15404E-47, 3.6782E-132],
                                [3.6782E-132, 2.15404E-47, 0.000347591, 0.000347591]])
        K_XU_correct = K_UX_correct.T
        K_UU_correct = np.array([[9.57E+01, 9.51E-85],
                                 [9.51E-85, 9.57E+01]])

        correct = K_XU_correct @ np.linalg.inv(K_UU_correct) @ K_UX_correct

        assert np.allclose(result, correct, atol=1E-2), "incorrect K_XX_FITC"


        # assert result is what_we_think
        # assert k_XU is what_we_think
        # assert k_UX_result is what_we_think
        # assert k_UU_result is what_we_think





        pass