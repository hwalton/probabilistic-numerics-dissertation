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
        hyperparameters_obj = Hyperparameters('p_se_composite')
        kernel = GaussianProcessKernel(hyperparameters_obj)
        dummy_X = np.array([1., 2., 3., 4.]).reshape(-1,1)
        dummy_y = np.array([2., 4., 6., 8.]).reshape(-1,1)
        dummy_U = np.array([1.5, 3.5]).reshape(-1,1)

        gp_nll_fitc_18_134 = GP_NLL_FITC_18_134(dummy_X,
                                                dummy_y,
                                                dummy_U,
                                                kernel,
                                                hyperparameters_obj)

        result, K_XU_result, K_UX_result, K_UU_result, K_XX_result, Q_XX_result, K_UU_inv_KUX_result = gp_nll_fitc_18_134.K_XX_FITC()

        # K_XU = np.squeeze(kernel.compute_kernel(dummy_X, dummy_U))
        # K_UX = np.squeeze(kernel.compute_kernel(dummy_U, dummy_X))
        # K_UU = np.squeeze(kernel.compute_kernel(dummy_U, dummy_U))
        #
        #
        #
        # correct = K_XU @ np.linalg.inv(K_UU) @ K_UX
        #
        # assert np.allclose(result, correct, atol=1E-2), "incorrect K_XX_FITC"


        assert result is what_we_think
        assert k_XU is what_we_think
        assert k_UX_result is what_we_think
        assert k_UU_result is what_we_think





        pass