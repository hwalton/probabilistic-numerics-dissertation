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

class TestGP_NLL_FITC(unittest.TestCase):
    def test__inverse_lower_triangular(self):
        obj = GP_NLL_FITC(1,2,3,4,5,6)

        matrix = np.array([
                [2, 0, 0],
                [3, 4, 0],
                [5, 6, 7]
        ])

        result = obj._inverse_lower_triangular(matrix)

        correct = np.array([
                [0.5, 0.0, 0.0],
                [-3./8., 0.25, 0.0],
                [-1./28., -3./14., 1./7.]
        ])

        debug_print(f"result = {result}")
        debug_print(f"correct = {correct}")

        assert np.allclose(result, correct, rtol= 0.01 )

        obj = GP_NLL_FITC(1,2,3,4,5,6)

        matrix2 = np.array([
            [-1.9883, 0, 0],
            [7.3051, -9.4199, 0],
            [5.7550, -6.4353, 1.7969]
        ])

        result2 = obj._inverse_lower_triangular(matrix2)

        correct2 = np.array([
            [-0.5029, 0, 0],
            [-0.3900, -0.1062, 0],
            [0.2140, -0.3802, 0.5565]
        ])

        debug_print(f"result2 = {result2}")
        debug_print(f"correct2 = {correct2}")
        debug_print(f"difference2 = {result2 - correct2}")


        assert np.allclose(result2, correct2, atol = 1E-3, rtol= 1E-3 )
