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
