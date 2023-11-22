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

    def test__inverse_upper_triangular(self):
        obj = GP_NLL_FITC(1, 2, 3, 4, 5, 6)

        # Upper triangular matrix
        matrix = np.array([
            [2, 3, 1],
            [0, 4, 5],
            [0, 0, 7]
        ])

        start_time = timer.time()

        result = obj._inverse_upper_triangular(matrix)

        end_time = timer.time()
        elapsed_time = end_time - start_time
        print(f"The _inverse_upper_triangular func ran in {elapsed_time} seconds")


        # The inverse of the upper triangular matrix calculated by hand or another method
        correct = np.array([
            [0.5, -3. / 8., 1. / 28.],
            [0, 0.25, -5. / 28.],
            [0, 0, 1. / 7.]
        ])

        start_time = timer.time()

        correct = np.linalg.inv(matrix)

        end_time = timer.time()
        elapsed_time = end_time - start_time
        print(f"The np.linalg.inv func ran in {elapsed_time} seconds")

        debug_print(f"result = {result}")
        debug_print(f"correct = {correct}")
        debug_print(f"difference = {result - correct}")

        assert np.allclose(result, correct, atol=1E-5, rtol=1E-5)

        # Another upper triangular matrix with different values
        matrix2 = np.array([
            [-1.9883, 2.0, 3.0],
            [0, -9.4199, 4.0],
            [0, 0, 1.7969]
        ])

        result2 = obj._inverse_upper_triangular(matrix2)

        # The inverse of the second upper triangular matrix
        correct2 = np.array([
            [-0.5029, 0.1065, -0.1876],
            [0, -0.1062, -0.2353],
            [0, 0, 0.5565]
        ])

        correct2 = np.linalg.inv(matrix2)

        debug_print(f"result2 = {result2}")
        debug_print(f"correct2 = {correct2}")
        debug_print(f"difference2 = {result2 - correct2}")

        assert np.allclose(result2, correct2, atol=1E-5, rtol=1E-5)