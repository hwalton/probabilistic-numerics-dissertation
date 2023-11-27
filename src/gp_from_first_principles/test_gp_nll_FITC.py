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
from typing import NamedTuple
class Hyperparameters(NamedTuple):
    sf2: float
    ll: float
    sn2: float

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
        start_time = timer.time()

        result2 = obj._inverse_upper_triangular(matrix2)

        end_time = timer.time()
        elapsed_time = end_time - start_time
        print(f"The _inverse_upper_triangular func ran in {elapsed_time} seconds")

        start_time = timer.time()

        correct2 = np.linalg.inv(matrix2)

        end_time = timer.time()
        elapsed_time = end_time - start_time
        print(f"The np.linalg.inv func ran in {elapsed_time} seconds")

        debug_print(f"result2 = {result2}")
        debug_print(f"correct2 = {correct2}")
        debug_print(f"difference2 = {result2 - correct2}")

        assert np.allclose(result2, correct2, atol=1E-5, rtol=1E-5)

    def test_K_y_hat_U_R(self):
        y_hat = np.array([1, 2, 3, 4, 5, 6])
        K_fu = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15],
            [16, 17, 18]
        ])

        R = np.array([
            [1, 2, 3],
            [0, 4, 5],
            [0, 0, 6]
        ])
        K_y_hat_U_R = y_hat.T @ K_fu @ np.linalg.inv(R)
        print(f"K_y_hat_U_R.T = {K_y_hat_U_R.T}")

        K_y_hat_U_R_T = np.linalg.inv(R).T @ K_fu.T @ y_hat
        print(f"K_y_hat_U_R_T = {K_y_hat_U_R_T}")

        assert np.allclose(K_y_hat_U_R.T, K_y_hat_U_R_T, atol=1E-5, rtol=1E-5)

    def test__compute_nll(self):
        from jax import config

        config.update("jax_enable_x64", True)

        from jax import numpy as jnp
        from jax.scipy.linalg import solve_triangular
        from jax.random import PRNGKey, split, normal

        from matplotlib import pyplot as plt

        from typing import Tuple, Callable, Self

        key = PRNGKey(0)

        hyps = Hyperparameters(1, 1, 1e-1)

        x = np.linspace(0, 10, 100)[:, None]
        u = x[::45, :]
        u = x.copy()
        y = 2 * np.sin(3 * x + 0.2) + hyps.sn2 * normal(key, shape=x.shape)

        force_response_kernel_type = 'squared_exponential'
        time = x
        force_response = y
        force_response_solver_type = 'metropolis_hastings'
        force_response_n_iter = 0
        force_response_nll_method = ['cholesky','FITC_18_134'][1]
        force_response_U_induced_method = 'even'
        M_one_in = 1

        force_response_model = GPModel(force_response_kernel_type,
                                       time,
                                       force_response,
                                       solver_type=force_response_solver_type,
                                       n_iter=force_response_n_iter,
                                       gp_algo=force_response_nll_method,
                                       U_induced_method=force_response_U_induced_method,
                                       M_one_in=M_one_in)

        print(f"Test NLL: {force_response_model.compute_nll(force_response_model.hyperparameters_obj)}")
