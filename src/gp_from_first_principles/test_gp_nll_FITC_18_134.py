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

class TestGP_NLL_FITC_18_134(unittest.TestCase):
    def test_compute(self):
        pass

    def test_K_XX_FITC(self):
        hyperparameters_obj = Hyperparameters('p_se_composite')
        kernel = GaussianProcessKernel(hyperparameters_obj)
        # dummy_X
        # dummy_Y
        # dummy_U

        # gp_nll_fitc_18_134 = GP_NLL_FITC_18_134()
        pass