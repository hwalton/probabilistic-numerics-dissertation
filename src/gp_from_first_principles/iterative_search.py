import numpy as np

from src.utils.utils import debug_print

class IterativeSearch:
    def __init__(self,initial_hyperparameters_array, bounds_array, compute_nll):
        self.initial_hyperparameters_array = initial_hyperparameters_array
        self.bounds_array = bounds_array
        self.compute_nll = compute_nll

    def solve(self, n_iter = '10'):
        best_hyperparameters = self.initial_hyperparameters_array
        best_nll = self.compute_nll(self.initial_hyperparameters_array)
        for j in range(n_iter):
            debug_print(f"Search Iteration: {j+1}/{n_iter}")
            for i, (lower, upper) in enumerate(self.bounds_array):
                for modifier in [0.75, 2]:
                    new_hyperparameters = best_hyperparameters.copy()
                    new_hyperparameters[i] = np.clip(new_hyperparameters[i] * modifier, lower, upper)
                    new_nll = self.compute_nll(new_hyperparameters)
                    if new_nll < best_nll:
                        best_nll = new_nll
                        best_hyperparameters = new_hyperparameters
        return best_hyperparameters
