import numpy as np

from utils import debug_print


def iterative_search_solve(initial_hyperparameters_array, bounds_array, compute_nll, n_iter = 10):
    best_hyperparameters = initial_hyperparameters_array
    best_nll = compute_nll(initial_hyperparameters_array)
    for j in range(n_iter):
        debug_print(f"Search Iteration: {j+1}/{n_iter}")
        for i, (lower, upper) in enumerate(bounds_array):
            for modifier in [0.75, 2]:
                new_hyperparameters = best_hyperparameters.copy()
                new_hyperparameters[i] = np.clip(new_hyperparameters[i] * modifier, lower, upper)
                new_nll = compute_nll(new_hyperparameters)
                if new_nll < best_nll:
                    best_nll = new_nll
                    best_hyperparameters = new_hyperparameters
    return best_hyperparameters
