import math

import numpy as np

from utils import debug_print
from map_number import map_number


def metropolis_hastings_solve(initial_hyperparameters_array, bounds_array,
                              compute_nll, n_iter=10, sample_length=0, restarts=5):
    # Initialize best hyperparameters and nll over all restarts
    overall_best_hyperparameters = initial_hyperparameters_array.copy()
    overall_best_nll = float('inf')

    # Loop for the number of restarts
    for r in range(restarts):
        best_hyperparameters = initial_hyperparameters_array.copy()
        hyperparameters = initial_hyperparameters_array.copy()
        best_nll = compute_nll(initial_hyperparameters_array)['nll']
        debug_print(f"Restart: {r+1}/{restarts} ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        debug_print(f"initial best_nll = {best_nll}")
        nll = compute_nll(initial_hyperparameters_array)['nll']
        iter_since_update = 0
        for j in range(n_iter):
            debug_print(f"Iteration: {j+1}/{n_iter} ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            if j >= 10 and best_nll < sample_length:
                debug_print(
                    f"Stopping after {j} iterations due to nll sufficiently low: {best_nll}")
                break
            else:
                for i, (lower, upper) in enumerate(bounds_array):
                    var = 3.5 * np.exp(-0.25 * j) + 2.5
                    exponent = np.random.normal(0, var)
                    modifier = np.exp(exponent)
                    hyperparameters_prime = hyperparameters.copy()
                    hyperparameters_prime[i] = np.clip(
                        hyperparameters_prime[i] * modifier, lower, upper)
                    nll_prime = compute_nll(hyperparameters_prime)['nll']

                    if nll_prime < best_nll:
                        best_nll = nll_prime
                        best_hyperparameters = hyperparameters_prime.copy()

                    A = map_number((nll_prime-nll), 0., 1000., 1.0, 0.0)
                    if math.isinf(nll) or math.isinf(nll_prime):
                        A = 0.0
                    debug_print(f"nll_prime: {nll_prime}")
                    debug_print(f"nll: {nll}")
                    debug_print((f"nll_prime - nll: {nll_prime - nll}"))
                    debug_print(f"A: {A}")
                    update = np.random.binomial(1, A)
                    if update:
                        nll = nll_prime
                        hyperparameters = hyperparameters_prime.copy()

        # Update the overall best hyperparameters and nll if necessary
        if best_nll < overall_best_nll:
            overall_best_nll = best_nll
            overall_best_hyperparameters = best_hyperparameters.copy()

    return overall_best_hyperparameters, overall_best_nll
