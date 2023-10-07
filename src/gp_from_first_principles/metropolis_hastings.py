import numpy as np

from utils import debug_print
from map_number import map_number


def metropolis_hastings_solve(initial_hyperparameters_array, bounds_array,
                              compute_nll, n_iter=10, sample_length = 0):
    best_hyperparameters = initial_hyperparameters_array.copy()
    hyperparameters = initial_hyperparameters_array.copy()
    best_nll = compute_nll(initial_hyperparameters_array)
    nll = compute_nll(initial_hyperparameters_array)

    for j in range(n_iter):
        debug_print(f"Iteration: {j+1}/{n_iter} ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        for i, (lower, upper) in enumerate(bounds_array):
            exponent = np.random.normal(0,1)
            modifier = np.exp(exponent)
            hyperparameters_prime = hyperparameters.copy()
            hyperparameters_prime[i] = np.clip(hyperparameters_prime[i] * modifier, lower, upper)
            nll_prime = compute_nll(hyperparameters_prime)
            if nll_prime < best_nll:
                best_nll = nll_prime
                best_hyperparameters = hyperparameters_prime
            A = map_number((nll_prime-nll),0,1000,1.0,0.0)
            #A = min(0, map_number_out)
            debug_print(f"nll_prime = :{nll_prime}")
            debug_print(f"nll = :{nll}")
            debug_print(A)
            update_nll = np.random.binomial(1,A)
            if update_nll:
                nll = nll_prime
                hyperparameters = hyperparameters_prime
        if j > 25 and best_nll < -sample_length:
            debug_print(f"Stopping after {j} iterations due to nll sufficiently low: {best_nll}")
            break
    return best_hyperparameters
