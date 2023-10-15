import numpy as np

from utils import debug_print
from map_number import map_number


def metropolis_hastings_solve(initial_hyperparameters_array, bounds_array,
                              compute_nll, n_iter=10, sample_length = 0):
    best_hyperparameters = initial_hyperparameters_array.copy()
    hyperparameters = initial_hyperparameters_array.copy()
    best_nll = compute_nll(initial_hyperparameters_array)['nll']
    nll = compute_nll(initial_hyperparameters_array)['nll']
    iter_since_update = 0
    for j in range(n_iter):

        debug_print(f"Iteration: {j+1}/{n_iter} ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        if j > 10 and best_nll < -sample_length:
        #if j > 2:
            debug_print(
                f"Stopping after {j} iterations due to nll sufficiently low: {best_nll}")
            break
        else:
            for i, (lower, upper) in enumerate(bounds_array):

                exponent = np.random.normal(0,3)
                modifier = np.exp(exponent)
                hyperparameters_prime = hyperparameters.copy()
                hyperparameters_prime[i] = np.clip(hyperparameters_prime[i] * modifier, lower, upper)
                nll_prime = compute_nll(hyperparameters_prime)['nll']
                if nll_prime < best_nll:
                    best_nll = nll_prime
                    best_hyperparameters = hyperparameters_prime.copy()
                    # debug_print(f"best_hyperparameters set as {best_hyperparameters} with nll {compute_nll(best_hyperparameters)}")

                if np.isinf(nll_prime-nll):
                    var34 = 0
                else:
                    var34 = 1


                A = map_number((nll_prime-nll),0,1000,1.0,0.0) * var34
                debug_print(f"nll_prime: {nll_prime}")
                debug_print(f"nll: {nll}")
                debug_print((f"nll_prime - nll: {nll_prime - nll}"))
                #A = min(0, map_number_out)
                #debug_print(f"nll_prime = :{nll_prime}")
                #debug_print(f"nll = :{nll}")
                debug_print(f"A: {A}")
                update = np.random.binomial(1,A)
                if update:
                    nll = nll_prime
                    hyperparameters = hyperparameters_prime.copy()


    # debug_print(f"metropolis_hastings_solve returning {best_hyperparameters} with nll {compute_nll(best_hyperparameters)}")
    return best_hyperparameters