from src.utils.utils import debug_print
class MetropolisHastings:
    def __init__(self,initial_hyperparameters_array, bounds_array, compute_nll):
        self.initial_hyperparameters_array = initial_hyperparameters_array
        self.bounds_array = bounds_array
        self.compute_nll = compute_nll

    def solve(self, n_iter = '10'):
        return
