

class Hyperparameters:
    def __init__(self, kernel_type):
        self.kernel_type = kernel_type
        self.initial_hyperparameters = {}
        self.hyperparameter_bounds = {}
        self._set_hyperparameters()

    def _set_hyperparameters(self):
        if self.kernel_type == 'squared_exponential':
            self.initial_hyperparameters = {
                'kernel_type': 'squared_exponential',
                'sigma': 0.1,
                'l': 0.01,
                'p': 1E-3,
                'mean_func_c': 1.0,
                'noise_level': 0.001
            }

            self.hyperparameter_bounds = {
                'kernel_type': 'squared_exponential',
                'sigma': (0.001,10),
                'l': (0.001,10),
                'p': (0.0001,1),
                'mean_func_c': (-1000,1000),
                'noise_level': (0.0001,1)
            }
        elif self.kernel_type == 'periodic':
            self.initial_hyperparameters = {
                'kernel_type': 'periodic',
                'sigma': 1,
                'l': 0.1,
                'p': 0.08,
                'mean_func_c': 1.0,
                'noise_level': 0.1
            }

            self.hyperparameter_bounds = {
                'kernel_type': 'periodic',
                'sigma': (0.001, 100),
                'l': (0.001, 10),
                'p': (0.0001, 1),
                'mean_func_c': (-1000,1000),
                'noise_level': (0.0001, 0.25)
            }
        elif self.kernel_type == 'p_se_composite':
            self.initial_hyperparameters = {
                'kernel_type': 'p_se_composite',
                'periodic_params': [
                {'sigma': 0.1, 'l': 0.01, 'p': 1E-3},
                {'sigma': 0.1, 'l': 0.02, 'p': 1E-3}
                ],
                'se_params': {'sigma': 0.1, 'l': 0.01},
                'mean_func_c': 1.0,
                'noise_level': 0.001
            }

            self.hyperparameter_bounds = {
                'kernel_type': 'p_se_composite',
                'periodic_param_bounds': [
                {'sigma': (0.0001,100), 'l': (0.0001,100), 'p': (0.0001,10)},
                {'sigma': (0.0001,100), 'l': (0.0001,100), 'p': (0.0001,10)}
                ],
                'se_param_bounds': {'sigma': (0.0001,100), 'l': (0.0001,100)},
                'mean_func_c': (-1000,1000),
                'noise_level': (0.0001,1)
            }
        elif self.kernel_type == 'white_noise':
            self.initial_hyperparameters = {
                'kernel_type': 'white_noise',
                'sigma': 1,
                'mean_func_c': 1.0,
                'noise_level': 0.1
            }

            self.hyperparameter_bounds = {
                'kernel_type': 'white_noise',
                'sigma': (0.001, 100),
                'mean_func_c': (-1000,1000),
                'noise_level': (0.0001, 0.25)
            }
        elif self.kernel_type == 'wn_se_composite':
            self.initial_hyperparameters = {
                'kernel_type': 'wn_se_composite',
                'wn_params': {'sigma': 0.1},
                'se_params': {'sigma': 0.1, 'l': 0.01},
                'mean_func_c': 1.0,
                'noise_level': 0.001
            }

            self.hyperparameter_bounds = {
                'kernel_type': 'wn_se_composite',
                'periodic_param_bounds': {'sigma': (0.001,10)},
                'se_param_bounds': {'sigma': (0.001,10), 'l': (0.001,10)},
                'mean_func_c': (-1000,1000),
                'noise_level': (0.0001,1)
            }
        else:
            raise ValueError("Invalid kernel_type")
