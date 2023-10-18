import numpy as np

class Hyperparameters:
    def __init__(self, kernel_type):
        self.kernel_type = kernel_type
        self._initial_hyperparameters = {}
        self._hyperparameter_bounds = {}
        self._set_initial_hyperparameters()
        self.template = self._initial_hyperparameters
        self._hyperparameters_dict = self._initial_hyperparameters
        self._hyperparameter_bounds_dict = self._hyperparameter_bounds

    def _set_initial_hyperparameters(self):
        if self.kernel_type == 'squared_exponential':
            self._initial_hyperparameters = {
                'kernel_type': 'squared_exponential',
                'sigma': 0.1,
                'l': 1.,
                'noise_level': 0.1
            }

            self._hyperparameter_bounds = {
                'kernel_type': 'squared_exponential',
                'sigma': (0.001,10),
                'l': (0.01,10),
                'noise_level': (0.1,1)
            }
        elif self.kernel_type == 'periodic':
            self._initial_hyperparameters = {
                'kernel_type': 'periodic',
                'sigma': 9.7834,
                'l': 0.019533,
                'p': 0.032121,
                'noise_level': 1
            }

            self._hyperparameter_bounds = {
                'kernel_type': 'periodic',
                'sigma': (0.001, 1000),
                'l': (0.01, 100),
                'p': (0.0001, 100),
                'noise_level': (0.01, 25)
            }
        elif self.kernel_type == 'p_se_composite':
            self._initial_hyperparameters = {
                'kernel_type': 'p_se_composite',
                'periodic_params': [
                {'sigma': 0.1, 'l': 1., 'p': 1E-3},
                {'sigma': 0.1, 'l': 1., 'p': 1E-3}
                ],
                'se_params': {'sigma': 0.1, 'l': 0.01},
                'noise_level': 1.
            }

            self._hyperparameter_bounds = {
                'kernel_type': 'p_se_composite',
                'periodic_param_bounds': [
                {'sigma': (0.0001,100), 'l': (0.01,100), 'p': (0.0001,10)},
                {'sigma': (0.0001,100), 'l': (0.01,100), 'p': (0.0001,10)}
                ],
                'se_param_bounds': {'sigma': (0.0001,100), 'l': (0.0001,100)},
                'noise_level': (0.1,1.)
            }
        elif self.kernel_type == 'white_noise':
            self._initial_hyperparameters = {
                'kernel_type': 'white_noise',
                'sigma': 0.0791306,
                'noise_level': 0.641445
            }

            self._hyperparameter_bounds = {
                'kernel_type': 'white_noise',
                'sigma': (0.001, 100),
                'noise_level': (0.1, 1)
            }
        elif self.kernel_type == 'wn_se_composite':
            self._initial_hyperparameters = {
                'kernel_type': 'wn_se_composite',
                'wn_params': {'sigma': 0.1},
                'se_params': {'sigma': 0.1, 'l': 1.},
                'noise_level': 0.1
            }

            self._hyperparameter_bounds = {
                'kernel_type': 'wn_se_composite',
                'periodic_param_bounds': {'sigma': (0.001,10)},
                'se_param_bounds': {'sigma': (0.001,10), 'l': (0.01,10)},
                'noise_level': (0.1,1)
            }
        else:
            raise ValueError("Invalid kernel_type")

    def flatten_params(self, params):
        flat_params = []
        for key, value in params.items():
            if isinstance(value, str):
                continue
            if isinstance(value, list):
                for item in value:
                    flat_params.extend(self.flatten_params(item))
            elif isinstance(value, dict):
                flat_params.extend(self.flatten_params(value))
            else:
                flat_params.append(value)
        return np.array(flat_params)

    # def _flatten_params(self):
    #     flat_params = []
    #     for key, value in self._hyperparameters_dict.items():
    #         if isinstance(value, str):
    #             continue
    #         if isinstance(value, list):
    #             for item in value:
    #                 flat_params.extend(self.flatten_params(item))
    #         elif isinstance(value, dict):
    #             flat_params.extend(self.flatten_params(value))
    #         else:
    #             flat_params.append(value)
    #     return np.array(flat_params)

    def _reconstruct_params_implementation(self, flat_params, template):
        reconstructed_params = {}
        index = 0
        for key, value in template.items():
            if isinstance(value, str):
                reconstructed_params[key] = value
                continue
            if isinstance(value, list):
                reconstructed_params[key] = []
                for item in value:
                    reconstructed_item, item_length = self._reconstruct_params_implementation(
                        flat_params[index:], item)
                    index += item_length
                    reconstructed_params[key].append(reconstructed_item)
            elif isinstance(value, dict):
                reconstructed_params[
                    key], item_length = self._reconstruct_params_implementation(
                    flat_params[index:], value)
                index += item_length
            else:
                reconstructed_params[key] = flat_params[index]
                index += 1
        return reconstructed_params, index

    def reconstruct_params(self, flat_params):
        reconstructed_params, index = self._reconstruct_params_implementation(
            flat_params, self.template)
        return reconstructed_params


    def dict(self, attribute = 'current'):
        if attribute == 'current':
            return self._hyperparameters_dict
        if attribute == 'bounds':
            return self._hyperparameter_bounds
        if attribute == 'initial':
            return self._initial_hyperparameters
        else:
            raise ValueError("Invalid attribute")

    def array(self, attribute = 'current'):
        if attribute == 'current':
            out = self.flatten_params(self._hyperparameters_dict)
            return out
        if attribute == 'bounds':
            out = self.flatten_params(self._hyperparameter_bounds)
            return out
        if attribute == 'initial':
            out = self.flatten_params(self._initial_hyperparameters)
            return out
        else:
            raise ValueError("Invalid attribute")

    def update(self, hyperparameters):
        if type(hyperparameters) == dict:
            self._hyperparameters_dict = hyperparameters
        if type(hyperparameters) == np.ndarray:
            self._hyperparameters_dict = self.reconstruct_params(hyperparameters)
        if type(hyperparameters) == Hyperparameters:
            self._hyperparameters_dict = hyperparameters.dict()
        return
