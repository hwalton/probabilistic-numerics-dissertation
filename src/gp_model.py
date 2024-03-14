import numpy as np
import scipy
#import freelunch
from numpy import linalg as npla
from gp_nll_FITC import GP_NLL_FITC
from hyperparameters import Hyperparameters
from gp_kernel import GaussianProcessKernel
from iterative_search import iterative_search_solve
from metropolis_hastings import metropolis_hastings_solve
from adam import adam_optimize
from utils import debug_print
from sklearn.cluster import KMeans
from scipy.optimize import minimize


class GPModel:
    def __init__(self, kernel_type, X, y, solver_type = 'iterative_search', n_iter=10, gp_algo ='cholesky', U_induced_method = 'even', M_one_in = 1):
        self.hyperparameters_obj = Hyperparameters(kernel_type)
        self.initial_hyperparameters = self.hyperparameters_obj._initial_hyperparameters.copy()
        self.hyperparameter_bounds = self.hyperparameters_obj._hyperparameter_bounds.copy()
        self.X = X
        self.y = y
        self.n_iter = n_iter
        self.template = self.hyperparameters_obj._initial_hyperparameters.copy()
        self.solver_type = solver_type
        self.gp_kernel = GaussianProcessKernel(self.hyperparameters_obj)
        self.U = self.U_induced(M_one_in, method = U_induced_method)
        self.gp_algo = gp_algo
        self.y_mean = np.mean(y)
        self.gp_nll_algo_obj = GP_NLL_FITC(self.X, self.y, self.y_mean, self.U,
                                           self.gp_kernel,
                                           self.hyperparameters_obj)
        self.M_one_in = M_one_in

    def fit_model(self):
        self.gp_kernel.set_params(self.hyperparameters_obj)
        optimal_hyperparameters, nll = self.get_optimal_hyperparameters()
        self.hyperparameters_obj.update(optimal_hyperparameters)
        debug_print(optimal_hyperparameters)
        debug_print(self.hyperparameters_obj.array())
        #nll = self.compute_nll(self.hyperparameters_obj)['nll']
        #debug_print(nll)
        return(nll)

    def solve(self, solver_type):
        if solver_type == 'iterative_search':
            return iterative_search_solve(self.hyperparameters_obj.array(), self.hyperparameters_obj.array(attribute='bounds'), self.compute_nll, n_iter=self.n_iter)
        elif solver_type == 'metropolis_hastings':
            #debug = self.hyperparameters_obj.array()
            return metropolis_hastings_solve(self.hyperparameters_obj.array(), self.hyperparameters_obj.array(attribute='bounds'), self.compute_nll, n_iter=self.n_iter, sample_length = len(self.X))
        elif solver_type == 'free_lunch':
            self.solver = freelunch.DE(self.compute_nll, bounds = self.hyperparameters_obj.array(attribute='bounds'))
            return self.solver()
        elif solver_type == 'adam':
            debug_print(f"nll before adam = {self.compute_nll(self.hyperparameters_obj.dict())}")
            optimal_hyperparameters = adam_optimize(self.compute_nll,self.X, self.y, self.U_X, self.hyperparameters_obj.array(), self.gp_kernel, self.hyperparameters_obj.reconstruct_params)
            #self.hyperparameters_obj.update(optimal_hyperparameters)
            nll = self.compute_nll(optimal_hyperparameters)
            debug_print(f"hyperparameters after adam {optimal_hyperparameters}")
            debug_print(f"nll after adam = {nll}")
            return optimal_hyperparameters, nll
        else:
            assert False, "invalid solver_type"

    def get_optimal_hyperparameters(self):
        debug_print("solving")
        optimal_hyperparameters, nll = self.solve(self.solver_type)
        optimal_hyperparameters = optimal_hyperparameters.T
        debug_print("solved")
        return self.hyperparameters_obj.reconstruct_params(optimal_hyperparameters), nll


    def U_induced(self, M_one_in = 1, method='k_means'):
        M = len(self.X) // M_one_in

        if method == 'k_means':
            # Perform k-means clustering on self.X
            kmeans = KMeans(n_clusters=M, n_init=3).fit(self.X)

            # Extract the cluster centers for self.X
            self.U_X = kmeans.cluster_centers_

            # Determine the corresponding self.y values for the inducing points
            self.U_y = np.zeros((M, self.y.shape[1] if self.y.ndim > 1 else 1))
            for i in range(M):
                # Find the indices of the points assigned to cluster i
                idx = np.where(kmeans.labels_ == i)[0]

                # Assign the mean of the self.y values of the points in cluster i
                # to the corresponding inducing output
                self.U_y[i] = np.mean(self.y[idx], axis=0)
        elif method == 'even':
            # Determine the number of inducing points M
            M = len(self.X) // M_one_in

            # Compute the step size for sampling from self.X
            step_size = len(self.X) // M

            # Initialize arrays for self.U_X and self.U_y
            self.U_X = np.zeros((M, self.X.shape[1]))
            self.U_y = np.zeros((M, self.y.shape[1] if self.y.ndim > 1 else 1))

            # Sample evenly spaced points from self.X and compute self.U_y
            for i in range(M):
                idx = i * step_size  # Compute the index for sampling
                self.U_X[i] = self.X[idx]  # Sample self.X
                self.U_y[i] = self.y[idx]  # Compute corresponding self.U_y
        else:
            raise ValueError("Invalid inducing method")
        #self.X = U_X
        #self.y = U_y
        return self.U_X


    def predict(self, X_star, method = 'FITC'):
        if method == 'FITC':
            self.mu, self.stdv = self.gp_nll_algo_obj.predict(X_star)
        elif method == 'cholesky':
            self.K_X_X = self.gp_kernel.compute_kernel(self.X,
                                                  self.X) + np.array(
                self.hyperparameters_obj.dict()[
                    'noise_level'] * np.eye(len(self.X)))[:, :, None]
            K_star_X = self.gp_kernel.compute_kernel(self.X, X_star)
            K_star_star = self.gp_kernel.compute_kernel(X_star, X_star)
            L = np.zeros_like(self.K_X_X)
            self.mu = np.zeros((K_star_X.shape[1], self.K_X_X.shape[2]))
            self.s2 = np.zeros((K_star_X.shape[1], self.K_X_X.shape[2]))
            for i in range(self.K_X_X.shape[2]):
                L[:, :, i] = npla.cholesky( self.K_X_X[:, :, i] + 1e-10 * np.eye(self.K_X_X.shape[0]))
                Lk = np.squeeze(npla.solve(L[:, :, i], K_star_X[:, :, i]))
                self.mu[:, i] = self.y_mean + np.dot(Lk.T,npla.solve(L[:, :, i],self.y - self.y_mean).flatten())
                self.s2[:, i] = np.diag(K_star_star[:, :, i]) - np.sum(
                    Lk ** 2, axis=0)
                self.stdv = np.sqrt(self.s2)
        return self.mu, self.stdv

    def K_SE_xi(self, xi, xi_prime):
        if xi.ndim == 1:
            xi = xi[:, None]
        if xi_prime.ndim == 1:
            xi_prime = xi_prime[:, None]

        delta_X = xi[:, None, :] - xi_prime[None, :, :]
        out =  np.exp(-0.5 * (delta_X) ** 2 / (self.hyperparameters_obj.dict()['l'] ** 2))
        out = np.squeeze(out)
        return out

    def spectrum_xi(self, xi_input, h):
        return np.exp(h).dot(self.K_SE_xi(xi_input, self.xi))

    def map_wrapper(self, n, xi_n):
        def neg_map(h):
            eq_15 = -np.abs(self.y_xi[n]) ** 2 / (self.spectrum_xi(xi_n, h) + self.hyperparameters_obj.dict()['noise_level']) + \
                    - np.log(np.pi * (self.spectrum_xi(xi_n, h) + self.hyperparameters_obj.dict()['noise_level']))
            debug = np.squeeze(self.xi)
            det_eq_13 = np.linalg.det(self.K_SE_xi(self.xi,self.xi))
            out = eq_15 + det_eq_13
            return -out

        return neg_map

    def doJ_doa(self, a):
        dol_doa = np.exp(a) * ((np.abs(self.y_xi)**2 - self.spectrum_xi(self.xi, a) + self.hyperparameters_obj.dict()['noise_level']) / (self.spectrum_xi(self.xi, a) + self.hyperparameters_obj.dict()['noise_level'])**2) @ self.K_SE_xi(self.xi, self.xi)
        dop_doa = -np.exp(a) * self.K_SE_xi(self.xi, self.xi) @ np.exp(a)
        out = dol_doa + dop_doa
        return out
    def grad_descent_se_fourier(self, a):
        while np.linalg.norm(self.doJ_doa(a)) > 3E-1:
            a +=  10E-1 * self.doJ_doa(a)
            #debug_print(f"norm = {np.linalg.norm(self.doJ_doa(a))}")
        return a

    def predict_fourier(self, X_star, method = 'GP'):
        if method == 'GP':
            hyp_l, w = self.GP_Mu(X_star)

            self.GP_STDV(hyp_l, w)

            return self.mu_fourier, self.stdv_fourier


        elif method == 'GP_2':
            hyp_l, w = self.GP_Mu(X_star)


            self.GP_STDV_2(X_star)

            return self.mu_fourier, self.stdv_fourier

        elif method == 'GP_3':
            hyp_l, w = self.GP_Mu(X_star)

            self.GP_STDV_3(X_star)

            return self.mu_fourier, self.stdv_fourier

        elif method == 'GP_4':
            hyp_l, w = self.GP_Mu(X_star)

            self.GP_STDV_4(X_star)

            return self.mu_fourier, self.stdv_fourier

        elif method == 'GP_5':
            hyp_l, w = self.GP_Mu(X_star)

            self.GP_STDV_5(X_star)

            return self.mu_fourier, self.stdv_fourier

        elif method == 'GP_6':
            hyp_l, w = self.GP_Mu(X_star)

            self.GP_STDV_6(X_star)

            return self.mu_fourier, self.stdv_fourier


        elif method == 'DFT':
            self.DFT_Mu()
            self.stdv_fourier = np.zeros_like(self.mu_fourier)
            return self.mu_fourier, self.stdv_fourier
        elif method == 'set':
            self.Set_Mu()
            self.stdv_fourier = np.zeros_like(self.mu_fourier)
            return self.mu_fourier, self.stdv_fourier
        else:
            assert 0, "Not yet implemented"



    def GP_STDV(self, hyp_l, w):
        window = np.hanning(len(self.xi))
        self.y_xi = np.fft.fft(np.squeeze(self.y) * window)
        self.stdv_fourier = np.zeros(len(self.xi), dtype=complex)
        for n, xi_n in enumerate(self.xi):
            a = np.ones(len(self.xi))
            h = self.grad_descent_se_fourier(a)

            debug_print(f"n = {n}")

            for k, w_k in enumerate(w):
                debug_21 = np.exp(h - (self.xi[n] - np.squeeze(self.xi)) ** 2 / (2 * hyp_l ** 2))
                debug_22 = np.exp(-1j * self.xi[n] * np.squeeze(self.X)[k])
                self.stdv_fourier[n] += np.sum(w_k * debug_21 * debug_22)
        debug_print("stdv_fourier calculated")

    def GP_STDV_2(self, X_star):
        X_star = np.asarray(X_star)
        window = np.hanning(len(self.xi))
        self.stdv_fourier = np.zeros(len(self.xi), dtype=complex)
        A = np.linalg.inv(np.squeeze(self.K_X_X) + self.hyperparameters_obj.dict()['noise_level'] * np.eye(len(self.X)))
        for n, xi_n in enumerate(self.xi):
            debug_print(f"n = {n}")
            for j in range(len(self.X)):
                for k in range(len(self.X)):
                    innn = A[j][k] * np.exp(-1j * xi_n * (np.squeeze(self.X)[j] + np.squeeze(self.X)[k]))
                    self.stdv_fourier[n] -= innn
            self.stdv_fourier[n] *= self.gp_kernel.compute_kernel_fourier_SE_squared(xi_n)
            self.stdv_fourier[n] += self.gp_kernel.compute_kernel_SE_fourier(xi_n)

            #self.stdv_fourier[0] = 0.0

    import numpy as np

    def GP_STDV_3(self, X_star):
        X_star = np.asarray(X_star)
        xi = np.asarray(self.xi)

        # Extract relevant variables and parameters
        noise_level = self.hyperparameters_obj.dict()['noise_level']
        X = np.squeeze(self.X)  # Assuming self.X is 2D and we need a 1D array

        # Cholesky decomposition for A matrix
        L = np.linalg.cholesky(np.squeeze(self.K_X_X) + noise_level * np.eye(len(X)))
        A_inv = np.linalg.solve(L, np.linalg.solve(L.T, np.eye(len(X))))

        # Initialize the Fourier standard deviation array
        stdv_fourier = np.zeros_like(xi, dtype=np.complex128)

        # Use a 2D meshgrid for pairwise differences in X
        X_mesh1, X_mesh2 = np.meshgrid(X, X, indexing='ij')
        pairwise_sum = X_mesh1 + X_mesh2

        # Iterate over frequencies to manage memory usage
        for i, freq in enumerate(xi):
            # Compute exponential factor for the current frequency using broadcasting
            exp_factor = np.exp(-1j * freq * pairwise_sum)

            # Element-wise multiplication with A_inv and sum
            stdv_contribution = A_inv * exp_factor
            stdv_fourier[i] = -np.sum(stdv_contribution) * self.gp_kernel.compute_kernel_fourier_SE_squared(freq)
            stdv_fourier[i] += self.gp_kernel.compute_kernel_SE_fourier(freq)
        self.stdv_fourier = np.squeeze(stdv_fourier)

    def GP_STDV_4(self, X_star):
        self.var_fourier = np.zeros(len(self.xi), dtype=complex)
        A = np.linalg.inv(np.squeeze(self.K_X_X) + self.hyperparameters_obj.dict()['noise_level'] * np.eye(len(self.X)))
        for n, xi_n in enumerate(self.xi):
            debug_print(f"n = {n}")
            for j in range(len(self.X)):
                for k in range(len(self.X)):
                    innn = A[j][k] * np.exp(-1j * xi_n * (np.squeeze(self.X)[j] - np.squeeze(self.X)[k]))
                    self.var_fourier[n] += innn    # CHECK!!!: SHOULD THIS BE -= or +=? Maths says -=, but positive and real values are expected, which occur with +=????
            self.var_fourier[n] *= self.gp_kernel.compute_kernel_SE_fourier(xi_n) * np.conj(self.gp_kernel.compute_kernel_SE_fourier(- xi_n))
            debug_k = self.gp_kernel.compute_kernel(-xi_n, 0) / 2 * np.pi
            self.var_fourier[n] += self.gp_kernel.compute_kernel(-xi_n, 0) / 2 * np.pi
            self.stdv_fourier = self.var_fourier



    def GP_STDV_5(self, X_star):
        '''
        This is a vectorised version of GP_STDV_4, but the += in the line
        self.var_fourier[n] += innn is replaced with -=. A mathematical reason for this has not been established, but it makes the output positive and real, as expected.
        '''
        A = np.linalg.inv(np.squeeze(self.K_X_X) + self.hyperparameters_obj.dict()['noise_level'] * np.eye(len(self.X)))
        X_squeezed = np.squeeze(self.X)
        self.stdv_fourier = np.zeros(len(self.xi), dtype=complex)

        for n, xi_n in enumerate(self.xi):
            debug = (X_squeezed[:, None] - X_squeezed)
            exponent_matrix = -1j * xi_n * debug
            contributions = A * np.exp(exponent_matrix)
            self.stdv_fourier[n] = np.sum(contributions) # CHECK!!!: SHOULD THIS BE -? Maths says -, but positive and real values are expected, which occur with +????

            kernel_fourier_sq = self.gp_kernel.compute_kernel_SE_fourier(xi_n) ** 2
            kernel_fourier_neg = self.gp_kernel.compute_kernel(-xi_n, 0) / 2 * np.pi

            self.stdv_fourier[n] *= kernel_fourier_sq
            self.stdv_fourier[n] += kernel_fourier_neg


    def Set_Mu(self):
        self.X = np.asarray(self.X)
        N = len(self.X)
        delta_t = (self.X[-1] - self.X[0]) / (N - 1)
        fs = 1 / delta_t
        delta_f = fs / N
        self.xi = np.linspace(-(fs/2 - delta_f), fs/2, N)
        # if N % 2 == 0:
        #     # Even number of samples: include Nyquist frequency
        #     self.xi = np.linspace(0, fs / 2, N // 2 + 1)
        # else:
        #     # Odd number of samples: exclude Nyquist frequency
        #     self.xi = np.linspace(0, fs / 2, (N - 1) // 2 + 1)
        self.xi = 2 * np.pi * self.xi  # convert from Hz to rad/s
        m = 1  # Mass
        c = 0.5  # Damping coefficient
        k = 100  # Stiffness
        self.mu_fourier = np.squeeze(np.exp(-1j * np.pi) / ((k - m * self.xi ** 2) + 1j * c * self.xi))

    def GP_Mu(self, X_star):
        hyp_l = self.hyperparameters_obj.dict()['l']
        X_star = np.asarray(X_star)
        N = len(X_star)
        delta_t = (X_star[-1] - X_star[0]) / (N - 1)
        fs = 1 / delta_t
        delta_f = fs / N
        self.xi = np.linspace(-(fs/2 - delta_f), fs/2, N)
        self.xi = 2 * np.pi * self.xi  # convert from Hz to rad/s
        self.K_xi = np.squeeze(self.gp_kernel.compute_kernel_SE_fourier(self.xi))
        # Extract the noise level and ensure it is a positive value
        noise_level = self.hyperparameters_obj.dict()['noise_level']
        assert noise_level > 0, "Noise level must be positive for Cholesky decomposition."
        # Add the noise level to the diagonal of K_X_X
        K = np.squeeze(self.K_X_X) + noise_level * np.eye(np.shape(self.K_X_X)[0])
        # Perform Cholesky decomposition
        L = np.linalg.cholesky(K)
        # Solve Lz = y for z (intermediate variable)
        y_squeezed = np.squeeze(self.y)
        z = np.linalg.solve(L, y_squeezed)
        # Solve L.T * w = z for w
        w = np.linalg.solve(L.T, z)
        self.mu_fourier = np.zeros(len(self.xi), dtype=complex)
        for i in range(len(self.xi)):
            exp = np.squeeze(np.exp(-1j * self.xi[i] * np.squeeze(self.X)))
            self.mu_fourier[i] = self.K_xi[i] * (w.dot(exp))

        #self.mu_fourier *= np.exp(-1j * (-np.pi + np.angle(self.mu_fourier[-1])))
        debug_print("mu_fourier calculated")
        return hyp_l, w

    def DFT_Mu(self):
        self.X = np.asarray(self.X)
        N = len(self.X)
        delta_t = (self.X[-1] - self.X[0]) / (N - 1)
        fs = 1 / delta_t
        delta_f = fs / N
        self.xi = np.linspace(-(fs/2 - delta_f), fs/2, N)
        # if N % 2 == 0:
        #     # Even number of samples: include Nyquist frequency
        #     self.xi = np.linspace(0, fs / 2, N // 2 + 1)
        # else:
        #     # Odd number of samples: exclude Nyquist frequency
        #     self.xi = np.linspace(0, fs / 2, (N - 1) // 2 + 1)
        self.xi = 2 * np.pi * self.xi  # convert from Hz to rad/s
        self.mu_fourier = np.fft.fft(np.squeeze(self.y) * np.hanning(len(np.squeeze(self.y)))) / (len(np.squeeze(self.y)) / 2) * 2 * np.pi
        #debug_a = self.mu_fourier
        #debug_b = np.fft.fftshift(self.mu_fourier)
        debug_c = np.concatenate((self.mu_fourier[(N//2+1):], self.mu_fourier[:(N//2+1)]))
        #self.mu_fourier *= np.exp(-1j * np.angle(self.mu_fourier[-1]))
        self.mu_fourier = debug_c


    def is_positive_definite(self, K):
        if K.ndim > 3:
            raise ValueError("Input must be at most 3D.")
        if K.ndim == 2:
            K = K[:, :, np.newaxis]
        if K.shape[0] != K.shape[1]:
            raise ValueError("The first two dimensions must be equal.")
        for i in range(K.shape[2]):
            if not np.all(np.linalg.eigvals(K[:, :, i]) > 0):
                return False
        return True





    def compute_nll(self, hyperparameters):
        self.update_hyperparameters_and_debug(hyperparameters)
        self.reshape_X_and_y()



        if self.gp_algo == 'cholesky':
            #self.hyperparameters_obj.update(hyperparameters)
            K = self.gp_kernel.compute_kernel(self.X, self.X)
            K += np.repeat(np.array(np.eye(len(self.X)) * 1e-3)[:,:, np.newaxis], self.X.shape[1], axis=2)
            debug_K = np.squeeze(K)
            L = scipy.linalg.cholesky(K[:, :, 0], lower=True)
            n = len(self.y)
            one_vector = np.ones(n)
            y_adj = self.y - self.y_mean
            #debug_print(f"y_adj: {y_adj}")

            alpha = scipy.linalg.cho_solve((L, True), y_adj)

            # term_1_c_array = np.array()
            # term_2_c_array = np.array()
            # term_3_c_array = np.array()


            term_1_c = (0.5 * y_adj.T @ alpha).item()
            term_2_c = np.sum(np.log(np.diag(L)))       ###### 2x this??????
            term_3_c = 0.5 * n * np.log(2 * np.pi)

            # # DEBUG
            #
            # term_1_c_array.append(term_1_c_array)
            # term_2_c_array.append(term_2_c_array)
            # term_3_c_array.append(term_3_c_array)
            #
            # # /DEBUG

            nll = term_1_c + term_2_c + term_3_c

            out_c = {
                'nll': nll,
                'term_1': term_1_c,
                'term_2': term_2_c,
                'term_3': term_3_c
            }
            debug_print(f"out_c = {out_c}")
            return out_c

        elif self.gp_algo == 'FITC_18_134':
            #out_f = self.run_FITC_18_134()

            out_f = self.gp_nll_algo_obj.compute_nll()
            return out_f

        else:
            raise ValueError("Invalid compute_nll method")


    def reshape_X_and_y(self):
        if self.X.ndim == 1: self.X = self.X.reshape(-1, 1)
        if self.y.ndim == 1: self.y = self.y.reshape(-1, 1)

    def update_hyperparameters_and_debug(self, hyperparameters):
        if type(hyperparameters) == dict or \
            type(hyperparameters) == Hyperparameters or \
            type(hyperparameters) == np.ndarray:
            self.hyperparameters_obj.update(hyperparameters)
        else:
            raise ValueError(
                "Incorrect hyperparameter type: must be 'dict' or 'ndarray'")
        debug_var = self.hyperparameters_obj.array()
        debug_print(
            f"compute_NLL {self.gp_algo} hyperparameters: {debug_var}")

    # def flatten_params(self, params):
    #     flat_params = []
    #     for key, value in params.items():
    #         if isinstance(value, str):
    #             continue
    #         if isinstance(value, list):
    #             for item in value:
    #                 flat_params.extend(self.flatten_params(item))
    #         elif isinstance(value, dict):
    #             flat_params.extend(self.flatten_params(value))
    #         else:
    #             flat_params.append(value)
    #     return flat_params
    #
    # def reconstruct_params_implementation(self, flat_params, template):
    #     reconstructed_params = {}
    #     index = 0
    #     for key, value in template.items():
    #         if isinstance(value, str):
    #             reconstructed_params[key] = value
    #             continue
    #         if isinstance(value, list):
    #             reconstructed_params[key] = []
    #             for item in value:
    #                 reconstructed_item, item_length = self.reconstruct_params_implementation(flat_params[index:], item)
    #                 index += item_length
    #                 reconstructed_params[key].append(reconstructed_item)
    #         elif isinstance(value, dict):
    #             reconstructed_params[key], item_length = self.reconstruct_params_implementation(flat_params[index:], value)
    #             index += item_length
    #         else:
    #             reconstructed_params[key] = flat_params[index]
    #             index += 1
    #     return reconstructed_params, index
    #
    # def reconstruct_params(self, flat_params):
    #     reconstructed_params, index = self.reconstruct_params_implementation(flat_params,self.template)
    #     return reconstructed_params
