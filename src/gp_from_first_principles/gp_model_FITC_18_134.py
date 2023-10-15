import gp_model


class GPModel_FITC_18_134:
    def __init__(self):


    def run_FITC_18_134(self):
        # self.hyperparameters_obj.update(hyperparameters)
        # K = self.gp_kernel.compute_kernel(self.X, self.X)
        # K += np.repeat(
        #    np.array(np.eye(len(self.X)) * 1e-3)[:, :, np.newaxis],
        #   self.X.shape[1], axis=2)
        n = len(self.y)
        #    L = scipy.linalg.cholesky(K[:, :, 0], lower=True)
        y_adj = np.squeeze(self.y - self.hyperparameters_obj.dict()[
            'mean_func_c'])
        K_XX_FITC, K_XU, K_UX, K_UU, K_XX, Q_XX, K_UU_inv_K_UX = self.K_XX_FITC()
        # beta = self.hyperparameters_obj.dict()['noise_level'] ** -1
        K_sigma_inv = self.K_sigma_inv()
        # eigs = np.linalg.eigvalsh(K_sigma_inv)
        # debug_print(f"cov eigenvalues: {eigs}")
        n = K_XX.shape[0]
        # var645 = self.hyperparameters_obj.dict()['noise_level'] ** 2 * np.eye(n)
        big_lambda = np.diag(np.diag(K_XX - Q_XX)) + \
                     self.hyperparameters_obj.dict()[
                         'noise_level'] ** 2 * np.eye(n)
        # eigs_big_lambda = np.linalg.eigvalsh(big_lambda)
        # debug_print(f"big lambda eigenvalues: {eigs_big_lambda}")
        in53 = Q_XX + big_lambda + 1E-3 * np.eye(n)
        cond_in53 = np.linalg.cond(in53)
        debug_print(f"cond_in53: {cond_in53}")
        fast_det = self.fast_det(K_XU, K_UU_inv_K_UX, big_lambda)
        debug_print(f"fast_det: {fast_det}")
        debug_print(f"K_sigma_inv: {K_sigma_inv}")
        term_1_f = 0.5 * np.log(fast_det)
        term_2_f = 0.5 * y_adj.T @ K_sigma_inv @ y_adj
        term_3_f = 0.5 * n * np.log(2 * np.pi)
        nll = term_1_f + term_2_f + term_3_f
        nll = np.array(-nll).item()
        out_f = {
            'nll': nll,
            'term_1': term_1_f,
            'term_2': term_2_f,
            'term_3': term_3_f
        }
        return out_f
