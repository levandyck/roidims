import numpy as np
import nimfa
from sklearn.base import BaseEstimator, TransformerMixin

from roidims.utils import submit_parallel_jobs

# ---------------------------- Bi-cross-validation --------------------------- #
class BiCrossValidation(BaseEstimator, TransformerMixin):
    def __init__(self, ks, n_perms, random_state=None):
        self.ks = ks
        self.n_perms = n_perms
        self.random_state = random_state
        self.reset_fit_params()

    def reset_fit_params(self):
        self._is_fitted = False
        self.fit_params = {
            "ks": self.ks,
            "n_perms": self.n_perms,
            "evars": {k: np.zeros(self.n_perms) for k in self.ks},
            "train_errs": {k: np.zeros(self.n_perms) for k in self.ks},
            "test_errs": {k: np.zeros(self.n_perms) for k in self.ks}
        }

    @property
    def evars(self):
        self.check_is_fitted()
        return self.fit_params["evars"]

    @property
    def train_errs(self):
        self.check_is_fitted()
        return self.fit_params["train_errs"]

    @property
    def test_errs(self):
        self.check_is_fitted()
        return self.fit_params["test_errs"]

    def get_params(self):
        self.check_is_fitted()
        return self.fit_params

    def check_is_fitted(self):
        if not self._is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")

    def permute_rows_and_cols(self, X):
        m, n = X.shape
        row_perm = np.random.permutation(m)
        col_perm = np.random.permutation(n)
        return X[row_perm, :][:, col_perm]

    def split_matrix_into_quadrants(self, matrix: np.ndarray):
        m, n = matrix.shape
        size_x = m // 2
        size_y = n // 2
        return (
            matrix[:size_x, :size_y], # Top-left quadrant
            matrix[:size_x, size_y:], # Top-right quadrant
            matrix[size_x:, :size_y], # Bottom-left quadrant
            matrix[size_x:, size_y:], # Bottom-right quadrant
        )

    def split_quadrants(self, X: np.ndarray):
        X = self.permute_rows_and_cols(X)
        A, B, C, D = self.split_matrix_into_quadrants(X)
        return A, B, C, D

    def fit_bnmf(self, X: np.ndarray, k_step: int):
        bnmf = nimfa.Bd(X, seed="random_c", rank=k_step, max_iter=1500, min_residuals=1e-5, alpha=np.zeros((X.shape[0], k_step)),
                        beta=np.zeros((k_step, X.shape[1])), theta=.0, k=.0, sigma=1., skip=500, stride=5,
                        n_w=np.zeros((k_step, 1)), n_h=np.zeros((k_step, 1)), n_run=1, n_sigma=False)
        bnmf_fit = bnmf()
        W = bnmf_fit.basis()
        H = bnmf_fit.coef()
        return np.array(W), np.array(H)

    def bi_crossvalidation(self, X: np.ndarray, k: int, n_perms: int):
        m = len(X)
        if k > m:
            raise ValueError(
                f"k must be less than the number of observations, got k={k} and {m} observations"
            )
        evars, train_errs, test_errs = np.zeros(n_perms), np.zeros(n_perms), np.zeros(n_perms)

        for i in range(n_perms):
            # Shuffle and split into quadrants
            A, B, C, D = self.split_quadrants(X)

            # Train on quadrant D
            W_D, H_D = self.fit_bnmf(D, k_step=k)
            D_hat = np.dot(W_D, H_D)
            train_rss = np.sum((D - D_hat) ** 2)
            train_errs[i] = train_rss

            # Predict A using B and C
            W_B = np.dot(B, np.linalg.pinv(H_D)) # hat(W)_A in the paper
            H_C = np.dot(np.linalg.pinv(W_D), C) # hat(H)_A in the paper
            A_hat = np.dot(W_B, H_C)
            test_rss = np.sum((A - A_hat) ** 2)
            test_errs[i] = test_rss
            evars[i] = 1.0 - test_rss / np.sum(A ** 2)

        evars = np.mean(evars)
        train_errs = np.mean(train_errs)
        test_errs = np.mean(test_errs)
        return evars, train_errs, test_errs

    def fit_parallel(self, X: np.ndarray, joblib_kwargs: dict={"n_jobs": -1, "verbose": 10}):
        self.reset_fit_params()
        args = [(X, k, self.n_perms) for k in self.ks]
        results = submit_parallel_jobs(self.bi_crossvalidation, args, joblib_kwargs)
        evars, train_errs, test_errs = zip(*results)
        for k, evar, train_err, test_err in zip(self.ks, evars, train_errs, test_errs):
            self.fit_params["evars"][k] = evar
            self.fit_params["train_errs"][k] = train_err
            self.fit_params["test_errs"][k] = test_err
        self._is_fitted = True
        return self

    def fit(self, X: np.ndarray, k: int):
        self.reset_fit_params()
        evars, train_errs, test_errs = self.bi_crossvalidation(X, k, self.n_perms)
        self.fit_params["evars"][k] = evars
        self.fit_params["train_errs"][k] = train_errs
        self.fit_params["test_errs"][k] = test_errs
        self._is_fitted = True
        return self
