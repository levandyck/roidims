import numpy as np
import nibabel as nib
from fracridge import FracRidgeRegressor
from sklearn.model_selection import KFold
from scipy import stats
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings

from roidims.utils import (
    SubjectLoader,
    VolumeConverter,
    load_resp_h5_lowmem,
    hoyer_sparseness
)

# ------------------ Voxel-wise fractional ridge regression ------------------ #
class FracRidgeVoxelwise:
    """Voxel-wise encoding model using fractional ridge regression with permutation testing."""
    def __init__(self, n_folds: int=5, fracs: list=None, fracridge_kws: dict=None, n_perms: int=3000, n_jobs: int=50):
        if fracs is None:
            fracs = np.hstack([np.arange(0.10, 1.00, 0.10), np.arange(0.91, 1.01, 0.01)])
        if fracridge_kws is None:
            fracridge_kws = dict(fit_intercept=True, normalize=True, jit=False)
        self.n_folds = n_folds
        self.fracs = fracs
        self.n_fracs = len(self.fracs)
        self.fr = FracRidgeRegressor(fracs=self.fracs, **fracridge_kws)
        self.frf = FracRidgeRegressor(fracs=0.01, **fracridge_kws)
        self.n_perms = n_perms
        self.n_jobs = n_jobs

    def tune_alpha(self, X_train: np.ndarray, y_train: np.ndarray):
        """Tune regularization using k-fold cross-validation."""
        # K-fold cross-validation on training set
        kfold = KFold(n_splits=self.n_folds, shuffle=True)
        n_vox = y_train.shape[1]
        r2s = np.zeros((self.n_folds, self.n_fracs, n_vox))

        # Iterate over folds
        for fold_i, (tune_ids, val_ids) in tqdm(
            enumerate(kfold.split(X_train)),
            desc="Tuning regularization on training set",
            total=self.n_folds
        ):
            # Split training set into tuning and validation folds
            X_tune, y_tune = X_train[tune_ids], y_train[tune_ids]
            X_val, y_val = X_train[val_ids], y_train[val_ids]

            # Fit on tuning fold and predict validation fold
            self.fr.fit(X_tune, y_tune)
            y_val_pred = self.fr.predict(X_val)

            # Compute prediciton performance (R2) for fraction candidates
            for frac_i in range(self.n_fracs):
                r_vals = pearsonr_nd(y_val, y_val_pred[:, frac_i, :])
                r2s[fold_i, frac_i] = r_vals ** 2

        # Average performance across folds and choose best fraction for each voxel
        mean_r2s = r2s.mean(axis=0)
        self.best_frac_ids = np.argmax(mean_r2s, axis=0)
        self.best_fracs = self.fracs[self.best_frac_ids]

        return mean_r2s

    def fit_eval(self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray):
        """Fit final model on training set using best fraction for each voxel."""
        n_vox = y_test.shape[1]
        perf_r = np.zeros(n_vox)
        coefs = np.zeros((X_train.shape[1], n_vox))
        intercepts = np.zeros(n_vox)

        # Iterate over fractions
        unique_frac_ids = np.unique(self.best_frac_ids)
        for frac_i in tqdm(unique_frac_ids, desc="Fitting best model and evaluating on test set", total=len(unique_frac_ids)):
            # Select corresponding voxels
            vox_ids = np.where(self.best_frac_ids == frac_i)[0]
            y_train_frac, y_test_frac = y_train[:, vox_ids], y_test[:, vox_ids]

            # Fit on training set using best fraction
            self.frf.fracs = np.array([self.fracs[frac_i]])
            self.frf.fit(X_train, y_train_frac)

            # Predict test set
            y_test_frac_pred = np.squeeze(self.frf.predict(X_test), axis=1)

            # Predict test set, evaluate correlation, keep beta coeffients
            # Single voxel edge case
            if len(vox_ids) == 1:
                vox_id = vox_ids[0]
                y_test_frac_pred = y_test_frac_pred.ravel()
                perf_r[vox_id] = pearsonr_nd(y_test_frac, y_test_frac_pred[:, None])[0]
                coefs[:, vox_id] = self.frf.coef_[0]
                intercepts[vox_id] = self.frf.intercept_[0]

            # Multi voxel main case
            else:
                perf_r[vox_ids] = pearsonr_nd(y_test_frac, y_test_frac_pred)
                coefs[:, vox_ids] = np.squeeze(self.frf.coef_, axis=1)
                intercepts[vox_ids] = self.frf.intercept_

        return perf_r, coefs, intercepts

    def perm_tests(self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray):
        """Perform permutation testing."""
        perf_perm = np.zeros((self.n_perms, y_test.shape[1]))
        n_jobs = self.n_jobs # // 2

        # Iterate over fracs
        unique_frac_ids = np.unique(self.best_frac_ids)
        for frac_i in tqdm(
            unique_frac_ids,
            desc="Performing permutation testing",
            total=len(unique_frac_ids),
            leave=True,
        ):
            # Select corresponding voxels
            vox_ids = np.where(self.best_frac_ids == frac_i)[0]
            y_train_frac, y_test_frac = y_train[:, vox_ids], y_test[:, vox_ids]

            # Fit on training set
            self.frf.fracs = [self.fracs[frac_i]]
            self.frf.fit(X_train, y_train_frac)

            # Predict test set
            y_test_frac_pred = np.squeeze(self.frf.predict(X_test), axis=1)

            # Perform permutation testing
            # Single voxel edge case
            if len(vox_ids) == 1:
                def compute_r_sample():
                    y_test_perm_frac = np.random.permutation(y_test_frac)
                    r = pearsonr_nd(y_test_perm_frac[:, np.newaxis], y_test_frac_pred[:, np.newaxis])[0]
                    return r

                perms = Parallel(n_jobs=n_jobs)(
                delayed(compute_r_sample)() for _ in range(self.n_perms)
                )
                perf_perm[:, vox_ids[0]] = perms

            # Multi voxel main case
            else:
                def compute_r_sample():
                    y_test_perm_frac = np.apply_along_axis(np.random.permutation, 0, y_test_frac)
                    r = pearsonr_nd(y_test_perm_frac, y_test_frac_pred)
                    return r

                perms = Parallel(n_jobs=n_jobs)(
                    delayed(compute_r_sample)()
                    for _ in range(self.n_perms)
                )
                perms = np.array(perms)
                perf_perm[:, vox_ids] = perms

        return perf_perm

    def calc_pvals(self, perf_true: np.ndarray, perf_perm: np.ndarray):
        """Compute p-values based on the fraction of permutations that equal or exceed true performance."""
        pvals = np.mean(perf_perm >= perf_true[np.newaxis, :], axis=0)
        return pvals

    def run(self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray):
        """Tune and fit fractional ridge regression model."""
        # Tune alpha and find best fractions using k-fold cross-validation on training set
        self.tune_alpha(X_train, y_train)

        # Fit final model and evaluate performance
        perf_r, coefs, intercepts = self.fit_eval(X_train, X_test, y_train, y_test)

        # Perform permutation testing
        if self.n_perms > 0:
            perf_perm = self.perm_tests(X_train, X_test, y_train, y_test)
            pvals = self.calc_pvals(perf_r, perf_perm)
        else:
            pvals = None

        return perf_r, coefs, intercepts, pvals, self.best_fracs

def pearsonr_nd(arr1: np.ndarray, arr2: np.ndarray, alongax: int=0):
    """
    Pearson correlation between respective variables in two arrays.
    arr1 and arr2 are 2d arrays. Rows correspond to observations, columns to variables.
    Returns:
        correlations: np.ndarray (shape nvariables)
    """
    # Center each feature
    arr1_c = arr1 - arr1.mean(axis=alongax)
    arr2_c = arr2 - arr2.mean(axis=alongax)

    # Get sum of products for each voxel (numerator)
    numerators = np.sum(arr1_c * arr2_c, axis=alongax)

    # Denominator
    arr1_sds = np.sqrt(np.sum(arr1_c**2, axis=alongax))
    arr2_sds = np.sqrt(np.sum(arr2_c**2, axis=alongax))
    denominators = arr1_sds * arr2_sds

    # For many voxels, this division will raise RuntimeWarnings for divide by zero. Ignore these.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = numerators / denominators
    return r

def fit_dim_encoding(subjects: list, rois: list, n_folds: int, fracs: list, n_perms: int):
    """Fit ROI dimensions to whole-cortex voxels responses."""
    for subject in subjects:
        sub = SubjectLoader(subject)
        Ws_train = np.load(sub.bnmf_dir / "W_train_con.npz", allow_pickle=True)
        Ws_test = np.load(sub.bnmf_dir / "W_test_con.npz", allow_pickle=True)

        # Whole-cortex voxel responses (= target)
        train_ids, test_ids = sub.load_set_ids()
        resp = load_resp_h5_lowmem(sub, "brain")
        y_train, y_test = resp[train_ids], resp[test_ids]

        for roi in rois:
            # Predictors
            X_train, X_test = Ws_train[roi], Ws_test[roi]

            # Fit fractional ridge regression
            model = FracRidgeVoxelwise(
                n_folds=n_folds,
                fracs=fracs,
                fracridge_kws=dict(fit_intercept=True, normalize=True, jit=False),
                n_perms=n_perms,
                n_jobs=40
            )
            perf_r, betas, intercepts, pvals, best_fracs = model.run(X_train, X_test, y_train, y_test)

            # Save to volume
            vol = VolumeConverter(subject)
            data_to_save = [
                ("perf_r", perf_r),
                ("betas", betas),
                ("intercepts", intercepts),
                ("pvals", pvals),
                ("best_fracs", best_fracs)
            ]
            for name, data in data_to_save:
                vol.save_mat2nii(name, data, seed_roi=roi, target_roi="brain")

            # Correct p-values for multiple comparisons
            if pvals is not None:
                _, pvals_corr, _, _ = multipletests(pvals, alpha=0.01, method="fdr_bh")
                vol.save_mat2nii("pvals_corr", pvals_corr, seed_roi=roi, target_roi="brain")


# ----------------------------------- Stats ---------------------------------- #
def compute_corr_H_beta(subjects: list, rois: list, ncsnr_thresh: float=0.2):
    for roi in rois:
        per_subj_corrs = []

        for subject in subjects:
            sub = SubjectLoader(subject)
            H_train_con = np.load(sub.bnmf_dir / "H_train_con.npz", allow_pickle=True)[roi]
            betas = nib.load(sub.encoding_dir / f"betas_{subject}_{roi}.nii.gz").get_fdata()

            # Get ROI voxels above SNR
            brain_mask = sub.load_brain_mask()[0]
            ncsnr = nib.load(sub.resp_dir / "ncsnr_train.nii.gz").get_fdata().flatten()[brain_mask]
            ncsnr = (ncsnr**2) / ((ncsnr**2) + (1/3))

            # Determine SNR threshold
            vox_ids = sub.load_roi_mask(roi, ids_ref="brain")[1]
            ncsnr_vox_ids = np.where(ncsnr[vox_ids] > ncsnr_thresh)[0]

            # Filter
            vox_ids = sub.load_roi_mask(roi, ids_ref="volume")[1]

            # Collect per-dimension correlations
            n_dims = H_train_con.shape[0]
            corr_vec = np.zeros(n_dims, dtype=float)
            for d in range(n_dims):
                h_d = H_train_con[d, :]
                beta_d = betas[..., d].flatten()[vox_ids][ncsnr_vox_ids]
                corr = np.corrcoef(h_d, beta_d)[0, 1]
                corr_vec[d] = corr

            per_subj_corrs.append(corr_vec)

        # Average across subjects
        corr_mat = np.stack(per_subj_corrs, axis=0)
        avg_corr_per_dim = corr_mat.mean(axis=0)

        # Summarize across dimensions
        mean_val = avg_corr_per_dim.mean()
        sd_val = avg_corr_per_dim.std(ddof=0)
        min_val = avg_corr_per_dim.min()
        max_val = avg_corr_per_dim.max()

        print(f"{roi:10s}: M = {mean_val:.2f}, SD = {sd_val:.2f}, range = {min_val:.2f}-{max_val:.2f}")

def selectivity_vs_sparseness(subjects: list, rois: list):
    floc_map = {"FFA": "faces", "EBA": "bodies", "PPA": "places"}

    results = []
    for roi in rois:
        floc = floc_map.get(roi, roi)

        corrs = []
        for subj in subjects:
            sub = SubjectLoader(subj)
            vox_ids = sub.load_roi_mask(roi, ids_ref="volume")[1]
            if len(vox_ids) == 0:
                continue

            # Load betas and d'
            betas = nib.load(sub.encoding_dir / f"betas_{subj}_{roi}.nii.gz").get_fdata()
            betas = betas.reshape(-1, betas.shape[3]).T[:, vox_ids]
            dprime = nib.load(sub.roi_mask_dir / f"dprime_{floc}.nii.gz").get_fdata().ravel()[vox_ids]
            sparse = hoyer_sparseness(betas, axis=0)

            corr, _ = stats.pearsonr(dprime, sparse)
            corrs.append(corr)

        if len(corrs):
            z = np.arctanh(corrs)
            corr_mean = np.tanh(z.mean())
            _, pval_group = stats.ttest_1samp(z, 0.0)
        else:
            corr_mean, pval_group = np.nan, np.nan

        print(f"{roi}: mean r={corr_mean:.2f}, p={pval_group:.3f}")

        results.append({
            "ROI": roi,
            "corr_mean": corr_mean,
            "pval_group": pval_group
        })
