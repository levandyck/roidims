import numpy as np

from roidims.bnmf import project_test_nnls
from roidims.encoding import fit_dim_encoding

# ---------------------- Fit voxel-wise encoding models ---------------------- #
def run_dim_encoding(subjects: list, rois: list, n_folds: int=10,
                    fracs: np.ndarray=np.hstack([np.arange(0.10, 1.0, 0.10), np.arange(0.91, 1.01, 0.01)]),
                    n_perms: int=3000):
    # Project new test samples to learned embedding using NNLS
    project_test_nnls(subjects, rois)

    # Fit voxel-wise encoding model using dimensions from ROI to predict voxel responses across cortex
    fit_dim_encoding(subjects, rois, n_folds, fracs, n_perms)
