from roidims.bnmf import perform_bnmf_runs
from roidims.consensus import find_consensus_dims
from roidims.consistent import find_consistent_dims
from roidims.utils import submit_parallel_jobs

# --------------------- Perform BNMF voxel decomposition --------------------- #
def run_bnmf(subjects: list, rois: list, n_runs: int=100, r_thresh: float=0.3):
    # Perform randomly initialized BNMF runs
    perform_bnmf_runs(subjects, rois, n_runs)

    # Find consensus dimensions within each subject
    args = [(subject, roi) for subject in subjects for roi in rois]
    _ = submit_parallel_jobs(find_consensus_dims, args, joblib_kwargs={"n_jobs": len(subjects)*len(rois), "verbose": 10})

    # Find consistent dimensions across subjects
    args = [(subjects, roi, r_thresh) for roi in rois]
    _ = submit_parallel_jobs(find_consistent_dims, args, joblib_kwargs={"n_jobs": len(rois), "verbose": 10})
