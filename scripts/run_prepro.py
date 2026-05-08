from roidims.config import *
from roidims.prepro import (
    download_sessions,
    prepro_img_meta,
    split_train_test,
    prepro_resp,
    define_rois,
    extract_roi_resp,
)

# -------------------- Preprocess NSD fMRI voxel responses ------------------- #
def run_prepro(subjects: list, rois: list, train_size: float=0.7, seed: int=0,
               t_thresh: int=2, ncsnr_thresh: float=0.2):
    # Download sessions
    download_sessions(subjects)

    # Preprocess image meta data
    prepro_img_meta()

    # Split training and test set
    split_train_test(subjects, train_size, seed)

    # Preprocess voxel responses
    prepro_resp(subjects)

    # Define ROIs
    define_rois(subjects, t_thresh)

    # Extract ROI voxel responses and apply baseline shift
    extract_roi_resp(subjects, rois, t_thresh, ncsnr_thresh)
