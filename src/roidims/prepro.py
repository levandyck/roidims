import os
import numpy as np
import pandas as pd
import h5py
import nibabel as nib
import requests
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

from roidims.config import *
from roidims.plotting import (
    plot_ncsnr_distribution,
    plot_shift_distribution,
    plot_minima_distribution,
)
from roidims.utils import (
    SubjectLoader,
    VolumeConverter,
    save_resp_h5_lowmem,
    load_resp_h5_lowmem,
)

SHARED_TYPE = "shared1000"

# ----------------------------- Download sessions ---------------------------- #
def download_sessions(subjects: list):
    """Download sessions with fMRI single trial responses."""
    for subject in subjects:
        sub = SubjectLoader(subject)
        base_url = f"https://natural-scenes-dataset.s3.amazonaws.com/nsddata_betas/ppdata/{subject}/func1pt8mm/betas_fithrf_GLMdenoise_RR/"
        for i in tqdm(range(sub.n_sessions), desc=f"Downloading sessions for {sub.subject}"):
            file_name = sub.raw_dir / f"betas_session{i+1:02d}.hdf5"
            session_url = base_url + f"betas_session{i+1:02d}.hdf5"
            with open(file_name, "wb") as f:
                f.write(requests.get(session_url).content)


# ------------------------ Preprocess image meta data ------------------------ #
def get_img_meta_subj(sub, img_meta_all: pd.DataFrame):
    """Get image meta data for subject."""
    trial_ids = img_meta_all[f"{sub.subject_name}_rep2"].values
    subj_mask = (trial_ids > 0) & (trial_ids <= sub.n_trials)
    img_meta_subj = img_meta_all[subj_mask]
    return img_meta_subj

def prepro_img_meta():
    """Preprocess image meta data."""
    subjects_all = [f"subj{s:02}" for s in [1, 2, 5, 7]]

    # Read img info
    nsd_info_dir = PROJ_DIR / "nsd_info"
    img_meta_all = pd.read_csv(nsd_info_dir / "nsd_stim_info_merged_anns.csv")

    # Get shared images
    shared_nsd_ids = set(img_meta_all["nsdId"])
    for subject in subjects_all:
        sub = SubjectLoader(subject)
        img_meta_subj = get_img_meta_subj(sub, img_meta_all)
        shared_nsd_ids &= set(img_meta_subj["nsdId"]) # Intersect with current subject"s nsdIds

    # Preprocess img meta data
    for subject in subjects_all:
        sub = SubjectLoader(subject)
        img_meta_subj = get_img_meta_subj(sub, img_meta_all).copy()
        cols_rep = [f"{sub.subject_name}_rep{i}" for i in range(3)]
        img_meta_subj["trialIds"] = img_meta_subj[cols_rep].apply(lambda row: [x-1 for x in row], axis=1)
        img_meta_subj["trialIds"] = img_meta_subj["trialIds"].apply(json.dumps)
        cols_keep = ["nsdId", "trialIds", "category_ids", "category_labels", "supercategory_labels", "caption1", "caption2", "caption3", "caption4", "caption5"]
        cols_keep.append(SHARED_TYPE)
        img_meta_subj = img_meta_subj.loc[:, cols_keep]
        img_meta_subj.to_csv(sub.resp_dir / "img_meta.csv", index=False)
    print(f"Preprocessed image metadata for all subjects.")

def split_train_test(subjects: list, train_size: float, seed: int=0):
    """Split images into a training and test set."""
    np.random.seed(seed)
    n_imgs = 10000
    n_train = int(n_imgs * train_size)
    n_test = n_imgs - n_train
    print(f"Nr. train = {n_train}; nr. test = {n_test}")

    for subject in subjects:
        sub = SubjectLoader(subject)
        img_meta = sub.load_img_meta()
        img_meta["trialIds"] = img_meta["trialIds"].apply(json.loads)

        # Separate shared and individual ids
        shared_ids = img_meta.index[img_meta[SHARED_TYPE]].to_numpy()
        indiv_ids = img_meta.index[~img_meta[SHARED_TYPE]].to_numpy()

        # Assign all shared ids to training set
        n_train_shared = len(shared_ids)
        n_train_indiv = n_train - n_train_shared
        if n_train_indiv < 0:
            raise ValueError("Number of shared images exceeds training size.")

        # Shuffle and split individual ids
        np.random.shuffle(indiv_ids)
        train_indiv_ids = indiv_ids[:n_train_indiv]
        train_ids = np.concatenate([shared_ids, train_indiv_ids])

        # Assign sets
        img_meta["set"] = "test" # Default to "test"
        img_meta.loc[train_ids, "set"] = "train"
        img_meta.to_csv(sub.resp_dir / "img_meta.csv", index=False)
        print(f"Split train and test images for {sub.subject}.")


# ------------------------ Preprocess voxel responses ------------------------ #
def zscore_trials(sub, sessions: list, img_meta_long: pd.DataFrame):
    """Z-score each voxel across trials and concatenate along trials."""
    brain_mask = sub.load_brain_mask()[0]
    train_mask = img_meta_long["set"] == "train"

    resp_z = []
    n_trials_session = 750
    for session_idx, _ in enumerate(
        tqdm(
            sessions,
            desc=f"Z-scoring responses within sessions for {sub.subject}"
            )
        ):
        # Get trial ids for session
        start_trial = session_idx * n_trials_session
        end_trial = (session_idx + 1) * n_trials_session
        trial_ids_curr = np.arange(start_trial, end_trial)

        # Load session responses
        with h5py.File(sub.raw_dir / sessions[session_idx], "r") as f:
            resp = np.array(f["betas"], dtype=np.int16).T.astype(np.float64) / 300.0
        resp = resp.reshape(-1, n_trials_session)[brain_mask]

        # Compute mean and std for train trials within session
        train_mask_session = train_mask[trial_ids_curr]
        n_train_trials = np.sum(train_mask_session)
        print(f"Training ratio of session{session_idx}: {n_train_trials/750:.2f}%")
        mean_train = np.nanmean(resp[:, train_mask_session], axis=1, keepdims=True)
        std_train = np.nanstd(resp[:, train_mask_session], axis=1, ddof=1, keepdims=True)

        # Z-score all trials in session accordingly
        with np.errstate(divide="ignore", invalid="ignore"):
            resp_z_session = (resp - mean_train) / std_train
            resp_z_session = np.nan_to_num(resp_z_session)
        resp_z.append(resp_z_session)

    # Concatenate sessions
    resp_z = np.concatenate(resp_z, axis=1).T
    return resp_z

def compute_ncsnr(resp_z: np.ndarray, img_meta_long: pd.DataFrame):
    """Compute voxel-wise NCSNR according to NSD."""
    # Group trials by nsdId and filter those with 3 trials
    trial_groups = img_meta_long.groupby("nsdId")["trialIds"].apply(list)
    filtered_groups = trial_groups[trial_groups.apply(len) == 3]
    if filtered_groups.empty:
        return np.zeros(resp_z.shape[1])

    # Compute within-image variance
    var_within = np.vstack([
        np.nanvar(resp_z[trial_ids, :], axis=0, ddof=1)
        for trial_ids in filtered_groups
    ])

    # Compute noise and signal variance
    var_noise = np.nanmean(var_within, axis=0)
    var_signal = np.maximum(0, 1 - var_noise)

    # Compute voxel-wise NCSNR
    with np.errstate(divide="ignore", invalid="ignore"):
        ncsnr = np.sqrt(var_signal) / np.sqrt(var_noise)
        ncsnr = np.nan_to_num(ncsnr) # Convert NaNs and infs to 0
    return ncsnr

def compute_ncsnr_splits(sub, resp_z: np.ndarray, img_meta_long: pd.DataFrame):
    """
    Compute voxel-wise noise-ceiling signal-to-noise ratio (ncsnr) separately
    for training and test images.
    """
    # Split sets
    train_meta = img_meta_long[img_meta_long["set"] == "train"]
    test_meta = img_meta_long[img_meta_long["set"] == "test"]

    # Compute ncsnr for sets
    ncsnr_train = compute_ncsnr(resp_z, train_meta)
    ncsnr_test = compute_ncsnr(resp_z, test_meta)

    # Save to volume
    vol = VolumeConverter(sub.subject)
    vol.save_nii(vol.to_volume(ncsnr_train, "brain"), vol.resp_dir / "ncsnr_train.nii.gz")
    vol.save_nii(vol.to_volume(ncsnr_test, "brain"), vol.resp_dir / "ncsnr_test.nii.gz")
    print(f"Computed noise ceilings for {sub.subject}.")

def average_reps(sub, resp_z: np.ndarray, img_meta: pd.DataFrame):
    """
    Average single-trial z-scored responses across repetitions for each unique
    nsdId.
    """
    trial_ids_list = [np.array(trial_ids) for trial_ids in img_meta["trialIds"]]
    resp_z_avg = np.array([
        resp_z[trial_ids].mean(axis=0)
        for trial_ids in tqdm(trial_ids_list, desc=f"Averaging repetitions for {sub.subject}")
    ])
    return resp_z_avg

def prepro_resp(subjects: list):
    """Preprocess single-trial voxel responses."""
    for subject in subjects:
        sub = SubjectLoader(subject)
        sessions = sorted([file for file in os.listdir(sub.raw_dir) if file.endswith(".hdf5")])

        # Prepare image meta data
        img_meta = sub.load_img_meta()
        img_meta["nsdId"] = img_meta["nsdId"].astype(int)
        img_meta["trialIds"] = img_meta["trialIds"].apply(json.loads)
        img_meta_long = img_meta.explode("trialIds")
        img_meta_long = img_meta_long.sort_values(by="trialIds").reset_index(drop=True)

        # Z-score across trials using training stats within each session
        resp_z = zscore_trials(sub, sessions, img_meta_long)

        # Compute voxel-wise ncsnr
        compute_ncsnr_splits(sub, resp_z, img_meta_long)

        # Average repetitions
        resp_z_avg = average_reps(sub, resp_z, img_meta)
        save_resp_h5_lowmem(sub, resp_z_avg, "brain")
        del resp_z_avg
        print(f"Preprocessed responses for {sub.subject}.")


# -------------------------------- Define ROIs ------------------------------- #
def define_rois(subjects: list, t_thresh: float):
    """Define ROIs based on fLOC."""
    roi_labels_dir = PROJ_DIR / "nsd_info" / "roi_labels"

    # Define ROI groups
    roi_groups = {
        "OFA": [["OFA"], [f"floc-faces_t{t_thresh}"]],
        "FFA": [["FFA-1", "FFA-2"], [f"floc-faces_t{t_thresh}"]],
        "EBA": [["EBA"], [f"floc-bodies_t{t_thresh}"]],
        "FBA": [["FBA-1", "FBA-2"], [f"floc-bodies_t{t_thresh}"]],
        "OPA": [["OPA"], [f"floc-places_t{t_thresh}"]],
        "PPA": [["PPA"], [f"floc-places_t{t_thresh}"]],
        "RSC": [["RSC"], [f"floc-places_t{t_thresh}"]],
        "Early": [["early"], ["streams"]],
        "Ventral": [["ventral"], ["streams"]],
        "Lateral": [["lateral"], ["streams"]],
        "Parietal": [["parietal"], ["streams"]]
        }

    # Get mask and label files
    lh_mask_files = {roi_group: "lh." + value[1][0] + ".nii.gz" for roi_group, value in roi_groups.items()}
    rh_mask_files = {roi_group: "rh." + value[1][0] + ".nii.gz" for roi_group, value in roi_groups.items()}
    label_files = {roi_group: label[1][0] + ".mgz.ctab" for roi_group, label in roi_groups.items()}

    # Create voxel meta data (indexing flattened brain mask)
    for subject in subjects:
        sub = SubjectLoader(subject)
        brain_mask = sub.load_brain_mask()[0]
        n_voxels_brain = brain_mask.sum()

        vox_meta = pd.DataFrame(index=np.arange(n_voxels_brain))
        for roi_group, roi_list in roi_groups.items():
            # Load mask and label files
            lh_mask = np.array(nib.load(sub.roi_mask_dir / lh_mask_files[roi_group]).get_fdata()).flatten()[brain_mask]
            rh_mask = np.array(nib.load(sub.roi_mask_dir / rh_mask_files[roi_group]).get_fdata()).flatten()[brain_mask]
            labels = pd.read_csv(roi_labels_dir / label_files[roi_group], sep=" ", header=None, names=["index", "label"])

            # Get ROI voxel indices
            roi_ids = []
            for roi in roi_list[0]:
                roi_ids.append(int(labels.loc[labels["label"] == roi, "index"].iloc[0]))

            # Initialize column with zeros
            vox_meta[roi_group] = 0
            # Set 1 and -1 for voxels in left and right hemisphere, respectively
            lh_ids = np.where(np.isin(lh_mask, roi_ids))[0]
            rh_ids = np.where(np.isin(rh_mask, roi_ids))[0]
            vox_meta.loc[lh_ids, roi_group] = -1
            vox_meta.loc[rh_ids, roi_group] = 1
            print(f"Nr. of voxels in {roi_group}: Both={len(lh_ids) + len(rh_ids)}, LH={len(lh_ids)}, RH={len(rh_ids)}")
        vox_meta.to_csv(sub.resp_dir / "vox_meta.csv", index=False)
        print(f"Defined ROIs for {sub.subject}.")


# --------------------------- Extract ROI responses -------------------------- #
def apply_baseline_shift(resp_train: np.ndarray, resp_test: np.ndarray):
    """
    Shift both training and testing data by subtracting global training minimum
    to ensure non-negativity in train and test data.
    """
    # Compute global minimum of training data
    min_train = np.min(resp_train)

    # Shift both training and test data by training minimum
    resp_train_shift = resp_train - min_train
    resp_test_shift = resp_test - min_train

    n_clipped = np.sum(resp_test_shift < 0)
    perc_clipped = (n_clipped / resp_test_shift.size) * 100
    print(f"Perc. clipped voxel responses: {perc_clipped:.2f}%")

    resp_test_shift = np.clip(resp_test_shift, a_min=0, a_max=None)
    assert (resp_train_shift >= 0).all()
    assert (resp_test_shift >= 0).all()

    return resp_train_shift, resp_test_shift

def extract_roi_resp(subjects: list, rois: list, t_thresh: float, ncsnr_thresh: float):
    """Extract ROI voxel responses and apply baseline shift."""
    for subject in subjects:
        sub = SubjectLoader(subject)
        ncsnr = sub.load_ncsnr(set="train")
        train_ids, test_ids = sub.load_set_ids()
        resp_brain = load_resp_h5_lowmem(sub, "brain")

        fig_ncsnr, axes_ncsnr = plt.subplots(
            1, len(rois), figsize=[15, 5], constrained_layout=True,
        )
        fig_shift, axes_shift = plt.subplots(
            1, len(rois), figsize=[15, 5], constrained_layout=True,
        )
        fig_minima, axes_minima = plt.subplots(
            1, len(rois), figsize=[15, 5], constrained_layout=True,
        )

        axes_ncsnr = [axes_ncsnr] if len(rois) == 1 else axes_ncsnr
        axes_shift = [axes_shift] if len(rois) == 1 else axes_shift
        axes_minima = [axes_minima] if len(rois) == 1 else axes_minima

        for r, roi in tqdm(enumerate(rois), desc=f"Extracted ROIs for {sub.subject}"):
            vox_ids = sub.load_roi_mask(roi, ids_ref="brain")[1]
            resp_roi = resp_brain[:, vox_ids]
            ncsnr_roi = ncsnr[vox_ids]

            # Determine SNR threshold and filter out low SNR voxels
            snr_vox_ids = np.where(ncsnr[vox_ids] > ncsnr_thresh)[0]
            resp_roi_snr = resp_roi[:, snr_vox_ids]
            n_vox_orig = resp_roi.shape[1]
            n_vox_snr = resp_roi_snr.shape[1]
            n_vox_removed = (n_vox_orig - n_vox_snr)
            print(f"{roi}: SNR threshold={ncsnr_thresh}; nr. of voxels: orig={n_vox_orig}; snr={n_vox_snr}; removed={n_vox_removed}")

            # Plot NCSNR distribution
            ax_ncsnr = axes_ncsnr[r]
            plot_ncsnr_distribution(ax_ncsnr, ncsnr_roi, ncsnr_thresh)

            # Split into sets
            resp_roi_train = resp_roi_snr[train_ids]
            resp_roi_test = resp_roi_snr[test_ids]

            # Apply baseline shift to ensure all responses are nonnegative
            resp_roi_train_shift, resp_roi_test_shift = apply_baseline_shift(resp_roi_train, resp_roi_test)
            np.save(sub.resp_dir / f"resp_{roi}_t{t_thresh}_train.npy", resp_roi_train_shift)
            np.save(sub.resp_dir / f"resp_{roi}_t{t_thresh}_test.npy", resp_roi_test_shift)

            # Plot distributions of values before and after shifting
            ax_shift = axes_shift[r]
            plot_shift_distribution(
                ax_shift, resp_roi_train, resp_roi_train_shift,
                resp_roi_test, resp_roi_test_shift,
            )

            # Plot distributions of the minima values
            ax_minima = axes_minima[r]
            plot_minima_distribution(
                ax_minima, resp_roi_train, resp_roi_train_shift,
                resp_roi_test, resp_roi_test_shift,
            )

            # Set individual titles for each subplot
            ax_ncsnr.set_title(roi)
            ax_shift.set_title(roi)
            ax_minima.set_title(roi)

            # Finalize and save figures
            fig_ncsnr.tight_layout()
            fig_ncsnr.savefig(sub.resp_dir / "ncsnr_dist.png", dpi=300, bbox_inches="tight")

            handles, labels = axes_shift[0].get_legend_handles_labels()
            fig_shift.legend(handles, labels, loc="center right", title="Legend", bbox_to_anchor=(1.1, 0.5))
            fig_shift.tight_layout()
            fig_shift.savefig(sub.resp_dir / "shift_dist.png", dpi=300, bbox_inches="tight")

            handles, labels = axes_minima[0].get_legend_handles_labels()
            fig_minima.legend(handles, labels, loc="center right", title="Legend", bbox_to_anchor=(1.1, 0.5))
            fig_minima.tight_layout()
            fig_minima.savefig(sub.resp_dir / "shift_minima.png", dpi=300, bbox_inches="tight")
