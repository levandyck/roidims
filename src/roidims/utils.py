import os
from os.path import exists as pexists
import numpy as np
import pandas as pd
import h5py
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cortex
from joblib import Parallel, delayed

from roidims.config import SUBJ_DIR, GROUP_DIR

# ---------------------------------- File IO --------------------------------- #
def make_dirs(*paths: str):
    """Create directories if they do not exist."""
    for path in paths:
        try:
            os.makedirs(path, exist_ok=True)
        except FileExistsError:
            pass

def submit_parallel_jobs(func, args, joblib_kwargs: dict = {"n_jobs": -1, "verbose": 10}):
    parallel = Parallel(**joblib_kwargs)
    return parallel(delayed(func)(*arg) for arg in args)

def update_npz(file_path: str, new_data: dict):
    """Update an existing .npz file with new arrays or create a new .npz file."""
    if pexists(file_path):
        existing_data = dict(np.load(file_path, allow_pickle=True))
        existing_data.update(new_data)
        np.savez(file_path, **existing_data)
    else:
        np.savez(file_path, **new_data)

def save_resp_h5_lowmem(sub, resp: np.ndarray, name: str="brain"):
    resp = (resp*300).astype(dtype=np.int16) # Low memory transformation
    with h5py.File(sub.resp_dir / f"resp_{name}.hdf5", "w") as f:
        f.create_dataset(f"resp_{name}", data=resp)

def load_resp_h5_lowmem(sub, name: str="brain"):
    with h5py.File(sub.resp_dir / f"resp_{name}.hdf5", "r") as f:
        resp = f[f"resp_{name}"][:]
    resp = resp.astype(dtype=np.int64) / 300 # Undo low memory transformation
    return resp


# ------------------------------- SubjectLoader ------------------------------ #
class SubjectLoader():
    """Load subject data."""
    def __init__(self, subject: str):
        self.subject = subject
        self.subject_name = f"subject{int(self.subject[4:]):d}"

        self.subject_dir = SUBJ_DIR / subject
        self.raw_dir, self.resp_dir, self.roi_mask_dir, self.bcv_dir, self.bnmf_dir, self.top_imgs_dir, self.tsne_dir, self.encoding_dir = [
            self.subject_dir / folder for folder in ["raw", "resp", "roi", "bcv", "bnmf", "top_imgs", "tsne", "encoding"]
            ]
        make_dirs(self.raw_dir, self.resp_dir, self.roi_mask_dir, self.bcv_dir, self.bnmf_dir, self.top_imgs_dir, self.tsne_dir, self.encoding_dir)

        n_sessions_all = {"subj01": 40, "subj02": 40, "subj03": 32, "subj04": 30, "subj05": 40, "subj06": 32, "subj07": 40, "subj08": 30}
        self.n_sessions = n_sessions_all[self.subject]
        self.n_trials = self.n_sessions * 750

        self.img_meta_loaded = False
        self.img_meta = None

    def load_img_meta(self):
        if not self.img_meta_loaded:
            self.img_meta = pd.read_csv(self.resp_dir / "img_meta.csv")
            self.img_meta_loaded = True
        return self.img_meta

    def load_shared_ids(self, train: bool):
        img_meta = self.load_img_meta()
        if train:
            img_meta = img_meta[img_meta["set"] == "train"]
        shared_ids = np.where(img_meta["shared1000"])[0]
        return shared_ids

    def load_ncsnr(self, set: str):
        brain_mask = self.load_brain_mask()[0]
        ncsnr = nib.load(self.resp_dir / f"ncsnr_{set}.nii.gz").get_fdata().flatten()[brain_mask]
        ncsnr = (ncsnr**2) / ((ncsnr**2) + (1/3))
        return ncsnr

    def load_resp(self, roi: str, set: str):
        V = np.load(self.resp_dir / f"resp_{roi}_t2_{set}.npy")
        return V

    def load_imgs(self, type: str):
        img_meta = pd.read_csv(self.resp_dir / "img_meta.csv")
        nsd_ids = np.unique(img_meta["nsdId"])
        with h5py.File("/home/levandyck/roi-dims/NSD/nsd/nsd_stimuli.hdf5", "r") as f:
            img_brick = f["/imgBrick"]
            imgs = img_brick[nsd_ids]
        if type == "train":
            train_ids = np.where(img_meta["set"] == "train")[0]
            imgs = imgs[train_ids]
        elif type == "test":
            test_ids = np.where(img_meta["set"] == "test")[0]
            imgs = imgs[test_ids]
        elif type == "shared":
            shared_ids = self.load_shared_ids(train=True)
            imgs = imgs[shared_ids]
        return imgs.T

    def load_set_ids(self):
        img_meta = self.load_img_meta()
        train_ids = np.where(img_meta["set"] == "train")[0]
        test_ids = np.where(img_meta["set"] == "test")[0]
        return train_ids, test_ids

    def load_vox_meta(self):
        vox_meta = pd.read_csv(self.resp_dir / "vox_meta.csv")
        return vox_meta

    def load_roi_mask(self, roi: str, ids_ref: str):
        vox_meta = self.load_vox_meta()
        roi_mask = vox_meta[roi] != 0
        if ids_ref == "brain":
            roi_ids = np.where(roi_mask)[0]
        elif ids_ref == "volume":
            brain_ids = self.load_brain_mask()[1]
            roi_ids = brain_ids[np.where(roi_mask)[0]]
        else:
            print("Select 'brain' or 'volume' as reference.")
        return roi_mask, roi_ids

    def load_brain_mask(self):
        brain_mask = nib.load(self.roi_mask_dir / "brainmask.nii.gz").get_fdata().astype(bool).flatten()
        brain_ids = np.where(brain_mask)[0]
        return brain_mask, brain_ids

    def load_volume_info(self):
        brain_mask_vol = nib.load(self.roi_mask_dir / "brainmask.nii.gz")
        affine = brain_mask_vol.affine
        volume_shape = brain_mask_vol.get_fdata().shape
        return affine, volume_shape

    def load_nc(self):
        brain_mask = self.load_brain_mask()[0]
        ncsnr = nib.load(self.resp_dir / f"ncsnr.nii.gz").get_fdata().flatten()[brain_mask]
        return (ncsnr**2) / ((ncsnr**2) + (1 / 3))


# ------------------------------ VolumeConverter ----------------------------- #
class VolumeConverter(SubjectLoader):
    """Convert matrix to volume."""
    def __init__(self, subject):
        SubjectLoader.__init__(self, subject)
        self.affine, self.volume_shape = self.load_volume_info()
        self.vox_meta = self.load_vox_meta()

    def to_volume(self, data, roi):
        brain_ids = self.load_brain_mask()[1]

        data = np.atleast_2d(data)
        volumes = []
        for d in data:
            volume = np.zeros(self.volume_shape).flatten()
            if roi == "brain":
                volume[brain_ids] = d
            else:
                roi_ids = self.load_roi_ids(roi)
                volume[brain_ids[roi_ids]] = d
            volumes.append(volume.reshape(self.volume_shape))
        return np.stack(volumes, axis=-1) if data.shape[0] > 1 else volumes[0]

    def save_nii(self, volume, filename):
        img = nib.Nifti1Image(volume, self.affine)
        img.header.get_xyzt_units()
        img.to_filename(filename)

    def save_mat2nii(self, name, data, seed_roi, target_roi):
        volume = self.to_volume(data, target_roi)
        self.save_nii(volume, self.encoding_dir / f"{name}_{self.subject}_{seed_roi}.nii.gz")


# ----------------------------------- BNMF ----------------------------------- #
def hoyer_sparseness(X: np.ndarray, axis: int):
    """Compute Hoyer sparseness."""
    l1 = np.sum(np.abs(X), axis=axis)
    l2 = np.sqrt(np.sum(np.square(X), axis=axis))
    n = X.shape[axis]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (np.sqrt(n) - l1 / l2) / (np.sqrt(n) - 1)
    s = np.nan_to_num(s, nan=0.0)
    return s

def compute_evar_all(V: np.ndarray, W: np.ndarray, H: np.ndarray):
    """Compute explained variance of all dimensions."""
    V_hat = np.dot(W, H)
    rss = np.sum(np.asarray(V_hat - V)**2)
    evar_all = 1. - rss / (V*V).sum()
    return evar_all

def compute_evar_indiv(V: np.ndarray, W: np.ndarray, H: np.ndarray, d: int):
    """Compute explained variance of a dimensions."""
    V_hat_d = np.outer(W[:, d], H[d, :])
    rss = np.sum(np.asarray(V_hat_d - V)**2)
    evar_indiv = 1. - rss / (V*V).sum()
    return evar_indiv

def compute_evar_unique(V: np.ndarray, W: np.ndarray, H: np.ndarray, d: int, evar_all: float):
    """Compute unique explained variance of a dimensions."""
    V_hat_wo_d = np.dot(W[:, np.arange(W.shape[1]) != d], H[np.arange(H.shape[0]) != d, :])
    rss = np.sum(np.asarray(V_hat_wo_d - V)**2)
    evar_rest = 1. - rss / (V*V).sum()
    evar_unique = evar_all - evar_rest
    return evar_unique


# --------------------------------- Plotting --------------------------------- #
def truncate_cmap(cmap, min_val=0.0, max_val=1.0):
    n_colors = cmap.N // 2
    cmap_trunc = mcolors.LinearSegmentedColormap.from_list(
        "truncated({},{:.2f},{:.2f})".format(cmap.name, min_val, max_val),
        cmap(np.linspace(min_val, max_val, n_colors))
    )
    return cmap_trunc

def cmap_thresh_div(cmap, threshold=None):
    colors = cmap(np.linspace(0, 1, 500))
    if threshold is not None:
        alpha_values = np.ones(len(colors))
        limit = int((threshold) * len(alpha_values)/2)
        lower_limit = int(len(alpha_values)/2 - limit)
        upper_limit = int(len(alpha_values)/2 + limit)
        alpha_values[lower_limit:upper_limit] = 0
        colors[:, -1] = alpha_values
    return mcolors.ListedColormap(colors)

def zoom_to_roi_combined(subject, roi):
    roi_verts = cortex.get_roi_verts(subject, roi)[roi]
    roi_map = cortex.Vertex.empty(subject)
    roi_map.data[roi_verts] = 1
    (lflatpts, lpolys), (rflatpts, rpolys) = cortex.db.get_surf(subject, 'flat', nudge=True)
    left_roi_pts = lflatpts[np.nonzero(roi_map.left)[0], :2]
    right_roi_pts = rflatpts[np.nonzero(roi_map.right)[0], :2]
    all_roi_pts = np.vstack((left_roi_pts, right_roi_pts))
    xmin, ymin = all_roi_pts.min(0)
    xmax, ymax = all_roi_pts.max(0)
    plt.axis([xmin, xmax, ymin, ymax])

def create_collage(imgs, size):
    collage = np.zeros((size[0] * imgs[0].shape[0], size[1] * imgs[0].shape[1], 3))
    for j in range(size[1]):
        for i in range(size[0]):
            idx = j * size[0] + i
            if idx >= len(imgs):
                break
            collage[i*imgs[0].shape[0]:(i+1)*imgs[0].shape[0], 
                    j*imgs[0].shape[1]:(j+1)*imgs[0].shape[1], :] = imgs[idx]
    return collage

def create_square_subplots(n_subplots, subplot_size):
    n_rows = int(np.sqrt(n_subplots))
    n_cols = int(np.ceil(n_subplots / n_rows))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=[n_cols*subplot_size, n_rows*subplot_size])
    axes = np.array(axes).flatten()
    for i in range(n_subplots, n_rows * n_cols):
        fig.delaxes(axes[i])
    return fig, axes

def load_dim_labels(roi):
    df_dim_labels = pd.read_csv(GROUP_DIR / "interpret" / "dim_labels.csv", delimiter=",")
    df_roi_labels = df_dim_labels[df_dim_labels["roi"] == roi]
    labels = list(df_roi_labels["label"])
    return labels
