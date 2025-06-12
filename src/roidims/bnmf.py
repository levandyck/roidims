import numpy as np
import nimfa
import scipy.optimize
from sklearn.metrics.pairwise import cosine_similarity

from roidims.utils import (
    SubjectLoader,
    submit_parallel_jobs,
    update_npz,
    compute_evar_all
)

# ------------------ Perform randomly initialized BNMF runs ------------------ #
def fit_bnmf(V: np.ndarray, k_optim: int):
    bnmf = nimfa.Bd(V, seed="random_c", rank=k_optim, max_iter=3000, min_residuals=1e-5, alpha=np.zeros((V.shape[0], k_optim)),
                    beta=np.zeros((k_optim, V.shape[1])), theta=.0, k=.0, sigma=1., skip=1000, stride=5,
                    n_w=np.zeros((k_optim, 1)), n_h=np.zeros((k_optim, 1)), n_run=1, n_sigma=False)
    bnmf_fit = bnmf()
    W = np.array(bnmf_fit.basis())
    H = np.array(bnmf_fit.coef())
    return W, H

def perform_bnmf_runs(subjects: list, rois: list, n_runs: int):
    """Perform randomly initialized BNMF runs."""
    for subject in subjects:
        sub = SubjectLoader(subject)

        for roi in rois:
            V = sub.load_resp(roi, set="train")
            k = np.load(sub.bcv_dir / f"bcv_metrics_{roi}.npz", allow_pickle=True)["k_optim_subj"]

            args = [(V, k) for r in range(n_runs)]
            results = submit_parallel_jobs(fit_bnmf, args, joblib_kwargs={"n_jobs": -1, "verbose": 50})

            Ws = np.stack([np.asarray(result[0]) for result in results], axis=2)
            Hs = np.stack([np.asarray(result[1]) for result in results], axis=2)
            update_npz(sub.bnmf_dir / "Ws.npz", {roi: Ws})
            update_npz(sub.bnmf_dir / "Hs.npz", {roi: Hs})
            print(f"Performed BNMF runs for {subject} {roi}.")


# ------------- Project test set to learned embedding using NNLS ------------- #
def solve_nnls(v, H_train):
    """Solve NNLS."""
    W, _ = scipy.optimize.nnls(H_train.T, v)
    return W

def solve_nnls_parallel(V_test, H_train):
    """Solve NNLS in parallel."""
    args = [(V_test[i], H_train) for i in range(V_test.shape[0])]
    results = submit_parallel_jobs(solve_nnls, args, joblib_kwargs={"n_jobs": -1, "verbose": 0})
    W_test = np.array(results)
    return W_test

def project_test_nnls(subjects: list, rois: list):
    """Project test samples to learned embedding with NNLS."""
    for subject in subjects:
        sub = SubjectLoader(subject)
        Hs_train = np.load(sub.bnmf_dir / "H_train.npz", allow_pickle=True)
        dims_con = np.load(sub.bnmf_dir / "dims_con.npz", allow_pickle=True)

        for roi in rois:
            H_train = Hs_train[roi]
            dims_con_roi = dims_con[roi]

            V_test = sub.load_resp(roi, set="test")
            W_test = solve_nnls_parallel(V_test, H_train)
            print(f"Projected test set to learned embedding for {subject} {roi}.")
            print(f"V_test: Full R2 = {compute_evar_all(V_test, W_test, H_train)}")
            update_npz(sub.bnmf_dir / "W_test.npz", {roi: W_test})

            W_test_con = W_test[:, dims_con_roi]
            update_npz(sub.bnmf_dir / "W_test_con.npz", {roi: W_test_con})
            print(f"V_test: Consistent R2 = {compute_evar_all(V_test, W_test_con, H_train[dims_con_roi])}")


# ----------------------------------- Stats ---------------------------------- #
def compute_evar_consistent(subjects: list, rois: list):
    """Compute average variance explained by consistent dimensions in each ROI."""
    for roi in rois:
        r2s = []
        for subject in subjects:
            sub = SubjectLoader(subject)
            V = sub.load_resp(roi, set="train")
            W = np.load(sub.bnmf_dir / "W_train_con.npz", allow_pickle=True)[roi]
            H = np.load(sub.bnmf_dir / "H_train_con.npz", allow_pickle=True)[roi]

            r2 = compute_evar_all(V, W, H)
            r2s.append(r2)
        mean_r2 = np.mean(r2s)
        sd_r2 = np.std(r2s)
        print(f"{roi}: M={round(mean_r2, 2)}, SD={round(sd_r2, 2)}")

def cross_group_mean(sim_mat: np.ndarray, g1: list, g2: list):
    """Mean similarity across groups."""
    return sim_mat[np.ix_(g1, g2)].mean()

def within_group_mean(sim_mat: np.ndarray, g: list):
    """Mean similarity within groups."""
    block = sim_mat[np.ix_(g, g)]
    iu = np.triu_indices_from(block, k=1)
    return block[iu].mean() if iu[0].size else np.nan

def compute_sim_dims_groups(subjects: list, rois: list, mean_center: bool=False):
    """Compute mean cosine similarity between two groups of dimensions."""
    sims = {roi: {"within_g1": [], "within_g2": [], "between": []} for roi in rois}
    for subj in subjects:
        sub = SubjectLoader(subj)
        Ws = np.load(sub.bnmf_dir / "W_train_con.npz", allow_pickle=True)

        for roi in rois:
            W = Ws[roi]
            if mean_center:
                W = W - W.mean(axis=1, keepdims=True)
            sim_mat = cosine_similarity(W.T)

            if roi == "FFA":
                group1_ids = np.array([0, 2])
                group2_ids = np.setdiff1d(np.arange(8), group1_ids)
            elif roi == "EBA":
                group1_ids = np.array([0, 2, 3, 5, 6, 11, 12, 14, 15, 16, 17])
                group2_ids = np.setdiff1d(np.arange(20), group1_ids)
            elif roi == "PPA":
                group1_ids = np.array([0, 2, 5, 7, 9])
                group2_ids = np.setdiff1d(np.arange(10), group1_ids)
            else:
                print("ROI not available.")

            sims[roi]["within_g1"].append(within_group_mean(sim_mat, group1_ids))
            sims[roi]["within_g2"].append(within_group_mean(sim_mat, group2_ids))
            sims[roi]["between"].append(cross_group_mean(sim_mat, group1_ids, group2_ids))

    for roi, vals in sims.items():
        w1 = np.asarray(vals["within_g1"])
        w2 = np.asarray(vals["within_g2"])
        bt = np.asarray(vals["between"])

        print(
            f"{roi}: "
            f"within pref. M={w1.mean():.2f} (SD={w1.std(ddof=0):.2f}), "
            f"within non-pref. M={w2.mean():.2f} (SD={w2.std(ddof=0):.2f}), "
            f"between M={bt.mean():.2f} (SD={bt.std(ddof=0):.2f})"
        )
