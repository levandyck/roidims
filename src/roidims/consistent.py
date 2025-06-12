import numpy as np
from itertools import combinations, product

from roidims.config import GROUP_DIR
from roidims.utils import (
    SubjectLoader,
    compute_evar_all,
    update_npz
)

# ------------------- Find consistent dims across subjects ------------------- #
def load_W_train_shared(subjects: list, roi: str):
    """Concatenate Ws for shared images across individual subjects."""
    Ws = []
    for subject in subjects:
        sub = SubjectLoader(subject)
        shared_ids = sub.load_shared_ids(train=True)
        W = np.load(sub.bnmf_dir / "W_train.npz")[roi][shared_ids]
        W = W / np.linalg.norm(W, axis=0, keepdims=True) # Apply L2 normalization across shared images
        Ws.append(W)
    return Ws

def compute_pairwise_corrs(Ws: list):
    """Compute pairwise correlations of individual dimensions across subjects."""
    n_subjects = len(Ws)
    n_samples = Ws[0].shape[0]

    # Precompute means and stds
    means = [W.mean(axis=0) for W in Ws]
    stds = [W.std(axis=0) for W in Ws]

    # Compute pairwise correlation matrices
    pair_corr_mats = {}
    for (a, b) in combinations(range(n_subjects), 2):
        Wa = Ws[a]
        Wb = Ws[b]
        Wa_centered = Wa - means[a]
        Wb_centered = Wb - means[b]
        numer = (Wa_centered[:, :, None] * Wb_centered[:, None, :]).sum(axis=0) / (n_samples - 1)
        denom = (stds[a][:, None] * stds[b][None, :])
        corr_mat = numer / denom
        pair_corr_mats[(a, b)] = corr_mat
    return pair_corr_mats

def match_dims(Ws: list, pair_corr_mats: dict, alpha: float=0.5):
    """Match corresponding dimensions based on a composite score of min and mean pairwise correlations."""
    n_subjects = len(Ws)
    dim_counts = [W.shape[1] for W in Ws]
    pairs = list(combinations(range(n_subjects), 2))

    # Generate all combinations of dims
    dim_combs = product(*[range(dc) for dc in dim_counts])
    comb_scores = []
    for dim_comb in dim_combs:
        pair_values = []

        # Gather correlations from each subject pair for specific dims
        for (a, b) in pairs:
            corr_mat = pair_corr_mats[(a, b)]
            pair_values.append(corr_mat[dim_comb[a], dim_comb[b]])
        pair_values = np.array(pair_values)
        min_corr = np.min(pair_values)
        mean_corr = np.mean(pair_values)

        # Composite score
        composite_score = alpha * min_corr + (1 - alpha) * mean_corr
        comb_scores.append((dim_comb, min_corr, mean_corr, composite_score))

    # Sort by composite score (descending)
    comb_scores.sort(key=lambda x: x[3], reverse=True)

    # Greedy selection: choose highest-score combos that don't reuse dim in any subject
    covered_dims = [set() for _ in range(n_subjects)]
    combs_selected = []
    for dim_comb, min_corr, mean_corr, comp_score in comb_scores:
        if all((dim not in covered_dims[s] for s, dim in enumerate(dim_comb))):
            combs_selected.append(dim_comb)
            for s, dim in enumerate(dim_comb):
                covered_dims[s].add(dim)
    return combs_selected

def find_consistent_dims(subjects: list, roi: str, r_thresh: float):
    """Aggregate dimensions across subjects."""
    Ws = load_W_train_shared(subjects, roi) # L2 normalization
    pair_corr_mats = compute_pairwise_corrs(Ws)

    # Match dims
    match_ids = match_dims(Ws, pair_corr_mats, alpha=0.5)

    # Get all pairwise correlations for matched dims
    n_subjects = len(subjects)
    pairs = list(combinations(range(n_subjects), 2))
    subj_corr_all_list = []
    for combo in match_ids:
        pair_values = []
        for (a, b) in pairs:
            corr_mat = pair_corr_mats[(a, b)]
            pair_values.append(corr_mat[combo[a], combo[b]])
        subj_corr_all_list.append(pair_values)
    subj_corr_all = np.array(subj_corr_all_list).T
    update_npz(GROUP_DIR / "consistency" / "match_ids.npz", {roi: match_ids})
    update_npz(GROUP_DIR / "consistency" / "corrs.npz", {roi: subj_corr_all})

    # Select consistent dims based on threshold
    means = subj_corr_all.mean(axis=0)
    dims_con_ids = [idx for idx, mean in enumerate(means) if mean > r_thresh]
    for s, subject in enumerate(subjects):
        sub = SubjectLoader(subject)
        V_train = sub.load_resp(roi, set="train")

        dims_con = [match_ids[d][s] for d in dims_con_ids]
        update_npz(sub.bnmf_dir / "dims_con.npz", {roi: dims_con})

        if dims_con:
            W_train = np.load(sub.bnmf_dir / "W_train.npz", allow_pickle=True)[roi]
            W_train_con = W_train[:, dims_con]
            update_npz(sub.bnmf_dir / "W_train_con.npz", {roi: W_train_con})

            H_train = np.load(sub.bnmf_dir / "H_train.npz", allow_pickle=True)[roi]
            H_train_con = H_train[dims_con, :]
            update_npz(sub.bnmf_dir / "H_train_con.npz", {roi: H_train_con})

            print(f"Consistent rank R2: {compute_evar_all(V_train, W_train_con, H_train_con)}")

    print(f"Aggregated consistent dimensions between subjects for {roi}.")
