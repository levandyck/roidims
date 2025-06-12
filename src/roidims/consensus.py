import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score

from roidims.utils import (
    SubjectLoader,
    compute_evar_all,
    update_npz
)

# ----------------- Aggregate randomly initialized BNMF runs ----------------- #
def l2_norm_dims(Ws: np.ndarray, Hs: np.ndarray):
    """L2 normalize responses and scale weights correspondingly."""
    l2_norms = np.linalg.norm(Ws, axis=2, keepdims=True)
    Ws = Ws / l2_norms
    Hs = Hs * l2_norms.transpose(0, 2, 1)
    return Ws, Hs

def remove_outliers(spectra: pd.DataFrame, n_dims: int, local_neighborhood_size: float):
    """Remove outliers from spectra."""
    # Compute pairwise cosine distances
    dist = cosine_distances(spectra.values)

    # Determine number of neighbors based on local neighborhood size
    n_neighbors = int(local_neighborhood_size * len(spectra) / n_dims)

    # Partition based on first n neighbors
    partitioning_order = np.argpartition(dist, n_neighbors+1)[:, :n_neighbors+1]

    # Calculate mean distance to nearest neighbors
    distance_to_nearest_neighbors = dist[np.arange(dist.shape[0])[:, None], partitioning_order]
    local_density = pd.DataFrame(distance_to_nearest_neighbors.sum(1) / n_neighbors,
                                columns=["local_density"],
                                index=spectra.index)

    # Set top 5th percentile as outlier threshold
    density_threshold = np.percentile(local_density.values[:, 0], 95)

    # Filter outliers based on density threshold
    density_filter = local_density.iloc[:, 0] < density_threshold
    spectra_filt = spectra.loc[density_filter]
    outlier_ids = spectra.loc[~density_filter].index
    print(f"Removed {len(outlier_ids)} outlier dimensions")
    return spectra_filt, outlier_ids

def perform_kmedoids(spectra_filt: pd.DataFrame, n_dims: int):
    """Perform k-medoids clustering."""
    kmedoids = KMedoids(n_clusters=n_dims, metric="cosine", method="pam", init="k-medoids++", max_iter=500)
    kmedoids.fit(spectra_filt)
    silhouette = silhouette_score(spectra_filt.values, kmedoids.labels_, metric="cosine")
    print(f"Silhouette score = {silhouette}")
    return kmedoids.labels_, kmedoids.cluster_centers_, kmedoids.medoid_indices_

def aggregate_W_train(sub, roi: str, Ws: np.ndarray):
    """Aggregate W consensus matrix via k-medoids clustering."""
    n_runs, n_dims, n_imgs = Ws.shape
    Ws = Ws.reshape(-1, n_imgs)
    spectra = pd.DataFrame(Ws, columns=range(n_imgs), index=["run_%d" % i for i in range(n_runs*n_dims)])

    # Detect and remove outliers based on local density
    spectra_filt = remove_outliers(spectra, n_dims, local_neighborhood_size=0.4)[1]

    # Perform k-medoids clustering
    cluster_labels = perform_kmedoids(spectra_filt, n_dims)[1]

    # Compute medians as consensus responses
    W_train = spectra_filt.groupby(cluster_labels).median().values.T
    return W_train

def aggregate_H_train(Ws: np.ndarray, Hs: np.ndarray, W_within: np.ndarray):
    """Aggregate consensus weights via cosine similarity."""
    n_runs, n_dims, n_imgs = Ws.shape
    H_train = []
    for d in range(n_dims):

        weights_runs = []
        for r in range(n_runs):

            # Compute cosine similarity between cluster median and all dims in run
            cos_sim = []
            for k in range(n_dims):
                cos_sim.append(cosine_similarity(W_within[:, d].reshape(1, -1), Ws[r, k, :].reshape(1, -1))[0][0])

            # Use coefficient weights of most similar dim
            best_idx = np.argmax(np.asarray(cos_sim))
            weights_runs.append(Hs[r, :, best_idx])

        # Compute median across runs
        weights_median = np.median(np.asarray(weights_runs), axis=0)
        H_train.append(weights_median)
    return np.asarray(H_train)

def find_consensus_dims(subject: str, roi: str):
    """Aggregate consensus dimensions within each subject."""
    sub = SubjectLoader(subject)
    V_train = sub.load_resp(roi, set="train")
    Ws = np.load(sub.bnmf_dir / "Ws.npz")[roi].T
    Hs = np.load(sub.bnmf_dir / "Hs.npz")[roi].T

    # L2 normalize dims
    Ws, Hs = l2_norm_dims(Ws, Hs)

    # Aggregate W_train
    W_train = aggregate_W_train(sub, roi, Ws)
    update_npz(sub.bnmf_dir / "W_train.npz", {roi: W_train})
    print(f"Aggregated consensus responses for {subject} {roi}.")

    # Aggregate H_train
    H_train = aggregate_H_train(Ws, Hs, W_train)
    update_npz(sub.bnmf_dir / "H_train.npz", {roi: H_train})
    print(f"Aggregated consensus weights for {subject} {roi}.")

    print(f"Full rank R2 for train set: {compute_evar_all(V_train, W_train, H_train)}")
    print(f"Aggregated consensus dims for {subject} {roi}.")
