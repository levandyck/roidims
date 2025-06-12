import pickle as pkl
import numpy as np
import pandas as pd
import nibabel as nib
import cortex
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
import matplotlib.ticker as ticker
import seaborn as sns
from wordcloud import WordCloud
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from roidims.config import SUBJ_DIR, GROUP_DIR, FIG_DIR
from roidims.bipolar import hotcold # https://github.com/endolith/bipolar-colormap
from roidims.utils import (
    SubjectLoader,
    create_collage,
    create_square_subplots,
    load_dim_labels,
    truncate_cmap,
    cmap_thresh_div,
    zoom_to_roi_combined,
    hoyer_sparseness
)

# ------------------------------ Global settings ----------------------------- #
CM2IN = 1 / 2.54
ROI_COLORS = {"FFA": "#D72638", "EBA": "#FBC02D", "PPA": "#005BB5"}
CMAP_HOT_COLD = hotcold(500, neutral=0.0)
CMAP_HOT = truncate_cmap(CMAP_HOT_COLD, min_val=0.5, max_val=1.0)
CMAP_HOT_COLD_THRESH = cmap_thresh_div(CMAP_HOT_COLD, threshold=0.10)

# Matplotlib defaults
params = {
    "svg.fonttype": "none",
    "axes.titlesize": 10,
    "axes.labelsize": 8,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "axes.titlepad": 1,
    "axes.labelpad": 1,
    "font.size": 10,
    "legend.fontsize": 8,
    "figure.dpi": 300
}
plt.rcParams.update(params)


# ------------------------------- MAIN FIGURES ------------------------------- #
def fig_consistency(rois: list, r_thresh: float):
    """Plot consistency of dimensions across subjects."""
    data = np.load(GROUP_DIR / "consistency" / "corrs.npz")
    fig, axes = plt.subplots(1, len(rois), figsize=(len(rois)*6*CM2IN, 6.5*CM2IN), squeeze=False)

    for r, roi in enumerate(rois):
        labels = load_dim_labels(roi)
        corrs = data[roi]
        mean_corrs = corrs.mean(axis=0)
        mask = mean_corrs > r_thresh
        corr_sel = corrs[:, mask]
        means = corr_sel.mean(axis=0)
        sems = corr_sel.std(axis=0) / np.sqrt(corr_sel.shape[0])

        # Plot
        ax = axes[0, r]
        ax.barh(
            np.arange(len(means)),
            means,
            xerr=sems,
            color=ROI_COLORS[roi],
            error_kw={"elinewidth": 0.8, "capsize": 0},
        )
        ax.set(
            title=roi,
            xlabel="Consistency (r)",
            xlim=(0, 0.8),
            xticks=np.arange(0, 1, 0.2),
            yticks=np.arange(len(means)),
            yticklabels=np.array(labels)[mask],
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.invert_yaxis()

        m, sd = means.mean(), means.std()
        print(f"{roi}: M={m:.2f}, SD={sd:.2f}")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_consistency.pdf")
    plt.show()

def fig_top_imgs(subjects: list, rois: list, n_top: int=5):
    """Plot collages of highest-scoring images combined across subjects."""
    for roi in rois:
        # Initialize dim count using first subject
        sub0 = SubjectLoader(subjects[0])
        W0 = np.load(sub0.bnmf_dir / "W_train_con.npz", allow_pickle=True)[roi]
        n_dims = W0.shape[1]

        # Gather top images across subjects
        top_imgs: dict[int, list[np.ndarray]] = {d: [] for d in range(n_dims)}
        for subj in subjects:
            sub = SubjectLoader(subj)
            imgs = sub.load_imgs(type=type)
            W = np.load(sub.bnmf_dir / "W_train_con.npz", allow_pickle=True)[roi]
            for d in range(n_dims):
                idxs = np.argsort(W[:, d])[-n_top:][::-1]
                for idx in idxs:
                    img = np.transpose(imgs[..., idx]) / 255.0
                    top_imgs[d].append(img)

        # Plot
        fig, axes = create_square_subplots(n_dims, 3)
        for d in range(n_dims):
            collage = create_collage(top_imgs[d], (n_top, len(subjects)))
            ax = axes[d]
            ax.imshow(collage)
            ax.axis("off")
            ax.set_title(f"Dim {d+1}", fontsize=12)

        fig.suptitle(roi, y=0.98, fontsize=18)
        fig.tight_layout(pad=1)
        output = FIG_DIR / "top_imgs" / f"fig_top_imgs_{roi}.pdf"
        fig.savefig(output)
        plt.show()

def fig_wordclouds(subjects: list, rois: list, n_top: int=10):
    """Plot word clouds of highest-scoring labels combined across subjects."""
    # Collect label scores
    combined: dict[tuple[str, int], dict[str, list[float]]] = {}
    for subj in subjects:
        path = GROUP_DIR / "interpret" / "wordclouds" / f"labels_sorted_{subj}.pkl"
        with open(path, "rb") as f:
            sorted_labels = pkl.load(f)
        for roi in rois:
            for d, entry in enumerate(sorted_labels[roi]):
                key = (roi, d)
                combined.setdefault(key, {})
                for label, score in entry["labels_sorted"]:
                    combined[key].setdefault(label, []).append(score)

    # Plot
    for roi in sorted({r for r, _ in combined}):
        dims = [(r, d) for r, d in combined if r == roi]
        n_dims = len(dims)
        cols = 3
        rows = int(np.ceil(n_dims / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
        for i, (r, d) in enumerate(dims):
            freqs = {lbl: np.mean(scores) for lbl, scores in combined[(r, d)].items()}
            wc = WordCloud(
                width=800,
                height=800,
                background_color="white",
                max_words=n_top,
                color_func=lambda *args, **kwargs: ROI_COLORS.get(roi, "black")
            ).generate_from_frequencies(freqs)
            ax = axes.flatten()[i]
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(f"Dim {d+1}", fontsize=8)

        # Remove empty axes
        for j in range(i+1, rows * cols):
            fig.delaxes(axes.flatten()[j])

        fig.tight_layout()
        output = FIG_DIR / "wordclouds" / f"fig_wordclouds_{roi}.pdf"
        fig.savefig(output, dpi=300)
        plt.show()


    # Collect all labels and scores across subjects
    labels_all = {}
    for subject in subjects:
        with open(GROUP_DIR / "interpret" / "wordclouds" / f"labels_sorted_{subject}.pkl", "rb") as f:
            labels_sorted = pkl.load(f)

        for roi in rois:
            n_dims = len(labels_sorted[roi])

            for d in range(n_dims):
                key = (roi, d)
                if key not in labels_all:
                    labels_all[key] = {}

                # Collect scores for each label
                labels_sorted_dim = labels_sorted[roi][d]["labels_sorted"]
                for label, score in labels_sorted_dim:
                    label_str = str(label)
                    if label_str not in labels_all[key]:
                        labels_all[key][label_str] = []
                    labels_all[key][label_str].append(score)

    # Word cloud per dim
    for roi in set(key[0] for key in labels_all.keys()):
        roi_dims = [key for key in labels_all.keys() if key[0] == roi]
        n_dims = len(roi_dims)

        n_cols = 3
        n_rows = int(np.ceil(n_dims / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=[n_cols*3, n_rows*3])
        for i, key in enumerate(roi_dims):
            roi, d = key
            label_scores = {label: np.mean(scores) for label, scores in labels_all[key].items()} # average scores

            roi_color = ROI_COLORS.get(roi, "lightgrey")
            def color_func(*args, **kwargs):
                return roi_color

            wordcloud = WordCloud(
                width=1000,
                height=1000,
                margin=0,
                max_words=n_top,
                max_font_size=100,
                min_font_size=10,
                background_color="white",
                prefer_horizontal=1,
                colormap=None,
                color_func=color_func
            ).generate_from_frequencies(label_scores)
            ax = axes.flatten()[i]
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(f"Dim{d+1}", fontsize=7)
            ax.set_box_aspect(1)

        for j in range(len(roi_dims), n_rows * n_cols):
            fig.delaxes(axes.flatten()[j])

        plt.tight_layout()
        plt.savefig(FIG_DIR / "wordclouds_comb" / f"wordclouds_comb_{roi}.pdf", dpi=900)
        plt.show()

def fig_dprime_vs_beta(subjects: list, rois: list):
    """Plot correlation between category selectivity (d') and encoding betas per dimension."""
    name_map = {"FFA": "faces", "EBA": "bodies", "PPA": "places"}
    fig, axes = plt.subplots(1, len(rois), figsize=(len(rois)*6*CM2IN, 6.5*CM2IN), squeeze=False)

    for r, roi in enumerate(rois):
        floc = name_map.get(roi, roi)

        # Determine dims
        sub0 = SubjectLoader(subjects[0])
        betas0 = nib.load(sub0.encoding_dir / f"betas_{subjects[0]}_{roi}.nii.gz").get_fdata()
        n_dims = betas0.shape[3]
        labels = load_dim_labels(roi)

        # Compute means and SEMs
        means = np.zeros(n_dims)
        sems = np.zeros(n_dims)
        for d in range(n_dims):
            rs = []
            for subj in subjects:
                sub = SubjectLoader(subj)
                mask = sub.load_roi_mask(roi, ids_ref="volume")[1]
                dprime = nib.load(sub.roi_mask_dir / f"dprime_{floc}.nii.gz").get_fdata().flatten()[mask]
                beta = nib.load(sub.encoding_dir / f"betas_{subj}_{roi}.nii.gz").get_fdata()[..., d].flatten()[mask]
                if dprime.size > 1:
                    rs.append(np.corrcoef(dprime, beta)[0, 1])
            if rs:
                means[d] = np.mean(rs)
                sems[d] = np.std(rs, ddof=1) / np.sqrt(len(rs))

        # Sort dims by effect
        order = np.argsort(means)[::-1]
        sorted_means = means[order]
        sorted_sems = sems[order]
        sorted_labels = np.array(labels)[order]

        # Plot
        ax = axes[0, r]
        ax.barh(
            np.arange(n_dims),
            sorted_means,
            xerr=sorted_sems,
            color=ROI_COLORS[roi],
            error_kw={"elinewidth": 0.8, "capsize": 0},
        )
        ax.set(
            title=roi,
            xlabel="Selectivity (r)",
            yticks=np.arange(n_dims),
            yticklabels=sorted_labels,
        )
        ax.axvline(0, ls="--", lw=1, color='black')
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_dprime_vs_beta.pdf")
    plt.show()



    floc_name_map = {"FFA": "faces", "EBA": "bodies", "PPA": "places"}

    n_rows, n_cols = 1, len(rois)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=[n_cols*6*CM2IN, n_rows*6.5*CM2IN], squeeze=False)
    for r, roi in enumerate(rois):
        floc_name = floc_name_map.get(roi, roi)

        # Get number of dims
        sub0 = SubjectLoader(subjects[0])
        betas0 = nib.load(sub0.encoding_dir / f"betas_{subjects[0]}_{roi}.nii.gz").get_fdata()
        n_dims = betas0.shape[3]

        # Get labels and colors for each dimension
        dim_labels = np.array(load_dim_labels(roi))
        mean_corrs = np.zeros(n_dims)
        sem_corrs = np.zeros(n_dims)

        for d in range(n_dims):
            subject_corrs = []
            for subject in subjects:
                sub = SubjectLoader(subject)

                # Load d' and beta for ROI
                vox_ids = sub.load_roi_mask(roi, ids_ref="volume")[1]
                dprime = nib.load(sub.roi_mask_dir / f"dprime_{floc_name}.nii.gz").get_fdata()
                betas = nib.load(sub.encoding_dir / f"betas_{subject}_{roi}.nii.gz").get_fdata()
                dprime_vals = dprime.flatten()[vox_ids]
                beta_vals = betas[..., d].flatten()[vox_ids]

                # Compute Pearson correlation
                if len(dprime_vals) > 1:
                    r_value = np.corrcoef(dprime_vals, beta_vals)[0, 1]
                    subject_corrs.append(r_value)

            # Compute mean and SEM across subjects
            if subject_corrs:
                mean_corrs[d] = np.mean(subject_corrs)
                sem_corrs[d] = np.std(subject_corrs, ddof=1) / np.sqrt(len(subject_corrs)) if len(subject_corrs) > 1 else 0
            else:
                mean_corrs[d] = 0
                sem_corrs[d] = 0

        floc_name = "Scene" if floc_name == "places" else floc_name
        if floc_name == "faces":
            floc_name = "Face"
        elif floc_name == "bodies":
            floc_name = "Body"
        elif floc_name == "places":
            floc_name = "Scene"

        # Sort dimensions by descending mean correlations
        sort_ids = np.argsort(mean_corrs)[::-1]
        mean_corrs_sorted = mean_corrs[sort_ids]
        sem_corrs_sorted = sem_corrs[sort_ids]
        dim_labels_sorted = dim_labels[sort_ids]

        x = np.arange(n_dims)
        ax = axes.flatten()[r]
        ax.barh(x, mean_corrs_sorted,
                xerr=sem_corrs_sorted,
                color=ROI_COLORS[roi],
                align="center",
                error_kw={"elinewidth": 0.8, "capsize": 0})
        ax.set_title(roi)
        ax.axvline(0, ls="--", color="black", lw=1)
        ax.set_xlabel(f"{floc_name} selectivity (r)")
        ax.set_yticks(x)
        ax.set_yticklabels(dim_labels_sorted)
        ax.spines[["right", "top"]].set_visible(False)
        ax.invert_yaxis()

    fig.tight_layout()
    plt.savefig(FIG_DIR / "fig_dprime_vs_beta.pdf", dpi=300)
    plt.show()

def fig_flatmaps_encoding_r2(subject: str, rois: list):
    """Plot flatmaps of prediction performance."""
    for roi in rois:
        perf_r = nib.load(SUBJ_DIR / subject / "encoding" / f"perf_r_{subject}_{roi}.nii.gz").get_fdata()
        perf_r2 = perf_r ** 2
        pvals_corr = nib.load(SUBJ_DIR / subject / "encoding" / f"pvals_corr_{subject}_{roi}.nii.gz").get_fdata()
        perf_r2 = np.where(pvals_corr < 0.01, perf_r2, np.nan)

        # Create volume
        volume = cortex.Volume(perf_r2.swapaxes(0, -1),
                                subject,
                                xfmname="auto-align",
                                vmin=np.nanmin(perf_r2),
                                vmax=np.nanpercentile(perf_r2, 99.99),
                                cmap=CMAP_HOT)

        # Create flatmap
        fig = plt.figure(figsize=(15*CM2IN, 8*CM2IN))
        fig = cortex.quickshow(volume,
                                pixelwise=True,
                                with_curvature=True,
                                with_sulci=False,
                                with_rois=False,
                                with_labels=True,
                                colorbar_location="center",
                                fig=fig)

        # Add other ROIs
        roi_list = ["FFA", "PPA", "EBA", "OFA", "aTL-faces", "OPA", "RSC", "FBA", "EVC"]
        for roi_ in roi_list:
            if roi_ != roi:
                target_color = ROI_COLORS.get(roi_, "white")
                _ = cortex.quickflat.composite.add_rois(fig,
                                                        volume,
                                                        roi_list=[roi_],
                                                        with_labels=True,
                                                        linewidth=4,
                                                        linecolor=target_color,
                                                        labelcolor="white",
                                                        labelsize=30)

        # Add seed ROI
        seed_color = ROI_COLORS.get(roi, "white")
        _ = cortex.quickflat.composite.add_rois(fig,
                                                volume,
                                                roi_list=[roi],
                                                with_labels=True,
                                                linewidth=6,
                                                linecolor=seed_color,
                                                labelcolor="white",
                                                labelsize=30)

        # Colorbar
        cbar_ax = fig.axes[-1]
        cbar_ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        cbar_ax.tick_params(axis="x", labelsize=5)
        cbar_ax.set_title("R²", fontsize=7, pad=2)

        plt.savefig(SUBJ_DIR / subject / "flatmaps" / "r2" / f"flatmap_{subject}_{roi}_r2.pdf", dpi=600)

def fig_r2_bar(subjects: list, rois: list):
    """Plot grouped bar chart of adjusted R2 across target ROIs for each seed ROI."""
    target_masks = ["Early", "Ventral", "Lateral", "Parietal",
                    "FFA", "OFA", "EBA", "FBA", "PPA", "OPA", "RSC"]
    
    # Compute means and SEMs
    data = {seed: {m: [] for m in target_masks} for seed in rois}
    for seed in rois:
        for subj in subjects:
            sub = SubjectLoader(subj)
            perf = nib.load(sub.encoding_dir / f"perf_r_{subj}_{seed}.nii.gz").get_fdata()
            perf_r2 = np.clip(perf**2, 0, 1)
            ncsnr = nib.load(sub.resp_dir / "ncsnr_test.nii.gz").get_fdata()
            ncsnr = (ncsnr**2) / ((ncsnr**2) + (1/3))
            adj = np.divide(perf_r2, ncsnr, out=np.full_like(perf_r2, np.nan), where=(ncsnr != 0))
            adj = np.clip(adj, 0, 1)
            for mask in target_masks:
                vox = adj.flatten()[sub.load_roi_mask(mask, "volume")[1]]
                vox = vox[~np.isnan(vox)]
                data[seed][mask].append(np.nanmean(vox) if vox.size else np.nan)
    means = {seed: [np.nanmean(data[seed][mask]) for mask in target_masks] for seed in rois}
    sems = {seed: [np.nanstd(data[seed][mask], ddof=1)/np.sqrt(len([v for v in data[seed][mask] if not np.isnan(v)])) if len([v for v in data[seed][mask] if not np.isnan(v)])>1 else 0 for mask in target_masks] for seed in rois}

    # Plot
    positions = np.arange(len(target_masks))
    width = 0.6 / len(rois)
    fig, ax = plt.subplots(figsize=(15*CM2IN, 5*CM2IN))
    for i, seed in enumerate(rois):
        pos = positions - 0.3 + i*width + width/2
        ax.bar(pos,
                means[seed],
                width=width,
                yerr=sems[seed],
                capsize=0,
                color=ROI_COLORS.get(seed),
                label=seed,
                error_kw={"elinewidth":0.8}
                )
    ax.set_xticks(positions)
    ax.set_xticklabels(target_masks, rotation=45, ha="right")
    ax.set_ylabel("Prediction performance (R² adj.)")
    ax.legend(title="Seed ROI", loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_r2_bar.pdf", dpi=300)
    plt.show()


    # Define target ROI masks
    target_masks = ["Early", "Ventral", "Lateral", "Parietal",
                    "FFA", "OFA", "EBA", "FBA", "PPA", "OPA", "RSC"]

    data = {seed: {target: [] for target in target_masks} for seed in rois}
    # Compute noise-corrected R2
    for seed_roi in rois:
        for subj in subjects:
            sub = SubjectLoader(subj)

            # Load R2 for seed ROI
            perf_r = nib.load(sub.encoding_dir / f"perf_r_{subj}_{seed_roi}.nii.gz").get_fdata()
            perf_r2 = perf_r ** 2
            perf_r2 = np.clip(perf_r2, 0, 1)

            # Get noise ceiling estimate and compute noise-ceiling corrected R2
            ncsnr = nib.load(sub.resp_dir / "ncsnr_test.nii.gz").get_fdata()
            ncsnr = (ncsnr**2) / ((ncsnr**2) + (1/3))
            perf_r2_adj = np.divide(
                perf_r2,
                ncsnr,
                out=np.full_like(perf_r2, np.nan),
                where=(ncsnr != 0)
            )
            perf_r2_adj = np.clip(perf_r2_adj, 0, 1)

            # For each target ROI mask, compute average R2 values
            for target in target_masks:
                mask = sub.load_roi_mask(target, "volume")[1]
                valid_vox = perf_r2_adj.flatten()[mask]
                valid_vox = valid_vox[~np.isnan(valid_vox)]
                if valid_vox.size > 0:
                    # Average across voxels
                    subject_avg = np.nanmean(valid_vox)
                    data[seed_roi][target].append(subject_avg)
                else:
                    data[seed_roi][target].append(np.nan)

    seed_rois = rois

    means = {seed: [] for seed in seed_rois}
    sems = {seed: [] for seed in seed_rois}

    for target in target_masks:
        for seed in seed_rois:
            subject_vals = np.array(data[seed][mask])
            subject_vals = subject_vals[~np.isnan(subject_vals)]
            if subject_vals.size > 0:
                mean_val = np.nanmean(subject_vals)
                sem_val = np.nanstd(subject_vals, ddof=1) / np.sqrt(subject_vals.size)
            else:
                mean_val = np.nan
                sem_val = np.nan
            means[seed].append(mean_val)
            sems[seed].append(sem_val)

    # Create grouped bar chart
    n_groups = len(target_masks)
    n_seeds = len(seed_rois)
    group_positions = np.arange(n_groups)
    bar_width = 0.6 / n_seeds

    fig, ax = plt.subplots(figsize=(15*CM2IN, 5*CM2IN))
    for i, seed in enumerate(seed_rois):
        bar_positions = group_positions - 0.3 + i * bar_width + bar_width/2
        color = ROI_COLORS.get(seed, "gray")
        ax.bar(bar_positions, means[seed], width=bar_width, yerr=sems[seed],
                capsize=0, label=seed, color=color, error_kw={"elinewidth": 0.8})

    ax.set_xticks(group_positions)
    ax.set_xticklabels(target_masks, rotation=45, ha="right")
    ax.set_ylabel("Prediction performance (R² adj.)")
    ax.legend(title="Dimensions", title_fontsize=6, loc="center left", bbox_to_anchor=(1.05, 0.5))
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_r2_bar.pdf", dpi=300)
    plt.show()

def fig_flatmaps_encoding_betas(subject: str, rois: list):
    """Plot flatmaps of tuning maps for each dimension."""
    for roi in rois:
        betas = nib.load(SUBJ_DIR / subject / "encoding", f"betas_{subject}_{roi}.nii.gz").get_fdata()
        pvals_corr = nib.load(SUBJ_DIR / subject / "encoding", f"pvals_corr_{subject}_{roi}.nii.gz").get_fdata()
        betas = np.where((pvals_corr < 0.01)[..., np.newaxis], betas, np.nan)

        abs_max_all = max(abs(np.nanpercentile(betas, 0.01)), abs(np.nanpercentile(betas, 99.99)))
        vmin_all, vmax_all = -abs_max_all, abs_max_all

        # Plot
        n_dims = betas.shape[3]
        n_cols = 5
        n_rows = int(np.ceil(n_dims/n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=[n_cols*3.6*CM2IN, n_rows*3.6*CM2IN])
        dim_labels = np.array(load_dim_labels(roi))
        for d in range(n_dims):
            # Create volume
            betas_dim = betas[..., d]
            volume = cortex.Volume(betas_dim.swapaxes(0, -1),
                                    subject,
                                    xfmname="auto-align",
                                    vmin=vmin_all,
                                    vmax=vmax_all,
                                    cmap=CMAP_HOT_COLD_THRESH)

            # Create flatmap
            ax = axes.flatten()[d]
            cbar_loc = "center" if d == n_dims - 1 else "left"
            fig = cortex.quickflat.make_figure(volume,
                                                pixelwise=True,
                                                with_curvature=True,
                                                with_sulci=False,
                                                with_rois=False,
                                                with_labels=True,
                                                colorbar_location=cbar_loc,
                                                dpi=600,
                                                fig=ax)

            # Add other ROIs
            roi_list = ["FFA", "PPA", "EBA", "OFA", "aTL-faces", "OPA", "RSC", "FBA", "EVC"]
            for roi_ in roi_list:
                if roi_ != roi:
                    target_color = ROI_COLORS.get(roi_, "white")
                    _ = cortex.quickflat.composite.add_rois(fig,
                                                            volume,
                                                            roi_list=[roi_],
                                                            with_labels=True,
                                                            linewidth=6,
                                                            linecolor=target_color,
                                                            labelcolor="white",
                                                            labelsize=35)

            # Add seed ROI
            seed_color = ROI_COLORS.get(roi, "white")
            _ = cortex.quickflat.composite.add_rois(fig,
                                                    volume,
                                                    roi_list=[roi],
                                                    with_labels=True,
                                                    linewidth=8,
                                                    linecolor=seed_color,
                                                    labelcolor="white",
                                                    labelsize=35)

            zoom_to_roi_combined(subject, roi="zoom")
            ax.set_box_aspect(1)
            ax.set_title(dim_labels[d], fontsize=8, y=0.9)

        # Colorbar
        cbar_ax = fig.axes[-1]
        cbar_ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        cbar_ax.tick_params(axis="x", labelsize=5)
        cbar_ax.set_title("β", fontsize=7, pad=2)

        for i in range(n_dims, n_rows * n_cols):
            fig.delaxes(axes.flatten()[i])
        plt.tight_layout(pad=0.03, w_pad=0.03, h_pad=0.03)
        plt.savefig(SUBJ_DIR / subject / "flatmaps" / "betas" / f"flatmap_{subject}_{roi}_betas.pdf", dpi=600)


# --------------------------- SUPPLEMENTARY FIGURES -------------------------- #
def suppfig_k_optim(subjects: list, rois: list):
    """Plot bi-cross-validation MSE and mark optimal dimensionality."""
    fig, axes = plt.subplots(1, len(rois), figsize=(len(rois)*2*CM2IN, 2.4*CM2IN), squeeze=False)
    for r, roi in enumerate(rois):
        ax = axes[0, r]
        palette = sns.light_palette(ROI_COLORS[roi], n_colors=len(subjects)+1)[1:]
        for subj_idx, subj in enumerate(subjects):
            sub = SubjectLoader(subj)
            metrics = np.load(sub.bcv_dir / f"bcv_metrics_{roi}.npz", allow_pickle=True)
            ks = metrics["ks"]
            errs = metrics["test_errs"]
            k_opt = metrics["k_optim_subj"]
            errs_norm = (errs - errs.min()) / (errs[0] - errs.min())

            # Plot
            ax.plot(ks, errs_norm, lw=1, color=palette[subj_idx], label=subj)
            ax.plot(k_opt, -0.05, marker="*", markersize=5, color=palette[subj_idx])

        ax.set(
            title=roi,
            xlabel="k",
            ylabel="Norm. MSE",
            xticks=np.arange(10, ks[-1] + 1, 10),
            ylim=(-0.1, 1),
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_box_aspect(1)

    # Legend
    grey_colors = sns.light_palette("#808080", n_colors=4)
    _, labels = ax.get_legend_handles_labels()
    grey_handles = [plt.Line2D([0], [0], color=grey_colors[s], lw=1, ms=1) for s in range(len(subjects))]
    fig.legend(grey_handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0), ncol=len(subjects), prop={"size": 5})

    fig.tight_layout(pad=1)
    fig.savefig(FIG_DIR / "suppfig_k_optim.pdf", dpi=300)
    plt.show()

def suppfig_consistency(rois: list, r_thresh: float):
    """Plot consistency of dimensions across subjects."""
    data = np.load(GROUP_DIR / "consistency" / "corrs.npz")
    fig, axes = plt.subplots(1, len(rois), figsize=(len(rois)*2*CM2IN, 2.4*CM2IN), squeeze=False)
    for r, roi in enumerate(rois):
        corrs = data[roi]
        means = corrs.mean(axis=0)

        ax = axes[0, r]
        ax.bar(np.arange(len(means)), means, color=ROI_COLORS[roi])
        ax.axhline(r_thresh, ls="--", color="black", lw=1)
        ax.set(
            title=roi,
            xlabel="Dimension",
            ylabel="Consistency",
            ylim=(0, 0.85),
            yticks=np.arange(0, 1, 0.2),
        )
        ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_box_aspect(1)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "suppfig_consistency.pdf", dpi=300)
    plt.show()

def suppfig_sim_dims(subjects: list, rois: list, mean_center: bool=False):
    """Plot averaged cosine similarity matrices of dimensions."""
    # Compute cosine similarity matrices
    sim_mats = {roi: [] for roi in rois}
    for subj in subjects:
        sub = SubjectLoader(subj)
        Ws = np.load(sub.bnmf_dir / "W_train_con.npz", allow_pickle=True)
        for roi in rois:
            W = Ws[roi]
            if mean_center:
                W = W - W.mean(axis=1, keepdims=True)
            sim_mats[roi].append(cosine_similarity(W.T))
    avg = {roi: np.mean(mats, axis=0) for roi, mats in sim_mats.items()}
    cmaps = {
        "FFA": LinearSegmentedColormap.from_list("w_r", ["white", ROI_COLORS["FFA"]]),
        "EBA": LinearSegmentedColormap.from_list("w_y", ["white", ROI_COLORS["EBA"]]),
        "PPA": LinearSegmentedColormap.from_list("w_b", ["white", ROI_COLORS["PPA"]])
    }

    # Plot
    fig, axes = plt.subplots(1, len(rois), figsize=(len(rois)*6*CM2IN, 8*CM2IN), squeeze=False)
    for r, (roi, mat) in enumerate(avg.items()):
        ax = axes[0, r]
        labels = load_dim_labels(roi)
        sns.heatmap(
            mat,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 2},
            square=True,
            cmap=cmaps.get(roi, "binary"),
            cbar_kws={"orientation": "horizontal", "shrink": 0.5, "pad": 0.15},
            xticklabels=[f"#{i+1}" for i in range(mat.shape[0])],
            yticklabels=labels,
            ax=ax
        )
        ax.set_xticks(np.arange(mat.shape[0])+0.5)
        ax.set_yticks(np.arange(mat.shape[0])+0.5)
        ax.set_title(roi)
        ax.tick_params(axis="x", rotation=90, labelsize=5)
        ax.tick_params(axis="y", rotation=0, labelsize=5)

        # Colorbar
        cbar = ax.collections[0].colorbar
        cbar.outline.set_edgecolor("black")
        cbar.outline.set_linewidth(1)
        cbar.set_ticks([cbar.vmin, cbar.vmax])
        cbar.set_ticklabels([f"{cbar.vmin:.2f}", f"{cbar.vmax:.2f}"])
        cbar.ax.xaxis.set_label_position("top")
        cbar.set_label("Cosine similarity", labelpad=4)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "suppfig_sim_dims.pdf", dpi=300)
    plt.show()

def suppfig_sparseness(subjects, rois, n_bins=3):
    """Plot sparseness of voxels binned by category selectivity."""
    # Compute sparseness
    recs = []
    floc = {"FFA": "faces", "EBA": "bodies", "PPA": "places"}
    for roi in rois:
        for subj in subjects:
            sub = SubjectLoader(subj)
            mask = sub.load_roi_mask(roi, ids_ref="volume")[1]
            if mask.size == 0:
                continue
            dprime = nib.load(sub.roi_mask_dir / f"dprime_{floc.get(roi,roi)}.nii.gz").get_fdata().ravel()[mask]
            betas = nib.load(sub.encoding_dir / f"betas_{subj}_{roi}.nii.gz").get_fdata()
            betas = betas.reshape(-1, betas.shape[3]).T[:, mask]
            sparse = hoyer_sparseness(betas, axis=0)
            recs.extend(zip([roi]*len(mask), dprime.tolist(), sparse.tolist()))
    df = pd.DataFrame(recs, columns=["ROI", "dprime", "sparse"]).dropna()

    # Plot
    b_labels = ["Low", "Mid", "High"] if n_bins==3 else [f"bin{i+1}" for i in range(n_bins)]
    df["bin"] = df.groupby("ROI")["dprime"].transform(lambda x: pd.qcut(x, n_bins, labels=b_labels, duplicates="drop"))
    plt.figure(figsize=(5,3))
    ax = sns.violinplot(
        data=df,
        x="bin",
        y="sparse",
        hue="ROI",
        palette={r:ROI_COLORS[r] for r in rois},
        inner="box",
        dodge=True,
        cut=0,
        linewidth=1,
    )
    ax.set(xlabel="Category selectivity (d')", ylabel="Representational sparseness")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "suppfig_sparseness.pdf", dpi=300)
    plt.show()

def suppfig_noiseceilings(subjects: list, rois: list, sign_vox: bool=False):
    """Plot prediction performance vs. noise ceiling estimates."""
    fig, axes = plt.subplots(len(subjects), len(rois), figsize=(len(rois)*6*CM2IN, len(subjects)*6*CM2IN), squeeze=False)
    edges = np.linspace(0, 1, 76)
    cmaps = {
        "FFA": LinearSegmentedColormap.from_list("w_r", ["white",ROI_COLORS['FFA']]),
        "EBA": LinearSegmentedColormap.from_list("w_y", ["white",ROI_COLORS['EBA']]),
        "PPA": LinearSegmentedColormap.from_list("w_b", ["white",ROI_COLORS['PPA']]),
    }

    for s, subj in enumerate(subjects):
        sub = SubjectLoader(subj)
        mask_all = sub.load_brain_mask()[0]
        ncsnr_all = nib.load(sub.resp_dir / "ncsnr_test.nii.gz").get_fdata().ravel()[mask_all]
        ncsnr_all = (ncsnr_all**2)/((ncsnr_all**2)+(1/3))

        for r, roi in enumerate(rois):
            ax = axes[s, r]
            perf = nib.load(sub.encoding_dir / f"perf_r_{subj}_{roi}.nii.gz").get_fdata().ravel()[mask_all]
            perf2 = perf**2
            if sign_vox:
                p = nib.load(sub.encoding_dir / f"pvals_corr_{subj}_{roi}.nii.gz").get_fdata().ravel()[mask_all]
                idx = p<0.01
                perf2 = perf2[idx]
                ncsnr = ncsnr_all[idx]
            else:
                ncsnr = ncsnr_all

            # Plot
            hist = ax.hist2d(ncsnr, perf2, bins=[edges, edges], norm=LogNorm(), cmap=cmaps.get(roi, "binary"))
            ax.plot([0,1],[0,1],'--',color='black',lw=1)
            ax.set(
                title=roi,
                xlabel="Noise ceiling (R²)",
                ylabel="Prediction performance (R²)",
                xlim=(0,1),
                ylim=(0,1),
                xticks=np.arange(0,1.2,0.2),
                yticks=np.arange(0,1.2,0.2),
            )
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_box_aspect(1)
            cbar = plt.colorbar(hist[3], ax=ax, orientation="vertical", pad=0.1, shrink=0.3)
            cbar.ax.tick_params(labelsize=5)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "suppfig_noiseceilings.pdf", dpi=300)
    plt.show()

def suppfig_flatmaps_encoding_betas_pca(subject: str, rois: list):
    """Plot flatmaps of PCs based on tuning maps for all dimensions."""
    fig, axes = plt.subplots(1, len(rois), figsize=[len(rois)*6*CM2IN, 1*6*CM2IN], squeeze=False)
    for r, roi in enumerate(rois):
        betas = nib.load(SUBJ_DIR / subject / "encoding" / f"betas_{subject}_{roi}.nii.gz").get_fdata()
        pvals_corr = nib.load(SUBJ_DIR / subject / "encoding" / f"pvals_corr_{subject}_{roi}.nii.gz").get_fdata()

        # Perform PCA
        x, y, z, d = betas.shape
        betas_z = StandardScaler().fit_transform(betas.reshape(-1, d))
        betas_pca = PCA(n_components=3).fit_transform(betas_z).reshape(x, y, z, 3)
        betas_pca_norm = MinMaxScaler().fit_transform(betas_pca.reshape(-1, 1)).reshape(betas_pca.shape)*255

        # Create RGBA volume
        red = cortex.Volume(betas_pca_norm[:, :, :, 0].swapaxes(0, -1).astype(np.uint8), subject, "auto-align")
        green = cortex.Volume(betas_pca_norm[:, :, :, 1].swapaxes(0, -1).astype(np.uint8), subject, "auto-align")
        blue = cortex.Volume(betas_pca_norm[:, :, :, 2].swapaxes(0, -1).astype(np.uint8), subject, "auto-align")
        alpha = np.full_like(betas_pca_norm[:, :, :, 0], 255)
        alpha = np.where(pvals_corr < 0.01, alpha, 0)
        rgba_volume = cortex.VolumeRGB(red,
                                        green,
                                        blue,
                                        subject,
                                        alpha=alpha.swapaxes(0, -1),
                                        shared_vmin=0,
                                        shared_vmax=255)

        # Create flatmap
        ax = axes.flatten()[r]
        fig = cortex.quickflat.make_figure(rgba_volume,
                                            pixelwise=True,
                                            with_curvature=True,
                                            with_sulci=False,
                                            with_rois=False,
                                            with_labels=True,
                                            colorbar_location="left",
                                            dpi=600,
                                            fig=ax)

        # Add other ROIs
        roi_list = ["FFA", "PPA", "EBA", "OFA", "aTL-faces", "OPA", "RSC", "FBA", "EVC"]
        for roi_ in roi_list:
            if roi_ != roi:
                target_color = ROI_COLORS.get(roi_, "white")
                _ = cortex.quickflat.composite.add_rois(fig,
                                                        rgba_volume,
                                                        roi_list=[roi_],
                                                        with_labels=True,
                                                        linewidth=6,
                                                        linecolor=target_color,
                                                        labelcolor="white",
                                                        labelsize=35)

        # Add seed ROI
        seed_color = ROI_COLORS.get(roi, "white")
        _ = cortex.quickflat.composite.add_rois(fig,
                                                rgba_volume,
                                                roi_list=[roi],
                                                with_labels=True,
                                                linewidth=8,
                                                linecolor=seed_color,
                                                labelcolor="white",
                                                labelsize=35)

        zoom_to_roi_combined(subject, roi="zoom")
        ax.set_box_aspect(1)
        ax.set_title(roi, fontsize=8, y=0.9)

    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
    plt.savefig(SUBJ_DIR / subject / "flatmaps" / "betas" / f"flatmap_{subject}_betas_pca.pdf", dpi=600)
