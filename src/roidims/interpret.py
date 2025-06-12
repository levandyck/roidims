import numpy as np
import pandas as pd
import torch
import clip
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from scipy.stats import zscore
import pickle as pkl
from tqdm import tqdm

from roidims.config import GROUP_DIR
from roidims.utils import SubjectLoader

# -------- Interpret dimensions using multimodal deep learning models -------- #
def load_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)
    model.eval()
    return model, preprocess, device

def get_clip_img_embeds(subjects: list, batch_size=64):
    """Encode images of each subject in CLIP."""
    model, preprocess, device = load_clip()

    for subject in subjects:
        sub = SubjectLoader(subject)
        imgs = sub.load_imgs(type="train").transpose()

        img_embeds = []
        for idx in tqdm(range(0, len(imgs), batch_size), desc=f"Extracting image embeddings for {subject}"):
            batch_imgs = imgs[idx:idx+batch_size]
            batch_pil = [Image.fromarray((img_array*255).astype(np.uint8)) for img_array in batch_imgs]
            batch_tensors = torch.stack([preprocess(img).to(device) for img in batch_pil])
            with torch.no_grad():
                batch_embeds = model.encode_image(batch_tensors).cpu().numpy()
            img_embeds.append(batch_embeds)

        img_embeds_subj = np.vstack(img_embeds)
        np.save(GROUP_DIR / "interpret" / "img_embeds" / f"img_embeds_{subject}.npy", img_embeds_subj)

def get_clip_text_embeds(batch_size=64):
    """Encode candidate labels in CLIP."""
    model, _, device = load_clip()
    df_labels = pd.read_csv(GROUP_DIR / "interpret" / "text_embeds" / "gpt4v_topimgs_labels.csv", delimiter=",")

    # Use different prompt templates
    prompts = [
        "a photo of {}",
        "an image of {}",
        "a picture of {}",
        "a good photo of {}",
        "a detailed image of {}",
        "a high-quality picture of {}",
        "a depiction of {}",
        "a photograph of {}",
        "a snapshot of {}",
        "a representation of {}",
        "a portrayal of {}",
        "a visual of {}",
        "an image showing {}",
        "an illustration of {}",
        "an image featuring {}"
    ]

    # Get all unique candidate labels
    df_labels["label"] = df_labels["label"].str.lower()
    labels_unique = df_labels["label"].unique()

    # Generate all prompt-label combinations
    prompts_all = [prompt.format(label) for label in labels_unique for prompt in prompts]
    text_tokens = clip.tokenize(prompts_all).to(device)

    # Compute text embeddings in batches
    text_embeds = []
    for idx in tqdm(range(0, len(text_tokens), batch_size), desc=f"Encoding text embeddings combined across subjects"):
        batch_tokens = text_tokens[idx:idx+batch_size]
        with torch.no_grad():
            batch_embeds = model.encode_text(batch_tokens).cpu().numpy()
        text_embeds.append(batch_embeds)
    text_embeds = np.vstack(text_embeds)

    # Organize embeddings into dict
    text_embeds_dict = {}
    idx = 0
    for label in labels_unique:
        text_embeds_dict[label] = text_embeds[idx:idx+len(prompts)]
        idx += len(prompts)

    with open(GROUP_DIR / "interpret" / "text_embeds" / "text_embeds_comb.pkl", "wb") as f:
        pkl.dump(text_embeds_dict, f)

def compute_cosine_embeds(subjects: list, rois: list):
    """Compute cosine similarity of images and candidate labels weighted by each dimension."""
    # Load and L2 normalize text embeddings
    with open(GROUP_DIR / "interpret" / "text_embeds" / "text_embeds_comb.pkl", "rb") as f:
        text_embeds = pkl.load(f)
    labels = list(text_embeds.keys())
    for label in labels:
        text_embeds[label] = normalize(text_embeds[label], norm="l2", axis=1)

    for subject in subjects:
        # Load and L2 normalize image embeddings
        img_embeds_subject = np.load(GROUP_DIR / "interpret" / "img_embeds" / f"img_embeds_{subject}.npy")
        img_embeds_subject = normalize(img_embeds_subject, norm="l2", axis=1)

        sim_mats = {}
        z_mats = {}
        labels_sorted_all = {}
        for roi in rois:
            sub = SubjectLoader(subject)
            W = np.load(sub.bnmf_dir / "W_train_con.npz", allow_pickle=True)[roi]
            n_dims = W.shape[1]
            n_labels = len(labels)

            # Compute weighted similarities for each dimension
            sim_mat = np.zeros((n_labels, n_dims))
            for d in range(n_dims):
                dim_resp = W[:, d]
                dim_weights = dim_resp / np.sum(dim_resp) if np.sum(dim_resp) != 0 else np.zeros_like(dim_resp)

                # Compute weighted average similarity for each label
                for l, label in enumerate(labels):
                    # Compute cosine similarities between all image and text embeddings
                    sims = cosine_similarity(img_embeds_subject, text_embeds[label])
                    sim_avg = sims.mean(axis=1)  # Average over prompts

                    # Compute average similarity weighted by dims
                    sim_mat[l, d] = np.average(sim_avg, weights=dim_weights)

            # Subtract global mean across dimensions for each label
            row_means = np.mean(sim_mat, axis=1, keepdims=True)
            sim_mat_adj = sim_mat - row_means

            # Z-score labels
            z_scores = zscore(sim_mat_adj, axis=1)
            z_mats[roi] = z_scores
            sim_mats[roi] = sim_mat_adj

            # Sort labels based on adjusted similarity for each dim
            labels_sorted_roi = []
            for d in range(n_dims):
                z_scores_dim = z_scores[:, d]
                labels_sorted = sorted(zip(labels, z_scores_dim), key=lambda x: x[1], reverse=True)
                labels_sorted_roi.append({
                    "roi": roi,
                    "dim": d,
                    "labels_sorted": labels_sorted
                })
            labels_sorted_all[roi] = labels_sorted_roi

        with open(GROUP_DIR / "interpret" / "wordclouds" / f"labels_sorted_{subject}.pkl", "wb") as f:
            pkl.dump(labels_sorted_all, f)
