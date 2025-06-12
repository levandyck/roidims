from roidims.interpret import (
    get_clip_img_embeds,
    get_clip_text_embeds,
    compute_cosine_embeds
)

# ---------------------- Run data-driven interpretation ---------------------- #
def run_interpret(subjects: list, rois: list):
    # Get CLIP embeddings for images
    get_clip_img_embeds(subjects)

    # Get CLIP embeddings for GPT-4V candidate labels
    get_clip_text_embeds()

    # Compute cosine similarity between embeddings
    compute_cosine_embeds(subjects, rois)
