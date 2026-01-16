# ------------------------------------------
#  Useful functions (visualizations etc.)
# ------------------------------------------

import os
import json
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid, save_image
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# ------------------------ Misc. ------------------------


# from a pytorch forum
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_global_seed(seed: int):
    """
    Seed all relevant RNGs so dataset sampling, dataloader shuffles,
    and plotting pick consistent examples across runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_experiment_folder(config, base_name="experiment"):
    if not os.path.exists(base_name):
        os.makedirs(base_name)

    # Save config
    with open(os.path.join(base_name, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    return base_name


# -------------------- Metrics and evaluation ---------------------


def calculate_psnr(mse_loss):
    # add epsilon to avoid log(0).
    return -10 * torch.log10(mse_loss + 1e-8)


# Helper to reconstruct images in the multi-resolution setting.
def reconstruct_single_image(model, img_idx, dataset, device, batch_size=4096):
    """
    Helper: Reconstructs a single image by batching the coordinate grid.
    Returns: Flat Tensor of shape (H*W, C)
    """
    model.eval()

    # Get all coordinates (H*W, 2)
    all_coords = dataset.shared_coords.to(device)
    total_pixels = all_coords.shape[0]

    pred_chunks = []

    with torch.no_grad():
        # Process in chunks to prevent GPU out-of-memory on large grids
        for i in range(0, total_pixels, batch_size):
            # 1. Prepare Chunk
            coord_chunk = all_coords[i : i + batch_size]
            chunk_len = coord_chunk.shape[0]

            # 2. Prepare Indices (All same image index)
            idx_chunk = torch.full(
                (chunk_len,), img_idx, dtype=torch.long, device=device
            )

            # 3. Model Forward
            # Works for both Spatial (AutoDecoderCNN) and Vector (Standard) models
            pred_batch, _ = model(idx_chunk, coord_chunk)

            pred_chunks.append(pred_batch)

    # Concatenate all chunks back into one big vector, preserving dataset coord order.
    return torch.cat(pred_chunks, dim=0)


def evaluate_dataset_psnr(model, dataset, device):
    psnr_dict = {}
    mse_fn = nn.MSELoss()

    for k in range(dataset.K):
        # reconstruct image (using helper)
        pred_flat = reconstruct_single_image(model, k, dataset, device)

        # get GT (Flat)
        gt_flat = dataset.flat_pixels[k].to(device)

        # compute metric
        mse = mse_fn(pred_flat, gt_flat)
        psnr = calculate_psnr(mse)
        psnr_dict[k] = psnr.item()

    model.train()  # return to training mode after evaluation
    return psnr_dict


#  --------------------- Plotting & Saving ---------------------


def _select_reference_indices(dataset, max_per_dataset):
    """
    Choose up to `max_per_dataset` indices per dataset source (if provided).
    Falls back to the first `max_per_dataset` images when sources are missing.
    """
    sources = getattr(dataset, "image_sources", None)
    if sources is None or len(sources) != dataset.K:
        return list(range(min(dataset.K, max_per_dataset)))

    counts = {}
    selected = []
    for idx, src in enumerate(sources):
        used = counts.get(src, 0)
        if used < max_per_dataset:
            selected.append(idx)
            counts[src] = used + 1
    return selected


def save_reference_reconstructions(
    model, dataset, epoch, run_folder, device, max_per_dataset=5
):
    """
    Save side-by-side GT/reconstruction pairs for up to `max_per_dataset` images
    per MedMNIST subset, using consistent indices across calls.
    """
    model.eval()
    H, W, C = dataset.H, dataset.W, dataset.C

    selection_indices = _select_reference_indices(dataset, max_per_dataset)
    if len(selection_indices) == 0:
        return

    img_list = []

    with torch.no_grad():
        for k in selection_indices:
            # Reconstruct -> we use the new helper now
            pred_pixels = reconstruct_single_image(model, k, dataset, device)

            # Reshape: (H, W, C) -> (C, H, W)
            recon_img = pred_pixels.reshape(H, W, C).permute(2, 0, 1).cpu()
            gt_img = dataset.flat_pixels[k].reshape(H, W, C).permute(2, 0, 1).cpu()

            # Stack [GT, Recon] vertically (GT on top, Recon on bottom)
            pair = torch.cat([gt_img, recon_img], dim=1)  # dim 1 is height
            img_list.append(pair)

    # Create a grid of pairs
    # make_grid expects (B, C, H, W)
    batch_tensor = torch.stack(img_list)
    grid_img = make_grid(batch_tensor, nrow=5, padding=2)

    save_image(grid_img, os.path.join(run_folder, f"recons_refs_epoch_{epoch}.png"))
    model.train()  # return to training mode after evaluation


def plot_tsne(
    model,
    dataset,
    epoch,
    run_folder,
    expected_num_clusters=1,
):
    """
    Visualize the latent codes with t-SNE and show the corresponding image
    thumbnail at 2D positions. For large numbers of images, only a
    cluster-representative is shown as a thumbnail; the rest are shown
    as colored points (color = average pixel color).

    Args
    ----
    model: nn.Module
        Needs a `.latents` tensor with shape (K, latent_dim) or (K, C, s, s).
    dataset: PixelPointDataset
        Used to reconstruct the original images corresponding to each latent.
        We assume it has attributes:
            - K   : number of images
            - H,W : height/width
            - C   : number of channels
            - flat_pixels: (K, H*W, C) tensor in [0, 1]
    epoch: int
        Current epoch index (only used for the filename).
    run_folder: str
        Where to store the PNG.
    expected_num_clusters: int, optional
        Expected number of semantic clusters in the latent space (e.g., number
        of dataset sources). Used for t-SNE perplexity tuning and KMeans
        representative selection (one thumbnail per cluster).
    """
    # Latents are no longer simple embeddings in the multi-resolution setting.
    # We flatten to a per-image feature vector for visualization.
    # Vector Model: (K, LatentDim)
    # Spatial Model: (K, Channels, s, s)
    z_data = model.latents.detach().cpu()

    # We flatten per image for visualization.
    # We want (K, Features).
    # For vector: stays (K, Dim).
    # For spatial: becomes (K, C*s*s). In the case s=1, we fall back to the basic setting.
    z_data_flat = z_data.view(z_data.shape[0], -1).numpy()
    K_samples = z_data_flat.shape[0]

    # Run t-SNE (perplexity must be < n_samples)
    if expected_num_clusters < 1:
        raise ValueError("expected_num_clusters must be >= 1")
    if K_samples < 2:
        raise ValueError("t-SNE requires at least 2 samples.")
    max_valid_perplexity = max(1, K_samples - 1)
    # Anchor perplexity to expected clusters so neighbors stay local enough to
    # reveal the separate modes we want to see.
    target_perplexity = min(30, max_valid_perplexity)
    target_perplexity = min(target_perplexity, max(5, expected_num_clusters * 3))
    perplexity = max(1, min(target_perplexity, max_valid_perplexity))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=0,  # keep orientation stable across calls while data drifts
    )
    z_embedded = tsne.fit_transform(z_data_flat)

    # Decide how many thumbnails to show (cluster representatives)
    n_clusters = max(1, min(int(expected_num_clusters), K_samples))

    # Compute average colors for all images for use as scatter colors
    colors = []
    for i in range(K_samples):
        img_flat = dataset.flat_pixels[i]
        img = img_flat.view(dataset.H, dataset.W, dataset.C).detach().cpu().numpy()
        avg_color = img.mean(axis=(0, 1))
        if avg_color.shape[0] == 1:
            avg_color = np.repeat(avg_color, 3)  # grayscale -> RGB
        colors.append(avg_color)
    colors = np.clip(np.array(colors), 0.0, 1.0)

    # Choose representative indices (closest to cluster centers) in embedded space.
    rep_indices = []
    if n_clusters == 1:
        rep_indices = [0] if K_samples > 0 else []
    else:
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
        labels = kmeans.fit_predict(z_embedded)
        centers = kmeans.cluster_centers_
        for c in range(n_clusters):
            idxs = np.where(labels == c)[0]
            if len(idxs) == 0:
                continue
            cluster_points = z_embedded[idxs]
            dists = np.linalg.norm(cluster_points - centers[c], axis=1)
            rep_indices.append(idxs[np.argmin(dists)])

    # Plot scatter for all points using the average colors
    point_size = 25  # scatter marker size
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        z_embedded[:, 0],
        z_embedded[:, 1],
        s=point_size,
        alpha=0.7,
        c=colors,
        edgecolors="k",
        linewidths=0.3,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.25)
    ax.tick_params(labelbottom=False, labelleft=False)

    # Overlay thumbnails for representative images
    for idx in rep_indices:
        if idx >= dataset.K:
            continue
        img_flat = dataset.flat_pixels[idx]
        img = img_flat.view(dataset.H, dataset.W, dataset.C).detach().cpu().numpy()
        img_to_show = img.squeeze(-1) if img.shape[-1] == 1 else img
        image_box = OffsetImage(img_to_show, zoom=0.25, cmap="gray")
        xy = (z_embedded[idx, 0], z_embedded[idx, 1])

        ab = AnnotationBbox(
            image_box,
            xy,
            frameon=True,
            pad=0.03,
            bboxprops=dict(edgecolor="white", linewidth=1.0, alpha=0.9),
        )
        ax.add_artist(ab)

    latent_spatial_dim = getattr(model, "latent_spatial_dim", 1)
    space_label = (
        "flattened latent grid space"
        if latent_spatial_dim > 1
        else "latent vector space"
    )
    ax.set_title(
        f"Latent space reduced to 2D with t-SNE (perplexity={perplexity})\n"
        f"Distance space: {space_label}"
    )

    # Show a small legend of distances in the original latent space (not t-SNE space).
    if K_samples > 1:
        latent_kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
        cluster_labels = latent_kmeans.fit_predict(z_data_flat)
        cluster_centers = latent_kmeans.cluster_centers_

        # Within-cluster pairwise distances
        within_all = []
        for c in range(n_clusters):
            idxs = np.where(cluster_labels == c)[0]
            if len(idxs) < 2:
                continue
            sub = z_data_flat[idxs]
            dists = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=-1)
            upper = dists[np.triu_indices_from(dists, k=1)]
            within_all.append(upper)
        within_all = np.concatenate(within_all) if len(within_all) > 0 else np.array([])

        # Between-cluster distances based on centroids
        center_dists = np.linalg.norm(
            cluster_centers[:, None, :] - cluster_centers[None, :, :], axis=-1
        )
        between_upper = (
            center_dists[np.triu_indices_from(center_dists, k=1)]
            if n_clusters > 1
            else np.array([])
        )

        def three_stats(arr):
            if arr.size == 0:
                return "NA, NA, NA"
            return f"{arr.min():.3f}, {arr.max():.3f}, {arr.mean():.3f}"

        within_str = three_stats(within_all)
        between_str = three_stats(between_upper)

        ax.text(
            0.02,
            0.98,
            f"Latent L2 ({space_label})\n"
            f"  within clusters: {within_str}\n"
            f"  between clusters: {between_str}",
            ha="left",
            va="top",
            transform=ax.transAxes,
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
        )

    fig.tight_layout()
    filename = f"tsne_2d_epoch_{epoch}.png"
    fig.savefig(os.path.join(run_folder, filename))
    plt.close(fig)
