# ------------------------------------------
#  Useful functions (visualizations etc.)
# ------------------------------------------

import os
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
from sklearn.manifold import TSNE

# ------------------------ Misc. ------------------------

# from a pytorch forum ->
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def setup_experiment_folder(config, base_name="experiment"):
    if not os.path.exists(base_name):
        os.makedirs(base_name)

    # Save config
    with open(os.path.join(base_name, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    return base_name

#--------------------- Metrics and evaluation ---------------------

def calculate_psnr(mse_loss):
    return -10 * torch.log10(mse_loss + 1e-8)

def evaluate_dataset_psnr(model, dataset, device):
    """
    Iterates over every image in the dataset, reconstructs it fully,
    and calculates the PSNR per image.
    Returns: dict {img_idx: psnr_value}
    """
    model.eval()
    psnr_dict = {}
    all_coords = dataset.shared_coords.to(device)
    mse_fn = nn.MSELoss()

    with torch.no_grad():
        for k in range(dataset.K):
            # 1. Get Latent for Image k
            idx_tensor = torch.tensor([k], device=device)
            z_vec = model.latents(idx_tensor) # (1, dim)

            # 2. Expand for full grid
            z_expanded = z_vec.repeat(all_coords.shape[0], 1)

            # 3. Predict
            pred_pixels = model.decoder(all_coords, z_expanded)

            # 4. Get GT
            gt_pixels = dataset.flat_pixels[k].to(device)

            # 5. Compute MSE & PSNR
            mse = mse_fn(pred_pixels, gt_pixels)
            psnr = calculate_psnr(mse)
            psnr_dict[k] = psnr.item()

    model.train()
    return psnr_dict

#  --------------------- Plotting & Saving ---------------------

def save_all_reconstructions(model, dataset, step, run_folder, device):
    """
    Reconstructs ALL images in the dataset and saves them side-by-side with GT.
    """
    model.eval()
    all_coords = dataset.shared_coords.to(device)

    img_list = []

    with torch.no_grad():
        for k in range(dataset.K):
            # Reconstruct
            idx_tensor = torch.tensor([k], device=device)
            z_vec = model.latents(idx_tensor)
            z_expanded = z_vec.repeat(all_coords.shape[0], 1)
            pred_pixels = model.decoder(all_coords, z_expanded)

            # Reshape: (H, W, C) -> (C, H, W)
            H, W, C = dataset.H, dataset.W, dataset.C
            recon_img = pred_pixels.reshape(H, W, C).permute(2, 0, 1).cpu()
            gt_img = dataset.flat_pixels[k].reshape(H, W, C).permute(2, 0, 1).cpu()

            # Stack [GT, Recon] vertically or horizontally
            # Here we stack them: GT on top, Recon on bottom
            pair = torch.cat([gt_img, recon_img], dim=1) # dim 1 is height
            img_list.append(pair)

    # Create a grid of pairs
    # make_grid expects (B, C, H, W)
    batch_tensor = torch.stack(img_list)
    grid_img = make_grid(batch_tensor, nrow=5, padding=2)

    save_image(grid_img, os.path.join(run_folder, f"recons_all_step_{step}.png"))
    model.train()

def plot_tsne(model, step, run_folder):
    z_data = model.latents.weight.detach().cpu().numpy()
    K_samples = z_data.shape[0]

    # t-SNE perplexity must be < n_samples.
    # For K=5, perplexity must be small (e.g., 2). For K=100, use 30.
    perplexity = min(30, max(1, K_samples - 1))

    tsne = TSNE(n_components=2, perplexity=perplexity, init='pca', learning_rate='auto')
    z_embedded = tsne.fit_transform(z_data)

    plt.figure(figsize=(6, 6))
    plt.scatter(z_embedded[:, 0], z_embedded[:, 1], c=range(len(z_data)), cmap='tab10', s=100)

    for i in range(len(z_data)):
        plt.text(z_embedded[i, 0]+0.02, z_embedded[i, 1]+0.02, str(i), fontsize=9)

    plt.title(f"Latent t-SNE at Step {step} (Perplexity={perplexity})")
    # Remove axis ticks as t-SNE units are arbitrary
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(run_folder, f"tsne_step_{step}.png"))
    plt.close()

def plot_psnr_per_image(psnr_logs, run_folder):
    """
    psnr_logs: dict of list -> {img_idx: [psnr_step_0, psnr_step_100...]}
    """
    steps = sorted(psnr_logs['steps'])

    plt.figure(figsize=(10, 6))

    for k in psnr_logs['data']:
        plt.plot(steps, psnr_logs['data'][k], label=f"Img {k}")

    plt.title("Reconstruction PSNR Evolution per Image")
    plt.xlabel("Step")
    plt.ylabel("PSNR (dB)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(run_folder, "psnr_per_image.png"))
    plt.close()
