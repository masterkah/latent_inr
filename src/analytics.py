import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from torchvision.utils import make_grid

from utils import reconstruct_single_image
import numpy as np

def enable_dropout(model):
    """
    Forces Dropout layers to be in train mode while keeping 
    the rest of the model (Batch Norm, etc.) in eval mode.
    """
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

def get_mc_dropout_stats(model, img_idx, dataset, device, T=20, batch_size=4096):
    """
    Performs T forward passes with Dropout enabled to estimate uncertainty.
    """
    # 1. Set model to eval mode (freezes BatchNorm stats)
    model.eval()
    # 2. Force Dropout layers to train mode (activates randomness)
    enable_dropout(model)
    
    # Get coordinates
    all_coords = dataset.shared_coords.to(device)
    total_pixels = all_coords.shape[0]
    
    predictions_list = []
    
    print(f"Running MC Dropout (T={T})...", end="", flush=True)
    
    with torch.no_grad():
        for t in range(T):
            chunk_preds = []
            for i in range(0, total_pixels, batch_size):
                coord_chunk = all_coords[i : i+batch_size]
                chunk_len = coord_chunk.shape[0]
                idx_chunk = torch.full((chunk_len,), img_idx, dtype=torch.long, device=device)
                
                # Forward Pass
                pred_batch = model(idx_chunk, coord_chunk)[0]
                chunk_preds.append(pred_batch)
            
            full_img_t = torch.cat(chunk_preds, dim=0) # (H*W, C)
            predictions_list.append(full_img_t)
            
    print(" Done.")
    
    stack = torch.stack(predictions_list)
    
    mean_pred = stack.mean(dim=0)
    variance = stack.var(dim=0)
    std_dev = stack.std(dim=0)
    
    return mean_pred, variance, std_dev

def plot_uncertainty_maps(mean, var, std, dataset, img_idx, run_folder, suffix=""):
    """
    Plots the Mean Reconstruction, Variance, and Std Dev side-by-side.
    """
    H, W, C = dataset.H, dataset.W, dataset.C
    
    # Reshape and move to CPU
    mean_img = mean.reshape(H, W, C).permute(2, 0, 1).cpu().numpy().squeeze()
    var_img = var.reshape(H, W, C).permute(2, 0, 1).cpu().numpy().mean(axis=0)
    std_img = std.reshape(H, W, C).permute(2, 0, 1).cpu().numpy().mean(axis=0)
    
    gt_img = dataset.flat_pixels[img_idx].reshape(H, W, C).permute(2, 0, 1).cpu().numpy().squeeze()

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 1. Ground Truth
    ax = axes[0]
    im0 = ax.imshow(gt_img, cmap='gray', vmin=0, vmax=1)
    ax.set_title(f"Ground Truth (Idx {img_idx})")
    ax.axis('off')
    plt.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)

    # 2. Mean Prediction
    ax = axes[1]
    im1 = ax.imshow(mean_img, cmap='gray', vmin=0, vmax=1)
    ax.set_title("MC Mean Reconstruction")
    ax.axis('off')
    plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
    
    # 3. Variance Map
    ax = axes[2]
    im2 = ax.imshow(var_img, cmap='inferno') 
    ax.set_title("Variance Map (Uncertainty)")
    ax.axis('off')
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    
    # 4. Std Dev Map
    ax = axes[3]
    im3 = ax.imshow(std_img, cmap='inferno')
    ax.set_title("Std Dev Map")
    ax.axis('off')
    plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_path = os.path.join(run_folder, f"uncertainty_img_{img_idx}{suffix}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Uncertainty map saved to {save_path}")

# -- Perturbation analysis --

def perturb_latent_and_reconstruct(model, img_idx, dataset, coords, epsilon, device):
    """
    Temporarily adds epsilon to the latent vector at grid position (y, x),
    reconstructs the image, and then restores the original latent.
    
    Args:
        coords: Tuple (grid_y, grid_x)
        epsilon: Float value to add to the latent vector.
    """
    gy, gx = coords
    
    # 1. Backup original latent vector at this position
    # Shape of latents: (K, C, S, S)
    original_slice = model.latents.data[img_idx, :, gy, gx].clone()
    
    # 2. Apply Perturbation (Add epsilon to ALL channels at this spatial location)
    # We use .data to avoid tracking gradients for this manipulation
    model.latents.data[img_idx, :, gy, gx] += epsilon
    
    # 3. Reconstruct
    # This uses the helper from utils to handle batching/memory
    try:
        recon_flat = reconstruct_single_image(model, img_idx, dataset, device)
    finally:
        # 4. Restore (CRITICAL: Always use try/finally to ensure restore happens)
        model.latents.data[img_idx, :, gy, gx] = original_slice
        
    return recon_flat

def visualize_grid_perturbation(model, dataset, img_idx, points_to_perturb, epsilon, device, run_folder):
    """
    Visualizes the effect of perturbing specific latent grid points.
    
    Args:
        points_to_perturb: List of tuples [(y1, x1), (y2, x2), ...]
                           Example: [(0,0), (4,4), (7,7)] for top-left, center, bottom-right (8x8 grid).
    """
    print(f"Running Spatial Perturbation Analysis on Image {img_idx} with epsilon={epsilon}...")
    
    H, W, C = dataset.H, dataset.W, dataset.C
    
    # 1. Get Baseline (No perturbation)
    baseline_flat = reconstruct_single_image(model, img_idx, dataset, device)
    baseline_img = baseline_flat.reshape(H, W, C).permute(2, 0, 1).cpu().numpy().squeeze()
    
    # 2. Get Ground Truth
    gt_img = dataset.flat_pixels[img_idx].reshape(H, W, C).permute(2, 0, 1).cpu().numpy().squeeze()
    
    num_points = len(points_to_perturb)
    # Rows: 1 (GT/Base) + 1 (Perturbed Recon) + 1 (Difference Map)
    # Cols: 1 (GT/Base) + num_points
    
    fig = plt.figure(figsize=(4 * (num_points + 1), 10))
    
    # --- Row 1: Baselines ---
    # Plot GT
    ax_gt = plt.subplot(3, num_points + 1, 1)
    ax_gt.imshow(gt_img, cmap='gray', vmin=0, vmax=1)
    ax_gt.set_title(f"Ground Truth (Idx {img_idx})")
    ax_gt.axis('off')
    
    # Plot Baseline Recon
    ax_base = plt.subplot(3, num_points + 1, num_points + 2) # Start of row 2
    ax_base.imshow(baseline_img, cmap='gray', vmin=0, vmax=1)
    ax_base.set_title("Baseline Reconstruction")
    ax_base.axis('off')

    # --- Iterate over points ---
    for i, (py, px) in enumerate(points_to_perturb):
        # Perform perturbation
        pert_flat = perturb_latent_and_reconstruct(model, img_idx, dataset, (py, px), epsilon, device)
        pert_img = pert_flat.reshape(H, W, C).permute(2, 0, 1).cpu().numpy().squeeze()
        
        # Calculate Difference
        diff_img = np.abs(pert_img - baseline_img)
        
        # Col index for this point (shifted by 1 because col 0 is references)
        col_idx = i + 2 
        
        # Plot Perturbed Image
        ax_p = plt.subplot(3, num_points + 1, col_idx)
        ax_p.imshow(pert_img, cmap='gray', vmin=0, vmax=1)
        ax_p.set_title(f"Perturbed Grid ({py}, {px})\nepsilon={epsilon}")
        ax_p.axis('off')
        
        # Plot Difference Map
        # We use 'seismic' or 'bwr' centered at 0, or just 'inferno' for absolute diff
        ax_d = plt.subplot(3, num_points + 1, col_idx + (num_points + 1))
        im_d = ax_d.imshow(diff_img, cmap='inferno') # Bright colors = high change
        ax_d.set_title(f"Difference Map ({py}, {px})")
        ax_d.axis('off')
        plt.colorbar(im_d, ax=ax_d, fraction=0.046)

        # Zoom/Crop (Optional: Plot a zoomed version of the difference in Row 3)
        # For now, let's just leave Row 3 empty or use it for specific channel metrics if needed.
        # Let's actually put the diff map in Row 2 and Perturbed in Row 1 (next to GT)?
        # Let's stick to the current layout but maybe center it better.
        
    plt.tight_layout()
    save_path = os.path.join(run_folder, f"perturb_analysis_img_{img_idx}_eps{epsilon}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Spatial perturbation plot saved to {save_path}")


# --- Patch-wise Variance Metrics ---

def compute_patch_wise_variance(model, dataset, img_idx, device, T=20):
    """
    1. Computes full MC Dropout Variance Map.
    2. Slices it into patches corresponding to the model's latent grid.
    3. Computes the mean variance for each patch.
    
    Returns:
        avg_patch_variance (float): The mean of all patch variances (a single global scalar).
        patch_grid (numpy array): The S x S grid where each cell is the mean variance of that patch.
    """
    # 1. Get Pixel-wise Variance
    _, var_flat, _ = get_mc_dropout_stats(model, img_idx, dataset, device, T=T)
    
    # Reshape to Image (H, W) - averaging channels if RGB
    H, W, C = dataset.H, dataset.W, dataset.C
    var_img = var_flat.reshape(H, W, C).permute(2, 0, 1).cpu().numpy().mean(axis=0)
    
    # 2. Get Grid Dimensions
    # model.latents is (K, C, S, S)
    S = model.latents.shape[-1]
    
    patch_h = H // S
    patch_w = W // S
    
    patch_variances = np.zeros((S, S))
    
    # 3. Iterate over grid and aggregate
    for gy in range(S):
        for gx in range(S):
            # Define pixel bounds for this grid cell
            y_start = gy * patch_h
            y_end = (gy + 1) * patch_h
            x_start = gx * patch_w
            x_end = (gx + 1) * patch_w
            
            # Extract patch from variance map
            patch = var_img[y_start:y_end, x_start:x_end]
            
            # Compute mean of this patch
            patch_variances[gy, gx] = np.mean(patch)
            
    # Global average (scalar)
    global_avg = np.mean(patch_variances)
    
    return global_avg, patch_variances

def plot_sweep_results(sweep_data, param_name, run_folder):
    """
    Plots the metric evolution.
    sweep_data: dict { param_value: metric_value }
    """
    params = sorted(sweep_data.keys())
    values = [sweep_data[p] for p in params]
    
    plt.figure(figsize=(8, 6))
    plt.plot(params, values, marker='o', linestyle='-', linewidth=2, markersize=8)
    plt.title(f"Reconstruction Uncertainty vs {param_name}")
    plt.xlabel(param_name)
    plt.ylabel("Average Patch Variance")
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(run_folder, f"sweep_variance_vs_{param_name}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Sweep plot saved to {save_path}")