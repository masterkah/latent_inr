import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from torchvision.utils import make_grid

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