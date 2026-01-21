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

def psnr(pred, target):
    """Calculate Peak Signal-to-Noise Ratio"""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def plot_psnr_curves(psnr_history, results_dir):
    """
    Plot PSNR curves for all datasets
    Args:
        psnr_history: dict mapping dataset names to PSNR values over epochs
        results_dir: directory to save plots
    """
    print("Plotting PSNR Curves...")
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(psnr_history['total']) + 1)
    
    # Plot total average with dashed line
    plt.plot(epochs, psnr_history['total'], label='Total Average', 
             color='black', linewidth=2.5, linestyle='--')
    
    # Color scheme for different datasets
    colors = {
        'breast': 'tab:blue', 
        'retina': 'tab:orange', 
        'pneumonia': 'tab:green', 
        'pathology': 'tab:red'
    }
    
    # Plot individual dataset curves
    for ds_name, color in colors.items():
        if ds_name in psnr_history:
            plt.plot(epochs, psnr_history[ds_name], 
                    label=f'{ds_name.capitalize()} MNIST', 
                    color=color, alpha=0.8)
            
    plt.title('Reconstruction PSNR over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_metrics_psnr.png'), dpi=150)
    plt.close()


def plot_codebook_usage(codebook_usage_history, results_dir):
    """
    Plot codebook utilization over training
    Args:
        codebook_usage_history: list of codebook usage percentages
        results_dir: directory to save plots
    """
    print("Plotting Codebook Usage...")
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(codebook_usage_history) + 1)
    plt.plot(epochs, codebook_usage_history, color='purple')
    plt.title('Codebook Utilization over Training')
    plt.xlabel('Epochs')
    plt.ylabel('Active Codes (%)')
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_metrics_codebook.png'), dpi=150)
    plt.close()


def plot_evolution(history, track_indices, all_images_original, vis_intervals, results_dir):
    """
    Generate evolution plots showing reconstruction quality over training
    Args:
        history: dict mapping epochs to reconstructed images
        track_indices: dict mapping dataset names to tracked image indices
        all_images_original: list of original images
        vis_intervals: list of epochs where snapshots were saved
        results_dir: directory to save plots
    """
    print("Generating Evolution Plots...")
    sorted_intervals = sorted([t for t in vis_intervals if t in history])
    
    for ds_name, indices in track_indices.items():
        rows = len(indices)
        cols = 1 + len(sorted_intervals)
        fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.5 * rows))
        if rows == 1:
            axes = axes[None, :]
        
        for r, global_idx in enumerate(indices):
            gt = all_images_original[global_idx]
            
            # Plot ground truth
            ax_gt = axes[r, 0]
            if gt.shape[-1] == 1:
                ax_gt.imshow(gt.squeeze(), cmap='gray')
            else:
                ax_gt.imshow(gt)
            ax_gt.axis('off')
            if r == 0:
                ax_gt.set_title("Ground Truth", fontsize=10, fontweight='bold')
            
            # Plot reconstructions at different epochs
            for c, epoch in enumerate(sorted_intervals):
                ax = axes[r, c+1]
                if global_idx in history[epoch]:
                    recon_img = history[epoch][global_idx]
                    if recon_img.shape[-1] == 1:
                        ax.imshow(recon_img.squeeze(), cmap='gray')
                    else:
                        ax.imshow(recon_img)
                    curr_psnr = psnr(recon_img, gt)
                    ax.set_title(f"Step {epoch}\n{curr_psnr:.1f} dB", fontsize=9)
                else:
                    ax.text(0.5, 0.5, "Missing", ha='center')
                ax.axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(results_dir, f'evolution_{ds_name}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def save_all_visualizations(module, all_images_original, vis_intervals, results_dir):
    """
    Save all visualization plots
    Args:
        module: trained MultiImageINRModule
        all_images_original: list of original images
        vis_intervals: list of epochs where snapshots were saved
        results_dir: directory to save plots
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot training metrics
    plot_psnr_curves(module.psnr_history, results_dir)
    plot_codebook_usage(module.codebook_usage_history, results_dir)
    
    # Plot evolution
    plot_evolution(module.history, module.track_indices, all_images_original, 
                  vis_intervals, results_dir)
    
    print(f"\n✅ All results saved to {results_dir}")
