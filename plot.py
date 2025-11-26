from typing import Tuple, List, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt

# We will track visual results every few epochs and visualize them after training
def plot_reconstructions(progress_ims: List[Tuple[int, torch.Tensor]], gt_im: torch.Tensor):
    ncols = len(progress_ims) + 1
    fig_width = 5
    fig, axs = plt.subplots(ncols=ncols, figsize=(ncols*fig_width, fig_width))
    # Plot all reconstructions images predicted by the model
    for i, (epoch, im, metric) in enumerate(progress_ims):
        im = im.cpu().numpy()
        ax = axs[i]
        ax.imshow(im, cmap='gray')
        ax.axis('off')
        title = f'Epoch: {epoch}, PSNR: {metric}'
        ax.set_title(title)
    # Plot ground-truth image
    gt_im = gt_im.cpu().numpy()
    axs[-1].imshow(gt_im, cmap='gray')
    axs[-1].axis('off')
    axs[-1].set_title('Ground Truth')
    plt.tight_layout()
    plt.savefig('reconstructions.png')

# We will also track the PSNR of our training samples
def psnr(pred, ref):
    max_value = ref.max()
    mse = torch.mean((pred - ref) ** 2, dim=(-2, -1))
    out = 20 * torch.log10(max_value / torch.sqrt(mse))
    return out.mean()

# Let's create a function to plot our psnr scores throughout training
def plot_scores(models: List['INRModule']):
    fig, ax = plt.subplots()
    # For each model, plot list of scores
    for model in models:
        epochs, scores = [i for i, _ in model.scores], [v for _, v in model.scores]
        name = getattr(model, 'name', 'VQ-INR')
        ax.plot(epochs, scores, label=name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('PSNR')
    ax.set_title('PSNR over epochs')
    ax.legend()
    plt.savefig('psnr_scores.png')