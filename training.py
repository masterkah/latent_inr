import torch
from torch.utils.data import Dataset, DataLoader
import lightning as pl
import numpy as np
import random
import matplotlib.pyplot as plt
import math
from datetime import datetime
from torchvision.transforms.functional import pil_to_tensor
import os

from medmnist import BreastMNIST, RetinaMNIST, PneumoniaMNIST
from vqcoin import SIREN_FACTOR, VQINR, MLP, SineLayer, ShiftSineLayer
from plot import plot_reconstructions, psnr

# ==========================================
# 1. Improved Dataset: Support Masking
# ==========================================
class MultiImageDataset(Dataset):
    def __init__(self, images: list, points_num: int = 4096):
        super().__init__()
        self.points_num = points_num
        self.images = []
        self.masks = []  
        
        for img in images:
            h, w, c = img.shape
            
            if c == 1:
                zeros = torch.zeros(h, w, 2, dtype=img.dtype, device=img.device)
                img_padded = torch.cat([img, zeros], dim=-1)
                mask = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
            elif c == 3:
                img_padded = img
                mask = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
            else:
                raise ValueError(f"Unsupported channel count: {c}")
            
            self.images.append(img_padded)
            self.masks.append(mask)

        self.dim_sizes = self.images[0].shape[:-1] 
        self.coord_size = len(self.dim_sizes)
        self.value_size = 3 
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx: int):
        image = self.images[idx] # (H, W, 3)
        mask = self.masks[idx]   # (3,)
        
        point_indices = [torch.randint(0, i, (self.points_num,)) for i in self.dim_sizes]
        
        point_values = image[tuple(point_indices)] # (Points, 3)
        
        point_coords = torch.stack(point_indices, dim=-1) # (Points, 2)
        spatial_dims = torch.tensor(self.dim_sizes).float()
        point_coords_norm = point_coords / (spatial_dims / 2) - 1
        
        return point_coords_norm, point_values, idx, mask

# ==========================================
# 2. Improved Lightning Module: Masked Loss
# ==========================================
class MultiImageINRModule(pl.LightningModule):
    def __init__(self,
                 network: VQINR,
                 gt_images_original: list, # Original unpadded images for validation
                 lr: float = 0.001,
                 eval_interval: int = 100,
                 visualization_intervals: list = []):
        super().__init__()
        self.save_hyperparameters(ignore=['network', 'gt_images_original'])
        self.lr = lr
        self.network = network
        self.gt_images = gt_images_original
        self.num_images = len(gt_images_original)
        
        self.eval_interval = eval_interval
        self.visualization_intervals = visualization_intervals
        
        self.progress_ims = [[] for _ in range(self.num_images)]
        self.scores = [[] for _ in range(self.num_images)]
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.network.parameters(), lr=self.lr)
    
    def forward(self, coords, latent_idx=0):
        return self.network(coords, latent_idx=latent_idx)
    
    def training_step(self, batch, batch_idx):
        coords, values, img_indices, masks = batch
        
        # Move data to device to avoid CPU/GPU transfers
        coords = coords.to(self.device)
        values = values.to(self.device)
        masks = masks.to(self.device)
        
        # Reshape inputs
        # coords: (B, Points, 2) -> (B*Points, 2)
        coords = coords.view(-1, coords.shape[-1])
        # values: (B, Points, 3) -> (B*Points, 3)
        values = values.view(-1, values.shape[-1])
        
        # Process Mask
        # repeat_interleave will convert (B, 3) to (B*Points, 3)
        points_per_sample = values.shape[0] // masks.shape[0]
        masks_expanded = masks.repeat_interleave(points_per_sample, dim=0)
        
        latent_idx = img_indices[0].item()
        
        # Forward pass
        outputs, indices, vq_loss = self.forward(coords, latent_idx=latent_idx)
        
        # --- Masked MSE Loss ---
        # 1. Calculate squared differences for all channels
        squared_diff = (outputs - values) ** 2
        # 2. Apply mask
        masked_diff = squared_diff * masks_expanded
        # Fixed normalization: (points * 3) to keep gradient scale consistent across grayscale/color
        recon_loss = masked_diff.sum() / (masked_diff.shape[0] * masked_diff.shape[1])
        
        loss = recon_loss + vq_loss
        
        self.log('train/recon_loss', recon_loss, prog_bar=True)
        self.log('train/vq_loss', vq_loss, prog_bar=True)
        
        return loss
    
    def on_train_epoch_end(self):
        """Evaluate on all images at regular intervals"""
        if (self.current_epoch + 1) % self.eval_interval == 0 or self.current_epoch == 0:
            avg_psnr = 0
            for img_idx in range(self.num_images):
                gt_image = self.gt_images[img_idx].to(self.device) # (H, W, C_original)
                original_c = gt_image.shape[-1]
                
                pred_im_full = self.sample_at_resolution(gt_image.shape[:-1], latent_idx=img_idx)
                
                pred_im = pred_im_full[..., :original_c]
                
                if pred_im.shape != gt_image.shape:
                     pred_im = pred_im.reshape(gt_image.shape)

                psnr_value = psnr(pred_im, gt_image).cpu().item()
                self.scores[img_idx].append((self.current_epoch + 1, psnr_value))
                avg_psnr += psnr_value
                
                if self.current_epoch + 1 in self.visualization_intervals:
                    self.progress_ims[img_idx].append((self.current_epoch + 1, pred_im.cpu(), psnr_value))
            
            avg_psnr /= self.num_images
            self.log('val/avg_psnr', avg_psnr, prog_bar=True)
    
    @torch.no_grad()
    def sample_at_resolution(self, resolution, latent_idx=0):
        meshgrid = torch.meshgrid([torch.arange(0, i, device=self.device) for i in resolution], indexing='ij')
        coords = torch.stack(meshgrid, dim=-1)
        coords_norm = coords / torch.tensor(resolution, device=self.device) * 2 - 1
        coords_norm_ = coords_norm.reshape(-1, coords.shape[-1])
        
        predictions_, _, _ = self.forward(coords_norm_, latent_idx=latent_idx)
        
        # predictions_ is always 3 channels
        target_shape = list(resolution) + [predictions_.shape[-1]]
        predictions = predictions_.reshape(*target_shape)
        
        return predictions

# ==========================================
# 3. Initialization Utilities
# ==========================================
def initialize_vqinr_weights(model):
    with torch.no_grad():
        for name, module in model.decoder.named_modules():
            if isinstance(module, (ShiftSineLayer, SineLayer)):
                is_first = (module == model.decoder.layers[0])
                in_dim = module.linear.weight.shape[1]
                
                if is_first:
                    bound = 1 / in_dim
                    module.linear.weight.uniform_(-bound, bound)
                else:
                    bound = np.sqrt(6 / in_dim) / model.decoder.layers[0].siren_factor
                    module.linear.weight.uniform_(-bound, bound)
                
                if module.linear.bias is not None:
                    module.linear.bias.fill_(0.0)

# ==========================================
# 4. Main Training 
# ==========================================
if __name__ == "__main__":
    POINTS_PER_SAMPLE = 4096
    HIDDEN_SIZE = 128
    NUM_LAYERS = 3
    LEARNING_RATE = 1e-4
    TRAINING_EPOCHS = 10000
    LATENT_DIM = 64    
    NUM_CODES = 128     
    COMMITMENT_COST = 0.25 
    VISUALIZATION_INTERVALS = [0, 1000, 3000, 5000, 7500, 10000]
    
    IMAGE_SIZE = 64  
    NUM_IMAGES_PER_DATASET = 20

    print("Loading datasets...")
    datasets = {
        'breast': BreastMNIST(split="val", download=True, size=IMAGE_SIZE),   # 1 Channel
        'retina': RetinaMNIST(split="val", download=True, size=IMAGE_SIZE),   # 3 Channels
        'pneumonia': PneumoniaMNIST(split="val", download=True, size=IMAGE_SIZE) # 1 Channel
    }
    
    all_images_original = [] 
    image_info = []
    
    print(f"\nSampling {NUM_IMAGES_PER_DATASET} images from each dataset...")
    
    for dataset_name, dataset in datasets.items():
        dataset_size = len(dataset)
        random_indices = random.sample(range(dataset_size), NUM_IMAGES_PER_DATASET)
        
        for idx in random_indices:
            pil_image, label = dataset[idx]
            
            gt_image = pil_to_tensor(pil_image) 
            gt_image = gt_image.moveaxis(0, -1) 
            gt_image = gt_image.to(torch.float32) / 255.0
            
            all_images_original.append(gt_image)
            
            image_info.append({
                'dataset': dataset_name,
                'index': idx,
                'shape': gt_image.shape
            })
            print(f"  {dataset_name.upper()}: Shape {gt_image.shape}")

    print("="*60)
    print("Setting up joint training with Masked Loss...")
    multi_dataset = MultiImageDataset(all_images_original, points_num=POINTS_PER_SAMPLE)
    multi_dataloader = DataLoader(multi_dataset, batch_size=1, shuffle=True, num_workers=0)
    
    # value_dim IS ALWAYS 3
    vqinr_net = VQINR(
        coord_dim=2,
        value_dim=3, 
        latent_dim=LATENT_DIM,
        num_codes=NUM_CODES,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        siren_factor=SIREN_FACTOR,
        commitment_cost=COMMITMENT_COST,
        activation='relu',
        num_latent_vectors=len(all_images_original)
    )
    initialize_vqinr_weights(vqinr_net)
    
    # Lightning Module
    vqinr_module = MultiImageINRModule(
        network=vqinr_net,
        gt_images_original=all_images_original, 
        lr=LEARNING_RATE,
        eval_interval=200,
        visualization_intervals=VISUALIZATION_INTERVALS
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=TRAINING_EPOCHS,
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
        log_every_n_steps=10
    )
    
    print(f"Starting training on {len(all_images_original)} images...")
    print(f"Grayscale images will only update the 1st channel of the decoder.")
    trainer.fit(vqinr_module, train_dataloaders=multi_dataloader)
    
    # ==========================================
    # 5. Results Visualization and Saving
    # ==========================================
    print("\n" + "="*60)
    print("Generating visualizations...")
    
    vqinr_net.eval()
    
    os.makedirs('results', exist_ok=True)
    
    # Save PSNR curves for all images to a single file
    import csv
    with open('results/psnr_scores.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image_Index', 'Dataset', 'Epoch', 'PSNR'])
        for img_idx, info in enumerate(image_info):
            for epoch, psnr_val in vqinr_module.scores[img_idx]:
                writer.writerow([img_idx, info['dataset'], epoch, psnr_val])
    print("Saved PSNR data to results/psnr_scores.csv")
    
    # Save full reconstruction sequences for each image (at VISUALIZATION_INTERVALS)
    for img_idx, info in enumerate(image_info):
        if len(vqinr_module.progress_ims[img_idx]) > 0:
            gt_image = all_images_original[img_idx]
            
            # Save all reconstructions at visualization_intervals
            num_recons = len(vqinr_module.progress_ims[img_idx])
            fig, axes = plt.subplots(1, num_recons + 1, figsize=(4*(num_recons+1), 4))
            
            if gt_image.shape[-1] == 1:
                axes[0].imshow(gt_image.squeeze(), cmap='gray')
            else:
                axes[0].imshow(gt_image)
            axes[0].set_title(f"GT: {info['dataset']}")
            axes[0].axis('off')
            
            for i, (epoch, recon_img, psnr_val) in enumerate(vqinr_module.progress_ims[img_idx]):
                if recon_img.shape[-1] == 1:
                    axes[i+1].imshow(recon_img.squeeze(), cmap='gray')
                else:
                    axes[i+1].imshow(torch.clamp(recon_img, 0, 1))
                axes[i+1].set_title(f"Epoch {epoch}\nPSNR: {psnr_val:.2f} dB")
                axes[i+1].axis('off')
            
            save_name = f"results/reconstruction_{info['dataset']}_{img_idx}.png"
            plt.tight_layout()
            plt.savefig(save_name, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved {save_name}")
    
    # Save combined PSNR curves plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for img_idx, info in enumerate(image_info):
        if len(vqinr_module.scores[img_idx]) > 0:
            epochs, psnrs = zip(*vqinr_module.scores[img_idx])
            label = f"{info['dataset'].upper()} (Img {img_idx})"
            ax.plot(epochs, psnrs, label=label, marker='o', markersize=3)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('Training Progress: PSNR vs Epoch (All Images)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/combined_psnr_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved results/combined_psnr_curves.png")

    print("\n" + "="*60)
    print("All results saved to 'results/' folder!")
    print("="*60)