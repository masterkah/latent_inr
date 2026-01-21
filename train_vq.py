"""
Vector Quantized Implicit Neural Representation Training Script
Reads configuration from JSON and trains VQINR model
"""
import os
import json
import random
import torch
from torch.utils.data import DataLoader
import lightning as pl
from torchvision.transforms.functional import pil_to_tensor
from medmnist import BreastMNIST, RetinaMNIST, PneumoniaMNIST, PathMNIST

from src.gpu_utils import auto_select_gpu
from src.vq_model import VQINR
from src.dataset import FullImageDataset
from src.trainer import MultiImageINRModule
from src.visualization import save_all_visualizations


def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def load_datasets(config):
    """
    Load and sample images from MedMNIST datasets
    Args:
        config: configuration dictionary
    Returns:
        all_images: list of all sampled images
        track_indices: dict mapping dataset names to tracked indices
    """
    image_size = config['IMAGE_SIZE']
    num_images_per_ds = config['NUM_IMAGES_PER_DS']
    
    print("Loading datasets...")
    datasets = {
        'breast': BreastMNIST(split="val", download=True, size=image_size),
        'retina': RetinaMNIST(split="val", download=True, size=image_size),
        'pneumonia': PneumoniaMNIST(split="val", download=True, size=image_size),
        'pathology': PathMNIST(split="val", download=True, size=image_size)
    }
    
    all_images_original = []
    track_indices = {}
    current_idx_offset = 0
    
    print(f"Sampling {num_images_per_ds} images from each dataset...")
    for ds_name, ds in datasets.items():
        count = min(len(ds), num_images_per_ds)
        indices = random.sample(range(len(ds)), count)
        
        dataset_global_indices = []
        for i, sample_idx in enumerate(indices):
            img = pil_to_tensor(ds[sample_idx][0]).moveaxis(0, -1).float() / 255.0
            all_images_original.append(img)
            # Track first 3 images from each dataset for visualization
            if i < 3:
                dataset_global_indices.append(current_idx_offset + i)
        
        track_indices[ds_name] = dataset_global_indices
        current_idx_offset += len(indices)
    
    print(f"Total training images: {len(all_images_original)}")
    print(f"Tracking indices for viz & stats: {track_indices}")
    
    return all_images_original, track_indices


def create_model(config, num_images):
    """
    Create VQINR model from configuration
    Args:
        config: configuration dictionary
        num_images: total number of images
    Returns:
        VQINR model
    """
    model = VQINR(
        coord_dim=config['COORD_DIM'],
        value_dim=config['VALUE_DIM'],
        latent_dim=config['LATENT_DIM'],
        num_codes=config['NUM_CODES'],
        hidden_size=config['HIDDEN_SIZE'],
        num_layers=config['NUM_LAYERS'],
        num_latent_vectors=config['NUM_LATENT_VECTORS'],
        num_images=num_images,
        commitment_cost=config['COMMITMENT_COST']
    )
    return model


def main():
    """Main training function"""
    # Auto-select GPU
    auto_select_gpu()
    
    # Load configuration
    config_path = './config/config_vq.json'
    config = load_config(config_path)
    print(f"Loaded configuration from {config_path}")
    print(json.dumps(config, indent=2))
    
    # Set random seed
    pl.seed_everything(config['SEED'])
    
    # Enable high-precision matrix multiplication
    torch.set_float32_matmul_precision('high')
    
    # Load datasets
    all_images_original, track_indices = load_datasets(config)
    
    # Create dataset and dataloader
    dataset = FullImageDataset(all_images_original)
    dataloader = DataLoader(
        dataset,
        batch_size=config['BATCH_SIZE'],
        shuffle=True,
        num_workers=config['NUM_WORKERS'],
        pin_memory=True,
        drop_last=False
    )
    
    # Create model
    vqinr = create_model(config, len(all_images_original))
    print("Model created successfully")
    
    # Define visualization intervals
    visualization_intervals = [0, 1000, 3000, 5000, 7500, 10000, 15000]
    
    # Create Lightning module
    module = MultiImageINRModule(
        vqinr,
        all_images_original,
        track_indices,
        visualization_intervals,
        lr=config['LR']
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config['EPOCHS'],
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        enable_checkpointing=False,
        logger=False,
        log_every_n_steps=10
    )
    
    print("\nStarting Training...")
    trainer.fit(module, dataloader)
    
    # Save results
    results_dir = config.get('RESULTS_DIR', 'results_vq')
    save_all_visualizations(
        module,
        all_images_original,
        visualization_intervals,
        results_dir
    )


if __name__ == "__main__":
    main()
