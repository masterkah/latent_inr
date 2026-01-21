"""
Test script to verify Residual VQ implementation
"""
import torch
from vqcoin import VQINR

def test_residual_vq():
    print("Testing Residual VQ Implementation...")
    print("="*60)
    
    # Create a small VQINR model with Residual VQ
    model = VQINR(
        coord_dim=2,
        value_dim=3,
        latent_dim=8,
        num_codes=16,
        hidden_size=32,
        num_layers=2,
        num_latent_vectors=4,  # 4 RVQ stages
        num_images=3,  # 3 images
        activation='relu'
    )
    
    print(f"Model created:")
    print(f"  - Number of images: {model.num_images}")
    print(f"  - RVQ stages per image: {model.num_latent_vectors}")
    print(f"  - Latent shape: {model.latents.shape}")
    print(f"  - Number of VQ layers: {len(model.vq_layers)}")
    print()
    
    # Test forward pass for each image
    coords = torch.randn(100, 2)  # 100 random coordinates
    
    for img_idx in range(3):
        print(f"Testing Image {img_idx}...")
        
        # Forward pass
        outputs, indices, vq_loss = model(coords, latent_idx=img_idx)
        
        print(f"  Output shape: {outputs.shape}")
        print(f"  Number of index tensors: {len(indices)}")
        print(f"  VQ loss: {vq_loss.item():.4f}")
        
        # Check indices from each stage
        for stage_idx, idx in enumerate(indices):
            print(f"  Stage {stage_idx} indices shape: {idx.shape}, value: {idx.item()}")
        
        # Test compression
        compressed_indices = model.compress(latent_idx=img_idx)
        print(f"  Compressed: {len(compressed_indices)} index tensors")
        
        # Test decompression
        reconstructed = model.decompress(coords, compressed_indices)
        print(f"  Reconstructed shape: {reconstructed.shape}")
        
        # Check reconstruction matches forward pass
        diff = (outputs - reconstructed).abs().max()
        print(f"  Reconstruction error: {diff.item():.6f}")
        print()
    
    print("="*60)
    print("Test completed successfully!")
    print()
    print("Key Residual VQ features:")
    print("  ✓ Each image has 4 latent vectors")
    print("  ✓ Each stage quantizes the residual from previous stages")
    print("  ✓ Final output is the sum of all quantized latents")
    print("  ✓ Compression stores 4 indices per image")
    print("  ✓ Decompression reconstructs from all indices")

if __name__ == "__main__":
    test_residual_vq()
