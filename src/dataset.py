import torch
from torch.utils.data import Dataset
from medmnist import PneumoniaMNIST
from torchvision.transforms.functional import pil_to_tensor

# ------------------------------------------
# Coordinate Dataset
# ------------------------------------------

class PixelPointDataset(Dataset):
    def __init__(self, images_tensor):
        """images_tensor: Shape (K, C, H, W). Assumed to be normalized [0, 1]."""

        self.K, self.C, self.H, self.W = images_tensor.shape

        # --- A. Create Coordinate Grid (Shared across all images) ---
        # range [-1, 1]
        y_coords = torch.linspace(-1, 1, self.H)
        x_coords = torch.linspace(-1, 1, self.W)

        # indexing='ij' ensures that:
        # grid_y varies along the height (rows)
        # grid_x varies along the width (cols)
        # thus matching the memory layout of the image tensor
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # Stack to (H, W, 2) then Flatten to (H*W, 2)
        # Now we have a tensor of coordinates "linspaced" from [-1,-1] to [1,1] (there is exactly H*W coordinates)
        # We also could have normalized pixel coordinates on the fly... Oui bon trop tard
        self.shared_coords = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)

        # --- B. Flatten Images ---
        # Current: (K, C, H, W)
        # Permute to: (K, H, W, C) -> Channel last is easier for pixel extraction (according to gemini)
        # Reshape to: (K, H*W, C) -> Flatten spatial dims
        self.flat_pixels = images_tensor.permute(0, 2, 3, 1).reshape(self.K, -1, self.C)

        self.num_pixels_per_img = self.H * self.W

    def __len__(self):
        # Total data points = Images * Pixels per image
        return self.K * self.num_pixels_per_img

    def __getitem__(self, idx):
        # Determine which image 'idx' belongs to
        img_idx = idx // self.num_pixels_per_img

        # Determine which pixel inside that image
        pixel_idx = idx % self.num_pixels_per_img

        # Now retrieve data
        # Use shared_coords for the spatial location (x,y)
        coord = self.shared_coords[pixel_idx]

        # Use img_idx and pixel_idx to get the specific pixel color
        pixel_val = self.flat_pixels[img_idx, pixel_idx]

        return img_idx, coord, pixel_val
    

def get_medmnist_data(image_size=64, num_images=5, download_root="./data"):
    """Helper to download and prep the data tensor"""
    
    dataset_raw = PneumoniaMNIST(split="val", download=True, root=download_root, size=image_size)
    data_acc = []
    for i in range(num_images):
        pil_img, _ = dataset_raw[i]
        t_img = pil_to_tensor(pil_img).float() / 255.0
        data_acc.append(t_img)
    
    return torch.stack(data_acc)