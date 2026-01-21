import torch
import random
import medmnist
from medmnist import INFO
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


class FullImageDataset(Dataset):
    def __init__(self, images: list):
        super().__init__()
        self.images = []
        self.masks = []
        print("Preprocessing images to RAM (Full Grid)...")
        for img in images:
            h, w, c = img.shape
            if c == 1:
                zeros = torch.zeros(h, w, 2, dtype=img.dtype)
                img_padded = torch.cat([img, zeros], dim=-1)
                mask = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
            elif c == 3:
                img_padded = img
                mask = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
            
            y = torch.linspace(-1, 1, h)
            x = torch.linspace(-1, 1, w)
            yy, xx = torch.meshgrid(y, x, indexing='ij')
            coords = torch.stack([yy, xx], dim=-1).reshape(-1, 2)
            values = img_padded.reshape(-1, 3)
            self.images.append((coords, values))
            self.masks.append(mask)

    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx: int):
        return self.images[idx][0], self.images[idx][1], idx, self.masks[idx]