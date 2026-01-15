import torch
import random
import medmnist
from medmnist import INFO
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

# ------------------------------------------
# Coordinate Dataset
# ------------------------------------------


class PixelPointDataset(Dataset):
    def __init__(self, images_tensor, image_sources=None, image_channels=None):
        """
        images_tensor: Shape (K, C, H, W). Assumed to be normalized [0, 1].
        image_sources: Optional list of dataset identifiers (len = K) to keep
                       track of which MedMNIST subset each image came from.
        image_channels: Optional list of original channel counts (len = K) before
                        any channel promotion; used for metrics.
        """

        self.K, self.C, self.H, self.W = images_tensor.shape
        if image_sources is not None and len(image_sources) != self.K:
            raise ValueError(
                f"image_sources length ({len(image_sources)}) must match number of images ({self.K})."
            )
        self.image_sources = image_sources
        if image_channels is not None and len(image_channels) != self.K:
            raise ValueError(
                f"image_channels length ({len(image_channels)}) must match number of images ({self.K})."
            )
        self.image_channels = image_channels

        # --- A. Create Coordinate Grid (Shared across all images) ---
        # range [-1, 1]
        y_coords = torch.linspace(-1, 1, self.H)
        x_coords = torch.linspace(-1, 1, self.W)

        # indexing='ij' ensures that:
        # grid_y varies along the height (rows)
        # grid_x varies along the width (cols)
        # thus matching the memory layout of the image tensor
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")

        # Stack to (H, W, 2) then flatten to (H*W, 2); reused for every image.
        self.shared_coords = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)

        # --- B. Flatten Images ---
        # Current: (K, C, H, W)
        # Permute to: (K, H, W, C) -> channel-last for pixel extraction
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


def _normalize_name(name: str) -> str:
    return name.lower().replace("_", "").replace("-", "")


def _resolve_medmnist_class(dataset_name: str):
    key = _normalize_name(dataset_name)
    if key not in INFO:
        raise ValueError(
            f"Unknown MedMNIST dataset key '{dataset_name}'. Available keys include: {list(INFO.keys())}"
        )
    cls_name = INFO[key]["python_class"]
    return getattr(medmnist, cls_name)


def get_medmnist_data(
    image_size=64,
    num_images=5,
    download_root="./data",
    dataset_names=None,
    split="train",
    seed=None,
):
    """
    Helper to download and prep data tensors from one or more MedMNIST subsets.
    dataset_names: list of dataset keys (e.g., ["pneumoniamnist", "pathmnist"]).
    The returned tensor contains an even split across the requested datasets
    (differences of at most 1 image), shuffled across datasets. If datasets
    differ in channel count, grayscale images are promoted to 3 channels by
    channel-wise repetition to keep RGB information intact. Always returns a
    tuple: (tensor, sources, channels) where sources is a list of dataset keys
    and channels contains the original channel count per sample.
    seed: optional int to make selection/shuffling deterministic across runs.
    """
    if dataset_names is None:
        dataset_names = ["pneumoniamnist"]

    if len(dataset_names) == 0:
        raise ValueError("dataset_names must contain at least one dataset key.")

    rng = random.Random(seed) if seed is not None else random

    n_sets = len(dataset_names)
    base = num_images // n_sets
    remainder = num_images % n_sets

    per_dataset_counts = []
    for idx in range(n_sets):
        count = base + (1 if idx < remainder else 0)
        if count > 0:
            per_dataset_counts.append(count)
        else:
            per_dataset_counts.append(0)

    samples = []
    # Determine target channel count (max across datasets); promotes 1->3 when mixed.
    channel_counts = []
    for name in dataset_names:
        info_entry = INFO[_normalize_name(name)]
        channel_counts.append(info_entry["n_channels"])
    target_channels = max(channel_counts)
    if target_channels not in (1, 3):
        raise ValueError(
            f"Unsupported target_channels={target_channels}; expected 1 or 3."
        )

    for name, count in zip(dataset_names, per_dataset_counts):
        if count == 0:
            continue
        dataset_key = _normalize_name(name)
        info_entry = INFO[dataset_key]
        dataset_cls = _resolve_medmnist_class(dataset_key)
        dataset_raw = dataset_cls(
            split=split, download=True, root=download_root, size=image_size
        )
        if len(dataset_raw) < count:
            raise ValueError(
                f"Requested {count} samples from '{name}', but dataset only has {len(dataset_raw)}."
            )
        for i in range(count):
            pil_img, _ = dataset_raw[i]
            t_img = pil_to_tensor(pil_img).float() / 255.0
            # Harmonize channels: if grayscale and target is RGB, repeat channels
            if t_img.shape[0] == 1 and target_channels == 3:
                t_img = t_img.repeat(3, 1, 1)
            if t_img.shape[0] != target_channels:
                raise ValueError(
                    f"Channel mismatch after promotion for dataset '{name}': "
                    f"got {t_img.shape[0]}, expected {target_channels}. "
                    "Ensure all chosen datasets have compatible channel counts."
                )
            samples.append((dataset_key, t_img, info_entry["n_channels"]))

    if len(samples) == 0:
        raise ValueError("No data collected from the requested datasets.")

    rng.shuffle(samples)
    sources_shuffled = [src for src, _, _ in samples]
    channels_shuffled = [ch for _, _, ch in samples]
    data_acc = [img for _, img, _ in samples]
    stacked = torch.stack(data_acc)
    return stacked, sources_shuffled, channels_shuffled
