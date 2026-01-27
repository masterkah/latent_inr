import torch
import random
import medmnist
from medmnist import INFO
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


# Returns flattened (x, y) coords in [-1, 1] to match grid_sample order.
def make_coord_grid(height, width):
    y = torch.linspace(-1, 1, height)
    x = torch.linspace(-1, 1, width)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([xx, yy], dim=-1).reshape(-1, 2)


class FullImageDataset(Dataset):
    def __init__(self, images: list, image_channels=None):
        super().__init__()
        if image_channels is not None and len(image_channels) != len(images):
            raise ValueError("image_channels length must match number of images.")
        self.images = []
        self.masks = []
        for idx, img in enumerate(images):
            h, w, c = img.shape
            orig_c = image_channels[idx] if image_channels is not None else c
            if orig_c not in (1, 3):
                raise ValueError(f"Unsupported original channel count: {orig_c}")
            if c not in (1, 3):
                raise ValueError(f"Unsupported image channel count: {c}")
            # Mask selects valid channels when grayscale was promoted to RGB.
            if orig_c == 1 and c == 3:
                mask = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
            else:
                mask = torch.ones(c, dtype=torch.float32)

            coords = make_coord_grid(h, w)
            values = img.reshape(-1, c)
            self.images.append((coords, values))
            self.masks.append(mask)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        return self.images[idx][0], self.images[idx][1], idx, self.masks[idx]


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

        # Shared coordinate grid in [-1, 1], flattened to (H*W, 2).
        self.shared_coords = make_coord_grid(self.H, self.W)

        # Flatten to (K, H*W, C) for direct indexing.
        self.flat_pixels = images_tensor.permute(0, 2, 3, 1).reshape(self.K, -1, self.C)

        self.num_pixels_per_img = self.H * self.W

    def __len__(self):
        return self.K * self.num_pixels_per_img

    def __getitem__(self, idx):
        img_idx = idx // self.num_pixels_per_img
        pixel_idx = idx % self.num_pixels_per_img
        coord = self.shared_coords[pixel_idx]
        pixel_val = self.flat_pixels[img_idx, pixel_idx]

        return img_idx, coord, pixel_val


def _normalize_name(name: str) -> str:
    return name.lower().replace("_", "").replace("-", "")


def build_track_indices(
    image_sources, dataset_names, max_per_dataset=3, drop_empty=True
):
    dataset_keys = [_normalize_name(name) for name in dataset_names]
    counts = {key: 0 for key in dataset_keys}
    track_indices = {key: [] for key in dataset_keys}
    for idx, src in enumerate(image_sources):
        if src in counts and counts[src] < max_per_dataset:
            track_indices[src].append(idx)
            counts[src] += 1
    if drop_empty:
        track_indices = {key: vals for key, vals in track_indices.items() if vals}
    return track_indices


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
