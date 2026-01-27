"""
Vector Quantized Implicit Neural Representation Training Script.
Reads configuration from JSON and trains a VQINR model.
"""

import argparse
import torch
import lightning as pl

from src.dataset import FullImageDataset, get_medmnist_data, build_track_indices
from src.model import VQINR
from src.trainer import MultiImageINRModule
from src.utils import (
    save_all_visualizations,
    set_global_seed,
    load_json_config,
    configure_device,
    build_dataloader,
    make_run_folder,
    config_name_from_path,
)

REQUIRED_CONFIG_KEYS = [
    "K_IMAGES",
    "IMAGE_SIZE",
    "BATCH_SIZE",
    "DATASET_NAMES",
    "NUM_EPOCHS",
    "VIZ_INTERVAL",
    "NUM_WORKERS",
    "PERSISTENT_WORKERS",
    "PREFETCH_FACTOR",
    "SEED",
    "COORD_DIM",
    "LATENT_DIM",
    "NUM_CODES",
    "NUM_LATENT_VECTORS",
    "HIDDEN_SIZE",
    "NUM_LAYERS",
    "COMMITMENT_COST",
    "WARMUP_EPOCHS",
    "LR",
]


def _create_model(config, num_images, value_dim):
    return VQINR(
        coord_dim=config["COORD_DIM"],
        value_dim=value_dim,
        latent_dim=config["LATENT_DIM"],
        num_codes=config["NUM_CODES"],
        hidden_size=config["HIDDEN_SIZE"],
        num_layers=config["NUM_LAYERS"],
        num_latent_vectors=config["NUM_LATENT_VECTORS"],
        num_images=num_images,
        commitment_cost=config["COMMITMENT_COST"],
        warmup_epochs=config["WARMUP_EPOCHS"],
        activation=config.get("ACTIVATION", "siren"),
        num_freqs=config.get("NUM_FREQS", 10),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-config",
        type=str,
        required=True,
        help="Path to a JSON config file (required).",
    )
    parser.add_argument(
        "-debug",
        type=int,
        default=0,
        choices=[0, 1],
        help="1=log every epoch, 0=log every VIZ_INTERVAL epochs.",
    )
    parser.add_argument(
        "-use-amp-tf32",
        type=int,
        default=1,
        choices=[0, 1],
        help="1=enable AMP + TF32 (CUDA only), 0=disable.",
    )
    parser.add_argument(
        "-output-folder",
        type=str,
        default=".",
        help="Base folder for all outputs (run folders, plots, data).",
    )
    args = parser.parse_args()

    config = load_json_config(args.config, required_keys=REQUIRED_CONFIG_KEYS)

    K_IMAGES = int(config["K_IMAGES"])
    IMAGE_SIZE = int(config["IMAGE_SIZE"])
    BATCH_SIZE = int(config["BATCH_SIZE"])
    DATASET_NAMES = config["DATASET_NAMES"]
    NUM_EPOCHS = int(config["NUM_EPOCHS"])
    VIZ_INTERVAL = int(config["VIZ_INTERVAL"])
    NUM_WORKERS = int(config["NUM_WORKERS"])
    PERSISTENT_WORKERS = bool(config["PERSISTENT_WORKERS"])
    PREFETCH_FACTOR = int(config["PREFETCH_FACTOR"])
    SEED = int(config["SEED"])
    LR = float(config["LR"])
    GRAD_LOSS_WEIGHT = float(config.get("GRAD_LOSS_WEIGHT", 0.0))

    activation = config.get("ACTIVATION", "siren").lower()
    device, use_amp = configure_device(args.use_amp_tf32)
    if activation == "siren" and use_amp:
        print("SIREN + AMP can produce NaNs; disabling AMP for this run.")
        use_amp = False
    print(f"Running on {device}")
    set_global_seed(SEED)

    output_folder = args.output_folder or "."
    config_name = config_name_from_path(args.config)

    images_tensor, image_sources, image_channels = get_medmnist_data(
        image_size=IMAGE_SIZE,
        num_images=K_IMAGES,
        dataset_names=DATASET_NAMES,
        seed=SEED,
    )
    print(f"Training on tensor shape: {images_tensor.shape}")
    # Infer value_dim from data.
    value_dim = images_tensor.shape[1]

    run_config = dict(config)
    run_config["config_path"] = args.config
    run_config["output_folder"] = output_folder
    run_config["device"] = device.type
    run_folder = make_run_folder(output_folder, f"run_vq_{config_name}", run_config)

    all_images_original = [img.permute(1, 2, 0).contiguous() for img in images_tensor]
    # Keep eval images at original channel count for PSNR/plots.
    all_images_eval = []
    for img, orig_c in zip(all_images_original, image_channels):
        if orig_c == 1:
            all_images_eval.append(img[..., :1])
        else:
            all_images_eval.append(img)
    track_indices = build_track_indices(image_sources, DATASET_NAMES)
    print(f"Tracking indices for viz & stats: {track_indices}")

    dataset = FullImageDataset(all_images_original, image_channels=image_channels)

    dataloader = build_dataloader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        seed=SEED,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    vqinr = _create_model(config, len(all_images_original), value_dim)

    if VIZ_INTERVAL <= 0:
        raise ValueError("VIZ_INTERVAL must be a positive integer.")
    # Include epoch 0 snapshot, then regular intervals.
    visualization_intervals = [0] + list(
        range(VIZ_INTERVAL, NUM_EPOCHS + 1, VIZ_INTERVAL)
    )

    module = MultiImageINRModule(
        vqinr,
        all_images_eval,
        track_indices,
        visualization_intervals,
        lr=LR,
        grad_loss_weight=GRAD_LOSS_WEIGHT,
        log_every_epoch=bool(args.debug),
    )

    precision = "16-mixed" if use_amp else 32
    accelerator = "gpu" if device.type == "cuda" else "cpu"

    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator=accelerator,
        devices=1,
        precision=precision,
        enable_checkpointing=False,
        logger=False,
        log_every_n_steps=1,
    )

    print("\nStarting Training...")
    trainer.fit(module, dataloader)

    save_all_visualizations(
        module,
        all_images_eval,
        visualization_intervals,
        run_folder,
    )


if __name__ == "__main__":
    main()
