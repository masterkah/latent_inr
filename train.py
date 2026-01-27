import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
import os

# local modules
from src.dataset import PixelPointDataset, get_medmnist_data
from src.model import AutoDecoderCNNWrapper, FourierFeatures
from src.utils import (
    load_json_config,
    configure_device,
    build_dataloader,
    make_run_folder,
    save_checkpoint,
    evaluate_dataset_psnr,
    save_reference_reconstructions,
    plot_tsne,
    count_parameters,
    set_global_seed,
)


REQUIRED_CONFIG_KEYS = [
    "K_IMAGES",
    "IMAGE_SIZE",
    "BATCH_SIZE",
    "DATASET_NAMES",
    "INIT_SIGMA",
    "NUM_EPOCHS",
    "VIZ_INTERVAL",
    "DECODER_LR",
    "LATENT_LR",
    "HIDDEN_DIM",
    "NUM_LAYERS",
    "MODEL_DROPOUT",
    "FF_FREQS",
    "FF_SCALE",
    "SEED",
    "LATENT_FEATURE_DIM",
    "CONV_KERNEL_SIZE",
    "LATENT_SIZES",
    "LATENT_SPATIAL_DIMS",
    "NUM_WORKERS",
    "PERSISTENT_WORKERS",
    "PREFETCH_FACTOR",
]


def train(config_path, debug=0, use_amp_tf32=1, output_folder="."):
    # Config note:
    # - Classic "latent size only" mode: use latent_spatial_dims=[1] and sweep latent_sizes.
    #   This disables the conv and uses one vector per image.
    # - Multires mode: set latent_spatial_dims to >1 values and choose latent_sizes divisible by (s^2).
    #   latent_size always means total latent parameters per image; latent_feature_dim controls decoder input width.
    config = load_json_config(config_path, required_keys=REQUIRED_CONFIG_KEYS)
    # -------------------------------------- [Config] --------------------------------------
    # dataset
    K_IMAGES = int(config["K_IMAGES"])
    IMAGE_SIZE = int(config["IMAGE_SIZE"])
    BATCH_SIZE = int(config["BATCH_SIZE"])
    DATASET_NAMES = config["DATASET_NAMES"]

    # latent init
    INIT_SIGMA = float(config["INIT_SIGMA"])

    # training
    DEVICE, use_amp = configure_device(use_amp_tf32)
    NUM_EPOCHS = int(config["NUM_EPOCHS"])
    VIZ_INTERVAL = int(config["VIZ_INTERVAL"])  # in epochs
    DECODER_LR = float(config["DECODER_LR"])
    LATENT_LR = float(config["LATENT_LR"])

    # model size
    HIDDEN_DIM = int(config["HIDDEN_DIM"])
    NUM_LAYERS = int(config["NUM_LAYERS"])
    MODEL_DROPOUT = float(config["MODEL_DROPOUT"])

    # Fourier features
    FF_FREQS = int(config["FF_FREQS"])
    FF_SCALE = float(config["FF_SCALE"])

    # rng
    SEED = int(config["SEED"])

    # latent resolution stuff
    # latent_size = total latent parameters per image
    # latent_feature_dim = decoder input channels after conv. Keep it fixed to compare
    # spatial resolutions fairly; if set to None, it follows latent_channels and
    # the decoder input width changes with latent_spatial_dim. For s=1, we always
    # use latent_feature_dim = latent_channels to recover the classic behavior.
    # conv_kernel_size controls the shared conv for s>1; padding preserves spatial dims.
    LATENT_FEATURE_DIM = config["LATENT_FEATURE_DIM"]
    conv_kernel_size_value = config["CONV_KERNEL_SIZE"]
    CONV_KERNEL_SIZE = None if conv_kernel_size_value is None else int(conv_kernel_size_value)
    latent_sizes = config["LATENT_SIZES"]
    latent_spatial_dims = config["LATENT_SPATIAL_DIMS"]
    num_workers = int(config["NUM_WORKERS"])
    persistent_workers = bool(config["PERSISTENT_WORKERS"])
    prefetch_factor = int(config["PREFETCH_FACTOR"])

    # -------------------------------------- [End Config] -----------------------------------------

    print(f"Running on {DEVICE}")
    set_global_seed(SEED)
    print(f"Using global seed: {SEED}")
    output_folder = output_folder or "."
    os.makedirs(output_folder, exist_ok=True)

    # --- Data Loading ---
    images_tensor, image_sources, image_channels = get_medmnist_data(
        image_size=IMAGE_SIZE,
        num_images=K_IMAGES,
        dataset_names=DATASET_NAMES,
        seed=SEED,
    )
    print(f"Training on tensor shape: {images_tensor.shape}")

    # --- Create coordinate dataset ---
    dataset = PixelPointDataset(
        images_tensor, image_sources=image_sources, image_channels=image_channels
    )
    # DataLoader is built per run to keep shuffles identical across runs.
    # --- SWEEP LATENT SIZES ---
    average_psnr_histories = {}  # {latent_size: {s: (epochs, vals)}}
    # Reuse one positional encoder so all runs share the exact same random features.
    fourier_features = FourierFeatures(
        coord_size=2, freq_num=FF_FREQS, freq_scale=FF_SCALE
    )

    for latent_size in latent_sizes:
        for latent_spatial_dim in latent_spatial_dims:
            # Reset RNGs so each run is deterministic and independent of prior runs.
            set_global_seed(SEED)

            # Recreate the loader with a fixed seed so shuffles match across runs.
            train_loader = build_dataloader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=num_workers,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
                seed=SEED,
                pin_memory=torch.cuda.is_available(),
                drop_last=False,
            )
            spatial_area = latent_spatial_dim * latent_spatial_dim
            if latent_size % spatial_area != 0:
                raise ValueError(
                    f"latent_size ({latent_size}) must be divisible by latent_spatial_dim^2 "
                    f"({spatial_area})."
                )
            latent_channels = latent_size // spatial_area
            if latent_channels == 0:
                raise ValueError(
                    f"latent_size ({latent_size}) is too small for latent_spatial_dim "
                    f"({latent_spatial_dim})."
                )
            if latent_spatial_dim == 1:
                # Classic single-vector behavior: decoder sees one latent per image.
                latent_feature_dim = latent_channels
            else:
                latent_feature_dim = (
                    latent_channels
                    if LATENT_FEATURE_DIM is None
                    else LATENT_FEATURE_DIM
                )

            print(
                f"\n=== Starting Run: Latent Size {latent_size} | s={latent_spatial_dim} ==="
            )
            print(
                f"Latent params per image: size={latent_size}, "
                f"spatial_dim={latent_spatial_dim}, "
                f"channels={latent_channels}, "
                f"feature_dim={latent_feature_dim}"
            )

            out_channels = dataset.C

            # Persist the run configuration for reproducibility.
            config = {
                "config_path": config_path,
                "latent_size": latent_size,
                "latent_feature_dim": latent_feature_dim,
                "conv_kernel_size": CONV_KERNEL_SIZE,
                "hidden_dim": HIDDEN_DIM,
                "latent_channels": latent_channels,
                "latent_spatial_dim": latent_spatial_dim,
                "num_layers": NUM_LAYERS,
                "img_size": IMAGE_SIZE,
                "nb_images": K_IMAGES,
                "batch_size": BATCH_SIZE,
                "max_epochs": NUM_EPOCHS,
                "device": DEVICE.__str__(),
                "init_latent_variance": INIT_SIGMA,
                "ff_frequency": FF_FREQS,
                "ff_scale": FF_SCALE,
                "out_channels": out_channels,
                "dataset_names": DATASET_NAMES,
                "dropout": MODEL_DROPOUT,
                "seed": SEED,
                "output_folder": output_folder,
            }
            # --> Setup run folder
            run_folder = make_run_folder(
                output_folder,
                f"run_latent_{latent_size}_s{latent_spatial_dim}",
                config,
            )

            # --> Logs stuff
            # Struct memo: {'epochs': [0, 10...], 'data': {0: [], 1: []...}}
            per_image_logs = {"epochs": [], "data": {k: [] for k in range(K_IMAGES)}}
            avg_psnr_history = []  # For the final latent comparison plot

            # --> Model init (multi-resolution wrapper)
            model = AutoDecoderCNNWrapper(
                num_images=K_IMAGES,
                hidden_dim=HIDDEN_DIM,
                pos_encoder=fourier_features,
                latent_feature_dim=latent_feature_dim,
                latent_channels=latent_channels,
                latent_spatial_dim=latent_spatial_dim,
                conv_kernel_size=CONV_KERNEL_SIZE,
                sigma=INIT_SIGMA,
                num_layers=NUM_LAYERS,
                out_channels=out_channels,
                dropout_prob=MODEL_DROPOUT,
            ).to(DEVICE)

            # --> Quick check model size and compression stats
            decoder_params = count_parameters(model.decoder)
            conv_params = count_parameters(model.conv)
            latent_params = model.latents.numel()
            print(
                f"\nDecoder params: {decoder_params} | Conv params: {conv_params} | "
                f"Latent params: {latent_params} | Total: {decoder_params + conv_params + latent_params}"
            )
            total_original_pixels = dataset.H * dataset.W * sum(dataset.image_channels)
            compression_model_only = (
                decoder_params + conv_params
            ) / total_original_pixels
            compression_with_latents = (
                decoder_params + conv_params + latent_params
            ) / total_original_pixels
            print(
                f"Total original pixels in dataset: {total_original_pixels} | "
                f"Compression (model only): {compression_model_only:.6f} | "
                f"Compression (model + latents): {compression_with_latents:.6f}"
            )

            # --- Optimizer ---
            optimizer = torch.optim.Adam(
                [
                    {
                        "params": list(model.decoder.parameters())
                        + list(model.conv.parameters()),
                        "lr": DECODER_LR,
                    },
                    {
                        "params": [model.latents],
                        "lr": LATENT_LR,
                    },
                ]
            )

            loss_criterion = nn.MSELoss()  # Reconstruction loss (L in the paper).
            scaler = torch.amp.GradScaler(enabled=use_amp)

            # -------------------------------------- [Training] --------------------------------------
            print("\n --- Starting Training... ---\n")
            last_avg_psnr = None  # track most recent avg psnr for lightweight logging
            for epoch in range(NUM_EPOCHS):
                last_rec_loss = None
                for batch_indices, batch_coords, batch_targets in train_loader:
                    batch_indices = batch_indices.to(DEVICE, non_blocking=True)
                    batch_coords = batch_coords.to(DEVICE, non_blocking=True)
                    batch_targets = batch_targets.to(DEVICE, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast(device_type=DEVICE.type, enabled=use_amp):
                        pred_vals, _ = model(batch_indices, batch_coords)

                        rec_loss = loss_criterion(pred_vals, batch_targets)

                    last_rec_loss = rec_loss

                    scaler.scale(rec_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                epoch_idx = epoch + 1
                # Visualization trigger (epoch-based)
                if epoch_idx % VIZ_INTERVAL == 0:
                    print(f"  [Epoch {epoch_idx}] logging metrics & images...")

                    # Evaluate PSNR (per-image)
                    psnr_dict = evaluate_dataset_psnr(model, dataset, DEVICE)

                    # Log data
                    per_image_logs["epochs"].append(epoch_idx)
                    current_avg = 0
                    for k, val in psnr_dict.items():
                        per_image_logs["data"][k].append(val)
                        current_avg += val
                    last_avg_psnr = current_avg / K_IMAGES
                    avg_psnr_history.append(last_avg_psnr)

                    # Save images & t-SNE
                    save_reference_reconstructions(
                        model, dataset, epoch_idx, run_folder, DEVICE
                    )
                    plot_tsne(
                        model,
                        dataset,
                        epoch_idx,
                        run_folder,
                        expected_num_clusters=len(DATASET_NAMES),
                    )

            if last_rec_loss is not None and (
                debug or ((not debug) and epoch_idx % VIZ_INTERVAL == 0)
            ):
                psnr_display = (
                    f"{last_avg_psnr:.2f} dB"
                    if last_avg_psnr is not None
                    else "n/a"
                )
                print(
                    f"Epoch {epoch_idx} | Rec loss: {last_rec_loss.item():.6f} | Avg PSNR: {psnr_display}"
                )

            # Save final model state for inference reuse (once per run).
            run_tag = f"run_latent_{latent_size}_s{latent_spatial_dim}"
            save_checkpoint(
                model,
                optimizer,
                step=epoch_idx,
                config=config,
                run_folder=output_folder,
                filename=f"{run_tag}.pth",
                subdir="models",
            )

            average_psnr_histories.setdefault(latent_size, {})[latent_spatial_dim] = (
                per_image_logs["epochs"],
                avg_psnr_history,
            )
            print(f"=== Finished Run {latent_size} | s={latent_spatial_dim} ===")

    if len(latent_spatial_dims) == 1 and latent_spatial_dims[0] == 1:
        # FINAL COMPARISON PLOT (avg psnr vs latent sizes)
        print("\nGenerating Final Latent Size Comparison...")
        plt.figure(figsize=(10, 6))

        only_s = latent_spatial_dims[0]
        for latent_size, s_dict in average_psnr_histories.items():
            epochs, vals = s_dict.get(only_s, ([], []))
            plt.plot(
                epochs,
                vals,
                marker="o",
                markersize=3,
                label=f"Latent Size {latent_size}",
            )
        plt.title("Average Dataset PSNR vs Latent Size")
        plt.xlabel("Epochs")
        plt.ylabel("Average PSNR (dB)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_folder, "latent_size_comparison_sweep.png"))
        print("Done.")
    else:
        print("\nGenerating Latent Resolution Comparisons...")
        # Combined view across all (latent_size, s) pairs.
        plt.figure(figsize=(10, 6))
        for latent_size, s_dict in average_psnr_histories.items():
            for latent_spatial_dim, (epochs, vals) in s_dict.items():
                plt.plot(
                    epochs,
                    vals,
                    marker="o",
                    markersize=3,
                    label=f"{latent_size}_s{latent_spatial_dim}",
                )
        plt.title("Average Dataset PSNR vs Epochs (All Runs)")
        plt.xlabel("Epochs")
        plt.ylabel("Average PSNR (dB)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_folder, "latent_all_runs_psnr_epochs.png"))

        for latent_size, s_dict in average_psnr_histories.items():
            plt.figure(figsize=(10, 6))
            for latent_spatial_dim, (epochs, vals) in s_dict.items():
                plt.plot(
                    epochs,
                    vals,
                    marker="o",
                    markersize=3,
                    label=f"s={latent_spatial_dim}",
                )
            plt.title(f"Average Dataset PSNR vs Epochs (Latent Size {latent_size})")
            plt.xlabel("Epochs")
            plt.ylabel("Average PSNR (dB)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(
                os.path.join(
                    output_folder,
                    f"latent_size_{latent_size}_spatial_sweep.png",
                )
            )

        # Final PSNR vs s at fixed latent size
        plt.figure(figsize=(10, 6))
        for latent_size, s_dict in average_psnr_histories.items():
            s_vals = []
            psnr_vals = []
            for s, (_, vals) in s_dict.items():
                if len(vals) == 0:
                    continue
                s_vals.append(s)
                psnr_vals.append(vals[-1])
            if len(s_vals) == 0:
                continue
            order = sorted(range(len(s_vals)), key=lambda i: s_vals[i])
            s_vals = [s_vals[i] for i in order]
            psnr_vals = [psnr_vals[i] for i in order]
            plt.plot(
                s_vals,
                psnr_vals,
                marker="o",
                markersize=3,
                label=f"Latent Size {latent_size}",
            )
        plt.title("Final PSNR vs Latent Spatial Dim")
        plt.xlabel("Latent Spatial Dim (s)")
        plt.ylabel("Average PSNR (dB)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_folder, "latent_spatial_comparison.png"))
        print("Done.")


if __name__ == "__main__":
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
        help="1=log every epoch, 0=log only at VIZ_INTERVAL",
    )
    parser.add_argument(
        "-use-amp-tf32",
        type=int,
        default=1,
        choices=[0, 1],
        help="1=enable AMP + TF32 (CUDA only), 0=disable",
    )
    parser.add_argument(
        "-output-folder",
        type=str,
        default=".",
        help="Base folder for all outputs (run folders, plots, data).",
    )
    args = parser.parse_args()
    train(
        config_path=args.config,
        debug=args.debug,
        use_amp_tf32=args.use_amp_tf32,
        output_folder=args.output_folder,
    )
