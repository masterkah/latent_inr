import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
import os
import json

# local modules
from src.dataset import PixelPointDataset, get_medmnist_data
from src.model import AutoDecoderCNNWrapper, FourierFeatures
from src.utils import (
    setup_experiment_folder,
    evaluate_dataset_psnr,
    save_reference_reconstructions,
    plot_tsne,
    count_parameters,
    set_global_seed,
)

def _load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    # Keep required keys explicit so config files are self-documenting and validated.
    required_keys = [
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
        "LATENT_SIZES",
        "LATENT_SPATIAL_DIMS",
        "NUM_WORKERS",
        "PERSISTENT_WORKERS",
        "PREFETCH_FACTOR",
    ]
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")
    return config


def train(config_path, debug=0, use_amp_tf32=1, output_folder="."):
    # Config note:
    # - Classic "latent size only" mode: use latent_spatial_dims=[1] and sweep latent_sizes.
    #   This disables the conv and uses one vector per image.
    # - Multires mode: set latent_spatial_dims to >1 values and choose latent_sizes divisible by (s^2).
    #   latent_size always means total latent parameters per image; latent_feature_dim controls decoder input width.
    config = _load_config(config_path)
    # -------------------------------------- [Config] --------------------------------------
    # dataset
    K_IMAGES = int(config["K_IMAGES"])
    IMAGE_SIZE = int(config["IMAGE_SIZE"])
    BATCH_SIZE = int(config["BATCH_SIZE"])
    DATASET_NAMES = config["DATASET_NAMES"]

    # latent init
    INIT_SIGMA = float(config["INIT_SIGMA"])

    # training
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_EPOCHS = int(config["NUM_EPOCHS"])
    VIZ_INTERVAL = int(config["VIZ_INTERVAL"])
    DECODER_LR = float(config["DECODER_LR"])
    LATENT_LR = float(config["LATENT_LR"])

    # model size
    HIDDEN_DIM = int(config["HIDDEN_DIM"])
    NUM_LAYERS = int(config["NUM_LAYERS"])
    MODEL_DROPOUT = float(config["MODEL_DROPOUT"])

    # fourrier features
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
    LATENT_FEATURE_DIM = config["LATENT_FEATURE_DIM"]
    latent_sizes = config["LATENT_SIZES"]
    latent_spatial_dims = config["LATENT_SPATIAL_DIMS"]
    num_workers = int(config["NUM_WORKERS"])
    persistent_workers = bool(config["PERSISTENT_WORKERS"])
    prefetch_factor = int(config["PREFETCH_FACTOR"])

    # -------------------------------------- [End Config] -----------------------------------------

    print(f"Running on {DEVICE}")
    set_global_seed(SEED)
    print(f"Using global seed: {SEED}")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(use_amp_tf32)
        torch.backends.cudnn.allow_tf32 = bool(use_amp_tf32)
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision(
                "high" if use_amp_tf32 else "highest"
            )
    use_amp = bool(use_amp_tf32) and torch.cuda.is_available()

    output_folder = output_folder or "."
    os.makedirs(output_folder, exist_ok=True)

    # --- Data Loading ---
    images_tensor, image_sources, image_channels = get_medmnist_data(
        image_size=IMAGE_SIZE,
        num_images=K_IMAGES,
        dataset_names=DATASET_NAMES,
        seed=SEED,
    )
    print(
        f"Training on tensor shape: {images_tensor.shape}"
    )  # (K_IMAGES, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)

    # --- Create coordinate dataset ---
    dataset = PixelPointDataset(
        images_tensor, image_sources=image_sources, image_channels=image_channels
    )
    data_loader_generator = torch.Generator()
    data_loader_generator.manual_seed(SEED)
    # Only set worker-specific options when workers are enabled to avoid DataLoader warnings.
    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "shuffle": True,
        "generator": data_loader_generator,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        dataset,
        **loader_kwargs,
    )

    # --- SWEEP LATENT SIZES ---
    average_psnr_histories = {}  # {latent_size: {s: (steps, vals)}}

    for latent_size in latent_sizes:
        for latent_spatial_dim in latent_spatial_dims:
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
                    latent_channels if LATENT_FEATURE_DIM is None else LATENT_FEATURE_DIM
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

            # --> Positional encoding
            fourier_features = FourierFeatures(
                coord_size=2, freq_num=FF_FREQS, freq_scale=FF_SCALE
            )

            out_channels = dataset.C

            # Persist the run configuration for reproducibility.
            config = {
                "config_path": config_path,
                "latent_size": latent_size,
                "latent_feature_dim": latent_feature_dim,
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
            run_folder = setup_experiment_folder(
                config,
                base_name=os.path.join(
                    output_folder, f"run_latent_{latent_size}_s{latent_spatial_dim}"
                ),
            )

            # --> Logs stuff
            # Struct memo: {'steps': [0, 100...], 'data': {0: [], 1: []...}}
            per_image_logs = {"steps": [], "data": {k: [] for k in range(K_IMAGES)}}
            avg_psnr_history = []  # For the final latent comparison plot

            # --> Model init (multi-resolution wrapper)
            model = AutoDecoderCNNWrapper(
                num_images=K_IMAGES,
                hidden_dim=HIDDEN_DIM,
                pos_encoder=fourier_features,
                latent_feature_dim=latent_feature_dim,
                latent_channels=latent_channels,
                latent_spatial_dim=latent_spatial_dim,
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
            step = 0  # global step counter across epochs
            last_avg_psnr = None  # track most recent avg psnr for lightweight logging
            for epoch in range(NUM_EPOCHS):
                print(f"--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")
                for batch_indices, batch_coords, batch_targets in train_loader:
                    batch_indices = batch_indices.to(DEVICE, non_blocking=True)
                    batch_coords = batch_coords.to(DEVICE, non_blocking=True)
                    batch_targets = batch_targets.to(DEVICE, non_blocking=True)

                    # Forward Pass
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast(device_type=DEVICE.type, enabled=use_amp):
                        pred_vals, _ = model(batch_indices, batch_coords)

                        # Loss Calculation (see Eq 9 in the paper)
                        rec_loss = loss_criterion(pred_vals, batch_targets)

                        total_loss = rec_loss

                    # Backprop
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    # Visualization trigger
                    if step % VIZ_INTERVAL == 0:
                        print(f"  [Step {step}] logging metrics & images...")

                        # Evaluate PSNR (per-image)
                        psnr_dict = evaluate_dataset_psnr(model, dataset, DEVICE)

                        # Log data
                        per_image_logs["steps"].append(step)
                        current_avg = 0
                        for k, val in psnr_dict.items():
                            per_image_logs["data"][k].append(val)
                            current_avg += val
                        last_avg_psnr = current_avg / K_IMAGES
                        avg_psnr_history.append(last_avg_psnr)

                        # Save images & t-SNE
                        save_reference_reconstructions(
                            model, dataset, step, run_folder, DEVICE
                        )
                        plot_tsne(
                            model,
                            dataset,
                            step,
                            run_folder,
                            expected_num_clusters=len(DATASET_NAMES),
                        )

                    if (debug and step % 100 == 0) or (
                        (not debug) and step % VIZ_INTERVAL == 0
                    ):
                        psnr_display = (
                            f"{last_avg_psnr:.2f} dB"
                            if last_avg_psnr is not None
                            else "n/a"
                        )
                        print(
                            f"Step {step} | Loss crit: {rec_loss.item():.6f} | Total: {total_loss.item():.6f} | Avg PSNR: {psnr_display}"
                        )

                    step += 1

            average_psnr_histories.setdefault(latent_size, {})[
                latent_spatial_dim
            ] = (per_image_logs["steps"], avg_psnr_history)
            print(f"=== Finished Run {latent_size} | s={latent_spatial_dim} ===")

    if len(latent_spatial_dims) == 1 and latent_spatial_dims[0] == 1:
        # FINAL COMPARISON PLOT (avg psnr vs latent sizes)
        print("\nGenerating Final Latent Size Comparison...")
        plt.figure(figsize=(10, 6))

        only_s = latent_spatial_dims[0]
        for latent_size, s_dict in average_psnr_histories.items():
            steps, vals = s_dict.get(only_s, ([], []))
            plt.plot(steps, vals, marker="o", label=f"Latent Size {latent_size}")
        plt.title("Average Dataset PSNR vs Latent Size")
        plt.xlabel("Training Steps")
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
            for latent_spatial_dim, (steps, vals) in s_dict.items():
                plt.plot(
                    steps,
                    vals,
                    marker="o",
                    label=f"{latent_size}_s{latent_spatial_dim}",
                )
        plt.title("Average Dataset PSNR vs Steps (All Runs)")
        plt.xlabel("Training Steps")
        plt.ylabel("Average PSNR (dB)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_folder, "latent_all_runs_psnr_steps.png"))

        for latent_size, s_dict in average_psnr_histories.items():
            plt.figure(figsize=(10, 6))
            for latent_spatial_dim, (steps, vals) in s_dict.items():
                plt.plot(steps, vals, marker="o", label=f"s={latent_spatial_dim}")
            plt.title(f"Average Dataset PSNR vs Steps (Latent Size {latent_size})")
            plt.xlabel("Training Steps")
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
            plt.plot(s_vals, psnr_vals, marker="o", label=f"Latent Size {latent_size}")
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
        help="1=verbose step logging, 0=log only at VIZ_INTERVAL",
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
