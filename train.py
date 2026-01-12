import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
import os

# local modules
from src.dataset import PixelPointDataset, get_medmnist_data
from src.model import AutoDecoderWrapper, FourierFeatures
from src.utils import (
    setup_experiment_folder,
    evaluate_dataset_psnr,
    save_reference_reconstructions,
    plot_tsne,
    count_parameters,
    set_global_seed,
)

def train(debug=1, use_amp_tf32=1, output_folder="."):
    # -------------------------------------- [Config] --------------------------------------
    # dataset
    K_IMAGES = 3000
    IMAGE_SIZE = 64  # 224
    BATCH_SIZE = 32768
    DATASET_NAMES = [
        "pneumoniamnist",
        "pathmnist",
        "bloodmnist",
    ]

    # latent init
    INIT_SIGMA = 0.01

    # training
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_EPOCHS = 1000
    VIZ_INTERVAL = 384 # corresponds to once per epoch in this setup
    REG_WEIGHT = 0
    DECODER_LR = 1e-5
    LATENT_LR = 5e-3

    # model size
    HIDDEN_DIM = 512
    NUM_LAYERS = 6
    MODEL_DROPOUT = 0

    # fourrier features
    FF_FREQS = 128
    FF_SCALE = 1.25

    # rng
    SEED = 42
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
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=data_loader_generator,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # --- SWEEP LATENT SIZES ---
    latent_sizes = [16, 32, 64, 128, 256, 512]
    average_psnr_histories = {}  # To compare latent sizes at the end

    for z_dim in latent_sizes:
        print(f"\n=== Starting Run: Latent Dim {z_dim} ===")

        # --> Positional encoding
        fourier_features = FourierFeatures(
            coord_size=2, freq_num=FF_FREQS, freq_scale=FF_SCALE
        )  # for now coord_size is hardcoded (we only play with image coordinates as a start)

        out_channels = dataset.C

        config = {
            "latent_dim": z_dim,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "img_size": IMAGE_SIZE,
            "nb_images": K_IMAGES,
            "batch_size": BATCH_SIZE,
            "max_epochs": NUM_EPOCHS,
            "device": DEVICE.__str__(),
            "regularization_weight": REG_WEIGHT,
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
            config, base_name=os.path.join(output_folder, f"run_latent_{z_dim}")
        )

        # --> Logs stuff
        # Struct memo: {'steps': [0, 100...], 'data': {0: [], 1: []...}}
        per_image_logs = {"steps": [], "data": {k: [] for k in range(K_IMAGES)}}
        avg_psnr_history = []  # For the final latent comparison plot

        # --> Model init
        model = AutoDecoderWrapper(
            num_images=K_IMAGES,
            hidden_dim=HIDDEN_DIM,
            pos_encoder=fourier_features,
            latent_dim=z_dim,
            sigma=INIT_SIGMA,
            num_layers=NUM_LAYERS,
            out_channels=out_channels,
            dropout_prob=MODEL_DROPOUT,
        ).to(DEVICE)

        # --> Quick check model size and compression stats
        decoder_params = count_parameters(model.decoder)
        latent_params = count_parameters(model.latents)
        print(
            f"\nDecoder params: {decoder_params} | Latent codes: {latent_params} | Total: {decoder_params + latent_params}"
        )
        total_original_pixels = dataset.H * dataset.W * sum(dataset.image_channels)
        compression_decoder_only = decoder_params / total_original_pixels
        compression_with_latents = (
            decoder_params + latent_params
        ) / total_original_pixels
        print(
            f"Total original pixels in dataset: {total_original_pixels} | Compression (decoder only): {compression_decoder_only:.6f} | Compression (decoder + latents): {compression_with_latents:.6f}"
        )

        # --- Optimizer ---
        optimizer = torch.optim.Adam(
            [
                {
                    "params": model.decoder.parameters(),
                    "lr": DECODER_LR,
                },
                {
                    "params": model.latents.parameters(),
                    "lr": LATENT_LR,
                },
            ]
        )

        loss_criterion = (
            nn.MSELoss()
        )  # [TODO] This is the L in the paper, I chose to use MSE for now. Try their clamped distance thing
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

                # 2 -> Forward Pass !!
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=DEVICE.type, enabled=use_amp):
                    pred_vals, z_batch = model(batch_indices, batch_coords)

                    # 3 -> Loss Calculation (see Eq 9 in the paper)
                    ### Reconstruction loss
                    rec_loss = loss_criterion(pred_vals, batch_targets)

                    ### Latent regularization loss
                    # It penalizes the magnitude of latent codes used in this batch
                    reg_loss = torch.mean(torch.norm(z_batch, dim=1) ** 2) * REG_WEIGHT

                    total_loss = rec_loss + reg_loss

                # 5. Backprop !!
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
                        f"Step {step} | Loss crit: {rec_loss.item():.6f} | Reg: {reg_loss.item():.6f} | Total: {total_loss.item():.6f} | Avg PSNR: {psnr_display}"
                    )

                step += 1

        average_psnr_histories[z_dim] = (per_image_logs["steps"], avg_psnr_history)
        print(f"=== Finished Run {z_dim} ===")

    # FINAL COMPARISON PLOT (avg psnr vs latent dims)
    print("\nGenerating Final Latent Sweep Comparison...")
    plt.figure(figsize=(10, 6))

    for z_dim, (steps, vals) in average_psnr_histories.items():
        plt.plot(steps, vals, marker="o", label=f"Latent Dim {z_dim}")
    plt.title("Average Dataset PSNR vs Latent Size")
    plt.xlabel("Training Steps")
    plt.ylabel("Average PSNR (dB)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_folder, "latent_size_comparison_sweep.png"))
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-debug",
        type=int,
        default=1,
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
        debug=args.debug,
        use_amp_tf32=args.use_amp_tf32,
        output_folder=args.output_folder,
    )
