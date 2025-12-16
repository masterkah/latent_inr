import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

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


def train():
    # -------------------------------------- [Config] --------------------------------------
    # dataset
    K_IMAGES = 300
    IMAGE_SIZE = 64  # 224
    BATCH_SIZE = 1024
    DATASET_NAMES = [
        "pneumoniamnist",
        "pathmnist",
        "bloodmnist",
    ]

    # latent init
    INIT_SIGMA = 0.01

    # training
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_EPOCHS = 100
    VIZ_INTERVAL = 500  # log and visualize every N steps
    REG_WEIGHT = 0
    DECODER_LR_BASE = (
        1e-5  # base lr, scaled by nb of distinct shapes per batch (DeepSDF: 1e-5 * B)
    )
    LATENT_LR = 5e-3

    # model size
    # If None, we auto-set to the minimal valid width (input_dim + 1)
    # If too small, we auto-set to minimal valid width
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
        dataset, batch_size=BATCH_SIZE, shuffle=True, generator=data_loader_generator
    )

    # --- SWEEP LATENT SIZES ---
    latent_sizes = [16, 128, 300]
    average_psnr_histories = {}  # To compare latent sizes at the end

    for z_dim in latent_sizes:
        print(f"\n=== Starting Run: Latent Dim {z_dim} ===")

        # --> Positional encoding
        fourier_features = FourierFeatures(
            coord_size=2, freq_num=FF_FREQS, freq_scale=FF_SCALE
        )  # for now coord_size is hardcoded (we only play with image coordinates as a start)
        input_dim = z_dim + fourier_features.out_size
        min_hidden_dim = (
            input_dim + 1
        )  # minimal working width to keep the skip-connection layer positive
        hidden_dim = (
            max(HIDDEN_DIM, min_hidden_dim)
            if HIDDEN_DIM is not None
            else min_hidden_dim
        )

        out_channels = dataset.C

        config = {
            "latent_dim": z_dim,
            "hidden_dim": hidden_dim,
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
        }
        # --> Setup run folder
        run_folder = setup_experiment_folder(config, base_name=f"run_latent_{z_dim}")

        # --> Logs stuff
        # Struct memo: {'steps': [0, 100...], 'data': {0: [], 1: []...}}
        per_image_logs = {"steps": [], "data": {k: [] for k in range(K_IMAGES)}}
        avg_psnr_history = []  # For the final latent comparison plot

        # --> Model init
        model = AutoDecoderWrapper(
            num_images=K_IMAGES,
            hidden_dim=hidden_dim,
            coord_size=fourier_features.out_size,  # important, we go from (x,y) to (x'1,...x'FF_FREQS*2) with fourier features
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
                    "lr": DECODER_LR_BASE,  # will be scaled per batch by distinct shapes
                },
                {
                    "params": model.latents.parameters(),
                    "lr": LATENT_LR,
                },
            ]
        )
        # LR plateau tracking
        decoder_base_lr = DECODER_LR_BASE
        latent_lr = LATENT_LR
        plateau_best_psnr = -float("inf")
        plateau_bad_epochs = 0
        plateau_factor = 0.8
        plateau_patience = 2
        plateau_min_lr = 1e-6
        # Treat metric as plateaued when it wiggles within a tiny band for a couple evals
        plateau_flat_window = 2
        plateau_flat_eps = 0.1  # dB band
        plateau_improve_eps = 0.05  # dB improvement needed
        plateau_metric_history = []

        loss_criterion = (
            nn.MSELoss()
        )  # [TODO] This is the L in the paper, I chose to use MSE for now. Try their clamped distance thing

        # -------------------------------------- [Training] --------------------------------------
        print("\n --- Starting Training... ---\n")
        step = 0  # global step counter across epochs
        last_avg_psnr = None  # track most recent avg psnr for lightweight logging
        for epoch in range(NUM_EPOCHS):
            print(f"--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")
            for batch_indices, batch_coords, batch_targets in train_loader:
                batch_indices = batch_indices.to(DEVICE)
                batch_coords = batch_coords.to(DEVICE)
                batch_targets = batch_targets.to(DEVICE)

                # Scale decoder LR by number of distinct images in this batch (DeepSDF heuristic: 1e-5 * B)
                shape_count = batch_indices.unique().numel()
                decoder_lr = decoder_base_lr * shape_count
                optimizer.param_groups[0]["lr"] = decoder_lr
                optimizer.param_groups[1]["lr"] = latent_lr

                # 2 -> Forward Pass !!
                optimizer.zero_grad()
                pred_vals, z_batch = model(batch_indices, batch_coords)

                # 3 -> Loss Calculation (see Eq 9 in the paper)
                ### Reconstruction loss
                rec_loss = loss_criterion(pred_vals, batch_targets)

                ### Latent regularization loss
                # It penalizes the magnitude of latent codes used in this batch
                reg_loss = torch.mean(torch.norm(z_batch, dim=1) ** 2) * REG_WEIGHT

                total_loss = rec_loss + reg_loss

                # 5. Backprop !!
                total_loss.backward()
                optimizer.step()

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
                    plateau_metric_history.append(last_avg_psnr)

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

                    # Reduce LR on plateau (monitor average PSNR; maximize)
                    if last_avg_psnr is not None:
                        if last_avg_psnr > plateau_best_psnr + plateau_improve_eps:
                            plateau_best_psnr = last_avg_psnr
                            plateau_bad_epochs = 0
                        else:
                            is_flat = False
                            if len(plateau_metric_history) >= plateau_flat_window:
                                recent = plateau_metric_history[-plateau_flat_window:]
                                if max(recent) - min(recent) < plateau_flat_eps:
                                    is_flat = True
                            plateau_bad_epochs = (
                                plateau_bad_epochs + 1 if is_flat else 0
                            )
                            if plateau_bad_epochs >= plateau_patience:
                                decoder_base_lr = max(
                                    decoder_base_lr * plateau_factor, plateau_min_lr
                                )
                                latent_lr = max(
                                    latent_lr * plateau_factor, plateau_min_lr
                                )
                                plateau_bad_epochs = 0
                                print(
                                    f"    Plateau detected: reducing decoder base LR to {decoder_base_lr:.2e}, latent LR to {latent_lr:.2e}"
                                )

                if step % 100 == 0:
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
    plt.savefig("latent_size_comparison_sweep.png")
    print("Done.")


if __name__ == "__main__":
    train()
