import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# local modules
from src.dataset import PixelPointDataset, get_medmnist_data
from src.model import AutoDecoderWrapper, FourierFeatures
from src.utils import (
    setup_experiment_folder, 
    evaluate_dataset_psnr, 
    save_all_reconstructions, 
    plot_tsne, 
    plot_psnr_per_image,
    count_parameters
)

# ------------------------------------------
#  Training
# ------------------------------------------

def train():

    # -------------------------------------- [Config] --------------------------------------
    # Using 64x64 images first, train in ~ 8 minutes with base parameters
    IMAGE_SIZE = 64 # [TODO] Let's try with 224 later
    BATCH_SIZE = 2048 # needs large batches of pixel (makes sense)
    MAX_STEPS = 10001
    VISUALIZATION_STEPS = [0, 100, 500, 1000, 5000, 10000]
    K_IMAGES = 15 # Extract first 5 images for now
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    REG_WEIGHT = 10000 # [TODO] fix Regularization weight (1 / sigma^2) where sigma=0.01 -> weight=10000
    HIDDEN_DIM = 512
    INIT_SIGMA = 10 # YESSSS A LOT OF VARIANCE SEEMS TO SOLVE MODE COLLAPSEEEEE

    # for fourrier features
    FF_FREQS = 96 #[TODO] play with this too
    FF_SCALE = 1.0
    # -------------------------------------- [End Config] -----------------------------------------

    print(f"Running on {DEVICE}")

    # --- Data Loading (PneumoniaMNIST) ---
    images_tensor = get_medmnist_data(image_size=IMAGE_SIZE, num_images=K_IMAGES)
    print(f"Training on tensor shape: {images_tensor.shape}") # (K_IMAGES, 1, IMAGE_SIZE, IMAGE_SIZE)

    # --- Create coordinate dataset ---
    dataset = PixelPointDataset(images_tensor)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- SWEEP LATENT SIZES (try many latent dims) ---
    latent_sizes = [8, 16, 32, 64, 96]
    average_psnr_histories = {} # To compare latent sizes at the end

    for z_dim in latent_sizes:
        print(f"\n=== Starting Run: Latent Dim {z_dim} ===")

        # --> Setup run folder
        config = {
            "latent_dim": z_dim,
            "hidden_dim": HIDDEN_DIM,
            "steps": MAX_STEPS,
            "img_size": IMAGE_SIZE,
            "nb_images" : K_IMAGES,
            "batch_size": BATCH_SIZE,
            "max_steps": MAX_STEPS,
            "device" : DEVICE,
            "regularization_weight": REG_WEIGHT,
            "init_latent_variance": INIT_SIGMA,
            "ff_frequency": FF_FREQS,
            "ff_scale" : FF_SCALE
        }
        run_folder = setup_experiment_folder(config, base_name=f"run_latent_{z_dim}")

        # --> Logs stuff
        # Struct memo: {'steps': [0, 100...], 'data': {0: [], 1: []...}}
        per_image_logs = {'steps': [], 'data': {k: [] for k in range(K_IMAGES)}}
        avg_psnr_history = [] # For the final latent comparison plot

        # --> Model init
        fourier_features = FourierFeatures(coord_size=2, freq_num=FF_FREQS, freq_scale=FF_SCALE) # for now coord_size is hardcoded (we only play with image coordinates as a start)
        model = AutoDecoderWrapper(
            num_images=K_IMAGES,
            hidden_dim=HIDDEN_DIM,
            coord_size=fourier_features.out_size, # important, we go from (x,y) to (x'1,...x'FF_FREQS*2) with fourier features
            pos_encoder=fourier_features,
            latent_dim=z_dim,
            sigma=INIT_SIGMA,
            out_channels=1 # [TODO] PneumoniaMNIST is grayscale but configure that later
        ).to(DEVICE)

        # --> Quick check
        nb_params = count_parameters(model)
        print(f"The model has ~{nb_params/1000.0:.1f} {nb_params} trainable parameters !\n")

        # --- Optimizer ---
        optimizer = torch.optim.Adam([
            {'params': model.decoder.parameters(), 'lr': 5e-5}, # [TODO] adapt, normally they use 1e-5 * B where B is the nb of "shapes" (images for us) per batch
            {'params': model.latents.parameters(), 'lr': 1e-3}  # "Learning rate for the latent vectors was set to be 1e-3"
        ])

        loss_criterion = nn.MSELoss() # [TODO] This is the L in the paper, I chose to use MSE for now. Try their clamped distance thing

        # -------------------------------------- [Training] --------------------------------------
        print("\n --- Starting Training... ---\n")
        step = 0
        data_iter = iter(train_loader) # yes, a dataloader is an iterable, not an iterator

        while step <= MAX_STEPS:

            # 1 -> Get batch ([TODO] we reload the iterator when we have reached the end of the dataset. Let's see how it works out like this for now)
            try:
                batch_indices, batch_coords, batch_targets = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch_indices, batch_coords, batch_targets = next(data_iter)

            batch_indices = batch_indices.to(DEVICE)
            batch_coords = batch_coords.to(DEVICE)
            batch_targets = batch_targets.to(DEVICE)

            # Visualization trigger (<3)
            if step in VISUALIZATION_STEPS:
                print(f"  [Step {step}] logging metrics & images...")

                # Evaluate PSNR (per-image)
                psnr_dict = evaluate_dataset_psnr(model, dataset, DEVICE)

                # Log data
                per_image_logs['steps'].append(step)
                current_avg = 0
                for k, val in psnr_dict.items():
                    per_image_logs['data'][k].append(val)
                    current_avg += val
                avg_psnr_history.append(current_avg / K_IMAGES)

                # Save images & t-SNE
                save_all_reconstructions(model, dataset, step, run_folder, DEVICE)
                plot_tsne(model, step, run_folder)

            # 2 -> Forward Pass !!
            optimizer.zero_grad()
            pred_vals, z_batch = model(batch_indices, batch_coords)

            # 3 -> Loss Calculation (see Eq 9 in the paper)
            ### Reconstruction loss
            rec_loss = loss_criterion(pred_vals, batch_targets)

            ### Latent regularization loss
            # It penalizes the magnitude of latent codes used in this batch
            reg_loss = torch.mean(torch.norm(z_batch, dim=1)**2) * REG_WEIGHT

            total_loss = rec_loss + reg_loss

            # 5. Backprop !!
            total_loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print(f"Step {step} | Loss crit: {rec_loss.item():.6f} | Reg: {reg_loss.item():.6f} | Total: {total_loss.item():.6f}")

            step += 1

        # End of run: save per-image plot
        plot_psnr_per_image(per_image_logs, run_folder)
        average_psnr_histories[z_dim] = (per_image_logs['steps'], avg_psnr_history)
        print(f"=== Finished Run {z_dim} ===")

    # FINAL COMPARISON PLOT (avg psnr vs latent dims)
    print("\nGenerating Final Latent Sweep Comparison...")
    plt.figure(figsize=(10, 6))

    for z_dim, (steps, vals) in average_psnr_histories.items():
        plt.plot(steps, vals, marker='o', label=f"Latent Dim {z_dim}")
    plt.title("Average Dataset PSNR vs Latent Size")
    plt.xlabel("Training Steps")
    plt.ylabel("Average PSNR (dB)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("latent_size_comparison_sweep.png")
    plt.show()
    print("Done.")


if __name__ == "__main__":
    train()