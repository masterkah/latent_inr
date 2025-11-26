from vqcoin import SIREN_FACTOR, INRLightningModule, RandomPointsDataset, VQINR, MLP, SineLayer, ShiftSineLayer, FixedINRLightningModule
from plot import plot_reconstructions, plot_scores, psnr

import torch
from torch.utils.data import Dataset, DataLoader

from medmnist import BreastMNIST, RetinaMNIST, PneumoniaMNIST
from torchvision.transforms.functional import pil_to_tensor
import matplotlib.pyplot as plt
import math
from datetime import datetime
import lightning as pl

import numpy as np

POINTS_PER_SAMPLE = 2048
HIDDEN_SIZE = 128
NUM_LAYERS = 3
LEARNING_RATE = 1e-3
TRAINING_EPOCHS = 10000

def initialize_siren_weights(network: MLP, omega: float):
    """ See SIREN paper supplement Sec. 1.5 for discussion """
    old_weights = network.layers[1].linear.weight.clone()
    with torch.no_grad():
        # First layer initialization
        num_input = network.layers[0].linear.weight.size(-1)
        network.layers[0].linear.weight.uniform_(-1 / num_input, 1 / num_input)
        # Subsequent layer initialization uses based on omega parameter
        for layer in network.layers[1:-1]:
            num_input = layer.linear.weight.size(-1)
            layer.linear.weight.uniform_(-math.sqrt(6 / num_input) / omega, math.sqrt(6 / num_input) / omega)
        # Final linear layer also uses initialization based on omega parameter
        num_input = network.layers[-1].weight.size(-1)
        network.layers[-1].weight.uniform_(-math.sqrt(6 / num_input) / omega, math.sqrt(6 / num_input) / omega)
        
    # Verify that weights did indeed change
    new_weights = network.layers[1].linear.weight
    assert (old_weights - new_weights).abs().sum() > 0.0

def initialize_vqinr_weights(model):
    with torch.no_grad():
        for name, module in model.decoder.named_modules():
            if isinstance(module, (ShiftSineLayer, SineLayer)):
                is_first = (module == model.decoder.layers[0])
                in_dim = module.linear.weight.shape[1]
                
                if is_first:
                    bound = 1 / in_dim
                    module.linear.weight.uniform_(-bound, bound)
                else:
                    bound = np.sqrt(6 / in_dim) / model.decoder.layers[0].siren_factor
                    module.linear.weight.uniform_(-bound, bound)
                
                if module.linear.bias is not None:
                    module.linear.bias.fill_(0.0)

if __name__ == "__main__":
    # Load ChestMNIST dataset
    IMAGE_SIZE = 224  # ChestMNIST offers sizes: 28, 64, 128, 224
    chest_dataset = PneumoniaMNIST(split="val",
                            download=True,
                            size=IMAGE_SIZE)

    # INRs are trained on only 1 scene. We only want 1 image.
    pil_image, _ = chest_dataset[1]

    gt_image = pil_to_tensor(pil_image)
    gt_image = gt_image.moveaxis(0, -1)  # Convert to torch.Tensor
    gt_image = gt_image.to(torch.float32) / 255.0  # Normalize image between [0.0, 1.0]
    print("Image shape:", gt_image.shape)
    print("Max:", gt_image.max(), "Min:", gt_image.min())
    plt.imshow(gt_image, cmap='gray')
    plt.title('Chest X-ray Image')
    plt.savefig('chest_xray_image.png')

    dataset = RandomPointsDataset(gt_image, points_num=POINTS_PER_SAMPLE)
    # We set a batch_size of 1 since our dataloader is already returning a batch of points.
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=True)
    """
    siren_inr = MLP(dataset.coord_size,
                dataset.value_size,
                hidden_size=HIDDEN_SIZE,
                num_layers=NUM_LAYERS,
                layer_class=SineLayer, 
                siren_factor=SIREN_FACTOR,
                )
    # Re-initialize the weights and make sure they are different
    initialize_siren_weights(siren_inr, SIREN_FACTOR)

    siren_module = INRLightningModule(network=siren_inr,
                                    gt_im=gt_image,
                                    lr=LEARNING_RATE,
                                    name='SIREN',
                                    )
    trainer = pl.Trainer(max_epochs=TRAINING_EPOCHS)
    s = datetime.now()
    trainer.fit(siren_module, train_dataloaders=dataloader)
    print(f"Fitting time: {datetime.now()-s}s.")
    plot_reconstructions(siren_module.progress_ims, gt_image)  
    # Plot PSNR scores
    plot_scores([siren_module])
    
    """

    LATENT_DIM = 32     # Latent vector dimension
    NUM_CODES = 256     # Codebook size
    COMMITMENT_COST = 0.25 

 
    vqinr_net = VQINR(
        coord_dim=dataset.coord_size,   
        value_dim=dataset.value_size,  
        latent_dim=LATENT_DIM,
        num_codes=NUM_CODES,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        siren_factor=SIREN_FACTOR,
        commitment_cost=COMMITMENT_COST,
        num_latent_vectors=1         
    )

    initialize_vqinr_weights(vqinr_net)

    vqinr_module = FixedINRLightningModule(
        network=vqinr_net,
        gt_im=gt_image,
        lr=LEARNING_RATE,
        name='VQINR',
        eval_interval=100,  
        visualization_intervals=[0, 100, 500, 1000, 5000, 10000]
    )

    trainer = pl.Trainer(
        max_epochs=TRAINING_EPOCHS,
        accelerator="auto", 
        devices=1
    )

    s = datetime.now()
    trainer.fit(vqinr_module, train_dataloaders=dataloader)
    print(f"VQINR Fitting time: {datetime.now()-s}s.")

    plot_reconstructions(vqinr_module.progress_ims, gt_image)

    plot_scores([vqinr_module])

    vqinr_net.eval()
    compressed_indices = vqinr_net.compress()
    print(f"Learned Compressed Indices: {compressed_indices.detach().cpu().numpy()}")
