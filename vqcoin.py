import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import random
import numpy as np

from typing import List, Tuple, Optional
from plot import plot_reconstructions, plot_scores, psnr
import lightning as pl


torch.cuda.is_available()

POINTS_PER_SAMPLE = 2048

SIREN_FACTOR = 30.0

class ReLULayer(nn.Module):
    def __init__(self,
                 in_size: int,
                 out_size: int,
                 **kwargs):
        super().__init__()
        self.linear = nn.Linear(in_size, out_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = torch.relu(x)
        return x
    
class SineLayer(nn.Module):
    """
        Implicit Neural Representations with Periodic Activation Functions
        Implementation based on https://github.com/vsitzmann/siren?tab=readme-ov-file
    """
    def __init__(self, in_size, out_size, siren_factor=30., **kwargs):
        super().__init__()
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        self.siren_factor = siren_factor
        self.linear = nn.Linear(in_size, out_size, bias=True)

    def forward(self, x):
        x = self.linear(x)
        x = torch.sin(self.siren_factor * x)
        return x

class MLP(nn.Module):
    def __init__(self,
                 in_size: int,
                 out_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 3,
                 layer_class: nn.Module = ReLULayer,
                 **kwargs):
        super().__init__()

        a = [layer_class(in_size, hidden_size, **kwargs)]
        for i in range(num_layers - 1):
            a.append(layer_class(hidden_size, hidden_size, **kwargs))
        a.append(nn.Linear(hidden_size, out_size))
        self.layers = nn.ModuleList(a)        

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x
    
class ShiftSineLayer(nn.Module):
    def __init__(self, in_size, out_size, siren_factor=30., **kwargs):
        super().__init__()
        self.siren_factor = siren_factor
        self.linear = nn.Linear(in_size, out_size, bias=True)

    def forward(self, x, beta=None):
        """
        x: (B, in_size)
        beta: (B, out_size)  
        """
        h = self.linear(x)               # Wh + b
        if beta is not None:
            # Automatically broadcast to batch dimension
            h = h + beta                 # Wh + b + β
        y = torch.sin(self.siren_factor * h)   # sin(ω0 (Wh + b + β))
        return y


class ShiftReLULayer(nn.Module):
    """ReLU layer with shift modulation (similar to ShiftSineLayer but with ReLU activation)"""
    def __init__(self, in_size, out_size, **kwargs):
        super().__init__()
        self.linear = nn.Linear(in_size, out_size, bias=True)

    def forward(self, x, beta=None):
        """
        x: (B, in_size)
        beta: (B, out_size)  
        """
        h = self.linear(x)               # Wh + b
        if beta is not None:
            h = h + beta                 # Wh + b + β
        y = torch.relu(h)
        return y
    
class ModulatedSIREN(nn.Module):
    def __init__(self,
                 in_size: int,
                 out_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 3,
                 siren_factor: float = 30.0,
                 **kwargs):
        super().__init__()

        layers = []
        layers.append(ShiftSineLayer(in_size, hidden_size,
                                     siren_factor=siren_factor, **kwargs))
        for _ in range(num_layers - 1):
            layers.append(ShiftSineLayer(hidden_size, hidden_size,
                                         siren_factor=siren_factor, **kwargs))
        layers.append(nn.Linear(hidden_size, out_size))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, betas=None):
        if betas is None:
            betas = [None] * (len(self.layers) - 1)

        for i, layer in enumerate(self.layers[:-1]):
            beta = betas[i]
            # print(f"[Layer {i}] β mean: {beta.mean().item():.4f}, std: {beta.std().item():.4f}")

            x = layer(x, beta=beta)

        x = self.layers[-1](x)
        return x


class ModulatedReLU(nn.Module):
    """Modulated MLP with ReLU activation (alternative to ModulatedSIREN)"""
    def __init__(self,
                 in_size: int,
                 out_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 3,
                 **kwargs):
        super().__init__()

        layers = []
        layers.append(ShiftReLULayer(in_size, hidden_size, **kwargs))
        for _ in range(num_layers - 1):
            layers.append(ShiftReLULayer(hidden_size, hidden_size, **kwargs))
        layers.append(nn.Linear(hidden_size, out_size))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, betas=None):
        if betas is None:
            betas = [None] * (len(self.layers) - 1)

        for i, layer in enumerate(self.layers[:-1]):
            beta = betas[i]
            x = layer(x, beta=beta)

        x = self.layers[-1](x)
        return x
    
class RandomPointsDataset(Dataset):
    def __init__(self, image: torch.Tensor, points_num: int = POINTS_PER_SAMPLE):
        super().__init__()
        self.device = "cpu"
        self.points_num = points_num
        assert image.dtype == torch.float32
        self.image = image.to(self.device)  # (H, W, ..., C)
        self.dim_sizes = self.image.shape[:-1]  # Size of each spatial dimension

        # To help us define the input/output sizes of our network later
        # we store the size of our input coordinates and output values
        self.coord_size = len(self.image.shape[:-1])  # Number of spatial dimensions
        self.value_size = self.image.shape[-1]  # Channel size

    def __len__(self):
        return 1

    def __getitem__(self, idx: int):
        # Create random sample of pixel indices
        point_indices = [torch.randint(0, i, (self.points_num,), device=self.device) for i in self.dim_sizes]

        # Retrieve image values from selected indices
        point_values = self.image[tuple(point_indices)]

        # Convert point indices into normalized [-1.0, 1.0] coordinates
        point_coords = torch.stack(point_indices, dim=-1)
        spatial_dims = torch.tensor(self.dim_sizes, device=self.device)
        point_coords_norm = point_coords / (spatial_dims / 2) - 1

        # The subject index is also returned in case the user wants to use subject-wise learned latents
        return point_coords_norm, point_values

class EMAVectorQuantizer(nn.Module):
    def __init__(self,
                 num_codes: int = 512,
                 code_dim: int = 64,
                 decay: float = 0.99,
                 epsilon: float = 1e-5,
                 commitment_cost: float = 0.25):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.decay = decay
        self.epsilon = epsilon
        self.commitment_cost = commitment_cost

        # Embedding used for quantization 
        embedding = torch.randn(num_codes, code_dim)
        embedding = embedding / embedding.norm(dim=1, keepdim=True)
        self.register_buffer("embedding", embedding)

        # EMA statistics
        self.register_buffer("ema_cluster_size",
                             torch.zeros(num_codes))
        self.register_buffer("ema_embedding",
                             self.embedding.clone())

    def forward(self, z: torch.Tensor):
        
        # Flatten to (N, D)
        z_shape = z.shape
        z_flat = z.view(-1, self.code_dim)  # (N, D)

        # Compute distances to codebook (N, K)
        distances = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * z_flat @ self.embedding.t()
            + self.embedding.pow(2).sum(dim=1)
        )  # (N, K)

        # Find nearest code
        indices = torch.argmin(distances, dim=1)           # (N,)
        z_q = F.embedding(indices, self.embedding)         # (N, D)

        # EMA update (only in training mode and when gradients are being computed)
        if self.training and torch.is_grad_enabled():
            # one-hot encodings: (N, K)
            encodings = F.one_hot(indices, self.num_codes).type(z_flat.dtype)

            # 1) cluster size EMA
            cluster_size = encodings.sum(0)  # (K,)
            self.ema_cluster_size.mul_(self.decay).add_(cluster_size * (1 - self.decay))

            # 2) embedding EMA
            dw = encodings.t() @ z_flat.detach()      # (K, D)
            self.ema_embedding.mul_(self.decay).add_(dw * (1 - self.decay))

            # 3) Normalize to prevent some codes from collapsing
            n = self.ema_cluster_size.sum()
            cluster_size = (
                (self.ema_cluster_size + self.epsilon)
                / (n + self.num_codes * self.epsilon) * n
            )

            # new embedding
            embedding = self.ema_embedding / cluster_size.unsqueeze(1)
            self.embedding.copy_(embedding)

        # Straight-through
        z_q_st = z_q + (z_flat - z_q).detach()

        # Only keep commitment loss (codebook updated via EMA)
        vq_loss = self.commitment_cost * F.mse_loss(z_q.detach(), z_flat)

        # Reshape back to original shape
        z_q_st = z_q_st.view(*z_shape)                   # (..., D)
        indices = indices.view(*z_shape[:-1])            # (...)

        return z_q_st, indices, vq_loss


class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs: int, include_input: bool = True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.register_buffer('freqs', 2.0 ** torch.arange(num_freqs))

    def forward(self, x):
        embed = [x] if self.include_input else []
        for freq in self.freqs:
            embed.append(torch.sin(x * freq * torch.pi))
            embed.append(torch.cos(x * freq * torch.pi))
        return torch.cat(embed, dim=-1)


class VQINR(nn.Module):
    """
    VQ-INR: Image Compression with Residual Vector Quantization and Modulated Network
    
    Architecture for image compression:
    1. Multiple learnable latent vectors per image (compressed representation)
    2. Residual VQ: Multiple codebooks quantize residuals progressively
       - First codebook approximates the first latent
       - Subsequent codebooks approximate the residuals
    3. Modulation: Summed quantized latents modulate network layers
    4. Decoder (ModulatedSIREN or ModulatedReLU): Maps coordinates to pixel values
    
    For compression:
    - Store: quantized indices from all codebooks (compact)
    - Reconstruct: decode from indices + coordinates
    
    Supports two activation types:
    - 'siren': SIREN with periodic activation (sin)
    - 'relu': ReLU with positional encoding (recommended: set num_freqs >= 10)
    """
    def __init__(self,
                 coord_dim: int,
                 value_dim: int,
                 latent_dim: int = 64,
                 num_codes: int = 512,
                 hidden_size: int = 128,
                 num_layers: int = 3,
                 siren_factor: float = 30.0,
                 commitment_cost: float = 0.25,
                 num_latent_vectors: int = 4,  # Number of residual VQ stages per image
                 num_images: int = 1,  # Number of images to encode
                 num_freqs: int = 25,  # Number of frequencies for positional encoding (important for ReLU)
                 activation: str = 'relu',  # 'siren' or 'relu'
                 **kwargs):
        super().__init__()
        
        self.coord_dim = coord_dim
        self.value_dim = value_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_latent_vectors = num_latent_vectors
        self.num_images = num_images
        self.num_freqs = num_freqs
        self.activation = activation
        
        # Positional Encoding (required for ReLU, optional for SIREN)
        if num_freqs > 0:
            self.pos_enc = PositionalEncoding(num_freqs)
            decoder_in_dim = coord_dim * (2 * num_freqs + 1)
        else:
            self.pos_enc = None
            decoder_in_dim = coord_dim
        
        # Learnable latent vectors for Residual VQ
        # Shape: (num_images, num_latent_vectors, latent_dim)
        # Each image has num_latent_vectors latent vectors for residual quantization
        self.latents = nn.Parameter(torch.randn(num_images, num_latent_vectors, latent_dim))
        
        # Multiple Vector Quantizers for Residual VQ
        # Shared across all images - each VQ stage quantizes the residual from previous stages
        self.vq_layers = nn.ModuleList([
            EMAVectorQuantizer(
                num_codes=num_codes,
                code_dim=latent_dim,
                commitment_cost=commitment_cost,
                **kwargs
            )
            for _ in range(num_latent_vectors)
        ])
        
        # Modulation layers: generate beta parameters for each layer
        self.modulation_layers = nn.ModuleList([
            nn.Linear(latent_dim, hidden_size) for _ in range(num_layers)
        ])
        
        # Decoder: Modulated SIREN or Modulated ReLU
        if activation == 'siren':
            self.decoder = ModulatedSIREN(
                in_size=decoder_in_dim,
                out_size=value_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                siren_factor=siren_factor
            )
        elif activation == 'relu':
            self.decoder = ModulatedReLU(
                in_size=decoder_in_dim,
                out_size=value_dim,
                hidden_size=hidden_size,
                num_layers=num_layers
            )
        else:
            raise ValueError(f"Unknown activation type: {activation}. Use 'siren' or 'relu'.")
    
    def forward(self, coords: torch.Tensor, latent_idx: int = 0):
        """
        Forward pass with Residual Vector Quantization
        
        Args:
            coords: (B, coord_dim) normalized coordinates in [-1, 1]
            latent_idx: which image to use (0 to num_images-1)
        
        Returns:
            values: (B, value_dim) reconstructed pixel values
            indices: list of (1,) codebook indices from each VQ stage
            vq_loss: scalar total VQ loss from all stages
        """
        # Get latent vectors for this image
        img_latents = self.latents[latent_idx]  # (num_latent_vectors, latent_dim)
        
        # Residual Vector Quantization
        # Stage 1: quantize the first latent
        # Stage 2-N: quantize the residual from previous stages
        
        z_q_sum = torch.zeros(1, self.latent_dim, device=coords.device)
        residual_target = torch.zeros(1, self.latent_dim, device=coords.device)
        
        all_indices = []
        total_vq_loss = 0.0
        
        for stage_idx in range(self.num_latent_vectors):
            # Get current stage's latent
            z_stage = img_latents[stage_idx:stage_idx+1]  # (1, latent_dim)
            
            if stage_idx == 0:
                # First stage: quantize the first latent directly
                z_current = z_stage
                residual_target = z_stage
            else:
                # Subsequent stages: quantize the residual
                # Target = accumulated latents up to this stage
                residual_target = residual_target + z_stage
                # Current input = residual (difference between target and quantized sum)
                z_current = residual_target - z_q_sum.detach()
            
            # Quantize current stage
            z_q_stage, indices_stage, vq_loss_stage = self.vq_layers[stage_idx](z_current)
            
            # Accumulate quantized results
            z_q_sum = z_q_sum + z_q_stage
            
            # Store indices and loss
            all_indices.append(indices_stage)
            total_vq_loss = total_vq_loss + vq_loss_stage
        
        # Use the sum of all quantized latents for modulation
        z_q = z_q_sum.expand(coords.shape[0], -1)  # (B, latent_dim)
        
        # Generate modulation parameters (betas) from quantized latent
        betas = [mod_layer(z_q) for mod_layer in self.modulation_layers]
        
        # Apply positional encoding if enabled
        decoder_input = coords
        if self.pos_enc is not None:
            decoder_input = self.pos_enc(coords)
        
        # Decode coordinates to pixel values with modulated network
        values = self.decoder(decoder_input, betas=betas)  # (B, value_dim)
        
        return values, all_indices, total_vq_loss
    
    def compress(self, latent_idx: int = 0):
        """
        Compress: Get the quantized indices from all VQ stages for storage
        
        Args:
            latent_idx: which image to compress (0 to num_images-1)
            
        Returns:
            indices: list of codebook indices from each stage (compressed representation)
        """
        img_latents = self.latents[latent_idx]  # (num_latent_vectors, latent_dim)
        
        z_q_sum = torch.zeros(1, self.latent_dim, device=self.latents.device)
        residual_target = torch.zeros(1, self.latent_dim, device=self.latents.device)
        all_indices = []
        
        for stage_idx in range(self.num_latent_vectors):
            z_stage = img_latents[stage_idx:stage_idx+1]
            
            if stage_idx == 0:
                z_current = z_stage
                residual_target = z_stage
            else:
                residual_target = residual_target + z_stage
                z_current = residual_target - z_q_sum.detach()
            
            z_q_stage, indices_stage, _ = self.vq_layers[stage_idx](z_current)
            z_q_sum = z_q_sum + z_q_stage
            all_indices.append(indices_stage)
        
        return all_indices
    
    def decompress(self, coords: torch.Tensor, indices: list):
        """
        Decompress: Reconstruct image from stored indices (Residual VQ)
        
        Args:
            coords: (B, coord_dim) coordinates to query
            indices: list of codebook indices from each VQ stage
        
        Returns:
            values: (B, value_dim) reconstructed pixel values
        """
        # Retrieve and sum quantized latents from all codebooks
        z_q_sum = torch.zeros(1, self.latent_dim, device=coords.device)
        
        for stage_idx, idx in enumerate(indices):
            z_q_stage = F.embedding(idx, self.vq_layers[stage_idx].embedding)
            z_q_sum = z_q_sum + z_q_stage
        
        z_q = z_q_sum
        
        # Expand to match batch size
        z_q = z_q.expand(coords.shape[0], -1)  # (B, latent_dim)
        
        # Generate modulation parameters
        betas = [mod_layer(z_q) for mod_layer in self.modulation_layers]
        
        # Apply positional encoding if enabled
        decoder_input = coords
        if self.pos_enc is not None:
            decoder_input = self.pos_enc(coords)
        
        # Decode to pixel values
        values = self.decoder(decoder_input, betas=betas)
        return values
    
    def reconstruct_full_image(self, H: int, W: int, latent_idx: int = 0, device='cuda'):
        """
        Reconstruct full image at given resolution
        
        Args:
            H: height
            W: width
            latent_idx: which latent to use
            device: device to run on
        
        Returns:
            image: (H, W, value_dim) reconstructed image
        """
        # Generate coordinate grid
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        coords = torch.stack([yy, xx], dim=-1).reshape(-1, 2)  # (H*W, 2)
        
        # Get compressed indices
        indices = self.compress(latent_idx)
        
        # Decompress
        with torch.no_grad():
            pixels = self.decompress(coords, indices)  # (H*W, value_dim)
        
        # Reshape to image
        image = pixels.reshape(H, W, self.value_dim)
        return image
    

    
    

class INRLightningModule(pl.LightningModule):
    def __init__(self,
                 network: nn.Module,
                 gt_im: torch.Tensor,
                 lr: float = 0.001,
                 name: str = "",
                 eval_interval: int = 100,
                 visualization_intervals: List[int] = [0, 100, 500, 1000, 5000, 10000],
                ):
        super().__init__()
        self.lr = lr
        self.network = network

        # Logging
        self.name = name
        self.gt_im = gt_im
        self.eval_interval = eval_interval
        self.visualization_intervals = visualization_intervals
        self.progress_ims = []
        self.scores = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.network.parameters(), lr=self.lr)

    def forward(self, coords):
        return self.network(coords)

    def training_step(self, batch, batch_idx):
        coords, values = batch
        coords = coords.view(-1, coords.shape[-1])
        values = values.view(-1, values.shape[-1])
        
        # Forward pass
        out = self.forward(coords)
        
        # Handle VQINR output (tuple) or standard MLP output (tensor)
        if isinstance(out, tuple):
            outputs, indices, vq_loss = out
            recon_loss = nn.functional.mse_loss(outputs, values)
            loss = recon_loss + vq_loss
            
            # Log specific losses
            self.log('train/recon_loss', recon_loss, prog_bar=True)
            self.log('train/vq_loss', vq_loss, prog_bar=True)
        else:
            outputs = out
            loss = nn.functional.mse_loss(outputs, values)
            
        self.log('train/loss', loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        """ At each visualization interval, reconstruct the image using our INR """
        if (self.current_epoch + 1) % self.eval_interval == 0 or self.current_epoch == 0:
            pred_im = self.sample_at_resolution(self.gt_im.shape[:-1])
            pred_im = pred_im.reshape(self.gt_im.shape)
            psnr_value = psnr(pred_im, self.gt_im.to(pred_im.device)).cpu().item()
            self.scores.append((self.current_epoch + 1, psnr_value))  # Log PSNR
            self.log('val/psnr', psnr_value, prog_bar=True)
            if self.current_epoch + 1 in self.visualization_intervals:
                self.progress_ims.append((self.current_epoch + 1, pred_im.cpu(), psnr_value))

    @torch.no_grad()
    def sample_at_resolution(self, resolution: Tuple[int, ...]):
        """ Evaluate our INR on a grid of coordinates in order to obtain an image. """
        meshgrid = torch.meshgrid([torch.arange(0, i, device=self.device) for i in resolution], indexing='ij')
        coords = torch.stack(meshgrid, dim=-1)
        coords_norm = coords / torch.tensor(resolution, device=self.device) * 2 - 1
        coords_norm_ = coords_norm.reshape(-1, coords.shape[-1])
        
        predictions_ = self.forward(coords_norm_)
        
        # Handle tuple output from VQINR (outputs, indices, vq_loss)
        if isinstance(predictions_, tuple):
            predictions_ = predictions_[0]
            
        predictions = predictions_.reshape(*resolution, -1)
        return predictions
    

class FixedINRLightningModule(INRLightningModule):
    @torch.no_grad()
    def sample_at_resolution(self, resolution: Tuple[int, ...]):
        meshgrid = torch.meshgrid([torch.arange(0, i, device=self.device) for i in resolution], indexing='ij')
        coords = torch.stack(meshgrid, dim=-1)
        coords_norm = coords / torch.tensor(resolution, device=self.device) * 2 - 1
        coords_norm_ = coords_norm.reshape(-1, coords.shape[-1])
        
        predictions_ = self.forward(coords_norm_)
        
        if isinstance(predictions_, tuple):
            predictions_ = predictions_[0]
            
        target_shape = list(resolution)
        if predictions_.shape[-1] > 1: 
             target_shape.append(predictions_.shape[-1])
             
        predictions = predictions_.reshape(*target_shape)
        return predictions