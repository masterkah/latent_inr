"""Vector Quantization and SIREN-based INR models"""
import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class SineLayer(nn.Module):
    """
    Standard SIREN Layer with FiLM modulation capability.
    y = sin(omega_0 * ( (Wx + b) * (1 + gamma) + beta ))
    """
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, x, gamma=None, beta=None):
        """
        Forward pass with optional FiLM modulation
        Args:
            x: input tensor [B_total, in_features]
            gamma: scaling factor for FiLM
            beta: shift factor for FiLM
        """
        out = self.linear(x)
        
        # FiLM Modulation: out = (Wx+b) * (1+gamma) + beta
        if gamma is not None and beta is not None:
            out = out * (1 + gamma) + beta
        elif beta is not None:
            out = out + beta
            
        return torch.sin(self.omega_0 * out)


class ModulatedSiren(nn.Module):
    """SIREN network with FiLM modulation capability"""
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers, omega_0=30.):
        super().__init__()
        self.net = nn.ModuleList()
        
        # First Layer (Coords -> Hidden)
        self.net.append(SineLayer(in_dim, hidden_dim, is_first=True, omega_0=omega_0))
        
        # Hidden Layers (Hidden -> Hidden)
        for _ in range(num_layers - 1):
            self.net.append(SineLayer(hidden_dim, hidden_dim, is_first=False, omega_0=omega_0))
            
        # Last Linear Layer (Hidden -> Value)
        self.last_layer = nn.Linear(hidden_dim, out_dim)
        
        with torch.no_grad():
            self.last_layer.weight.uniform_(-np.sqrt(6 / hidden_dim) / omega_0, 
                                             np.sqrt(6 / hidden_dim) / omega_0)
            
    def forward(self, x, gammas=None, betas=None):
        """
        Forward pass with optional FiLM modulation parameters
        Args:
            x: input coordinates
            gammas: list of gamma parameters for each layer
            betas: list of beta parameters for each layer
        """
        for i, layer in enumerate(self.net):
            if i == 0:
                x = layer(x, gamma=None, beta=None)  # First layer without FiLM
            else:
                g = gammas[i] if gammas is not None else None
                b = betas[i] if betas is not None else None
                x = layer(x, gamma=g, beta=b)
        return self.last_layer(x)


class EMAVectorQuantizer(nn.Module):
    """Exponential Moving Average Vector Quantizer"""
    def __init__(self, num_codes, code_dim, decay=0.99, epsilon=1e-5, commitment_cost=0.25):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.decay = decay
        self.epsilon = epsilon
        self.commitment_cost = commitment_cost

        # Initialize codebook
        embedding = torch.randn(num_codes, code_dim)
        embedding = embedding / embedding.norm(dim=1, keepdim=True)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_cluster_size", torch.zeros(num_codes))
        self.register_buffer("ema_embedding", self.embedding.clone())

    def forward(self, z):
        """
        Quantize input tensor z using EMA-updated codebook
        Args:
            z: input latent tensor
        Returns:
            z_q: quantized tensor (with straight-through estimator)
            indices: codebook indices
            vq_loss: vector quantization loss
        """
        original_dtype = z.dtype
        z_fp32 = z.float()
        z_flat = z_fp32.view(-1, self.code_dim)

        # Compute distances to codebook entries
        distances = (z_flat.pow(2).sum(dim=1, keepdim=True)
                     - 2 * z_flat @ self.embedding.t()
                     + self.embedding.pow(2).sum(dim=1))

        indices = torch.argmin(distances, dim=1)
        z_q = F.embedding(indices, self.embedding)

        # Update codebook using EMA
        if self.training and torch.is_grad_enabled():
            encodings = F.one_hot(indices, self.num_codes).float()
            self.ema_cluster_size.mul_(self.decay).add_(encodings.sum(0) * (1 - self.decay))
            dw = encodings.t() @ z_flat.detach()
            self.ema_embedding.mul_(self.decay).add_(dw * (1 - self.decay))
            n = self.ema_cluster_size.sum()
            cluster_size = (self.ema_cluster_size + self.epsilon) / (n + self.num_codes * self.epsilon) * n
            self.embedding.copy_(self.ema_embedding / cluster_size.unsqueeze(1))

        # Straight-through estimator
        z_q_st = z_fp32 + (z_q - z_fp32).detach()
        vq_loss = self.commitment_cost * F.mse_loss(z_q.detach(), z_fp32)
        
        return z_q_st.to(original_dtype), indices, vq_loss


class VQINR(nn.Module):
    """
    Vector Quantized Implicit Neural Representation
    Uses residual VQ with FiLM-conditioned SIREN decoder
    """
    def __init__(self, coord_dim, value_dim, latent_dim, num_codes, hidden_size, 
                 num_layers, num_latent_vectors, num_images, commitment_cost, **kwargs):
        super().__init__()
        self.coord_dim = coord_dim
        self.latent_dim = latent_dim
        self.num_latent_vectors = num_latent_vectors
        self.hidden_size = hidden_size
        
        # Learnable latent codes for each image
        self.latents = nn.Parameter(torch.randn(num_images, num_latent_vectors, latent_dim))
        
        # Multi-stage VQ layers for residual quantization
        self.vq_layers = nn.ModuleList([
            EMAVectorQuantizer(num_codes=num_codes, code_dim=latent_dim, commitment_cost=commitment_cost)
            for _ in range(num_latent_vectors)
        ])
        
        # FiLM Generator: outputs 2 * hidden_size (gamma and beta)
        self.modulation_layers = nn.ModuleList([
            nn.Linear(latent_dim, hidden_size * 2) for _ in range(num_layers)
        ])
        
        # SIREN decoder with FiLM modulation
        self.decoder = ModulatedSiren(
            in_dim=coord_dim, 
            out_dim=value_dim, 
            hidden_dim=hidden_size, 
            num_layers=num_layers,
            omega_0=30.0
        )
        self.current_epoch = 0

    def forward(self, coords, latent_indices):
        """
        Forward pass through VQINR
        Args:
            coords: coordinate tensor [B, N, 2]
            latent_indices: indices of images in batch
        Returns:
            values: predicted values [B, N, 3]
            None: placeholder for consistency
            total_vq_loss: vector quantization loss
        """
        batch_size = coords.shape[0]
        num_points = coords.shape[1]
        
        # Get latent codes for batch
        img_latents = self.latents[latent_indices]
        z = img_latents.sum(dim=1)
        
        # Residual VQ: quantize in stages
        residual = z
        z_q_sum = torch.zeros_like(z)
        total_vq_loss = 0.0
        
        for stage_idx in range(self.num_latent_vectors):
            z_q_stage, _, vq_loss = self.vq_layers[stage_idx](residual)
            z_q_sum = z_q_sum + z_q_stage
            residual = (residual - z_q_stage).detach()
            total_vq_loss += vq_loss
        total_vq_loss = total_vq_loss / self.num_latent_vectors
        
        # Cosine warmup for quantized codes
        effective_z = z_q_sum
        if self.training:
            warmup_epochs = 5000
            if self.current_epoch < warmup_epochs:
                scale = 0.5 * (1 - math.cos(math.pi * self.current_epoch / warmup_epochs))
            else:
                scale = 1.0
            effective_z = z_q_sum * scale
        
        # Generate FiLM parameters (Gamma, Beta)
        gammas = []
        betas = []
        
        for layer in self.modulation_layers:
            mod_out = layer(effective_z)  # [B, 2 * Hidden]
            g, b = mod_out.chunk(2, dim=-1)  # [B, Hidden] each
            
            # Expand to per-pixel: [B, Hidden] -> [B*N, Hidden]
            g = g.unsqueeze(1).expand(-1, num_points, -1).reshape(-1, self.hidden_size)
            b = b.unsqueeze(1).expand(-1, num_points, -1).reshape(-1, self.hidden_size)
            
            gammas.append(g)
            betas.append(b)
            
        # Flatten coordinates [B, N, 2] -> [B*N, 2]
        coords_flat = coords.reshape(-1, self.coord_dim)
        
        # Decode with FiLM modulation
        values_flat = self.decoder(coords_flat, gammas=gammas, betas=betas)
        
        return values_flat.view(batch_size, num_points, -1), None, total_vq_loss

    @torch.no_grad()
    def get_image(self, resolution, latent_idx, device):
        """
        Generate full image from latent code
        Args:
            resolution: (H, W) tuple
            latent_idx: index of the latent code
            device: torch device
        Returns:
            predicted image tensor [H, W, C]
        """
        H, W = resolution
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        coords = torch.stack([yy, xx], dim=-1).reshape(1, -1, 2)
        indices = torch.tensor([latent_idx], device=device)
        pred, _, _ = self(coords, indices)
        return pred.reshape(H, W, -1)
