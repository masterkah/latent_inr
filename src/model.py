"""Vector Quantization and SIREN-based INR models"""
import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.parametrizations import weight_norm as param_weight_norm

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


class PositionalEncoding(nn.Module):
    """Fourier feature positional encoding for ReLU networks"""
    def __init__(self, in_dim, num_freqs):
        super().__init__()
        self.num_freqs = num_freqs
        self.in_dim = in_dim
        # Output dimension: in_dim + 2 * in_dim * num_freqs (original + sin + cos for each frequency)
        self.out_dim = in_dim + 2 * in_dim * num_freqs
        
        # Frequency bands (learnable or fixed)
        freq_bands = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)
        self.register_buffer('freq_bands', freq_bands)
    
    def forward(self, x):
        """
        Apply Fourier feature encoding
        Args:
            x: input coordinates [B, in_dim]
        Returns:
            encoded features [B, in_dim + 2*in_dim*num_freqs]
        """
        if self.num_freqs == 0:
            return x
        
        # x: [B, in_dim]
        encoded = [x]
        for freq in self.freq_bands:
            encoded.append(torch.sin(2 * np.pi * freq * x))
            encoded.append(torch.cos(2 * np.pi * freq * x))
        
        return torch.cat(encoded, dim=-1)


class ReLULayer(nn.Module):
    """ReLU Layer with FiLM modulation capability"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        # Xavier initialization for ReLU networks
        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)
    
    def forward(self, x, gamma=None, beta=None):
        """
        Forward pass with optional FiLM modulation
        Args:
            x: input tensor
            gamma: scaling factor for FiLM
            beta: shift factor for FiLM
        """
        out = self.linear(x)
        
        # FiLM Modulation: out = (Wx+b) * (1+gamma) + beta
        if gamma is not None and beta is not None:
            out = out * (1 + gamma) + beta
        elif beta is not None:
            out = out + beta
            
        return F.relu(out)


class ModulatedReLU(nn.Module):
    """ReLU network with FiLM modulation and positional encoding capability"""
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers, num_freqs=10):
        super().__init__()
        self.net = nn.ModuleList()
        
        # Positional encoding (Fourier features)
        self.pos_encoding = PositionalEncoding(in_dim, num_freqs)
        encoded_dim = self.pos_encoding.out_dim
        
        # First Layer (Encoded Coords -> Hidden)
        self.net.append(ReLULayer(encoded_dim, hidden_dim))
        
        # Hidden Layers (Hidden -> Hidden)
        for _ in range(num_layers - 1):
            self.net.append(ReLULayer(hidden_dim, hidden_dim))
            
        # Last Linear Layer (Hidden -> Value)
        self.last_layer = nn.Linear(hidden_dim, out_dim)
        nn.init.xavier_uniform_(self.last_layer.weight)
        nn.init.zeros_(self.last_layer.bias)
            
    def forward(self, x, gammas=None, betas=None):
        """
        Forward pass with optional FiLM modulation parameters
        Args:
            x: input coordinates
            gammas: list of gamma parameters for each layer
            betas: list of beta parameters for each layer
        """
        # Apply positional encoding to input coordinates
        x = self.pos_encoding(x)
        
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
    Uses residual VQ with FiLM-conditioned decoder (SIREN or ReLU)
    """
    def __init__(self, coord_dim, value_dim, latent_dim, num_codes, hidden_size, 
                 num_layers, num_latent_vectors, num_images, commitment_cost, 
                 activation='siren', **kwargs):
        super().__init__()
        self.coord_dim = coord_dim
        self.latent_dim = latent_dim
        self.num_latent_vectors = num_latent_vectors
        self.hidden_size = hidden_size
        self.activation = activation.lower()
        
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
        
        # Decoder with FiLM modulation (SIREN or ReLU)
        if self.activation == 'siren':
            self.decoder = ModulatedSiren(
                in_dim=coord_dim, 
                out_dim=value_dim, 
                hidden_dim=hidden_size, 
                num_layers=num_layers,
                omega_0=30.0
            )
        elif self.activation == 'relu':
            num_freqs = kwargs.get('num_freqs', 0)  # Get NUM_FREQS from config
            self.decoder = ModulatedReLU(
                in_dim=coord_dim, 
                out_dim=value_dim, 
                hidden_dim=hidden_size, 
                num_layers=num_layers,
                num_freqs=num_freqs
            )
        else:
            raise ValueError(f"Unknown activation: {activation}. Choose 'siren' or 'relu'.")
        
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

# ---------------------------------------------
# NN Architecture (DeepSDF + Pos encoding)
# ---------------------------------------------


class DeepSDFNet(nn.Module):
    def __init__(
        self,
        latent_dim=64,
        hidden_dim=256,
        num_layers=8,
        out_channels=1,
        coord_size=2,
        pos_encoder=None,
        dropout_prob=0.1,
    ):
        super().__init__()
        # If a positional encoder is provided, we use its output dimensionality;
        # otherwise fall back to the raw coordinate size.
        extra_size = pos_encoder.out_size if pos_encoder is not None else coord_size
        self.input_dim = latent_dim + extra_size
        self.pos_encoder = pos_encoder

        # In the paper they use 8 fully connected layers with weight norm.
        self.layer_0 = param_weight_norm(nn.Linear(self.input_dim, hidden_dim))

        self.layers = nn.ModuleList()
        for i in range(1, num_layers - 1):
            self.layers.append(param_weight_norm(nn.Linear(hidden_dim, hidden_dim)))

        # Output layer: Projects to 1 channel (Grayscale) or 3 (RGB)
        self.last_layer = nn.Linear(hidden_dim, out_channels)

        if not 0.0 <= dropout_prob < 1.0:
            raise ValueError(f"dropout_prob must be in [0, 1), got {dropout_prob}")
        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()

    def forward(self, coords, latent_code):
        # coords: (Batch, coord_size)
        # latent_code: (Batch, latent_dim)

        # ---> Apply pos_encoder (fourrier features) if applicable
        if self.pos_encoder is not None:
            pos_coords = self.pos_encoder(coords)
        else:
            pos_coords = coords

        # Concatenate latents z with inputs (x, y, ...) (may be pos_encodings)
        model_input = torch.cat([latent_code, pos_coords], dim=1)

        x = self.layer_0(model_input)
        x = self.relu(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)
            x = self.relu(x)
            x = self.dropout(x)

        x = self.last_layer(x)

        # Sigmoid instead of tanh in the paper to force pixel values to [0, 1]
        return torch.sigmoid(x)


# ---------------- [Fourrier features from INR TUTORIAL] ----------------


class FourierFeatures(nn.Module):
    """Positional encoder from Fourier Features [Tancik et al. 2020]
    Implementation based on https://github.com/tancik/fourier-feature-networks/blob/master/Demo.ipynb
    """

    def __init__(self, coord_size: int, freq_num: int, freq_scale: float = 1.0):
        super().__init__()
        self.freq_num = freq_num  # Number of frequencies
        self.freq_scale = freq_scale  # Standard deviation of the frequencies
        # Note: B_gauss is not registered as a buffer, so it won't be in state_dict.
        self.B_gauss = (
            torch.normal(0.0, 1.0, size=(coord_size, self.freq_num)) * self.freq_scale
        )

        # We store the output size of the module so that the INR knows what input size to expect
        self.out_size = 2 * self.freq_num

    def forward(self, coords):
        # Map the coordinates to a higher dimensional space using the randomly initialized features
        b_gauss_pi = 2.0 * torch.pi * self.B_gauss.to(coords.device)
        prod = coords @ b_gauss_pi
        # Pass the features through a sine and cosine function
        out = torch.cat((torch.sin(prod), torch.cos(prod)), dim=-1)
        return out


# ================= Multi-resolution wrapper =================
# Spatial latents that can recover classic single-vector behavior when s=1.


class AutoDecoderCNNWrapper(nn.Module):
    def __init__(
        self,
        num_images,
        latent_spatial_dim=8,
        latent_channels=32,
        latent_feature_dim=32,
        hidden_dim=256,
        num_layers=8,
        out_channels=1,
        coord_size=2,
        pos_encoder=None,
        sigma=1e-4,
        dropout_prob=0.1,
    ):
        super().__init__()
        self.decoder = DeepSDFNet(
            latent_dim=latent_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            out_channels=out_channels,
            coord_size=coord_size,
            pos_encoder=pos_encoder,
            dropout_prob=dropout_prob,
        )
        self.latent_spatial_dim = latent_spatial_dim
        self.latent_channels = latent_channels
        self.latent_feature_dim = latent_feature_dim

        # Instead of nn.Embedding, we create a 4D Parameter Tensor
        # Shape: (N_Images, channels, height (s), width (s))
        # like -> (5, 32, 8, 8) (in the paper s <-> latent_spatial_dim; c <-> latent_channels; C <-> latent_feature_dim)
        self.latents = nn.Parameter(
            torch.zeros(
                num_images, latent_channels, latent_spatial_dim, latent_spatial_dim
            )
        )

        # Initialize with Gaussian prior (just like DeepSDF)
        nn.init.normal_(self.latents, 0.0, sigma)

        # This gives structure to the latent space via a shared conv layer.
        if latent_spatial_dim == 1 and latent_channels != latent_feature_dim:
            raise ValueError(
                "latent_feature_dim must match latent_channels when latent_spatial_dim=1 "
                "because the conv is disabled."
            )
        self.conv = (
            nn.Conv2d(latent_channels, latent_feature_dim, kernel_size=3, padding=1)
            if latent_spatial_dim != 1
            else nn.Identity()
        )

    def forward(self, image_indices, coords):

        # -> RETRIEVE THE GRIDS
        # We slice the big parameter tensor to get the grids for these specific images
        # Shape: (Batch, latent_channels, s, s)
        current_grids = self.latents[image_indices]

        # -> ADD SPATIAL STRUCTURE
        # Apply the shared convolution, mixes information locally
        # Shape: (Batch, latent_feature_dim, s, s) bc 3x3 conv with 1 padding (so w,h doesn't change, only channels)
        modulated_grids = self.conv(current_grids)

        if self.latent_spatial_dim == 1:
            # Single latent vector per image: recover the classic behavior
            z_batch = modulated_grids.squeeze(-1).squeeze(-1)
        else:
            # -> INTERPOLATE (bilinear interpolation of the latents, given the coords to sample from the grid)

            # need to find the feature vector at exact position (x,y)
            # Reshape coords for grid_sample: (Batch, 1, 1, 2) (-> expected shape, see grid_sample documentation)
            sample_coords = coords.view(-1, 1, 1, 2)

            # Sample features: Output shape (Batch, latent_feature_dim, 1, 1)
            # align_corners=True matches the [-1, 1] coordinate grid used in the dataset.
            z_features = F.grid_sample(
                modulated_grids, sample_coords, align_corners=True
            )

            # Flatten: (Batch, latent_feature_dim)
            z_batch = z_features.view(z_features.shape[0], -1)

        # Feed (z, coords) to MLP
        pred_pixel_vals = self.decoder(coords, z_batch)

        return pred_pixel_vals, z_batch
