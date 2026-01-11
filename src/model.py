import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm as param_weight_norm
import torch.nn.functional as F # for grid_sample

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

        # Decide where to apply the skip connection relative to depth.
        interior_layers = max(1, num_layers - 2)  # ModuleList length
        self.skip_concat_idx = max(1, interior_layers // 2 + 1)

        # In the paper they use 8 fully connected layers with weight norm (let's try without later)
        self.layer_0 = param_weight_norm(nn.Linear(self.input_dim, hidden_dim))

        self.layers = nn.ModuleList()
        for i in range(1, num_layers - 1):
            input_size = (
                hidden_dim + self.input_dim if i == self.skip_concat_idx else hidden_dim
            )
            output_size = hidden_dim

            self.layers.append(param_weight_norm(nn.Linear(input_size, output_size)))
            print(f"layer {i} added in:{input_size} out:{output_size}")

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

        for i, layer in enumerate(self.layers, 1):
            # "Skip connection" logic (we cat the layer with the input)
            if i == self.skip_concat_idx:
                x = torch.cat([x, model_input], dim=1)

            x = layer(x)
            x = self.relu(x)
            x = self.dropout(x)

        x = self.last_layer(x)

        # Sigmoid instead of tanh in the paper to force pixel values to [0, 1]
        return torch.sigmoid(x)


# ---------------- [Fourrier features from INR TUTORIAL] ----------------


class FourierFeatures(nn.Module):
    """Positional encoder from Fourite Features [Tancik et al. 2020]
    Implementation based on https://github.com/tancik/fourier-feature-networks/blob/master/Demo.ipynb
    """

    def __init__(self, coord_size: int, freq_num: int, freq_scale: float = 1.0):
        super().__init__()
        self.freq_num = freq_num  # Number of frequencies
        self.freq_scale = freq_scale  # Standard deviation of the frequencies
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


# ------------------------------------------------
# Auto-Decoder Wrapper (like in the DeepSDF paper)
# -> added positional encoder (Fourier Features)
# ------------------------------------------------


class AutoDecoderWrapper(nn.Module):
    def __init__(
        self,
        num_images,
        latent_dim=64,
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
            latent_dim,
            hidden_dim,
            num_layers=num_layers,
            out_channels=out_channels,
            coord_size=coord_size,
            pos_encoder=pos_encoder,
            dropout_prob=dropout_prob,
        )

        # the latent codes {z_i} we store within the wrapper
        # we use an embedding layer as a lookup table for learnable parameters -> thanks gemini
        self.latents = nn.Embedding(
            num_images, latent_dim
        )  # there is one latent vector per image

        # gaussian prior over the latent z + we initialize them with mean 0 and variance 0.01^2
        # this function runs in torch no grad
        nn.init.normal_(self.latents.weight.data, 0.0, sigma)

    def forward(self, image_indices, coords):
        # 1. Lookup z_i for this batch of pixels
        z_batch = self.latents(image_indices)

        # 2. Feed (z, coords) to MLP
        pred_pixel_vals = self.decoder(coords, z_batch)

        return pred_pixel_vals, z_batch


# ================= New wrapper for multi resolution BRRRR =================
# Mainly the same as before but with that thing in him
# Feel free to cleen the code/comments etc as always

class AutoDecoderCNNWrapper(nn.Module):
    def __init__(self, num_images, latent_spatial_dim=8, latent_channels=32, latent_dim=32, hidden_dim=256, out_channels=1, coord_size=2, pos_encoder=None, sigma=1e-4):
        super().__init__()
        self.decoder = DeepSDFNet(latent_dim=latent_dim, hidden_dim=hidden_dim, out_channels=out_channels, coord_size=coord_size, pos_encoder=pos_encoder)

        # Instead of nn.Embedding, we create a 4D Parameter Tensor
        # Shape: (N_Images, channels, height (s), width (s))
        # like -> (5, 32, 8, 8) (in the paper s <-> latent_spatial_dim; c <-> latent_channels; C <-> latent_dim (modulated latent channel))
        self.latents = nn.Parameter(
            torch.zeros(num_images, latent_channels, latent_spatial_dim, latent_spatial_dim)
        )

        # Initialize with Gaussian prior (just like DeepSDF)
        nn.init.normal_(self.latents, 0.0, sigma)

        # This gives structure to the latent space (simple conv layer, [TODO] try with a 1x1 conv and same size for c and C)
        self.conv = nn.Conv2d(latent_channels, latent_dim, kernel_size=3, padding=1)

    def forward(self, image_indices, coords):

        # -> RETRIEVE THE GRIDS
        # We slice the big parameter tensor to get the grids for these specific images
        # Shape: (Batch, latent_channels, s, s)
        current_grids = self.latents[image_indices]

        # -> ADD SPATIAL STRUCTURE
        # Apply the shared convolution, mixes information locally
        # Shape: (Batch, latent_dim, s, s) bc 3x3 conv with 1 padding (so w,h doesn't change, only channels)
        modulated_grids = self.conv(current_grids)

        # -> INTERPOLATE (bilinear interpolation of the latents, given the coords to sample from the grid)
        # need to find the feature vector at exact position (x,y)

        # Reshape coords for grid_sample: (Batch, 1, 1, 2) (-> expected shape, see grid_sample documentation)
        sample_coords = coords.view(-1, 1, 1, 2)

        # Sample features: Output shape (Batch, latent_dim, 1, 1)
        z_features = F.grid_sample(modulated_grids, sample_coords, align_corners=True) # align corner useful given that we used linspace for coords (see documentation)

        # Flatten: (Batch, latent_dim)
        z_batch = z_features.view(z_features.shape[0], -1)

        ### LEGACY modified ###
        # 1. Lookup z_i for this batch of pixels
        # (done, it's in  z_batch)

        # 2. Feed (z, coords) to MLP
        pred_pixel_vals = self.decoder(coords, z_batch)

        return pred_pixel_vals, z_batch