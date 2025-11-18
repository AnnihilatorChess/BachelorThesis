import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from torch.nn.utils.parametrizations import spectral_norm


class BaseModel(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        n_spatial_dims: int,
        spatial_resolution: tuple[int, ...],
    ):
        super().__init__()
        self.n_spatial_dims = n_spatial_dims
        self.spatial_resolution = spatial_resolution


class NestedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, input):
        return F.linear(input, self.gamma * self.weight, self.bias)


class SigmaNormLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        self.inner = spectral_norm(
            NestedLinear(
                in_features, out_features, bias=bias, device=device, dtype=dtype
            )
        )

    def forward(self, input):
        return self.inner(input)


class MLP(nn.Module):
    def __init__(self, hidden_dim, exp_factor=4.0):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, int(hidden_dim * exp_factor))
        self.fc2 = nn.Linear(int(hidden_dim * exp_factor), hidden_dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class SN_MLP(nn.Module):
    def __init__(self, hidden_dim, exp_factor=4.0):
        super().__init__()
        self.fc1 = SigmaNormLinear(hidden_dim, int(hidden_dim * exp_factor))
        self.fc2 = SigmaNormLinear(int(hidden_dim * exp_factor), hidden_dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

###################
# here are some helper classes for FiLM
###################

class FourierEncoding(nn.Module):
    """Fourier encoding: NeRF style, cite Mildenhall et al. (2020)."""
    def __init__(self, num_bands, include_input=False):
        super().__init__()
        self.num_bands = num_bands
        self.include_input = include_input
        self.freq_bands = 2 ** torch.linspace(0, num_bands-1, num_bands)

    def forward(self, x):
        # expects input: (B, 1)
        enc = [] if not self.include_input else [x]

        for f in self.freq_bands:
            enc.append(torch.sin(math.pi * f * x))
            enc.append(torch.cos(math.pi * f * x))
        return torch.cat(enc, dim=1)


class EmbedFeatures(nn.Module):
    """Uses Sinusoidal Fourier encoding and MLP to embed a feature scalar."""
    def __init__(self, hidden_dim, num_bands, include_input=False):
        super().__init__()
        self.FourierEncoding = FourierEncoding(num_bands, include_input)
        self.MLP = MLP(hidden_dim)

    def forward(self, inputs):
        encodings = [self.FourierEncoding(x) for x in inputs]
        encodings = torch.cat(encodings, dim=1)
        return self.MLP(encodings)

class FiLMLayer(nn.Module):
    """This generates FiLM Layers. It returns gamma and beta from conditioning parameters"""
    def __init__(self, t_cool_embed_dim, time_embed_dim, feature_channels, hidden_dim):
        super().__init__()

        # embed time and t_cool via MLP
        self.time_embedding = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.t_cool_embedding = nn.Sequential(
            nn.Linear(1, t_cool_embed_dim),
            nn.GELU(),
            nn.Linear(t_cool_embed_dim, t_cool_embed_dim),
        )
        # MLP for generating beta and gamma
        self.generator = nn.Sequential(
            nn.Linear(time_embed_dim + t_cool_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2*feature_channels),
        )
        # Initialize gamma to 1 and beta to 0
        nn.init.constant_(self.generator.bias[feature_channels:], 0)
        nn.init.constant_(self.generator.bias[:feature_channels], 1)

    def forward(self, t_cool, time):
        # embed conditioning params
        embedded_time = self.time_embedding(time)
        embedded_t_cool = self.t_cool_embedding(t_cool)
        conditioning_input = torch.cat([embedded_time, embedded_t_cool], dim=1)
        # convert to gamma beta
        gammas_betas = self.generator(conditioning_input)
        gamma, beta = torch.chunk(gammas_betas, 2, dim=1)
        # reshape: [B, C] --> [B, C, 1, 1]
        gamma, beta = gamma.unsqueeze(-1).unsqueeze(-1), beta.unsqueeze(-1).unsqueeze(-1)
        return gamma, beta