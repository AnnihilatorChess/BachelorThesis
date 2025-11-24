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
    def __init__(self, in_dim, exp_factor=4.0, out_dim=None, hidden_dim=None):
        super().__init__()
        if out_dim is None: out_dim = in_dim
        if hidden_dim is None: hidden_dim = in_dim

        self.fc1 = nn.Linear(in_dim, int(hidden_dim * exp_factor))
        self.fc2 = nn.Linear(int(hidden_dim * exp_factor), out_dim)
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

#######################################
# here are some helper classes for FiLM
#######################################

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
    """Uses Sinusoidal Fourier encoding and MLP to embed features."""
    def __init__(self, output_dim, num_bands, num_inputs=2, include_input=False):
        super().__init__()
        self.FourierEncoding = FourierEncoding(num_bands, include_input)
        # calculate input dimensions
        encoding_dim = (2 * num_bands + int(include_input)) * num_inputs

        self.MLP = MLP(in_dim=encoding_dim, out_dim=output_dim, hidden_dim=output_dim)    # hidden gets multiplied by 4
        # init to 0
        with torch.no_grad():
            self.MLP.fc2.weight.fill_(0.0)
            self.MLP.fc2.bias.fill_(0.0)

    def forward(self, inputs: list):
        encodings = [self.FourierEncoding(x) for x in inputs if x is not None]
        encodings = torch.cat(encodings, dim=1)
        # idea here to solve dim issue: use if_instantiated_MLP to get the shape of encodings as hidden dim
        return self.MLP(encodings)

class FiLMLayers(nn.Module):
    """This generates FiLM Layers. It returns gamma and beta from conditioning parameters for all layers"""
    def __init__(self, n_layers=4, feature_channels=128, hidden_factor=4.0, num_inputs=2):
        super().__init__()
        self.n_layers = n_layers
        self.feature_channels = feature_channels
        self.embed_features = EmbedFeatures(output_dim=128, num_bands=8, num_inputs=num_inputs)
        # We output a tensor containing gammas and betas for n_layers
        output_size = n_layers * feature_channels * 2
        self.generator = MLP(in_dim=128, out_dim=output_size, hidden_dim=128)    # hidden gets multiplied by 4

        # Initialize gamma to 1 and beta to 0
        # We do this by init to 0 and then adding 1 to gamma in the forward
        with torch.no_grad():
            self.generator.fc2.weight.fill_(0)
            self.generator.fc2.bias.fill_(0)

    def forward(self, t_cool, time):
        # embed conditioning params
        emb = self.embed_features([t_cool, time])

        # convert to gamma beta
        out = self.generator(emb)

        # Reshape: [B, n_layers, C, 2]
        out = out.view(-1, self.n_layers, self.feature_channels, 2)

        gamma = out[..., 0] + 1    # we add 1 to initialize with identity
        beta = out[..., 1]

        # reshape: [B, n_layers, C] --> [B, n_layers, C, 1, 1]
        gamma, beta = gamma.unsqueeze(-1).unsqueeze(-1), beta.unsqueeze(-1).unsqueeze(-1)
        return gamma, beta