"""
Convolutional Neural Operator (CNO)

Based on:
    Raonic et al., "Convolutional Neural Operators for robust and accurate learning of PDEs", NeurIPS 2023.
    https://arxiv.org/abs/2302.01178
    Reference implementation: https://github.com/bogdanraonic3/ConvolutionalNeuralOperator

Adapted to support 1D, 2D, and 3D spatial dimensions and follow the BaseModel interface.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from the_well.benchmark.models.common import BaseModel


def _get_conv(n_spatial_dims):
    return {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[n_spatial_dims]


def _get_bn(n_spatial_dims):
    return {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}[n_spatial_dims]


def _interpolate(x, size, n_spatial_dims):
    """Resolution-invariant interpolation using F.interpolate."""
    mode = {1: "linear", 2: "bicubic", 3: "trilinear"}[n_spatial_dims]
    return F.interpolate(x, size=size, mode=mode, antialias=True)


class CNOActivation(nn.Module):
    """CNO activation: upsample to 2x, apply LeakyReLU, downsample to target size.

    This ensures the activation is applied at a higher resolution to avoid aliasing,
    which is a key design choice in the CNO paper.
    """

    def __init__(self, in_size, out_size, n_spatial_dims):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.n_spatial_dims = n_spatial_dims
        self.act = nn.LeakyReLU()
        self.upsampled_size = tuple(2 * s for s in in_size)

    def forward(self, x):
        x = _interpolate(x, self.upsampled_size, self.n_spatial_dims)
        x = self.act(x)
        x = _interpolate(x, self.out_size, self.n_spatial_dims)
        return x


class CNOBlock(nn.Module):
    """Conv -> BatchNorm (optional) -> CNOActivation (with up/downsampling)."""

    def __init__(self, in_channels, out_channels, in_size, out_size, n_spatial_dims, use_bn=True):
        super().__init__()
        Conv = _get_conv(n_spatial_dims)
        self.conv = Conv(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = _get_bn(n_spatial_dims)(out_channels) if use_bn else nn.Identity()
        self.act = CNOActivation(in_size, out_size, n_spatial_dims)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class LiftProjectBlock(nn.Module):
    """Lifting (input -> hidden) or projection (hidden -> output) block."""

    def __init__(self, in_channels, out_channels, size, n_spatial_dims, latent_dim=64):
        super().__init__()
        Conv = _get_conv(n_spatial_dims)
        self.block = CNOBlock(in_channels, latent_dim, size, size, n_spatial_dims, use_bn=False)
        self.conv = Conv(latent_dim, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(self.block(x))


class ResidualBlock(nn.Module):
    """Conv -> BN -> CNOActivation -> Conv -> BN + skip connection."""

    def __init__(self, channels, size, n_spatial_dims, use_bn=True):
        super().__init__()
        Conv = _get_conv(n_spatial_dims)
        BN = _get_bn(n_spatial_dims)
        self.conv1 = Conv(channels, channels, kernel_size=3, padding=1)
        self.conv2 = Conv(channels, channels, kernel_size=3, padding=1)
        self.bn1 = BN(channels) if use_bn else nn.Identity()
        self.bn2 = BN(channels) if use_bn else nn.Identity()
        self.act = CNOActivation(size, size, n_spatial_dims)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return x + out


class ResNet(nn.Module):
    """Stack of ResidualBlocks."""

    def __init__(self, channels, size, num_blocks, n_spatial_dims, use_bn=True):
        super().__init__()
        self.blocks = nn.Sequential(
            *[ResidualBlock(channels, size, n_spatial_dims, use_bn) for _ in range(num_blocks)]
        )

    def forward(self, x):
        return self.blocks(x)


class CNO(BaseModel):
    """Convolutional Neural Operator.

    A multi-resolution encoder-decoder architecture that uses interpolation-based
    up/downsampling (resolution-invariant) with skip connections between encoder
    and decoder levels.

    Args:
        dim_in: Number of input channels (T_in * n_fields + n_constant_fields).
        dim_out: Number of output channels (n_fields).
        n_spatial_dims: Number of spatial dimensions (1, 2, or 3).
        spatial_resolution: Spatial resolution tuple, e.g., (128, 128).
        N_layers: Number of encoder/decoder levels.
        N_res: Number of residual blocks per encoder level.
        N_res_neck: Number of residual blocks in the bottleneck.
        channel_multiplier: Base channel count; channels grow as 2^i * channel_multiplier.
        use_bn: Whether to use batch normalization.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        n_spatial_dims: int,
        spatial_resolution: tuple,
        N_layers: int = 3,
        N_res: int = 4,
        N_res_neck: int = 4,
        channel_multiplier: int = 16,
        use_bn: bool = True,
    ):
        super().__init__(n_spatial_dims, spatial_resolution)

        self.N_layers = N_layers
        lift_dim = channel_multiplier // 2

        # Channel evolution: encoder
        encoder_features = [lift_dim]
        for i in range(N_layers):
            encoder_features.append(2 ** i * channel_multiplier)

        # Channel evolution: decoder
        decoder_features_in = list(reversed(encoder_features[1:]))
        decoder_features_out = list(reversed(encoder_features[:-1]))
        # After the first decoder level, inputs are concatenated with skip connections
        for i in range(1, N_layers):
            decoder_features_in[i] = 2 * decoder_features_in[i]

        # Spatial size evolution
        encoder_sizes = [tuple(s // (2 ** i) for s in spatial_resolution) for i in range(N_layers + 1)]
        decoder_sizes = [tuple(s // (2 ** (N_layers - i)) for s in spatial_resolution) for i in range(N_layers + 1)]

        # Lift and project
        self.lift = LiftProjectBlock(dim_in, encoder_features[0], spatial_resolution, n_spatial_dims)
        self.project = LiftProjectBlock(
            encoder_features[0] + decoder_features_out[-1], dim_out, spatial_resolution, n_spatial_dims
        )

        # Encoder blocks (downsampling)
        self.encoder = nn.ModuleList([
            CNOBlock(encoder_features[i], encoder_features[i + 1],
                     encoder_sizes[i], encoder_sizes[i + 1], n_spatial_dims, use_bn)
            for i in range(N_layers)
        ])

        # ED expansion blocks (match encoder sizes to decoder sizes for skip connections)
        self.ed_expansion = nn.ModuleList([
            CNOBlock(encoder_features[i], encoder_features[i],
                     encoder_sizes[i], decoder_sizes[N_layers - i], n_spatial_dims, use_bn)
            for i in range(N_layers + 1)
        ])

        # Decoder blocks (upsampling)
        self.decoder = nn.ModuleList([
            CNOBlock(decoder_features_in[i], decoder_features_out[i],
                     decoder_sizes[i], decoder_sizes[i + 1], n_spatial_dims, use_bn)
            for i in range(N_layers)
        ])

        # ResNet blocks at each encoder level
        self.res_nets = nn.ModuleList([
            ResNet(encoder_features[i], encoder_sizes[i], N_res, n_spatial_dims, use_bn)
            for i in range(N_layers)
        ])

        # Bottleneck ResNet
        self.res_net_neck = ResNet(
            encoder_features[N_layers], encoder_sizes[N_layers], N_res_neck, n_spatial_dims, use_bn
        )

    def forward(self, x):
        x = self.lift(x)
        skip = []

        # Encoder path
        for i in range(self.N_layers):
            y = self.res_nets[i](x)
            skip.append(y)
            x = self.encoder[i](x)

        # Bottleneck
        x = self.res_net_neck(x)

        # Decoder path
        for i in range(self.N_layers):
            if i == 0:
                x = self.ed_expansion[self.N_layers - i](x)
            else:
                x = torch.cat((x, self.ed_expansion[self.N_layers - i](skip[-i])), dim=1)
            x = self.decoder[i](x)

        # Final skip connection and projection
        x = torch.cat((x, self.ed_expansion[0](skip[0])), dim=1)
        x = self.project(x)

        return x
