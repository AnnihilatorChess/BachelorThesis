"""
Mixed adaptation from:

    Liu et al. 2022, A ConvNet for the 2020s.
    Source: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py

    Ronneberger et al., 2015. Convolutional Networks for Biomedical Image Segmentation.

If you use this implementation, please cite original work above.

This version is adapted once more to integrate input conditioning and FiLM.
The code is based on the unet_convnext implementation in unet_convnext/__init__.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath

from the_well.benchmark.models.common import BaseModel, EmbedFeatures, FiLMLayers, FiLMGenerator

conv_modules = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
conv_transpose_modules = {
    1: nn.ConvTranspose1d,
    2: nn.ConvTranspose2d,
    3: nn.ConvTranspose3d,
}

permute_channel_strings = {
    2: [
        "N C H W -> N H W C",
        "N H W C -> N C H W",
    ],
    3: [
        "N C D H W -> N D H W C",
        "N D H W C -> N C D H W",
    ],
}


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(
        self, normalized_shape, n_spatial_dims, eps=1e-6, data_format="channels_last"
    ):
        super().__init__()
        if data_format == "channels_last":
            padded_shape = (normalized_shape,)
        else:
            padded_shape = (normalized_shape,) + (1,) * n_spatial_dims
        self.weight = nn.Parameter(torch.ones(padded_shape))
        self.bias = nn.Parameter(torch.zeros(padded_shape))
        self.n_spatial_dims = n_spatial_dims
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            x = F.normalize(x, p=2, dim=1, eps=self.eps) * self.weight
            return x


class Upsample(nn.Module):
    r"""Upsample layer."""

    def __init__(self, dim_in, dim_out, n_spatial_dims=2):
        super().__init__()
        self.block = nn.Sequential(
            LayerNorm(dim_in, n_spatial_dims, eps=1e-6, data_format="channels_first"),
            conv_transpose_modules[n_spatial_dims](
                dim_in, dim_out, kernel_size=2, stride=2
            ),
        )

    def forward(self, x):
        return self.block(x)


class Downsample(nn.Module):
    r"""Downsample layer."""

    def __init__(self, dim_in, dim_out, n_spatial_dims=2):
        super().__init__()
        self.block = nn.Sequential(
            LayerNorm(dim_in, n_spatial_dims, eps=1e-6, data_format="channels_first"),
            conv_modules[n_spatial_dims](dim_in, dim_out, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


class Block(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
            self,
            dim,
            n_spatial_dims,
            drop_path=0.0,
            layer_scale_init_value=1e-6,
            film_naive=False,
            num_inputs=2
    ):
        super().__init__()
        self.film_naive = film_naive
        self.n_spatial_dims = n_spatial_dims
        self.dwconv = conv_modules[n_spatial_dims](
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        self.norm = LayerNorm(dim, n_spatial_dims, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, gamma=None, beta=None):
        input = x
        x = self.dwconv(x)
        # (N, C, H, W) -> (N, H, W, C)
        x = rearrange(x, permute_channel_strings[self.n_spatial_dims][0])
        x = self.norm(x)

        # Apply FiLM Modulation if parameters are provided
        if gamma is not None and beta is not None:
            # gamma/beta shape: [B, C]. We need to broadcast to [B, 1, 1, C]
            # Reshape to (B, *spatial_ones, C)
            shape = [x.shape[0]] + [1] * self.n_spatial_dims + [x.shape[-1]]
            gamma_b = gamma.view(*shape)
            beta_b = beta.view(*shape)
            x = (gamma_b * x) + beta_b

        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma * x
        # (N, H, W, C) -> (N, C, H, W)
        x = rearrange(x, permute_channel_strings[self.n_spatial_dims][1])
        x = input + self.drop_path(x)
        return x


class Stage(nn.Module):
    r"""ConvNeXt Stage.
    Args:
        dim_in (int): Number of input channels.
        dim_out (int): Number of output channels.
        n_spatial_dims (int): Number of spatial dimensions.
        depth (int): Number of blocks in the stage.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        mode (str): Down, Up, Neck. Default: "down"
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        n_spatial_dims,
        depth=1,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        mode="down",
        skip_project=False,
        film_naive=False,
        num_inputs=2
    ):
        super().__init__()
        self.film_naive = film_naive
        if skip_project:
            self.skip_proj = conv_modules[n_spatial_dims](2 * dim_in, dim_in, 1)
        else:
            self.skip_proj = nn.Identity()
        if mode == "down":
            self.resample = Downsample(dim_in, dim_out, n_spatial_dims)
        elif mode == "up":
            self.resample = Upsample(dim_in, dim_out, n_spatial_dims)
        else:
            self.resample = nn.Identity()

        self.blocks = nn.ModuleList(
            [
                Block(dim_in, n_spatial_dims, drop_path, layer_scale_init_value, film_naive, num_inputs)
                for _ in range(depth)
            ]
        )

    def forward(self, x, film_params=None):
        """
        film_params: dict with keys 'gamma', 'beta'.
                     Values are tensors of shape [B, depth, C]
        """
        x = self.skip_proj(x)

        for i, block in enumerate(self.blocks):
            # Extract specific params for this block
            gamma, beta = None, None
            if film_params is not None:
                gamma = film_params['gamma'][:, i, :]  # [B, C]
                beta = film_params['beta'][:, i, :]  # [B, C]

            x = block(x, gamma=gamma, beta=beta)

        x = self.resample(x)
        return x


class UNetConvNextFiLM(BaseModel):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        n_spatial_dims: int,
        spatial_resolution: tuple[int, ...],
        stages: int = 4,
        blocks_per_stage: int = 1,
        blocks_at_neck: int = 1,
        init_features: int = 32,
        film_naive: bool = False,
        film_naive_use_embedding: bool = True,
        film_t_cool: bool = False,
        film_time: bool = False,
    ):
        super().__init__(n_spatial_dims, spatial_resolution)

        # augment input channels of model for naive FiLM
        self.num_inputs = int(film_time + film_t_cool)
        if film_naive:
            scaler = 4 if film_naive_use_embedding else 1
            self.extra_channels = self.num_inputs * scaler
            dim_in = dim_in + self.extra_channels

        self.naive_use_embedding = film_naive_use_embedding
        self.film_naive = film_naive
        self.time = film_time
        self.t_cool = film_t_cool

        self.n_spatial_dims = n_spatial_dims
        features = init_features
        encoder_dims = [features * 2**i for i in range(stages + 1)]
        decoder_dims = [features * 2**i for i in range(stages, -1, -1)]
        encoder = []
        decoder = []
        self.in_proj = conv_modules[n_spatial_dims](
            dim_in, features, kernel_size=3, padding=1
        )
        self.out_proj = conv_modules[n_spatial_dims](
            features, dim_out, kernel_size=3, padding=1
        )

        # We need to collect structural info to build the Centralized Generator
        self.film_structure = []
        for i in range(stages):
            dim = encoder_dims[i]
            self.film_structure.append((dim, blocks_per_stage))
            encoder.append(
                Stage(
                    dim,
                    encoder_dims[i + 1],
                    n_spatial_dims,
                    blocks_per_stage,
                    mode="down",
                    film_naive=film_naive,
                    num_inputs=self.num_inputs
                )
            )

        for i in range(stages):
            dim = decoder_dims[i]
            # Record structure
            self.film_structure.append((dim, blocks_per_stage))
            decoder.append(
                Stage(
                    dim,
                    decoder_dims[i + 1],
                    n_spatial_dims,
                    blocks_per_stage,
                    mode="up",
                    skip_project=i != 0,
                    film_naive=film_naive,
                    num_inputs=self.num_inputs
                )
            )
        self.encoder = nn.ModuleList(encoder)

        # Record structure for Neck
        neck_dim = encoder_dims[-1]
        self.film_structure.insert(stages, (neck_dim, blocks_at_neck))  # Neck is between enc and dec

        self.neck = Stage(
            neck_dim,
            neck_dim,
            n_spatial_dims,
            blocks_at_neck,
            mode="neck",
            film_naive=film_naive,
            num_inputs=self.num_inputs
        )

        self.decoder = nn.ModuleList(decoder)

        # FiLM adaptive Layers
        if not film_naive and self.num_inputs > 0:
            self.film_generator = FiLMGenerator(self.film_structure, num_inputs=self.num_inputs)

        # embedding method
        if film_naive and film_naive_use_embedding:
            self.embed_features = EmbedFeatures(
                self.extra_channels,
                2,
                num_inputs=int(film_time + film_t_cool)
            )


    def forward(self, x, t_cool=None, time=None):
        """Forward pass of UNetConvNext: adjusted for input conditioning and FiLM"""
        t_cool, time = t_cool if self.t_cool else None, time if self.time else None
        if self.film_naive:
            if self.naive_use_embedding:
                x = self.embed_concatenate_channels(x, t_cool, time)
            else:
                x = self.concatenate_channels(x, t_cool, time)

        # 1. Generate Global FiLM Parameters if active
        all_film_params = None
        if not self.film_naive and hasattr(self, 'film_generator'):
            # Returns a list of dicts: [{'gamma':..., 'beta':...}, ...]
            all_film_params = self.film_generator(t_cool, time)

        x = self.in_proj(x)
        skips = []
        param_idx = 0
        # Encoder Pass
        for i, enc in enumerate(self.encoder):
            skips.append(x)

            stage_params = all_film_params[param_idx] if all_film_params else None
            x = enc(x, film_params=stage_params)
            param_idx += 1

        # Neck Pass
        stage_params = all_film_params[param_idx] if all_film_params else None
        x = self.neck(x, film_params=stage_params)
        param_idx += 1

        # Decoder Pass
        for j, dec in enumerate(self.decoder):
            if j > 0:
                x = torch.cat([x, skips[-j]], dim=1)

            stage_params = all_film_params[param_idx] if all_film_params else None
            x = dec(x, film_params=stage_params)
            param_idx += 1

        x = self.out_proj(x)
        return x


    def concatenate_channels(self, inputs_tensor, t_cool, t) -> torch.Tensor:
        """Concatenates t_cool and time with the physical input channels."""
        B, C, H, W = inputs_tensor.shape
        params = []

        # converts [B, 1] -> [B, 1, H, W]
        if self.t_cool:
            params.append(t_cool.view(B, 1, 1, 1).expand(B, 1, H, W))
        if self.time:
            params.append(t.view(B, 1, 1, 1).expand(B, 1, H, W))

        # concat along channel dim: -> [B, C_in + params, H, W]
        inputs_with_params = torch.cat([inputs_tensor] + params, dim=1)
        return inputs_with_params

    def embed_concatenate_channels(self, inputs_tensor, t_cool, t) -> torch.Tensor:
        B, C, H, W = inputs_tensor.shape
        params = []
        if self.t_cool:
            params.append(t_cool)
        if self.time:
            params.append(t)

        embedded_params = self.embed_features(params)

        # reshape to match inputs
        params_spatial = embedded_params.view(B, self.extra_channels, 1, 1).expand(B, self.extra_channels, H, W)
        return torch.cat([inputs_tensor, params_spatial], dim=1)