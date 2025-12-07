"""
Adapted from:

    Takamoto et al. 2022, PDEBENCH: An Extensive Benchmark for Scientific Machine Learning
    Source: https://github.com/pdebench/PDEBench/blob/main/pdebench/models/unet/unet.py

If you use this implementation, please cite original work above.

This version is adapted once more to integrate input conditioning and FiLM.
The code is based on the unet_classic implementation in unet_classic/__init__.py
"""


from collections import OrderedDict

import torch
import torch.nn as nn

from the_well.benchmark.models.common import BaseModel, EmbedFeatures, FiLMLayers

conv_modules = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
conv_transpose_modules = {
    1: nn.ConvTranspose1d,
    2: nn.ConvTranspose2d,
    3: nn.ConvTranspose3d,
}
pool_modules = {1: nn.MaxPool1d, 2: nn.MaxPool2d, 3: nn.MaxPool3d}
norm_modules = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}


class UNetBlockFiLM(nn.Module):
    """Modified Block that supports FiLM conditioning"""
    def __init__(self, in_channels, features, n_spatial_dims=2, film_naive=False, num_inputs=2):
        super().__init__()
        self.film_naive=film_naive
        self.conv1 = conv_modules[n_spatial_dims](
            in_channels=in_channels, out_channels=features, kernel_size=3, padding=1, bias=False
        )
        self.norm1 = norm_modules[n_spatial_dims](num_features=features)
        self.act1 = nn.Tanh()

        self.conv2 = conv_modules[n_spatial_dims](
            in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False
        )
        self.norm2 = norm_modules[n_spatial_dims](num_features=features)
        self.act2 = nn.Tanh()

        # FiLM Layers class from common.py
        if not film_naive:
            self.film_layers = FiLMLayers(n_layers=2, feature_channels=features, num_inputs=num_inputs)

    def forward(self, x, t_cool=None, time=None):
        out = self.norm1(self.conv1(x))

        if not self.film_naive:  # not film_naive --> real FiLM
            gamma, beta = self.film_layers(t_cool, time) # returns [B, 2, C, 1, 1] because of n_layers
            gamma1, beta1 = gamma[:, 0], beta[:, 0]  # convert to [B, C, 1, 1]
            gamma2, beta2 = gamma[:, 1], beta[:, 1]
            out = (gamma1 * out) + beta1

        out = self.act1(out)
        out = self.norm2(self.conv2(out))

        if not self.film_naive:
            out = (gamma2 * out) + beta2

        return self.act2(out)

class UNetClassicFiLM(BaseModel):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        n_spatial_dims: int,
        spatial_resolution: tuple[int, ...],
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

        def make_block_film(in_c, out_c):
            return UNetBlockFiLM(in_c, out_c, n_spatial_dims, film_naive, num_inputs=self.num_inputs)

        features = init_features
        self.encoder1 = make_block_film(dim_in, features)
        self.pool1 = pool_modules[n_spatial_dims](kernel_size=2, stride=2)
        self.encoder2 = make_block_film(features, features * 2)
        self.pool2 = pool_modules[n_spatial_dims](kernel_size=2, stride=2)
        self.encoder3 = make_block_film(features * 2, features * 4)
        self.pool3 = pool_modules[n_spatial_dims](kernel_size=2, stride=2)
        self.encoder4 = make_block_film(features * 4, features * 8)
        self.pool4 = pool_modules[n_spatial_dims](kernel_size=2, stride=2)

        self.bottleneck = make_block_film(features * 8, features * 16)

        self.upconv4 = conv_transpose_modules[n_spatial_dims](
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = make_block_film((features * 8) * 2, features * 8)
        self.upconv3 = conv_transpose_modules[n_spatial_dims](
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = make_block_film((features * 4) * 2, features * 4)
        self.upconv2 = conv_transpose_modules[n_spatial_dims](
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = make_block_film((features * 2) * 2, features * 2)
        self.upconv1 = conv_transpose_modules[n_spatial_dims](
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = make_block_film(features * 2, features)

        self.conv = conv_transpose_modules[n_spatial_dims](
            in_channels=features, out_channels=dim_out, kernel_size=1
        )
        # embedding method
        if film_naive and film_naive_use_embedding:
            self.embed_features = EmbedFeatures(
                self.extra_channels,
                2,
                num_inputs=int(film_time + film_t_cool)
            )

    def forward(self, x, t_cool, time):
        # handle naive conditioning
        t_cool, time = t_cool if self.t_cool else None, time if self.time else None
        if self.film_naive:
            if self.naive_use_embedding:
                x = self.embed_concatenate_channels(x, t_cool, time)
            else:
                x = self.concatenate_channels(x, t_cool, time)

        enc1 = self.encoder1(x, t_cool, time)
        enc2 = self.encoder2(self.pool1(enc1), t_cool, time)
        enc3 = self.encoder3(self.pool2(enc2), t_cool, time)
        enc4 = self.encoder4(self.pool3(enc3), t_cool, time)

        bottleneck = self.bottleneck(self.pool4(enc4), t_cool, time)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4, t_cool, time)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3, t_cool, time)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2, t_cool, time)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1, t_cool, time)
        return self.conv(dec1)

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