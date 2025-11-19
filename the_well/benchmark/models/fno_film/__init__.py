import torch
import torch.nn as nn
from neuralop.models import FNO as neuralop_FNO
from neuralop.layers.fno_block import FNOBlocks

from the_well.benchmark.models.common import BaseModel, MLP, EmbedFeatures, FiLMLayer


class SpectralBlockFiLM(FNOBlocks):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, index=0, output_shape=None, gamma=None, beta=None):
        """Forward pass from FNOBlocks of Neuraloperator Library with added FiLM Layer"""
        if self.fno_skips is not None:
            x_skip_fno = self.fno_skips[index](x)
            x_skip_fno = self.convs[index].transform(x_skip_fno, output_shape=output_shape)

        if self.use_channel_mlp and self.channel_mlp_skips is not None:
            x_skip_channel_mlp = self.channel_mlp_skips[index](x)
            x_skip_channel_mlp = self.convs[index].transform(x_skip_channel_mlp, output_shape=output_shape)


        x_fno = self.convs[index](x, output_shape=output_shape)

        if self.norm is not None:    # standard settings have norm=None
            x_fno = self.norm[self.n_norms * index](x_fno)

        # We add the FiLM layer before the non_linearity and before the skip connection
        x_fno = (gamma * x_fno) + beta

        x = x_fno + x_skip_fno if self.fno_skips is not None else x_fno

        if index < (self.n_layers - 1):
            x = self.non_linearity(x)

        if self.use_channel_mlp:
            if self.channel_mlp_skips is not None:
                x = self.channel_mlp[index](x) + x_skip_channel_mlp
            else:
                x = self.channel_mlp[index](x)

        if self.norm is not None:
            x = self.norm[self.n_norms * index + 1](x)

        if index < (self.n_layers - 1):
            x = self.non_linearity(x)

        return x

class FNOFiLM(neuralop_FNO):
    """Integrates FiLM layers into FNO."""
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        n_spatial_dims: int,
        spatial_resolution: tuple[int, ...],
        modes1: int,
        modes2: int,
        hidden_channels: int = 128,
        film_naive: bool = False,
        film_naive_use_embedding: bool = True,
        film_time: bool = True,
        film_t_cool: bool = True,
    ):
        # augment input channels of model for naive FiLM
        if film_naive:
            scaler = 16 if film_naive_use_embedding else 1
            extra_channels = (film_time + film_t_cool) * scaler
            dim_in = dim_in + extra_channels

        super().__init__(
            n_modes=(modes1, modes2),
            in_channels=dim_in,
            out_channels=dim_out,
            hidden_channels=hidden_channels,
        )
        self.naive_use_embedding = film_naive_use_embedding
        self.film_naive = film_naive
        self.film_time = film_time
        self.t_cool = film_t_cool
        mlp_in = extra_channels if film_naive else 32
        self.embed_features = EmbedFeatures(mlp_in, 8)

        if not self.film_naive:
            self.fno_blocks = SpectralBlockFiLM(
                in_channels = hidden_channels,
                out_channels = hidden_channels,
                n_modes = (modes1, modes2),
                n_layers = 4,
            )
            self.film_layer = FiLMLayer()

    def forward(self, x, t_cool, time):
        if self.film_naive:
            return self.forward_naive(x, t_cool, time)
        else:
            return self.forward_film(x, t_cool, time)

    def forward_naive(self, x, t_cool, time):
        """Forward pass from original FNO class above with channel conditioning"""

        if self.naive_use_embedding:
            x = self.embed_concatenate_channels(x, t_cool, time)
        else:
            x = self.concatenate_channels(x, t_cool, time)

        x = self.lifting(x)

        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        for layer_idx in range(self.n_layers):
            x = self.fno_blocks(x, layer_idx)

        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        x = self.projection(x)

        return x

    def forward_film(self, x, t_cool, time):
        """Forward pass from original FNO class above with new SpectralBlockFiLM"""
        x = self.lifting(x)

        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        for layer_idx in range(self.n_layers):
            gamma, beta = self.film_layer(t_cool, time)
            x = self.fno_blocks(x, layer_idx, gamma=gamma, beta=beta)

        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        x = self.projection(x)

        return x

    def concatenate_channels(self, inputs_tensor, t_cool, t) -> torch.Tensor:
        """Concatenates t_cool and time with the physical input channels."""
        B, C, H, W = inputs_tensor.shape

        # converts [B, 1] -> [B, 1, H, W]
        if self.film_t_cool:
            t_cool = t_cool.view(B, 1, 1, 1).expand(B, 1, H, W)
        else:
            t_cool = None
        # converts: [B,1 -> [B, 1, H, W]
        if self.film_time:
            t = t.view(B, 1, 1, 1).expand(B, 1, H, W)
        else:
            t = None
        # concat along channel dim: -> [B, C_in + params, H, W]
        params = [p for p in [t, t_cool] if p]
        inputs_with_params = torch.cat([inputs_tensor] + params, dim=1)
        return inputs_with_params

    def embed_concatenate_channels(self, inputs_tensor, t_cool, t) -> torch.Tensor:
        B, C, H, W = inputs_tensor.shape
        params = []
        if self.film_t_cool:
            params.append(t_cool)
        if self.film_time:
            params.append(t)
        embedded_params = self.embed_features(params)
        params = embedded_params.unsqueeze(-1).unsqueeze(-1).expand(B, len(params)*16, H, W)
        return torch.cat([inputs_tensor, params], dim=1)