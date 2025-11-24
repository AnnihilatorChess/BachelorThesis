import torch
from neuralop.models import FNO as neuralop_FNO
from neuralop.layers.fno_block import FNOBlocks

from the_well.benchmark.models.common import EmbedFeatures, FiLMLayers


class SpectralBlockFiLM(FNOBlocks):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, index=0, output_shape=None, gamma=None, beta=None):
        """Forward pass from FNOBlocks of Neuraloperator Library with added FiLM Layer and removed conditional parts"""
        x_skip_fno = self.fno_skips[index](x)
        x_skip_fno = self.convs[index].transform(x_skip_fno, output_shape=output_shape)

        x_fno = self.convs(x, index, output_shape=output_shape)

        if self.norm is not None:
            x_fno = self.norm[self.n_norms * index](x_fno)

        # Here we add the Scaling and shifting of FiLM in the Fourier Path of the FNO
        # x_fno = (gamma * x_fno) + beta

        x = x_fno + x_skip_fno
        x = (gamma * x) + beta

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
        self.num_inputs = int(film_time + film_t_cool)
        if film_naive:
            scaler = 16 if film_naive_use_embedding else 1
            self.extra_channels = self.num_inputs * scaler
            dim_in = dim_in + self.extra_channels

        super().__init__(
            n_modes=(modes1, modes2),
            in_channels=dim_in,
            out_channels=dim_out,
            hidden_channels=hidden_channels,
        )
        self.naive_use_embedding = film_naive_use_embedding
        self.film_naive = film_naive
        self.time = film_time
        self.t_cool = film_t_cool

        # embedding method
        if film_naive and film_naive_use_embedding:
            self.embed_features = EmbedFeatures(
                self.extra_channels,
                8,
                num_inputs=int(film_time + film_t_cool)
            )
        # override fno_blocks with FiLM-Spectral block for real FiLM
        if not self.film_naive:
            self.fno_blocks = SpectralBlockFiLM(
                in_channels = hidden_channels,
                out_channels = hidden_channels,
                n_modes = (modes1, modes2),
                n_layers = 4,
            )
            self.film_layers = FiLMLayers(
                n_layers=4,
                feature_channels=hidden_channels,
                num_inputs=self.num_inputs
            )

    def forward(self, x, t_cool, time):
        """
        Forward pass from original FNO class with FiLM modifications.
        There are 3 different forward paths depending on film_naive and film_naive_use_embedding:
            1. Naive channel conditioning: Add params naively to input
            2. Channel conditioning with embedding: Fourier Encoding --> MLP --> add to channels
            3. FiLM layers inside the fno_blocks: no channel augmentation
        """
        # augment input channels if film_naive
        if self.film_naive:
            if self.naive_use_embedding:
                x = self.embed_concatenate_channels(x, t_cool, time)
            else:
                x = self.concatenate_channels(x, t_cool, time)

        x = self.lifting(x)
        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)


        # normal fno_blocks if film_naive
        if self.film_naive:
            for layer_idx in range(self.n_layers):
                x = self.fno_blocks(x, layer_idx)
        # fno_blocks wit FiLM layers otherwise
        else:
            t_cool, time = t_cool if self.t_cool else None, time if self.time else None
            gammas, betas = self.film_layers(t_cool, time)
            for layer_idx in range(self.n_layers):
                g, b = gammas[:, layer_idx], betas[:, layer_idx]
                x = self.fno_blocks(x, layer_idx, gamma=g, beta=b)


        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        x = self.projection(x)

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

