# from typing import Dict, Tuple

import torch
import torch.nn as nn
from neuralop.models import FNO as neuralop_FNO
from neuralop.layers.fno_block import FNOBlocks
from torch.utils.checkpoint import checkpoint

from the_well.benchmark.models.common import BaseModel


class NeuralOpsCheckpointWrapper(neuralop_FNO):
    """
    Quick wrapper around neural operator's model to apply checkpointing
    for really big inputs.
    """

    def __init__(self, *args, **kwargs):
        super(NeuralOpsCheckpointWrapper, self).__init__(*args, **kwargs)
        if "gradient_checkpointing" in kwargs:
            self.gradient_checkpointing = kwargs["gradient_checkpointing"]

    def optional_checkpointing(self, layer, *inputs, **kwargs):
        if self.gradient_checkpointing:
            return checkpoint(layer, *inputs, use_reentrant=False, **kwargs)
        else:
            return layer(*inputs, **kwargs)

    def forward(self, x: torch.Tensor, output_shape=None, **kwargs):
        """TFNO's forward pass

        Args:
            x: Input tensor
            output_shape: {tuple, tuple list, None}, default is None
                Gives the option of specifying the exact output shape for odd shaped inputs.
                * If None, don't specify an output shape
                * If tuple, specifies the output-shape of the **last** FNO Block
                * If tuple list, specifies the exact output-shape of each FNO Block
        """

        if output_shape is None:
            output_shape = [None] * self.n_layers
        elif isinstance(output_shape, tuple):
            output_shape = [None] * (self.n_layers - 1) + [output_shape]

        x = self.optional_checkpointing(self.lifting, x)

        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        for layer_idx in range(self.n_layers):
            x = self.optional_checkpointing(
                self.fno_blocks, x, layer_idx, output_shape=output_shape[layer_idx]
            )

        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)


        x = self.optional_checkpointing(self.projection, x)

        return x


class FNO(BaseModel):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        n_spatial_dims: int,
        spatial_resolution: tuple[int, ...],
        modes1: int,
        modes2: int,
        modes3: int = 16,
        hidden_channels: int = 64,
        gradient_checkpointing: bool = False,
    ):
        super().__init__(n_spatial_dims, spatial_resolution)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.hidden_channels = hidden_channels
        self.model = None
        self.initialized = False
        self.gradient_checkpointing = gradient_checkpointing

        if self.n_spatial_dims == 2:
            self.n_modes = (self.modes1, self.modes2)
        elif self.n_spatial_dims == 3:
            self.n_modes = (self.modes1, self.modes2, self.modes3)

        self.model = NeuralOpsCheckpointWrapper(
            n_modes=self.n_modes,
            in_channels=self.dim_in,
            out_channels=self.dim_out,
            hidden_channels=self.hidden_channels,
            gradient_checkpointing=gradient_checkpointing,
        )

    def forward(self, input) -> torch.Tensor:
        return self.model(input)

# new classes for FiLM block integration into FNO

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

class SpectralBlockFiLM(FNOBlocks):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x,index=0, output_shape=None, gamma=None, beta=None):
        """Forward pass from FNOBlocks of Neuraloperator Library with added FiLM Layer"""
        if self.fno_skips is not None:
            x_skip_fno = self.fno_skips[index](x)
            x_skip_fno = self.convs[index].transform(x_skip_fno, output_shape=output_shape)

        if self.use_channel_mlp and self.channel_mlp_skips is not None:
            x_skip_channel_mlp = self.channel_mlp_skips[index](x)
            x_skip_channel_mlp = self.convs[index].transform(x_skip_channel_mlp, output_shape=output_shape)

        if self.stabilizer == "tanh":
            x = torch.tanh(x)

        x_fno = self.convs[index](x, output_shape=output_shape)

        if self.norm is not None:
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

class FNO_FiLM(neuralop_FNO):
    """Integrates FiLM layers into FNO."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fno_blocks = SpectralBlockFiLM()

    def forward(self, x: torch.Tensor, output_shape=None, **kwargs):
        """Forward pass from FNO class above with new SpectralBlockFiLM"""

        if output_shape is None:
            output_shape = [None] * self.n_layers
        elif isinstance(output_shape, tuple):
            output_shape = [None] * (self.n_layers - 1) + [output_shape]

        x = self.optional_checkpointing(self.lifting, x)

        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)



        for layer_idx in range(self.n_layers):
            x = self.optional_checkpointing(
                self.fno_blocks, x, layer_idx, output_shape=output_shape[layer_idx]
            )

        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)


        x = self.optional_checkpointing(self.projection, x)

        return x

