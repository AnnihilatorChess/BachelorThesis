import torch
import torch.nn as nn

from vit_pytorch import MAE, ViT
from the_well.benchmark.models.common import BaseModel

class VisionTransformer(BaseModel):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        n_spatial_dims: int,
        spatial_resolution: tuple[int, ...],
        depth: int = 6,
        mlp_dim: int = 2048,
        dim: int = 1024,
        heads: int = 8,
        patch_size: int = 16,
        decoder_dim: int = 512,
        decoder_depth: int = 6,
        masking_ratio: float = 0.0,
    ):
        super().__init__(n_spatial_dims, spatial_resolution)

        # ViT, we only use some parts of this for the MAE:
        # encoder, transformer, pos_embedding, to_patch_embedding, pool
        self.v = ViT(
            image_size=spatial_resolution,
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            heads=heads,
            channels=dim_in,
            num_classes= 1000,  # not used because we do not use the output head of the ViT, required for init
            mlp_dim=mlp_dim,
        )

        # Masked Auto Encoder: will output 4 timesteps as reconstruction
        self.model = MAE(
            encoder=self.v,
            masking_ratio=masking_ratio,
            decoder_dim=decoder_dim,
            decoder_depth=decoder_depth,
        )

        # 2d conv to project from 4 steps to 1
        self.projection_head = nn.Conv2d(
            in_channels=dim_in, # 4 steps x 4 channels
            out_channels=dim_out, # 1 step x 4 channels
            kernel_size=1,
        )

    def forward(self, input) -> torch.Tensor:
        x = self.model(input)
        x = self.projection_head(x)
        return x





