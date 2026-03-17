from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch
from einops import rearrange

from .datasets import WellMetadata


class AbstractDataFormatter(ABC):
    def __init__(self, metadata: WellMetadata):
        self.metadata = metadata

    @abstractmethod
    def process_input(self, data: Dict) -> Tuple:
        raise NotImplementedError

    @abstractmethod
    def process_output_channel_last(self, output: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def process_output_expand_time(self, output: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def process_output_unbundle(self, output: torch.Tensor, bundle_size: int, n_fields: int) -> torch.Tensor:
        """Reshape bundled model output [B, K*C, *spatial] -> [B, K, *spatial, C].

        For bundle_size=1 this is equivalent to process_output_channel_last + process_output_expand_time.
        """
        if bundle_size == 1:
            out = self.process_output_channel_last(output)
            return self.process_output_expand_time(out)
        # output shape: [B, K*C, *spatial_dims]
        B = output.shape[0]
        spatial = output.shape[2:]  # e.g., (H, W)
        # Reshape: [B, K*C, *spatial] -> [B, K, C, *spatial] -> [B, K, *spatial, C]
        output = output.view(B, bundle_size, n_fields, *spatial)
        # Move channels to last: [B, K, C, *spatial] -> [B, K, *spatial, C]
        # For nD spatial: dims are (0=B, 1=K, 2=C, 3..=spatial) -> (0=B, 1=K, 3.., 2=C)
        n_spatial = len(spatial)
        perm = [0, 1] + list(range(3, 3 + n_spatial)) + [2]
        return output.permute(*perm).contiguous()


class DefaultChannelsFirstFormatter(AbstractDataFormatter):
    """
    Default preprocessor for data in channels first format.

    Stacks time as individual channel.
    """

    def process_input(self, data: Dict) -> Tuple:
        x = data["input_fields"]
        x = rearrange(x, "b t ... c -> b (t c) ...")
        if "constant_fields" in data:
            flat_constants = rearrange(data["constant_fields"], "b ... c -> b c ...")
            x = torch.cat(
                [
                    x,
                    flat_constants,
                ],
                dim=1,
            )
        y = data["output_fields"]
        # TODO - Add warning to output if nan has to be replaced
        # in some cases (staircase), its ok. In others, it's not.
        return (torch.nan_to_num(x),), torch.nan_to_num(y)

    def process_output_channel_last(self, output: torch.Tensor) -> torch.Tensor:
        return rearrange(output, "b c ... -> b ... c")

    def process_output_expand_time(self, output: torch.Tensor) -> torch.Tensor:
        return rearrange(output, "b ... c -> b 1 ... c")


class DefaultChannelsLastFormatter(AbstractDataFormatter):
    """
    Default preprocessor for data in channels last format.

    Stacks time as individual channel.
    """

    def process_input(self, data: Dict) -> Tuple:
        x = data["input_fields"]
        x = rearrange(x, "b t ... c -> b ... (t c)")
        if "constant_fields" in data:
            flat_constants = data["constant_fields"]
            x = torch.cat(
                [
                    x,
                    flat_constants,
                ],
                dim=-1,
            )
        y = data["output_fields"]
        # TODO - Add warning to output if nan has to be replaced
        # in some cases (staircase), its ok. In others, it's not.
        return (torch.nan_to_num(x),), torch.nan_to_num(y)

    def process_output_channel_last(self, output: torch.Tensor) -> torch.Tensor:
        return output

    def process_output_expand_time(self, output: torch.Tensor) -> torch.Tensor:
        return rearrange(output, "b ... c -> b 1 ... c")
