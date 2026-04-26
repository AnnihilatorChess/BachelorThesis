# from typing import Dict, Tuple

import torch

# --- Monkey patch for cuFFT float16 power-of-two limitation ---
if not getattr(torch.fft, "_is_patched_for_amp", False):
    _original_rfftn = torch.fft.rfftn
    _original_irfftn = torch.fft.irfftn
    _original_fft2 = getattr(torch.fft, "fft2", None)
    _original_ifft2 = getattr(torch.fft, "ifft2", None)

    _fft_fallback_cache = set()

    def _patched_rfftn(input, *args, **kwargs):
        if input.dtype == torch.float16:
            if input.shape in _fft_fallback_cache:
                return _original_rfftn(input.float(), *args, **kwargs)
            try:
                return _original_rfftn(input, *args, **kwargs)
            except RuntimeError as e:
                if "powers of two" in str(e).lower():
                    _fft_fallback_cache.add(input.shape)
                    return _original_rfftn(input.float(), *args, **kwargs)
                raise
        return _original_rfftn(input, *args, **kwargs)

    def _patched_irfftn(input, *args, **kwargs):
        if input.dtype == torch.complex32:
            if input.shape in _fft_fallback_cache:
                return _original_irfftn(input.cfloat(), *args, **kwargs)
            try:
                return _original_irfftn(input, *args, **kwargs)
            except RuntimeError as e:
                if "powers of two" in str(e).lower():
                    _fft_fallback_cache.add(input.shape)
                    return _original_irfftn(input.cfloat(), *args, **kwargs)
                raise
        return _original_irfftn(input, *args, **kwargs)

    def _patched_fft2(input, *args, **kwargs):
        if _original_fft2 is None: return input
        if input.dtype == torch.float16:
            if input.shape in _fft_fallback_cache:
                return _original_fft2(input.float(), *args, **kwargs)
            try:
                return _original_fft2(input, *args, **kwargs)
            except RuntimeError as e:
                if "powers of two" in str(e).lower():
                    _fft_fallback_cache.add(input.shape)
                    return _original_fft2(input.float(), *args, **kwargs)
                raise
        return _original_fft2(input, *args, **kwargs)

    def _patched_ifft2(input, *args, **kwargs):
        if _original_ifft2 is None: return input
        if input.dtype == torch.complex32:
            if input.shape in _fft_fallback_cache:
                return _original_ifft2(input.cfloat(), *args, **kwargs)
            try:
                return _original_ifft2(input, *args, **kwargs)
            except RuntimeError as e:
                if "powers of two" in str(e).lower():
                    _fft_fallback_cache.add(input.shape)
                    return _original_ifft2(input.cfloat(), *args, **kwargs)
                raise
        return _original_ifft2(input, *args, **kwargs)

    torch.fft.rfftn = _patched_rfftn
    torch.fft.irfftn = _patched_irfftn
    if _original_fft2 is not None:
        torch.fft.fft2 = _patched_fft2
    if _original_ifft2 is not None:
        torch.fft.ifft2 = _patched_ifft2

    torch.fft._is_patched_for_amp = True
# --------------------------------------------------------------

from neuralop.models import FNO as neuralop_FNO
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
