"""
Convolutional Neural Operator (CNO)

Based on:
    Raonic et al., "Convolutional Neural Operators for robust and accurate learning of PDEs", NeurIPS 2023.
    https://arxiv.org/abs/2302.01178
    Reference implementation: https://github.com/bogdanraonic3/ConvolutionalNeuralOperator

Adapted to support 1D, 2D, and 3D spatial dimensions and follow the BaseModel interface.

Two activation backends are supported (selected via the `activation` constructor
argument, threaded through from the model config):

- ``"torch"`` (default): pure-PyTorch path using ``F.interpolate`` for up/down
  resampling around a ``LeakyReLU``. Supports 1D, 2D, and 3D inputs. The
  ``antialias`` argument controls whether the bicubic interpolation uses a
  scale-aware low-pass filter; defaults to False for training throughput.
  This path is slow under autograd because ``F.interpolate(antialias=True)``
  has no cuDNN backend and a particularly unoptimised backward kernel.

- ``"compiled"``: vendored fused CUDA op from the official CNO repo (itself
  ported from StyleGAN3's alias-free primitive). Applies bias, upsamples with a
  Kaiser low-pass filter, runs LeakyReLU, then downsamples with another Kaiser
  filter — all in a single fused CUDA kernel. **2D only.** Requires a Linux
  training environment with a matching CUDA toolkit (nvcc), gcc, ninja, and
  scipy. Triggers a one-time JIT compile (~5-15 min) on first invocation;
  subsequent runs use the cached ``.so``. For ``n_spatial_dims != 2`` we emit
  a warning and silently fall back to the ``"torch"`` path so a single config
  can drive both 1D Burgers and 2D datasets.

See ``docs/models/cno.md`` for the full speed/accuracy discussion and the
``the_well/benchmark/models/cno/_filtered_lrelu/README.md`` for the vendored
extension's origin and license.
"""

import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from the_well.benchmark.models.common import BaseModel

# Both ``scipy`` (used by CompiledCNOActivation for Kaiser filter design) and
# the vendored ``filtered_lrelu`` extension are imported lazily inside the
# CompiledCNOActivation class. This keeps the torch-only code path importable
# on minimal envs (and on Windows dev boxes) where scipy / nvcc are absent.


def _get_conv(n_spatial_dims):
    return {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[n_spatial_dims]


def _get_bn(n_spatial_dims):
    return {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}[n_spatial_dims]


def _interpolate(x, size, n_spatial_dims, antialias=False):
    """Resolution-invariant interpolation using F.interpolate.

    `antialias=True` matches the paper's formal alias-free formulation (the reference
    implementation uses compiled CUDA kernels for this). The pure PyTorch fallback
    for antialiased bicubic has no cudnn path and a slow backward kernel, making it
    ~12-18x slower than FNO on our hardware. We default to antialias=False to keep
    training tractable; see docs/benchmark-comparability.md for the speed/accuracy
    trade-off and the CNO paper's Appendix C.5 ablation.

    `antialias` is only valid for bilinear/bicubic modes; ignored otherwise.
    """
    mode = {1: "linear", 2: "bicubic", 3: "trilinear"}[n_spatial_dims]
    kwargs = dict(size=tuple(int(s) for s in size), mode=mode)
    if mode in ("bilinear", "bicubic"):
        kwargs["antialias"] = antialias
    return F.interpolate(x, **kwargs)


class CNOActivation(nn.Module):
    """CNO activation (pure-PyTorch path): upsample to 2x, apply LeakyReLU, downsample.

    This ensures the activation is applied at a higher resolution to avoid aliasing,
    which is a key design choice in the CNO paper. ``antialias`` controls whether the
    bicubic interpolation uses a scale-aware low-pass filter; defaults to False for
    training throughput (see ``_interpolate`` docstring).

    Used directly when the model is constructed with ``activation="torch"``, and as
    the silent fallback for ``activation="compiled"`` when ``n_spatial_dims != 2``.
    """

    def __init__(self, in_size, out_size, n_spatial_dims, antialias=False):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.n_spatial_dims = n_spatial_dims
        self.antialias = antialias
        self.act = nn.LeakyReLU()
        self.upsampled_size = tuple(2 * s for s in in_size)

    def forward(self, x):
        x = _interpolate(x, self.upsampled_size, self.n_spatial_dims, self.antialias)
        x = self.act(x)
        x = _interpolate(x, self.out_size, self.n_spatial_dims, self.antialias)
        return x


class CompiledCNOActivation(nn.Module):
    """Alias-free CNO activation using the fused ``filtered_lrelu`` CUDA op.

    Ports the reference ``LReLu`` implementation from
    ``CNO2d_classic/training/filtered_networks.py`` (Raonić et al. 2023).
    Designs separable Kaiser low-pass filters at construction time, then in
    ``forward`` invokes the vendored StyleGAN3 fused kernel that does
    ``bias -> upsample -> LP filter -> LeakyReLU -> LP filter -> downsample``
    in a single CUDA pass. The first call on each unique cache key triggers a
    ninja+nvcc compile; subsequent calls are essentially free.

    **2D only** — the vendored kernel is `upfirdn2d`, not N-D. For non-2D
    inputs, the parent dispatcher (`_make_activation`) falls back to
    ``CNOActivation``.

    For non-square ``in_size`` / ``out_size`` (e.g. TRL 128x384), the sampling
    rate is taken as ``max(in_size)`` / ``max(out_size)`` since the
    StyleGAN3 filter is designed against a single scalar rate. The padding is
    computed per-axis from the actual ``(H, W)`` tuple, so non-square inputs
    still produce correctly-shaped outputs; only the filter cutoff is
    isotropic. For square inputs (the common case in the CNO paper) this is
    identical to the reference implementation.

    Args:
        in_size: Input spatial size — int or 2-tuple (H, W).
        out_size: Output spatial size — int or 2-tuple (H, W).
        channels: Number of channels (for the per-channel bias parameter).
        lrelu_upsampling: Relative sampling-rate multiplier for the LeakyReLU.
            Default 2, matching the paper. The activation is applied at this
            multiple of the working resolution so high-frequency content from
            the nonlinearity is suppressed by the downsampling filter.
        filter_size: Number of Kaiser taps per up/down factor. Default 6,
            matching the paper (`M = 6` per Appendix C.5).
        cutoff_den: Cutoff-frequency denominator: ``f_c = sampling_rate /
            cutoff_den``. Default 2.0001 — slightly below Nyquist so the
            transition band fits without aliasing.
        half_width_mult: Filter transition-band parameter. Default 0.8.
    """

    def __init__(
        self,
        in_size,
        out_size,
        channels: int,
        lrelu_upsampling: int = 2,
        filter_size: int = 6,
        cutoff_den: float = 2.0001,
        half_width_mult: float = 0.8,
    ):
        super().__init__()

        # Lazy import: keeps Windows/CPU-only environments importable when
        # they never construct a CompiledCNOActivation.
        from ._filtered_lrelu.torch_utils.ops import filtered_lrelu as _fl_mod

        self._fl_mod = _fl_mod

        # Broadcast scalar to 2-tuple along (H, W).
        in_size_np = np.broadcast_to(np.asarray(in_size), [2]).astype(np.int64)
        out_size_np = np.broadcast_to(np.asarray(out_size), [2]).astype(np.int64)

        # Sampling rates are scalars in the StyleGAN3 formulation; for
        # non-square inputs use the finer axis as the reference rate. This is
        # equivalent to the reference impl when in_size is square, which is the
        # CNO-paper case.
        in_sampling_rate = float(int(max(in_size_np)))
        out_sampling_rate = float(int(max(out_size_np)))
        tmp_sampling_rate = max(in_sampling_rate, out_sampling_rate) * lrelu_upsampling

        in_cutoff = in_sampling_rate / cutoff_den
        out_cutoff = out_sampling_rate / cutoff_den
        in_half_width = half_width_mult * in_sampling_rate - in_sampling_rate / cutoff_den
        out_half_width = half_width_mult * out_sampling_rate - out_sampling_rate / cutoff_den

        # Up-sampling filter
        up_factor = int(np.rint(tmp_sampling_rate / in_sampling_rate))
        up_taps = filter_size * up_factor if up_factor > 1 else 1
        up_filter = self._design_lowpass_filter(
            numtaps=up_taps,
            cutoff=in_cutoff,
            width=in_half_width * 2,
            fs=tmp_sampling_rate,
        )

        # Down-sampling filter
        down_factor = int(np.rint(tmp_sampling_rate / out_sampling_rate))
        down_taps = filter_size * down_factor if down_factor > 1 else 1
        down_filter = self._design_lowpass_filter(
            numtaps=down_taps,
            cutoff=out_cutoff,
            width=out_half_width * 2,
            fs=tmp_sampling_rate,
        )

        # Per-axis padding (Appendix C.3 of the StyleGAN3 paper).
        pad_total = (out_size_np - 1) * down_factor + 1
        pad_total = pad_total - in_size_np * up_factor
        pad_total = pad_total + up_taps + down_taps - 2
        pad_lo = (pad_total + up_factor) // 2
        pad_hi = pad_total - pad_lo
        # filtered_lrelu padding order is [px0, px1, py0, py1] where x = W, y = H.
        # in_size/out_size are (H, W), so index 0 is H -> y and index 1 is W -> x.
        self.padding = [
            int(pad_lo[1]),
            int(pad_hi[1]),
            int(pad_lo[0]),
            int(pad_hi[0]),
        ]

        self.up_factor = up_factor
        self.down_factor = down_factor
        self.in_size = tuple(int(s) for s in in_size_np)
        self.out_size = tuple(int(s) for s in out_size_np)
        self.channels = int(channels)

        self.bias = nn.Parameter(torch.zeros(self.channels))
        # `register_buffer` happily accepts None when the filter is the identity
        # (numtaps == 1); the fused op handles a None filter as a 1x1 identity.
        self.register_buffer("up_filter", up_filter)
        self.register_buffer("down_filter", down_filter)

    @staticmethod
    def _design_lowpass_filter(numtaps, cutoff, width, fs):
        """Separable Kaiser low-pass filter (the only mode CNO uses).

        Returns a 1D float32 tensor of length ``numtaps`` suitable for passing
        as ``fu``/``fd`` to ``filtered_lrelu`` — the underlying ``upfirdn2d``
        convolves separably with the 1D filter along H and W. Returns ``None``
        for the identity case (``numtaps == 1``), which the fused op treats as
        a 1x1 unit kernel.
        """
        if numtaps == 1:
            return None
        # Lazy import: scipy is not in the_well's hard dependency list (it
        # comes in transitively via neuraloperator). Importing here keeps the
        # torch-only activation path usable on envs without scipy.
        import scipy.signal  # noqa: WPS433

        f = scipy.signal.firwin(numtaps=numtaps, cutoff=cutoff, width=width, fs=fs)
        return torch.as_tensor(f, dtype=torch.float32)

    def forward(self, x):
        # filtered_lrelu kernel supports fp16/fp32; fp64 silently falls back to
        # the slow PyTorch reference impl. We keep inputs fp32 for parity with
        # the rest of the model (no AMP support yet).
        return self._fl_mod.filtered_lrelu(
            x=x,
            fu=self.up_filter,
            fd=self.down_filter,
            b=self.bias.to(x.dtype),
            up=self.up_factor,
            down=self.down_factor,
            padding=self.padding,
            gain=float(np.sqrt(2)),
            slope=0.2,
            clamp=None,
        )

    def extra_repr(self) -> str:
        return (
            f"in_size={self.in_size}, out_size={self.out_size}, "
            f"channels={self.channels}, up={self.up_factor}, down={self.down_factor}, "
            f"padding={self.padding}"
        )


def _make_activation(in_size, out_size, n_spatial_dims, channels, *,
                     activation: str = "torch", antialias: bool = False):
    """Activation factory: dispatch between the torch path and the compiled kernel.

    Falls back to the torch path with a warning when ``activation="compiled"``
    is requested for a non-2D input. This lets a single config drive both 2D
    runs (where the compiled kernel is the point) and 1D Burgers (where it
    isn't available) without per-dataset overrides.
    """
    if activation == "torch":
        return CNOActivation(in_size, out_size, n_spatial_dims, antialias=antialias)
    if activation == "compiled":
        if n_spatial_dims != 2:
            warnings.warn(
                f"activation='compiled' is only implemented for 2D inputs (upfirdn2d-based); "
                f"falling back to the torch path for n_spatial_dims={n_spatial_dims}.",
                UserWarning,
                stacklevel=2,
            )
            return CNOActivation(in_size, out_size, n_spatial_dims, antialias=antialias)
        return CompiledCNOActivation(in_size, out_size, channels=channels)
    raise ValueError(
        f"Unknown activation {activation!r}; expected 'torch' or 'compiled'."
    )


class CNOBlock(nn.Module):
    """Conv -> BatchNorm (optional) -> activation (with up/downsampling)."""

    def __init__(self, in_channels, out_channels, in_size, out_size, n_spatial_dims,
                 use_bn=True, antialias=False, activation: str = "torch"):
        super().__init__()
        Conv = _get_conv(n_spatial_dims)
        self.conv = Conv(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = _get_bn(n_spatial_dims)(out_channels) if use_bn else nn.Identity()
        # Channel count for the per-channel bias in the compiled activation matches
        # the conv's output channels (the activation runs *after* the conv).
        self.act = _make_activation(
            in_size, out_size, n_spatial_dims, channels=out_channels,
            activation=activation, antialias=antialias,
        )

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class LiftProjectBlock(nn.Module):
    """Lifting (input -> hidden) or projection (hidden -> output) block."""

    def __init__(self, in_channels, out_channels, size, n_spatial_dims,
                 latent_dim=64, antialias=False, activation: str = "torch"):
        super().__init__()
        Conv = _get_conv(n_spatial_dims)
        self.block = CNOBlock(
            in_channels, latent_dim, size, size, n_spatial_dims,
            use_bn=False, antialias=antialias, activation=activation,
        )
        self.conv = Conv(latent_dim, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(self.block(x))


class ResidualBlock(nn.Module):
    """Conv -> BN -> activation -> Conv -> BN + skip connection."""

    def __init__(self, channels, size, n_spatial_dims,
                 use_bn=True, antialias=False, activation: str = "torch"):
        super().__init__()
        Conv = _get_conv(n_spatial_dims)
        BN = _get_bn(n_spatial_dims)
        self.conv1 = Conv(channels, channels, kernel_size=3, padding=1)
        self.conv2 = Conv(channels, channels, kernel_size=3, padding=1)
        self.bn1 = BN(channels) if use_bn else nn.Identity()
        self.bn2 = BN(channels) if use_bn else nn.Identity()
        self.act = _make_activation(
            size, size, n_spatial_dims, channels=channels,
            activation=activation, antialias=antialias,
        )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return x + out


class ResNet(nn.Module):
    """Stack of ResidualBlocks."""

    def __init__(self, channels, size, num_blocks, n_spatial_dims,
                 use_bn=True, antialias=False, activation: str = "torch"):
        super().__init__()
        self.blocks = nn.Sequential(*[
            ResidualBlock(
                channels, size, n_spatial_dims, use_bn,
                antialias=antialias, activation=activation,
            )
            for _ in range(num_blocks)
        ])

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
        antialias: Used only by the torch activation path. See ``_interpolate`` docstring.
        activation: ``"torch"`` (default) or ``"compiled"``. The compiled path
            requires the vendored ``filtered_lrelu`` plugin to build (Linux +
            matching nvcc + ninja + scipy) and is 2D-only; 1D/3D models silently
            fall back to the torch path with a warning.
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
        antialias: bool = False,
        activation: str = "torch",
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

        common_kwargs = dict(antialias=antialias, activation=activation)

        # Lift and project
        self.lift = LiftProjectBlock(
            dim_in, encoder_features[0], spatial_resolution, n_spatial_dims, **common_kwargs,
        )
        self.project = LiftProjectBlock(
            encoder_features[0] + decoder_features_out[-1], dim_out, spatial_resolution, n_spatial_dims,
            **common_kwargs,
        )

        # Encoder blocks (downsampling)
        self.encoder = nn.ModuleList([
            CNOBlock(encoder_features[i], encoder_features[i + 1],
                     encoder_sizes[i], encoder_sizes[i + 1], n_spatial_dims, use_bn, **common_kwargs)
            for i in range(N_layers)
        ])

        # ED expansion blocks (match encoder sizes to decoder sizes for skip connections)
        self.ed_expansion = nn.ModuleList([
            CNOBlock(encoder_features[i], encoder_features[i],
                     encoder_sizes[i], decoder_sizes[N_layers - i], n_spatial_dims, use_bn, **common_kwargs)
            for i in range(N_layers + 1)
        ])

        # Decoder blocks (upsampling)
        self.decoder = nn.ModuleList([
            CNOBlock(decoder_features_in[i], decoder_features_out[i],
                     decoder_sizes[i], decoder_sizes[i + 1], n_spatial_dims, use_bn, **common_kwargs)
            for i in range(N_layers)
        ])

        # ResNet blocks at each encoder level
        self.res_nets = nn.ModuleList([
            ResNet(encoder_features[i], encoder_sizes[i], N_res, n_spatial_dims, use_bn, **common_kwargs)
            for i in range(N_layers)
        ])

        # Bottleneck ResNet
        self.res_net_neck = ResNet(
            encoder_features[N_layers], encoder_sizes[N_layers], N_res_neck, n_spatial_dims, use_bn,
            **common_kwargs,
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
