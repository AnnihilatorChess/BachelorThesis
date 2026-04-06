import numpy as np
import torch

from the_well.benchmark.metrics.common import Metric
from the_well.benchmark.metrics.spectral import fftn, ifftn
from the_well.data.datasets import WellMetadata


class PearsonR(Metric):
    @staticmethod
    def eval(
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
        meta: WellMetadata,
        eps: float = 1e-7,
    ) -> torch.Tensor:
        """
        Pearson Correlation Coefficient

        Args:
            x: Input tensor.
            y: Target tensor.
            meta: Metadata for the dataset.

        Returns:
            Pearson correlation coefficient between x and y.
        """
        x_flat = torch.flatten(x, start_dim=-meta.n_spatial_dims - 1, end_dim=-2)
        y_flat = torch.flatten(y, start_dim=-meta.n_spatial_dims - 1, end_dim=-2)

        # Calculate means along flattened axis
        x_mean = torch.mean(x_flat, dim=-2, keepdim=True)
        y_mean = torch.mean(y_flat, dim=-2, keepdim=True)

        # Calculate covariance
        covariance = torch.mean((x_flat - x_mean) * (y_flat - y_mean), dim=-2)
        # Calculate standard deviations
        std_x = torch.std(x_flat, dim=-2)
        std_y = torch.std(y_flat, dim=-2)

        # Calculate Pearson correlation coefficient
        correlation = covariance / (std_x * std_y + eps)
        return correlation


class MSE(Metric):
    @staticmethod
    def eval(
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
        meta: WellMetadata,
    ) -> torch.Tensor:
        """
        Mean Squared Error

        Args:
            x: Input tensor.
            y: Target tensor.
            meta: Metadata for the dataset.

        Returns:
            Mean squared error between x and y.
        """
        n_spatial_dims = tuple(range(-meta.n_spatial_dims - 1, -1))
        return torch.mean((x - y) ** 2, dim=n_spatial_dims)


class MAE(Metric):
    @staticmethod
    def eval(
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
        meta: WellMetadata,
    ) -> torch.Tensor:
        """
        Mean Absolute Error

        Args:
            x: Input tensor.
            y: Target tensor.
            meta: Metadata for the dataset.

        Returns:
            Mean absolute error between x and y.
        """
        n_spatial_dims = tuple(range(-meta.n_spatial_dims - 1, -1))
        return torch.mean((x - y).abs(), dim=n_spatial_dims)


class NMSE(Metric):
    @staticmethod
    def eval(
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
        meta: WellMetadata,
        eps: float = 1e-7,
        norm_mode: str = "norm",
    ) -> torch.Tensor:
        """
        Normalized Mean Squared Error

        Args:
            x: Input tensor.
            y: Target tensor.
            meta: Metadata for the dataset.
            eps: Small value to avoid division by zero. Default is 1e-7.
            norm_mode:
                Mode for computing the normalization factor. Can be 'norm' or 'std'. Default is 'norm'.

        Returns:
            Normalized mean squared error between x and y.
        """
        n_spatial_dims = tuple(range(-meta.n_spatial_dims - 1, -1))
        if norm_mode == "norm":
            norm = torch.mean(y**2, dim=n_spatial_dims)
        elif norm_mode == "std":
            norm = torch.std(y, dim=n_spatial_dims) ** 2
        else:
            raise ValueError(f"Invalid norm_mode: {norm_mode}")
        return MSE.eval(x, y, meta) / (norm + eps)


class RMSE(Metric):
    @staticmethod
    def eval(
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
        meta: WellMetadata,
    ) -> torch.Tensor:
        """
        Root Mean Squared Error

        Args:
            x: torch.Tensor | np.ndarray
                Input tensor.
            y: torch.Tensor | np.ndarray
                Target tensor.
            meta: WellMetadata
                Metadata for the dataset.

        Returns:
            Root mean squared error between x and y.
        """
        return torch.sqrt(MSE.eval(x, y, meta))


class NRMSE(Metric):
    @staticmethod
    def eval(
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
        meta: WellMetadata,
        eps: float = 1e-7,
        norm_mode: str = "norm",
    ) -> torch.Tensor:
        """
        Normalized Root Mean Squared Error

        Args:
            x: Input tensor.
            y: Target tensor.
            meta: Metadata for the dataset.
            eps: Small value to avoid division by zero. Default is 1e-7.
            norm_mode : Mode for computing the normalization factor. Can be 'norm' or 'std'. Default is 'norm'.

        Returns:
            Normalized root mean squared error between x and y.

        """
        return torch.sqrt(NMSE.eval(x, y, meta, eps=eps, norm_mode=norm_mode))


class VMSE(Metric):
    @staticmethod
    def eval(
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
        meta: WellMetadata,
    ) -> torch.Tensor:
        """
        Variance Scaled Mean Squared Error

        Args:
            x: Input tensor.
            y: Target tensor.
            meta: Metadata for the dataset.

        Returns:
            Variance mean squared error between x and y.
        """
        return NMSE.eval(x, y, meta, norm_mode="std")


class VRMSE(Metric):
    @staticmethod
    def eval(
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
        meta: WellMetadata,
    ) -> torch.Tensor:
        """
        Root Variance Scaled Mean Squared Error

        Args:
            x: Input tensor.
            y: Target tensor.
            meta: Metadata for the dataset.

        Returns:
            Root variance mean squared error between x and y.
        """
        return NRMSE.eval(x, y, meta, norm_mode="std")


class LInfinity(Metric):
    @staticmethod
    def eval(
        x: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
        meta: WellMetadata,
    ) -> torch.Tensor:
        """
        L-Infinity Norm

        Args:
            x: Input tensor.
            y: Target tensor.
            meta: Metadata for the dataset.

        Returns:
            L-Infinity norm between x and y.
        """
        spatial_dims = tuple(range(-meta.n_spatial_dims - 1, -1))
        return torch.max(
            torch.abs(x - y).flatten(start_dim=spatial_dims[0], end_dim=-2), dim=-2
        ).values


class SobolevH1(Metric):
    """Sobolev H1 semi-norm error: sqrt(MSE + ||grad(pred - ref)||^2).

    Penalizes high-frequency and gradient errors that pixel-space MSE misses.
    Uses FFT-based gradients: d/dx_i f = ifftn(i * k_i * fftn(f)).

    Returns:
        [T, C] tensor of H1 error per timestep and field.
    """

    @staticmethod
    def eval(
        x: torch.Tensor,
        y: torch.Tensor,
        meta: WellMetadata,
    ) -> torch.Tensor:
        spatial_dims = tuple(range(-meta.n_spatial_dims - 1, -1))
        spatial_shape = tuple(x.shape[dim] for dim in spatial_dims)
        ndim = len(spatial_dims)

        # L2 error component
        l2_mse = MSE.eval(x, y, meta)  # [T, C]

        # Gradient error via FFT
        error = x - y
        error_fft = fftn(error, meta)  # FFT of error field

        grad_mse = torch.zeros_like(l2_mse)
        for d in range(ndim):
            N = spatial_shape[d]
            # Wavenumber vector for dimension d
            k = 2 * np.pi * torch.fft.fftfreq(N, device=x.device)
            # Reshape for broadcasting: insert size-1 dims for other spatial dims
            shape = [1] * error_fft.ndim
            shape[spatial_dims[d]] = N
            k = k.reshape(shape)

            # Spectral derivative: multiply by i*k
            grad_fft = 1j * k * error_fft
            grad_real = ifftn(grad_fft, meta).real

            # MSE of gradient component, averaged over spatial dims
            n_spatial = tuple(range(-meta.n_spatial_dims - 1, -1))
            grad_mse = grad_mse + torch.mean(grad_real**2, dim=n_spatial)

        # H1 = sqrt(L2_MSE + gradient_MSE)
        return torch.sqrt(l2_mse + grad_mse)
