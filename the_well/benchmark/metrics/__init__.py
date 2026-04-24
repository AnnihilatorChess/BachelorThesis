from .common import SummaryMetric
from .plottable_data import (
    field_histograms,
    make_video,
    plot_all_time_metrics,
    plot_power_spectrum_by_field,
)
from .spatial import MAE, MSE, NMSE, NRMSE, RMSE, VMSE, VRMSE, LInfinity, PearsonR, cRMSE
from .spectral import binned_spectral_mse
from .temporal import (
    CorrelationTime,
    ErrorGrowthRate,
    ValidRolloutLength,
)

__all__ = [
    "MAE",
    "NRMSE",
    "RMSE",
    "MSE",
    "NMSE",
    "LInfinity",
    "VMSE",
    "VRMSE",
    "cRMSE",
    "binned_spectral_mse",
    "PearsonR",
    "ValidRolloutLength",
    "ErrorGrowthRate",
    "CorrelationTime",
    "SummaryMetric",
]

long_time_metrics = ["VRMSE", "RMSE", "binned_spectral_mse", "PearsonR"]
validation_metric_suite = [
    RMSE(),
    NRMSE(),
    LInfinity(),
    VRMSE(),
    cRMSE(),
    binned_spectral_mse(),
    PearsonR(),
]
# Extended [T, C] metrics — appended to validation_suite when enabled.
# Currently empty: SobolevH1 + HighFreqEnergyRatio were removed as redundant
# with binned_spectral_mse and not clearly interpretable for our thesis.
extended_validation_suite = []
# Summary metrics (scalar per field) — processed separately from split_up_losses.
# NRMSEAreaUnderCurve was removed: it's within a boundary-point rounding of
# NRMSE_T=all, which is already logged and more directly interpretable.
summary_metric_suite = [
    ValidRolloutLength(),
    ErrorGrowthRate(),
    CorrelationTime(),
]
validation_plots = [plot_power_spectrum_by_field, field_histograms]
time_plots = [plot_all_time_metrics]
time_space_plots = [make_video]
