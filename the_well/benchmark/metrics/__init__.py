from .common import SummaryMetric
from .plottable_data import (
    field_histograms,
    make_video,
    plot_all_time_metrics,
    plot_power_spectrum_by_field,
)
from .spatial import MAE, MSE, NMSE, NRMSE, RMSE, VMSE, VRMSE, LInfinity, PearsonR, SobolevH1
from .spectral import HighFreqEnergyRatio, binned_spectral_mse
from .temporal import (
    CorrelationTime,
    ErrorGrowthRate,
    NRMSEAreaUnderCurve,
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
    "binned_spectral_mse",
    "PearsonR",
    "SobolevH1",
    "HighFreqEnergyRatio",
    "ValidRolloutLength",
    "NRMSEAreaUnderCurve",
    "ErrorGrowthRate",
    "CorrelationTime",
    "SummaryMetric",
]

long_time_metrics = ["VRMSE", "RMSE", "binned_spectral_mse", "PearsonR", "SobolevH1", "HighFreqEnergyRatio"]
validation_metric_suite = [
    RMSE(),
    NRMSE(),
    LInfinity(),
    VRMSE(),
    binned_spectral_mse(),
    PearsonR(),
]
# Extended [T, C] metrics — appended to validation_suite when enabled
extended_validation_suite = [SobolevH1(), HighFreqEnergyRatio()]
# Summary metrics (scalar per field) — processed separately from split_up_losses
summary_metric_suite = [
    ValidRolloutLength(),
    NRMSEAreaUnderCurve(),
    ErrorGrowthRate(),
    CorrelationTime(),
]
validation_plots = [plot_power_spectrum_by_field, field_histograms]
time_plots = [plot_all_time_metrics]
time_space_plots = [make_video]
