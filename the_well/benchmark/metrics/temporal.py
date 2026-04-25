import torch

from the_well.benchmark.metrics.common import SummaryMetric
from the_well.benchmark.metrics.spatial import NRMSE, PearsonR
from the_well.data.datasets import WellMetadata


class ValidRolloutLength(SummaryMetric):
    """Number of timesteps before per-step nRMSE exceeds a threshold.

    Reports "how long does the model remain accurate?" rather than
    "what is the total error?". Conceptually the nRMSE-threshold analogue of
    the correlation-time / prediction-horizon framing in Zhou & Barati
    Farimani (2024), "Predicting Change, Not States: An Alternate Framework
    for Neural PDE Surrogates". The exact formulation (threshold=0.2 on
    nRMSE) is our own. APEBench and MPP do not define this metric.

    Returns:
        dict with:
        - "valid_rollout_length": [C] — timestep count before threshold breach
        - "valid_rollout_fraction": [C] — fraction of trajectory completed
    """

    def __init__(self, threshold: float = 0.2):
        super().__init__()
        self.threshold = threshold

    def eval(self, x, y, meta: WellMetadata, **kwargs):
        # x, y: [T, H, W, C] (batch already reduced) or [B, T, H, W, C]
        # NRMSE.eval returns [T, C] (or [B, T, C] with batch)
        nrmse = NRMSE.eval(x, y, meta)  # [T, C] or [B, T, C]

        # If batch dim present, average over batch first
        if nrmse.ndim == 3:
            nrmse = nrmse.mean(0)  # [T, C]

        T = nrmse.shape[0]
        exceeded = nrmse > self.threshold  # [T, C]

        # Find first timestep exceeding threshold per field
        # If never exceeded, valid length = T
        any_exceeded = exceeded.any(dim=0)  # [C]
        # argmax on bool tensor returns first True index (as int64)
        first_exceed = exceeded.long().argmax(dim=0)  # [C]
        # If never exceeded, valid length = T; keep as int64 throughout
        valid_length = torch.where(any_exceeded, first_exceed, torch.tensor(T, dtype=torch.long, device=nrmse.device))

        return {
            "valid_rollout_length": valid_length.float(),
            "valid_rollout_fraction": valid_length.float() / T,
        }


class ErrorGrowthRate(SummaryMetric):
    """Exponential error growth rate (Lyapunov-like exponent).

    Fits E(t) ~ C * exp(lambda * t) by linear regression in log-space:
    log(nRMSE(t)) = log(C) + lambda * t

    A lower lambda means a more stable autoregressive surrogate.
    This metric is independent of trajectory length.

    Cite Nayak & Goswami (2025), "Data-Efficient Time-Dependent PDE
    Surrogates: Graph Neural Simulators vs Neural Operators", for the
    phenomenon of rapid autoregressive error accumulation that this
    metric formalises. The exponential-fit formulation here is ours; the
    cited paper does not define the metric itself.

    Returns:
        dict with "error_growth_rate": [C]
    """

    def __init__(self, min_steps: int = 5, eps: float = 1e-8):
        super().__init__()
        self.min_steps = min_steps
        self.eps = eps

    def eval(self, x, y, meta: WellMetadata, **kwargs):
        nrmse = NRMSE.eval(x, y, meta)  # [T, C] or [B, T, C]

        if nrmse.ndim == 3:
            nrmse = nrmse.mean(0)  # [T, C]

        T, C = nrmse.shape
        device = nrmse.device

        if T < self.min_steps:
            # Not enough data to fit; return zeros
            return {"error_growth_rate": torch.zeros(C, device=device)}

        # log(nRMSE + eps) for numerical stability
        log_nrmse = torch.log(nrmse + self.eps)  # [T, C]

        # Check for non-finite values (Inf/NaN) which crash the lstsq backend
        if not torch.isfinite(log_nrmse).all():
            return {"error_growth_rate": torch.zeros(C, device=device)}

        # Linear regression: log_nrmse = a + lambda * t
        # Design matrix [T, 2]: column of ones and column of t
        t = torch.arange(T, dtype=nrmse.dtype, device=device)
        A = torch.stack([torch.ones(T, device=device), t], dim=1)  # [T, 2]

        # Solve for each field: A @ [a, lambda]^T = log_nrmse
        # lstsq returns (solution, residuals, rank, singular_values)
        # Run on CPU: torch.linalg.lstsq on CUDA requires MAGMA and a driver arg
        result = torch.linalg.lstsq(A.cpu(), log_nrmse.cpu())  # solution: [2, C]
        growth_rate = result.solution[1].to(device)  # [C] — the lambda coefficient

        return {"error_growth_rate": growth_rate}


class CorrelationTime(SummaryMetric):
    """Number of timesteps before PearsonR drops below a threshold.

    For turbulent systems, pixel-wise matching fails eventually even for
    perfect solvers. This measures how long spatial structure is preserved,
    complementing the nRMSE-based Valid Rollout Length.

    Source: Zhou & Barati Farimani (2024), "Predicting Change, Not States:
    An Alternate Framework for Neural PDE Surrogates", which defines
    correlation time as the timestep where Pearson correlation drops below
    a threshold (e.g. 0.8) — matching our default threshold.

    Returns:
        dict with:
        - "correlation_time": [C] — timestep count before correlation drops
        - "correlation_time_fraction": [C] — fraction of trajectory completed
    """

    def __init__(self, threshold: float = 0.8):
        super().__init__()
        self.threshold = threshold

    def eval(self, x, y, meta: WellMetadata, **kwargs):
        pearson = PearsonR.eval(x, y, meta)  # [T, C] or [B, T, C]

        if pearson.ndim == 3:
            pearson = pearson.mean(0)  # [T, C]

        T = pearson.shape[0]
        dropped = pearson < self.threshold  # [T, C]

        any_dropped = dropped.any(dim=0)  # [C]
        # argmax on bool tensor returns first True index (as int64)
        first_drop = dropped.long().argmax(dim=0)  # [C]
        # If never dropped, correlation time = T; keep as int64 throughout
        corr_time = torch.where(any_dropped, first_drop, torch.tensor(T, dtype=torch.long, device=pearson.device))

        return {
            "correlation_time": corr_time.float(),
            "correlation_time_fraction": corr_time.float() / T,
        }
