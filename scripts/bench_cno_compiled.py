"""
Smoke test + speed benchmark for the CNO compiled-CUDA activation path.

Run this on the Linux training server (the box where ``nvcc`` is on ``$PATH``
and the vendored ``filtered_lrelu`` plugin can JIT-compile) **before** firing
off any full training run with ``activation=compiled``. It does three things:

1. **Smoke / shape parity** — instantiates both backends with identical
   architecture args, runs a forward + backward on a fixed input, and checks
   that the output shape matches and the backward pass doesn't crash. The
   *numerical* outputs don't have to match (different filters; the compiled
   path is the "real" alias-free formulation while the torch path is a
   bicubic-AA approximation) — we just want both to produce sensible tensors.

2. **Forward + backward timings** — warm-up loop then 20 timed iterations, with
   ``torch.cuda.synchronize`` around the timed window. Compare ``s/iter`` and
   parameter count between ``activation="torch"`` (current production path)
   and ``activation="compiled"`` (the new path). Expect roughly 10-20x speedup
   on the fwd+bwd time at the paper-spec CNO config; if the speedup is much
   less than 5x, something is wrong with the compile cache or the wiring.

3. **Fallback warning check** — instantiates a 1D model with
   ``activation="compiled"`` and verifies the user-facing warning fires
   (so 1D Burgers configs don't silently train on a different path than the
   user thought).

The configs benchmarked here mirror the entries in
``docs/models/cno.md`` so results are directly comparable to the historical
torch-path timings recorded there.
"""

from __future__ import annotations

import time
import warnings

import torch

from the_well.benchmark.models.cno import CNO


def _count_params(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def _build(activation: str, *, n_spatial_dims: int = 2,
           spatial_resolution=(128, 128), channel_multiplier: int = 96,
           N_res: int = 4, N_res_neck: int = 4) -> CNO:
    return CNO(
        dim_in=3,
        dim_out=3,
        n_spatial_dims=n_spatial_dims,
        spatial_resolution=spatial_resolution,
        N_layers=3,
        N_res=N_res,
        N_res_neck=N_res_neck,
        channel_multiplier=channel_multiplier,
        activation=activation,
        antialias=False,  # torch path -> aggressive bicubic; closer to historical default
    ).cuda()


def _time_fwd_bwd(model: CNO, x: torch.Tensor, warmup: int = 5, repeats: int = 20):
    # Warm-up (also triggers the compiled-plugin JIT build on the first call).
    for _ in range(warmup):
        y = model(x)
        y.sum().backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeats):
        y = model(x)
        y.sum().backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return (time.time() - t0) / repeats


def _time_fwd_only(model: CNO, x: torch.Tensor, warmup: int = 5, repeats: int = 20):
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(x)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeats):
        with torch.no_grad():
            _ = model(x)
    torch.cuda.synchronize()
    return (time.time() - t0) / repeats


def _bench_one(activation: str, label: str, *, spatial_resolution=(128, 128),
               channel_multiplier=96, N_res=4, N_res_neck=4, batch=2):
    model = _build(
        activation,
        spatial_resolution=spatial_resolution,
        channel_multiplier=channel_multiplier,
        N_res=N_res, N_res_neck=N_res_neck,
    )
    x = torch.randn(batch, 3, *spatial_resolution, device="cuda")
    n_params = _count_params(model)

    fwd = _time_fwd_only(model, x)
    fwd_bwd = _time_fwd_bwd(model, x)
    print(
        f"  {label:32s}  params={n_params:5.2f}M  "
        f"fwd={fwd * 1e3:7.2f} ms  fwd+bwd={fwd_bwd * 1e3:7.2f} ms"
    )
    return {"label": label, "params_M": n_params, "fwd_s": fwd, "fwd_bwd_s": fwd_bwd}


def smoke_parity():
    """Instantiate both backends, forward+backward, check shapes match."""
    print("[1/3] Shape parity (random init, same input):")
    torch.manual_seed(0)
    m_t = _build("torch")
    torch.manual_seed(0)
    m_c = _build("compiled")

    x = torch.randn(2, 3, 128, 128, device="cuda", requires_grad=False)
    y_t = m_t(x)
    y_c = m_c(x)
    assert y_t.shape == y_c.shape, f"shape mismatch: torch={y_t.shape}, compiled={y_c.shape}"
    print(f"      ok: both produce shape {tuple(y_t.shape)}")

    # Backward sanity
    y_t.sum().backward()
    y_c.sum().backward()
    has_grads_t = all(p.grad is not None for p in m_t.parameters() if p.requires_grad)
    has_grads_c = all(p.grad is not None for p in m_c.parameters() if p.requires_grad)
    assert has_grads_t and has_grads_c, "backward did not populate grads on every parameter"
    print("      ok: backward populates grads on both")


def fallback_warning_check():
    """1D model with activation='compiled' should warn and fall back."""
    print("\n[3/3] 1D fallback warning:")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m_1d = CNO(
            dim_in=2, dim_out=2, n_spatial_dims=1,
            spatial_resolution=(128,), N_layers=2,
            N_res=1, N_res_neck=1, channel_multiplier=32,
            activation="compiled",
        ).cuda()
        # Smoke: forward should still work via the torch fallback.
        x = torch.randn(2, 2, 128, device="cuda")
        _ = m_1d(x)
    fallback_warnings = [
        wi for wi in w if "compiled" in str(wi.message).lower() and "fall" in str(wi.message).lower()
    ]
    if not fallback_warnings:
        print("      WARN: expected a fallback warning for n_spatial_dims=1; none fired.")
    else:
        print(f"      ok: {len(fallback_warnings)} fallback warning(s) raised, 1D forward succeeded")


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark; run on the training server.")

    print("Device:", torch.cuda.get_device_name(0))
    print()

    smoke_parity()

    print("\n[2/3] Speed: 2D, batch=2, 128x128, paper-spec CNO (ch=96, N_res=4, N_res_neck=4):")
    rows = []
    rows.append(_bench_one("torch", "torch  (antialias=False)"))
    rows.append(_bench_one("compiled", "compiled (filtered_lrelu)"))

    print("\n     Smaller config (ch=64, N_res=2, N_res_neck=2) for sanity:")
    rows.append(_bench_one("torch", "torch  ch=64, R=2, RN=2",
                            channel_multiplier=64, N_res=2, N_res_neck=2))
    rows.append(_bench_one("compiled", "compiled ch=64, R=2, RN=2",
                            channel_multiplier=64, N_res=2, N_res_neck=2))

    # Speed-up summary
    print("\n  Speedup (compiled vs. torch) on fwd+bwd:")
    for t, c in [(rows[0], rows[1]), (rows[2], rows[3])]:
        speedup = t["fwd_bwd_s"] / c["fwd_bwd_s"]
        print(f"    {c['label']:32s}  {speedup:5.2f}x")

    fallback_warning_check()


if __name__ == "__main__":
    main()
