#!/usr/bin/env python3
"""Render a 4x4 grid of dataset snapshots.

Rows  = datasets (turbulent_radiative_layer_2D, active_matter, pdebench_swe,
        pdebench_1d_burgers)
Cols  = snapshots at t in {0, T/3, 2T/3, T}

The 2D datasets are shown as images (imshow); the 1D Burgers dataset is shown as
u(x) line plots (a 1D "snapshot at t" is a curve, not an image).

The script reads The Well HDF5 layout directly with h5py:

    {base}/{dataset}/data/{split}/*.hdf5
        dimensions/time
        t0_fields/<name>   shape (n_traj, T, *spatial)              # scalars
        t1_fields/<name>   shape (n_traj, T, *spatial, dim)         # vectors
        t2_fields/<name>   shape (n_traj, T, *spatial, dim, dim)    # tensors

Run on the server (active_matter lives on the publicdata path, the rest on the
local path -- defaults below come from configs/server/{local,publicdata}.yaml):

    python scripts/plot_dataset_snapshots.py \
        --local-base ../../datasets/ \
        --publicdata-base /system/user/publicdata/the_well/datasets/ \
        --out dataset_snapshots.png

Add --labels for row/column labels.
"""
import argparse
import glob
import os
import re

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# Per-dataset configuration. `base` is filled in from CLI args at runtime.
# `fields` lists preferred field names (first match wins); falls back to the
# first available field if none are present.
# ---------------------------------------------------------------------------
DATASETS = [
    dict(label="turbulent_radiative_layer_2D", source="local",
         name="turbulent_radiative_layer_2D", fields=["density", "pressure"],
         cmap="inferno", is_1d=False, transpose=True),  # 128x384 -> long axis vertical
    dict(label="active_matter", source="publicdata",
         name="active_matter", fields=["concentration"],
         cmap="viridis", is_1d=False),
    dict(label="pdebench_swe", source="local",
         name="pdebench_swe", fields=["h"],
         cmap="cividis", is_1d=False),
    dict(label="pdebench_1d_burgers", source="local",
         name="pdebench_1d_burgers", fields=["u"],
         line_color="#16466b", is_1d=True),
]

COL_TITLES = [r"$t=0$", r"$t=T/3$", r"$t=2T/3$", r"$t=T$"]


def _decode(names):
    return [n.decode() if isinstance(n, bytes) else str(n) for n in names]


def find_files(base, name):
    """Locate the HDF5 files for a dataset, trying the usual split layout."""
    for split in ("train", "valid", "test"):
        files = sorted(glob.glob(os.path.join(base, name, "data", split, "*.hdf5")))
        if files:
            return files
    for pat in (os.path.join(base, name, "data", "*.hdf5"),
                os.path.join(base, name, "*.hdf5")):
        files = sorted(glob.glob(pat))
        if files:
            return files
    return []


def choose_file(files, ds, am_file=None):
    """Pick which HDF5 file (parameter set) to visualise.

    For active_matter the files are one-per-parameter-set; the alphabetically
    first one (zeta_1.0/alpha_-1.0) is the *least* active and homogenises. We
    instead pick the most active set (highest zeta, then most negative alpha),
    or honour an explicit --am-file substring.
    """
    if ds["name"] == "active_matter" and len(files) > 1:
        if am_file:
            hits = [f for f in files if am_file in os.path.basename(f)]
            if hits:
                return hits[0]

        def activity(path):
            b = os.path.basename(path)
            z = re.search(r"zeta_(-?\d+(?:\.\d+)?)", b)
            a = re.search(r"alpha_(-?\d+(?:\.\d+)?)", b)
            zeta = float(z.group(1)) if z else 0.0
            alpha = float(a.group(1)) if a else 0.0
            return (zeta, -alpha)  # high zeta, most negative alpha = most active

        return max(files, key=activity)
    return files[0]


def pick_field(f, preferred):
    """Return (group, field_name) for the first preferred field, else the first."""
    for order in ("t0_fields", "t1_fields", "t2_fields"):
        if order in f:
            names = _decode(f[order].attrs.get("field_names", []))
            for p in preferred:
                if p in names:
                    return order, p
    for order in ("t0_fields", "t1_fields", "t2_fields"):
        if order in f and len(f[order].attrs.get("field_names", [])):
            return order, _decode(f[order].attrs["field_names"])[0]
    raise RuntimeError("no fields found in file")


def four_time_indices(n_t):
    return [0,
            int(round((n_t - 1) / 3)),
            int(round(2 * (n_t - 1) / 3)),
            n_t - 1]


def reduce_components(snap):
    """Collapse trailing vector/tensor component axes into a scalar magnitude."""
    snap = np.squeeze(np.asarray(snap, dtype=np.float64))
    if snap.ndim > 2:  # 2D field with component dim(s) -> magnitude
        snap = np.linalg.norm(snap.reshape(snap.shape[0], snap.shape[1], -1), axis=-1)
    return snap


def load_snapshots(path, preferred, traj):
    """Return (snapshots, field_label) for one trajectory at the 4 time indices."""
    with h5py.File(path, "r") as f:
        order, name = pick_field(f, preferred)
        dset = f[order][name]
        sample_varying = bool(dset.attrs.get("sample_varying", True))
        time_varying = bool(dset.attrs.get("time_varying", True))
        n_samples = dset.shape[0] if sample_varying else 1
        traj = min(traj, n_samples - 1)
        full = dset[traj] if sample_varying else dset[:]   # (T, *spatial, *comp)
        n_t = full.shape[0] if time_varying else 1
        idxs = four_time_indices(n_t)
        snaps = [reduce_components(full[i]) for i in idxs]
    return snaps, f"{order}/{name}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local-base", default="../../datasets/",
                    help="base path for local datasets (configs/server/local.yaml)")
    ap.add_argument("--publicdata-base",
                    default="/system/user/publicdata/the_well/datasets/",
                    help="base path for active_matter (configs/server/publicdata.yaml)")
    ap.add_argument("--out", default="dataset_snapshots.png", help="output image path")
    ap.add_argument("--traj", type=int, default=0, help="trajectory index to show")
    ap.add_argument("--dpi", type=int, default=300, help="output resolution")
    ap.add_argument("--panel", type=float, default=2.6, help="panel size in inches")
    ap.add_argument("--pct", type=float, nargs=2, default=(1.0, 99.0),
                    help="percentile clip for image colour range")
    ap.add_argument("--norm", choices=["row", "panel"], default="row",
                    help="colour scale shared across a row (default) or per panel")
    ap.add_argument("--aspect", choices=["auto", "equal"], default="auto",
                    help="'auto' fills each cell (uniform grid); 'equal' keeps "
                         "physical proportions (e.g. TRL stays 1:3)")
    ap.add_argument("--am-file", default=None,
                    help="substring selecting the active_matter parameter file "
                         "(default: highest-activity set)")
    ap.add_argument("--labels", action="store_true",
                    help="draw row (dataset) and column (time) labels")
    args = ap.parse_args()

    bases = {"local": args.local_base, "publicdata": args.publicdata_base}

    n_rows = len(DATASETS)
    fig, axes = plt.subplots(
        n_rows, 4,
        figsize=(4 * args.panel, n_rows * args.panel),
        gridspec_kw=dict(wspace=0.04, hspace=0.04),
    )

    for r, ds in enumerate(DATASETS):
        base = bases[ds["source"]]
        files = find_files(base, ds["name"])
        row_axes = axes[r]

        if not files:
            for ax in row_axes:
                ax.set_xticks([]); ax.set_yticks([])
            row_axes[0].text(0.02, 0.5, f"{ds['label']}: NOT FOUND under {base!r}",
                             transform=row_axes[0].transAxes, color="red",
                             va="center", fontsize=9)
            print(f"[WARN] {ds['label']}: no files under {os.path.join(base, ds['name'])}")
            continue

        path = choose_file(files, ds, args.am_file)
        snaps, field_label = load_snapshots(path, ds["fields"], args.traj)
        if ds.get("transpose"):
            snaps = [s.T for s in snaps]
        print(f"[ok]   {ds['label']:32s} field={field_label:24s} "
              f"shape={snaps[0].shape} file={os.path.basename(path)}")

        if ds["is_1d"]:
            ymin = min(float(s.min()) for s in snaps)
            ymax = max(float(s.max()) for s in snaps)
            pad = 0.05 * (ymax - ymin + 1e-9)
            for c, s in enumerate(snaps):
                ax = row_axes[c]
                ax.plot(np.linspace(0, 1, s.shape[0]), s,
                        color=ds["line_color"], lw=1.4)
                ax.set_ylim(ymin - pad, ymax + pad)
                ax.set_xlim(0, 1)
                ax.set_xticks([]); ax.set_yticks([])
        else:
            if args.norm == "row":
                stacked = np.concatenate([s.ravel() for s in snaps])
                lo, hi = np.percentile(stacked, args.pct)
                ranges = [(lo, hi)] * len(snaps)
            else:  # per-panel scaling reveals structure when amplitude decays
                ranges = [tuple(np.percentile(s.ravel(), args.pct)) for s in snaps]
            for c, s in enumerate(snaps):
                ax = row_axes[c]
                vmin, vmax = ranges[c]
                ax.imshow(s, cmap=ds["cmap"], vmin=vmin, vmax=vmax,
                          aspect=args.aspect, origin="lower", interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])

        if args.labels:
            row_axes[0].set_ylabel(ds["label"].replace("_", r"\_") if False
                                   else ds["label"], fontsize=9)

    if args.labels:
        for c, title in enumerate(COL_TITLES):
            axes[0, c].set_title(title, fontsize=11)

    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", pad_inches=0.02)
    print(f"\nSaved {args.out} ({args.dpi} dpi)")


if __name__ == "__main__":
    main()
