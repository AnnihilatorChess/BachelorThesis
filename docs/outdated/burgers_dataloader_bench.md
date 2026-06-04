# 1D Burgers DataLoader Benchmark

Empirical study of DataLoader settings for the 1D Burgers HDF5 layout, motivated by the ~54 % GPU utilisation observed on server runs (see [summary.md §1D Burgers Training Throughput](summary.md)). Chunking was already tested on this dataset and made things worse — The Well's default contiguous layout appears to be the correct choice — so this benchmark focuses on **DataLoader-level knobs** instead.

## Setup

- **Script:** [`scripts/benchmarks/bench_burgers_dataloader.py`](../../scripts/benchmarks/bench_burgers_dataloader.py)
- **Synthetic dataset:** [`scripts/benchmarks/gen_synthetic_burgers.py`](../../scripts/benchmarks/gen_synthetic_burgers.py) — writes a 1.6 GB HDF5 with exactly the same layout the real conversion script produces (contiguous `t0_fields/u`, shape `(2000, 201, 1024)`, no chunking, no compression, plus the required dimensions / scalars / BC groups). 2000 training trajectories → **394 000 windows** at `n_steps_input=4, n_steps_output=1`.
- **Hardware:** laptop, 16 logical CPU cores, Windows 10, torch 2.10, local SSD.
- **Protocol:** each config runs 2 epochs × 50 batches. Epoch-2 steady-state numbers are the ones that matter (E1 includes worker spawn; E2 shows what training actually sees).

> **Important caveat on absolute numbers.** The real server runs measured **1.54 s/iter** at `bs=1024, num_workers=8`. The same config on the laptop measures **0.26 s/iter**. We are ~6× faster locally — likely a mix of Linux-vs-Windows `fork`/worker-start overhead and different I/O characteristics. **The relative ranking between configurations is what this benchmark informs**; absolute speedups on the server will be smaller in wall-clock terms but the ordering should hold.

## Results — sorted by steady-state throughput

| Config | bs | workers | persistent | prefetch | pin | iter (s) | samples/s | E2 first-batch (s) |
|---|---:|---:|:---:|---:|:---:|---:|---:|---:|
| **w16_persistent_prefetch4_bs512** | 512 | 16 | ✓ | 4 | ✓ | **0.097** | **5254** | 1.8 |
| w16_persistent_prefetch4 | 1024 | 16 | ✓ | 4 | ✓ | 0.201 | 5085 | 3.5 |
| w16_persistent_prefetch8 | 1024 | 16 | ✓ | 8 | ✓ | 0.202 | 5060 | 3.4 |
| w16 | 1024 | 16 | ✗ | 2 | ✓ | 0.218 | 4706 | 7.3 |
| w12_persistent_prefetch4 | 1024 | 12 | ✓ | 4 | ✓ | 0.239 | 4288 | 2.9 |
| **baseline_w8** *(current)* | 1024 | 8 | ✗ | 2 | ✓ | 0.260 | 3945 | 4.6 |
| w8_persistent | 1024 | 8 | ✓ | 2 | ✓ | 0.267 | 3838 | **2.2** |
| w8_prefetch8 | 1024 | 8 | ✗ | 8 | ✓ | 0.267 | 3829 | 5.1 |
| w8_prefetch4 | 1024 | 8 | ✗ | 4 | ✓ | 0.307 | 3335 | 5.2 |
| bs512_w8 | 512 | 8 | ✗ | 2 | ✓ | 0.159 | 3229 | 3.5 |
| w8_persistent_prefetch4_nopin | 1024 | 8 | ✓ | 4 | ✗ | 0.317 | 3228 | 2.7 |
| w8_nopin | 1024 | 8 | ✗ | 2 | ✗ | 0.317 | 3226 | 4.8 |
| bs2048_w8 | 2048 | 8 | ✗ | 2 | ✓ | 0.638 | 3213 | 7.5 |
| w8_persistent_prefetch8 | 1024 | 8 | ✓ | 8 | ✓ | 0.323 | 3170 | 2.7 |
| bs256_w8 | 256 | 8 | ✗ | 2 | ✓ | 0.081 | 3145 | 2.9 |
| w8_persistent_prefetch4 | 1024 | 8 | ✓ | 4 | ✓ | 0.335 | 3053 | 3.0 |
| w4 | 1024 | 4 | ✗ | 2 | ✓ | 0.446 | 2297 | 3.9 |
| w2 | 1024 | 2 | ✗ | 2 | ✓ | 0.843 | 1214 | 3.5 |
| w0 | 1024 | 0 | — | — | ✓ | 1.660 | 617 | 1.6 |

Raw JSON: [`scripts/benchmarks/bench_results_full.json`](../../scripts/benchmarks/bench_results_full.json).

## Findings

### 1. `num_workers` is by far the dominant knob

| workers | 0 | 2 | 4 | 8 | 12 | 16 |
|---:|---:|---:|---:|---:|---:|---:|
| iter (s) | 1.660 | 0.843 | 0.446 | 0.260 | 0.239* | 0.218 |
| samples/s | 617 | 1214 | 2297 | 3945 | 4288* | 4706 |

\* 12-worker number is with persistent+prefetch on, so not fully apples-to-apples.

Near-linear scaling up through 8, then diminishing to 16 on this 16-core machine. **The current server config of `num_workers=8` is under-provisioned** — raising to 16 would improve steady-state throughput by ~19 % in this benchmark (4706 vs 3945 samples/s). This alone should be the first change.

### 2. `persistent_workers` is mostly about first-batch latency

At `w=8, pref=2`:

| | first-batch E1 | first-batch E2 | steady iter |
|---|---:|---:|---:|
| non-persistent | 4.8 s | 4.6 s | 0.260 s |
| persistent | 4.9 s | **2.2 s** | 0.267 s |

Steady-state iter time is essentially unchanged. What persistent workers buy is: **every epoch after the first skips ~2–3 s of worker respawn**. On Burgers with ~1500 iters/epoch, this is <0.2 % of epoch time — negligible. But validation is short (~30 iters), so persistent workers do help the val loops more meaningfully in the real training loop.

Flip it on when combined with more workers (see §5 below).

### 3. `prefetch_factor` above 2 does not help here

At `w=8`, raising prefetch 2 → 4 → 8 stays within noise or worse (0.260 → 0.307 → 0.267 s). At `w=16 persistent`, prefetch 4 vs 8 is a wash (0.201 vs 0.202 s). The default of 2 is fine for this workload because workers are already producing faster than the (GPU-less) consumer is pulling.

### 4. `pin_memory` — keep it on

Disabling pin_memory went 0.260 → 0.317 s/iter (+22 %), which is counter-intuitive for a CPU-only consumer but the pinned branch of torch's DataLoader has its own producer thread that overlaps work. On the server — where the next step is actually a `.cuda()` — pin_memory is unambiguously the right choice.

### 5. Batch size does not trade against throughput much

| batch | iter (s) | samples/s | notes |
|---:|---:|---:|---|
| 256 | 0.081 | 3145 | overhead per batch is almost fixed |
| 512 | 0.159 | 3229 | |
| 1024 | 0.260 | **3945** | best at w=8 |
| 2048 | 0.638 | 3213 | single-worker still has to assemble 2048 samples |

At `w=8` the loader prefers `bs=1024`. At `w=16` both `bs=1024` and `bs=512` are competitive (5085 vs 5254 samples/s). So if gradient-noise considerations push toward smaller batches, the throughput cost at `w=16` is near-zero.

### 6. Best combination

The top config in this benchmark is `w=16, persistent_workers=True, prefetch_factor=4, pin_memory=True`, run at bs=1024 (≈5085 samples/s) or bs=512 (≈5254 samples/s). That's a **29 % throughput improvement over the current baseline** (3945 samples/s), purely from DataLoader settings, with no change to the HDF5 layout.

## Caveats

1. **Synthetic data is a simplification.** Real Burgers values may compress better in the OS page cache (all zero-mean noise has high entropy). We exercise the **same HDF5 layout and the same `WellDataset` code path**, so the shape of the results (which knob helps, by how much relative to others) transfers; absolute numbers do not.
2. **Windows `spawn` is slower than Linux `fork`.** Worker startup cost is higher on the laptop than on the server, which makes `persistent_workers` look more valuable here than it probably is on Linux.
3. **No GPU compute is in the loop.** The server measurements were GPU-bound-waiting-on-I/O; this benchmark is purely I/O + preprocessing. That means the improvements measured here should *translate* to GPU utilisation on the server (they fill the pipe faster), but the actual wall-clock savings depend on how much of the server's 1.54 s/iter is CPU-side.
4. **Epoch-count effect.** With 100 epochs × ~1500 iters, the worker-spawn cost that `persistent_workers` eliminates is amortised. It matters more for short validation loops than for training.

## Recommended server config changes

In priority order:

1. **`data_workers: 16`** in `server/student_1d.yaml` (up from 8) — largest single win.
2. **`persistent_workers: True`** in the DataLoader calls in `the_well/data/datamodule.py`. Requires exposing it as a config option (currently hard-coded absent).
3. Leave `prefetch_factor` at 2 and `pin_memory=True` — no improvement from changing them in this benchmark.
4. Batch size: keep 1024 unless switching to 16 workers, in which case 512 becomes equally fast and may improve training dynamics.

## Reproducing

```bash
# 1.6 GB of synthetic Burgers data
python scripts/benchmarks/gen_synthetic_burgers.py --n_traj 2000

# full sweep, writes bench_results_full.json
python scripts/benchmarks/bench_burgers_dataloader.py --iters_per_epoch 50

# subset
python scripts/benchmarks/bench_burgers_dataloader.py \
    --iters_per_epoch 50 \
    --only w16_persistent_prefetch4,baseline_w8
```
