# CNO — Convolutional Neural Operator

**Paper:** Raonić et al., "Convolutional Neural Operators for robust and accurate learning of PDEs"
**Venue:** NeurIPS 2023
**arXiv:** https://arxiv.org/abs/2302.01178
**Code:** https://github.com/camlab-ethz/ConvolutionalNeuralOperator

## Key Idea

CNO is designed to preserve continuous-discrete equivalence: the discretised network approximates the same operator regardless of resolution. This avoids aliasing artifacts that FNO can suffer from. It achieves this via alias-free activations — upsample ×2, apply LeakyReLU, low-pass filter, downsample — embedded in a U-Net-style encoder/decoder.

## Architecture (paper's default)

| Component | Value |
|-----------|-------|
| Channel progression | 17 → 64 → 128 → 256 |
| D/U blocks (down/upsampling) | 3 each |
| I blocks (integral/spectral) | 6 |
| R blocks (resampling) | 6 |
| Kernel size | 3 |
| Anti-aliasing filter taps (M) | 16 |
| Filter cutoff | f_s / 2.0001 |

## Model Size

~5.3 million parameters (paper's default). Our current config at `channel_multiplier=96, N_res=4, N_res_neck=4` produces ~18 M parameters so the model is compute-scale-matched to our FNO/UNet — but see the performance section below for why this backfires in pure PyTorch.

---

## Training-speed pitfall (the hidden cost of alias-free activations)

**Status:** documented, config unchanged. Decision pending.

The paper claims CNO trains *faster* than FNO per epoch (Fig. 3 center, p. 13). **In our setup it is 18–21× slower.** This section explains why, so we choose a deliberate mitigation rather than debug a non-bug.

### Root cause

The official repo ships three activation variants:

| Variant | Implementation | Speed |
|---|---|---|
| `LReLu` (classic, paper's default) | Compiled CUDA op `filtered_lrelu` ported from StyleGAN3 (Kaiser low-pass filters, written in CUDA/C++) | Fast; requires CUDA toolkit + ninja + C++ compiler |
| `LReLu_torch` | Pure PyTorch: three `F.interpolate(mode='bicubic', antialias=True)` calls per activation | **Slow** |
| `LReLu_regular` | Plain pointwise LeakyReLU (paper's "CNO w/o Filters" ablation) | Fastest; breaks the alias-free guarantee |

Our implementation (`the_well/benchmark/models/cno/__init__.py`) follows `CNO2d_simplified/CNO2d.py` — two `F.interpolate` calls per activation with `antialias=True`. Structurally identical to the reference; no bug. The speed gap comes entirely from using PyTorch-native `F.interpolate` instead of the compiled kernel.

PyTorch's antialiased bicubic has no cuDNN backend. The backward pass is especially unoptimised — see open issues in `pytorch/pytorch`:

- #82397 (bilinear AA 5× slowdown)
- #51206 (interpolate VERY slow under AMP)
- #48163 (interpolate fp16 backward extremely slow)
- #23531 (interpolate backward >10× slowdown, open since 2019)
- #103589 (antialias=True on CUDA fails at large scale ratios)
- #113445 (AA only supported for bilinear/bicubic)

None of these have a planned cuDNN fix.

### Measured timings (TRL 128×384, batch=2, RTX 3070 Ti)

| Config | Params | Forward | Fwd+Bwd | × FNO |
|---|---|---|---|---|
| **FNO ref** (`modes=16, h=128`) | 19.01 M | 14.8 ms | 41.0 ms | 1.0× |
| **UNet Classic ref** (`f=48`) | 17.47 M | 8.6 ms | 28.6 ms | 0.7× |
| CNO `ch=96, N_res=4, N_res_neck=4`, `antialias=True` | 17.93 M | 58.6 ms | 696.5 ms | 18.4× |
| CNO `ch=96, N_res=4, N_res_neck=4`, `antialias=False` | 17.93 M | 69.5 ms | 437–877 ms | 11–21× |
| CNO `ch=96, N_res=2, N_res_neck=2`, `antialias=False` | 10.87 M | 43.9 ms | 339.9 ms | 8.3× |
| CNO `ch=64, N_res=2, N_res_neck=2`, `antialias=False` | 4.86 M | 33.5 ms | 109.9 ms | 2.7× |
| CNO `ch=48, N_res=2, N_res_neck=2`, `antialias=False` | 2.75 M | 30.0 ms | 84.5 ms | 2.1× |
| CNO `ch=64, N_res=1, N_res_neck=1`, `antialias=False` | 3.29 M | 16.2 ms | 80.5 ms | 2.0× |
| CNO `ch=48, N_res=1, N_res_neck=2`, `antialias=False` | 2.53 M | 14.3 ms | 64.5 ms | 1.6× |

Scripts reproducing these numbers are in the session transcript; re-run with any inline python block using `bench(...)`.

### Why `antialias=False` isn't a full fix

Flipping `antialias=False` removes the scale-aware low-pass filter inside `F.interpolate`, giving a **1.6× speedup** on `fwd+bwd`. This matches the CNO paper's own Appendix C.5 ablation (p. 61), which reports 1.1–1.75× speedup for their "CNO w/o Filters" variant (nearest-neighbour up + avg-pool down + pointwise LReLU).

Drawbacks of `antialias=False`:

1. **Breaks continuous–discrete equivalence.** CNO's theoretical guarantee is band-limited→band-limited mapping; plain bicubic admits high-frequency ringing and lets the nonlinearity's high-frequency content alias back on downsample.
2. **Loss of resolution invariance.** Relevant if you plan to test at resolutions other than training. *Irrelevant for our thesis — we train and test at the same resolution per dataset.*
3. **Increased spectral error.** Expect a hit specifically in the high-frequency bin of `binned_spectral_mse`. Paper's own ablation reports test-error costs ranging from <10% (Compressible Euler, Darcy) to **~300%** (Poisson). For TRL 2D the penalty is uncharacterised.
4. **Amplified rollout error.** Aliasing artifacts can compound autoregressively — most relevant for the stabilisation topic of this thesis.

### What the community does about this

Summary from a web search (full citations at the end of this section):

- **Official repo has no issues at all about training speed.** 9 issues lifetime, none about speed or filter-related compile problems. The 200+ citations ride on the paper's reported speeds, not on anyone reproducing them.
- **CNO is not in `neuraloperator/neuraloperator`** (the official Kovachki/Azizzadenesheli library). Not in NVIDIA Modulus either. There is no widely-maintained reimplementation.
- **`filtered_lrelu` is effectively Linux-only in practice.** NVlabs/stylegan3 issues #2, #38, #48 document Windows build failures (`'if constexpr' is a C++17 language extension`, stuck at "Setting up PyTorch plugin..."). Standard fixes are "clear the torch_extensions cache and retry" or "use Docker/WSL."
- **Nobody has publicly flagged the 10–20× slowdown.** No blog posts, papers, or threads that benchmarked pure-PyTorch CNO against FNO. Our measurement is apparently novel — or other users quietly hit it and moved on without reporting.
- **StyleGAN3's own filtered_lrelu.py** comments on the pure-PyTorch reference path being "slow and memory-inefficient" — inherited by CNO's `LReLu_torch` branch.
- **Alias-free follow-ups exist** — see `ReNO` (OpenReview `7LSEkvEGCM`) and `MambaNO` (NeurIPS 2024, OpenReview `gUEBXGV8JM`) which reframe alias-free operator learning without `filtered_lrelu`. Potential related-work citations.

### Options we've considered (decision pending)

| Option | Effort | Effect | Drawback |
|---|---|---|---|
| 1. Compile `filtered_lrelu` on Windows | High (VS + CUDA toolkit + ninja; likely needs WSL) | Matches paper's speed | Platform-specific pain; thesis-scope creep |
| 2. Shrink CNO to compute-match FNO (`ch=64, N_res=2, N_res_neck=2`) | Minimal — one config change | ~5 M params, 2.7× FNO wall-clock. Full ablation fits in ~12 days. | CNO now has ~4× fewer params than FNO/UNet; not param-matched |
| 3. Shrink further to wall-clock match (`ch=48, N_res=1, N_res_neck=2`) | Minimal | ~2.5 M params, 1.6× FNO | Very reduced capacity; barely CNO-like |
| 4. Drop CNO stabilisation ablation, keep baseline only | None | 3 CNO runs (~5 days extra) | Only covers architecture comparison, not stabilisation techniques |
| 5. Accept `antialias=False` as the new default | Already done in code | 1.6× speed-up at param-matched size | Still 11–21× FNO, not enough on its own |

Current state:
- `antialias` is now a configurable constructor arg on `CNO` (default `False`).
- Config file `configs/model/cno.yaml` still uses `channel_multiplier=96, N_res=4, N_res_neck=4`. **No config change applied yet.**

### References (external sources consulted for this section)

- CNO paper Appendix C.5 (p. 60–61): "CNO w/o Filters" ablation, 1.1–1.75× speed-up, 0–300% test-error cost depending on PDE.
- StyleGAN3 troubleshooting: https://github.com/NVlabs/stylegan3/blob/main/docs/troubleshooting.md
- NVlabs/stylegan3 issue #2 (Ninja build fails on Windows): https://github.com/NVlabs/stylegan3/issues/2
- NVlabs/stylegan3 issue #38 (`filtered_lrelu_plugin problem`): https://github.com/NVlabs/stylegan3/issues/38
- NVlabs/stylegan3 issue #48 (hang at plugin setup): https://github.com/NVlabs/stylegan3/issues/48
- PyTorch issues on `F.interpolate` speed: #82397, #51206, #48163, #23531, #103589, #113445
- camlab-ethz/ConvolutionalNeuralOperator issues list (no speed-related entries): https://github.com/camlab-ethz/ConvolutionalNeuralOperator/issues
- MambaNO (explicit alternative alias-free operator): https://openreview.net/forum?id=gUEBXGV8JM
- ReNO (representation-equivalent operators): https://openreview.net/forum?id=7LSEkvEGCM
