# BPTT vs PDEBench — Implementation, Configs, Hardware

A close reading of `the_well/benchmark/trainer/bptt_training.py`, our PDEBench-protocol experiment configs, the PDEBench reference repo, and the PDEBench paper. Plus the hardware findings from running it on the JKU server.

**Status (2026-05-10):** the BPTT trainer now mirrors PDEBench's two training algorithms — FNO autoregressive (full BPTT) and U-Net pushforward (PF-K with no_grad warmup + grad-tracked tail) — at exact step-count parity. The remaining deviations are deliberate, documented, and listed in §5.

## 1. What PDEBench actually does (reference)

### FNO ([`pdebench/models/fno/train.py`](file:///C:/Users/simon/Documents/GitHub/PDEBench/pdebench/models/fno/train.py))

The training loop in `fno/train.py` is `training_type == "autoregressive"` (the paper's default — Appendix C):

```python
loss = 0
pred = yy[..., :initial_step, :]
for t in range(initial_step, t_train):           # 10 → 101  ⇒ 91 steps for SWE
    inp = xx.reshape(inp_shape)
    y   = yy[..., t:t+1, :]
    im  = model(inp, grid)                       # GRADIENT-TRACKED
    loss += loss_fn(im.reshape(_batch, -1), y.reshape(_batch, -1))
    pred = torch.cat((pred, im), -2)
    xx   = torch.cat((xx[..., 1:, :], im), dim=-2)  # autoregressive feedback
optimizer.zero_grad()
loss.backward()                                  # ONE backward over 91 graphs
optimizer.step()
```

Critical properties:

1. **Full BPTT through the entire trajectory.** For 2D SWE that's 91 grad-tracked forward passes per `optimizer.step()`. For 1D Burgers (after their `reduced_resolution_t=5` time downsampling), 31 steps.
2. **Loss is *summed*, not averaged.** Effective per-step gradient is ~91× a one-step gradient on SWE.
3. **No noise injection, no curriculum, no temporal bundling.** Pure deterministic full-trajectory BPTT.
4. **No `torch.no_grad()` warmup.** Every step is gradient-tracked.

### U-Net ([`pdebench/models/unet/train.py`](file:///C:/Users/simon/Documents/GitHub/PDEBench/pdebench/models/unet/train.py))

The U-Net has three modes selected by `(ar_mode, pushforward)` flags. The one PDEBench *reports* is `PF-20` (paper §4.3, Figure 5: *"we used the U-Net with pushforward trick as our baseline score for U-Net"*):

| Mode | Flags | What happens |
|---|---|---|
| `1-step` | `ar_mode=False` | Teacher-forced with σ=0.001 input noise. Loss summed across all `t`. No autoregressive feedback. |
| `AR` | `ar_mode=True, pushforward=False` ⇒ `unroll_step = t_train` | Same as FNO autoregressive — full BPTT. |
| **`PF-20`** (reported baseline) | `ar_mode=True, pushforward=True, unroll_step=20` | First `t_train - initial_step - unroll_step` steps under `torch.no_grad()`; last 20 steps gradient-tracked, summed-MSE loss. Single backward. |

### Optimizer / scheduler / loss / data

From [`pdebench/models/config/config.yaml`](file:///C:/Users/simon/Documents/GitHub/PDEBench/pdebench/models/config/config.yaml) and the per-dataset configs (`config_rdb.yaml` for SWE, `config_Bgs.yaml` for Burgers):

| Knob | PDEBench default |
|---|---|
| Optimizer | `Adam(lr=1e-3, weight_decay=1e-4)` |
| Scheduler | `StepLR(step_size=100, gamma=0.5)` |
| Epochs | 500 |
| Loss | `nn.MSELoss(reduction='mean')` per step, **summed** across the unrolled loop |
| Batch size | 5 (SWE), 50 (Burgers) |
| `initial_step` | 10 |
| `t_train` | 101 (SWE), 200 (Burgers, downsampled by 5 in time → 41 effective frames) |
| `reduced_resolution` | 1 (SWE), **4** (Burgers — 1024 → 256 spatial) |
| `reduced_resolution_t` | 1 (SWE), **5** (Burgers — 201 → 41 temporal) |
| `unroll_step` | 20 (UNet PF only) |
| FNO size | `modes=12, width=20` (~150 K params) |
| UNet size | `init_features=32` (~7.7 M params) |
| Train/test split | 90 / 10 (no separate validation) |

## 2. What our BPTT trainer actually does (current state)

[`bptt_training.py`](../the_well/benchmark/trainer/bptt_training.py) implements both PDEBench training algorithms via two knobs plus a reduction selector:

| Knob | Default | Meaning |
|---|---|---|
| `bptt_unroll_steps` | 5 (a safety fallback; **always overridden** by the experiment configs to 91/31/20) | # of gradient-tracked unrolled steps per batch |
| `pf_warmup_steps` | 0 | # of leading `torch.no_grad()` rollout steps. `0` → FNO-AR; `>0` → U-Net PF-K. |
| `loss_reduction` | `"sum"` | Matches PDEBench. `"mean"` (the original behaviour) is opt-in. |

The four PDEBench-protocol experiment configs in [`configs/experiment/`](../the_well/benchmark/configs/experiment/) bake in the right per-(model, dataset) combination:

| Experiment file | Model | Dataset | unroll | warmup | batch | What it reproduces |
|---|---|---|---:|---:|---:|---|
| [`fno_pdebench_swe.yaml`](../the_well/benchmark/configs/experiment/fno_pdebench_swe.yaml) | FNO (modes=12, h=20) | `pdebench_swe` (128×128, 101 steps) | 91 | 0 | 5 | PDEBench FNO-AR |
| [`fno_pdebench_burgers.yaml`](../the_well/benchmark/configs/experiment/fno_pdebench_burgers.yaml) | FNO (modes=12, h=20) | `pdebench_1d_burgers_pdebench` (256×41) | 31 | 0 | 50 | PDEBench FNO-AR (Burgers) |
| [`unet_pdebench_swe.yaml`](../the_well/benchmark/configs/experiment/unet_pdebench_swe.yaml) | UNet (f=32) | `pdebench_swe` | 20 | 71 | 5 | PDEBench UNet-PF-20 |
| [`unet_pdebench_burgers.yaml`](../the_well/benchmark/configs/experiment/unet_pdebench_burgers.yaml) | UNet (f=32) | `pdebench_1d_burgers_pdebench` | 20 | 11 | 50 | PDEBench UNet-PF-20 (Burgers) |

### Where the unroll horizons came from

| Dataset | Source | Effective unroll |
|---|---|---|
| 2D SWE | `config_rdb.yaml`: `t_train=101, initial_step=10, reduced_resolution_t=1` | `min(101, 101) - 10 = 91` |
| 1D Burgers | `config_Bgs.yaml`: `t_train=200, initial_step=10, reduced_resolution_t=5` | source has 201 steps; `[::5]` → 41 effective; `min(200, 41) - 10 = 31` |
| UNet PF | `unroll_step=20` (dataset-independent, every PDEBench `args/*.yaml`) | warmup = `unroll_total - 20` |

### How the dataloader is fed

[`train.py:64-71`](../the_well/benchmark/train.py) sets

```python
cfg.data["n_steps_output_train"] = (bptt_unroll_steps + pf_warmup_steps) * bundle_size
```

For SWE that's 91 future timesteps per training window — exactly the full trajectory (10 input + 91 output = 101). One window per trajectory, just like PDEBench.

## 3. Config parity table — current state vs PDEBench

Composed from `experiment/fno_pdebench_swe.yaml` + `trainer/bptt.yaml` + `optimizer/adam_pdebench.yaml`:

| Knob | PDEBench | Ours (`fno_pdebench_swe`) | Match? |
|---|---|---|---|
| Optimizer | Adam | Adam | ✅ |
| `lr` | `1e-3` | `1e-3` (from `adam_pdebench.yaml`) | ✅ |
| `weight_decay` | `1e-4` | `1e-4` | ✅ |
| Loss reduction | sum | sum | ✅ |
| `bptt_unroll_steps` | 91 (full trajectory) | 91 | ✅ |
| `pf_warmup_steps` | 0 | 0 | ✅ |
| `n_steps_input` (= PDEBench `initial_step`) | 10 | 10 | ✅ |
| `n_steps_output_train` | full trajectory (91) | 91 (= unroll + warmup) | ✅ |
| Batch size | 5 (SWE) | 5 | ✅ |
| FNO `modes` / `hidden` | 12 / 20 | 12 / 20 | ✅ |
| **Scheduler** | `StepLR(100, 0.5)` | `LinearWarmupCosineAnnealingLR(warmup=5)` | ❌ deliberate (see §5) |
| **Epochs** | 500 | 100 | ❌ deliberate (see §5) |
| **Mixed precision** | off | on (default in `bptt.yaml`); off in current LR-sweep override | ⚠️ depends on run |
| Train / test split | 90 / 10 | 80 / 10 / 10 (real val set) | ⚠️ acceptable deviation |
| Normalization | none (raw) | z-score | ⚠️ acceptable deviation |

For UNet PF-20 the table is the same except `bptt_unroll_steps=20, pf_warmup_steps=71`, and `init_features=32`.

For Burgers, the data variant is the parity dataset `pdebench_1d_burgers_pdebench` (256 × 41), produced by `python scripts/convert_pdebench_1d_burgers.py --variant pdebench`. **One real difference** there: PDEBench's default `config_Bgs.yaml` uses `1D_Burgers_Sols_Nu1.0.hdf5` (smooth regime); we converted ν=10⁻³ (shock-forming regime). The paper itself sweeps ν, but headline numbers in their figures use ν=1.0.

## 4. Hardware findings (server runs, 2026-05-10)

Empirical memory + speed numbers from running the parity configs on the JKU GPU server.

### FNO-AR-SWE (FNO at modes=12, hidden=20, BPTT depth 91)

| GPU | VRAM | AMP | Max batch | Per-step time | GPU util | Notes |
|---|---:|---|---:|---|---:|---|
| **Titan V** | 12 GB | off | **1** | slow | ~20 % | dispatch-bound: tiny model + batch=1 + 91 sequential forwards = kernels too small to fill the GPU |
| **Quadro GV100** | 32 GB | off | **4** (28 GB used) | "a lot faster" | not measured | batch=5 OOMs |

Per-step compute on a 150K-param FNO at fp32 batch=1 is maybe 1–3 ms on the GPU itself. Each `optimizer.step()` does 91 forwards + 91 backwards sequentially, and at batch=1 the per-step Python/dispatch overhead is comparable to the GPU work — hence the 20 % utilisation on Titan V. On the Quadro GV100, batch=4 multiplies kernel size by 4× and the dispatch fraction drops; that's most of the speedup.

The Quadro GV100 sweet spot is **batch=4**. Batch=5 — which would be exact PDEBench parity — OOMs because the BPTT autograd graph holds activations for all 91 steps and the marginal sample brings us above the 32 GB VRAM ceiling.

PDEBench themselves report running on RTX 2080 (8 GB, paper §4.1) and V100 (Appendix F). Their batch=5 fits because their hand-rolled `FNO2d` is leaner than ours (`neuraloperator.models.FNO`) — same param count, more compact activation footprint.

### UNet-PF-20-SWE (UNet at init_features=32, BPTT 20 grad-tracked + 71 no_grad)

| GPU | AMP | Batch | Peak VRAM |
|---|---|---:|---:|
| Quadro GV100 | off | 5 (PDEBench parity) | **~5 GB** |

UNet PF-20 is dramatically lighter on memory than FNO-AR despite the UNet having ~50× more parameters (~7.7 M vs ~150 K):

1. **Only 20 forward passes are in autograd.** The 71 warmup forwards run under `torch.no_grad()` and don't keep activations.
2. **UNet is more memory-lean per forward.** UNet stores ~1 MB / level / batch element at 128×128, vs FNO's spectral conv intermediates which are FFT-of-input-sized at multiple modes. The parameter count differs by 50×, but the activation memory differs by ~4× the other way.

Net effect: ~25× less BPTT VRAM than FNO-AR on the same dataset. The UNet experiments are not memory-constrained; they fit the PDEBench batch size on any decent GPU.

### `torch.compile` — does not work for FNO

The `neuraloperator` library's `SpectralConv` builds its weight via parameterisation and indexes it with a list of slices:

```python
# neuralop/layers/spectral_convolution.py:453
weight = self._get_weight(indices)[slices_w]
```

`torch._dynamo` raises a hard `AssertionError: ListVariable(length=4)` inside its `nn_module` `__getitem__` handler — *not* a graph break, an actual tracing-level crash. Setting `torch._dynamo.config.suppress_errors = True` makes it fall back to eager for that scope (we wired this up automatically when `compile.enabled=true` in [train.py](../the_well/benchmark/train.py)), but since the spectral conv is the expensive op, the realistic speedup is essentially zero. **Conclusion:** compile is left default-off; for FNO experiments we accept the dispatch overhead.

UNet should compile cleanly (no parametrised weights, no FFTs), but the UNet experiments aren't dispatch-bound to begin with, so it's not a priority.

### Implication for LR search

- **FNO-AR on Titan V:** batch=1 only. Low GPU utilisation but functional. LR sweep at batch=1 is the only option on this GPU.
- **FNO-AR on Quadro GV100:** batch=4. ~4× lower per-step gradient noise than batch=1, plus much higher utilisation. **Use this GPU for the LR sweep if it's available.**
- **UNet-PF-20:** any GPU. Use PDEBench parity batch=5.

## 5. Remaining deviations from PDEBench (deliberate)

These are the things that still don't match PDEBench, and the reasoning for each.

| Knob | PDEBench | Ours | Why we deviate |
|---|---|---|---|
| Scheduler | `StepLR(100, 0.5)` | `LinearWarmupCosineAnnealingLR` | Used everywhere else in the thesis; cross-experiment comparability matters more than per-experiment PDEBench parity. `step_lr.yaml` exists if exact parity is wanted. |
| Epochs | 500 | 100 | Compute budget; 5× longer runs aren't justified. We accept this is undertraining vs PDEBench and treat it as a deviation. |
| Train/test split | 90/10 | 80/10/10 | We need a real validation set for checkpoint selection; PDEBench treats test as validation, which conflates selection and reporting. |
| Normalization | none (raw data) | z-score | Improves numerical conditioning; standard across the thesis. |
| Mixed precision (`enable_amp`) | off (fp32) | `True` default in `bptt.yaml`; **disabled** in the SWE LR-sweep CLI override | Memory pressure on Titan V/GV100 forced AMP-off for FNO-AR; UNet PF-20 keeps AMP on. |
| `lr` | `1e-3` | `1e-3` for AMP-off runs; LR-sweep CLI tries `{3e-3, 1e-3, 3e-4}` | We ran a coarse LR search on this exact protocol because BPTT-91 with sum-loss has fundamentally different gradient scaling than 1-step training (~91× larger per-step gradient signal). Reusing the baseline LR was not defensible. |
| Burgers ν | 1.0 (smooth) | 0.001 (shock-forming) | Our converted dataset uses ν=10⁻³; PDEBench's headline numbers use ν=1.0. Direct number-comparison only valid at the same ν. |
| `temporal_loss_boundaries` | n/a | `[0, 6, 12, 13, 30]` (TRL bins) | Cosmetic logging-only artifact; produces a degenerate `[12,13]` bin on SWE/Burgers but doesn't affect any reported metric. |

### Burgers spatial+temporal downsampling — solved at conversion time

PDEBench's Burgers training is at 256 × 41, not 1024 × 201. We added `--variant pdebench` to [`scripts/convert_pdebench_1d_burgers.py`](../scripts/convert_pdebench_1d_burgers.py); running it produces a separate dataset `pdebench_1d_burgers_pdebench/` with PDEBench's exact `[::4, ::5]` naive subsampling baked in. The thesis-wide full-resolution `pdebench_1d_burgers/` is unaffected; the parity experiments point at the new variant.

## 6. Cookbook

```bash
# FNO autoregressive on SWE — PDEBench parity for batch, unroll, model, loss
python the_well/benchmark/train.py \
    experiment=fno_pdebench_swe server=$SERVER seed=$SEED

# UNet pushforward PF-20 on SWE
python the_well/benchmark/train.py \
    experiment=unet_pdebench_swe server=$SERVER seed=$SEED

# FNO autoregressive on Burgers (requires running the conversion script once first):
#   python scripts/convert_pdebench_1d_burgers.py --variant pdebench
python the_well/benchmark/train.py \
    experiment=fno_pdebench_burgers server=$SERVER seed=$SEED

# UNet pushforward PF-20 on Burgers
python the_well/benchmark/train.py \
    experiment=unet_pdebench_burgers server=$SERVER seed=$SEED

# LR search over the BPTT-specific protocol (relevant for FNO-AR-SWE specifically;
# do separately for UNet PF-20 if needed — different optimization regime)
python the_well/benchmark/train.py --multirun \
    experiment=fno_pdebench_swe server=$SERVER seed=1 \
    optimizer.lr=3e-3,1e-3,3e-4 trainer.enable_amp=False data.batch_size=4 \
    wandb_project_name=LR_search wandb_group=FNO-SMALL-BPTT
```

For the FNO-AR-SWE LR search:
- On Quadro GV100: `data.batch_size=4` (closest to PDEBench's 5 that fits in 32 GB).
- On Titan V: `data.batch_size=1` (only thing that fits in 12 GB).

After picking the LR, do 3–5 multi-seed runs with that LR for mean ± std. Same comment about which GPU.

## References

- PDEBench paper: Takamoto et al., NeurIPS 2022. https://arxiv.org/abs/2210.07182
- PDEBench repo (locally cloned): `C:\Users\simon\Documents\GitHub\PDEBench`
- Our trainer: [`the_well/benchmark/trainer/bptt_training.py`](../the_well/benchmark/trainer/bptt_training.py)
- Cross-benchmark comparability: [`benchmark-comparability.md`](benchmark-comparability.md) (older doc; this audit supersedes its §3 re: BPTT specifically)
