# Related Work — Annotated Bibliography & Training-Practice Synthesis

Reference material for **§1 Introduction** and **§2 Background & Related Work** of the thesis
*"Stabilizing Autoregressive Neural PDE Surrogates: Overcoming the Capacity–Stability
Trade-off via Lightweight Regularization."*

BibTeX for every entry below lives in [`references.bib`](references.bib) (keys given in
parentheses). Citation details were verified against primary sources (arXiv abstracts,
NeurIPS/ICML/ICLR/JMLR/Nature pages). Confidence on the 9 core entries is high
(3-vote adversarial verification); the remaining entries were cross-checked individually.

> **Provenance:** compiled 2026-06-04 via the `deep-research` workflow (105 agents,
> 23 primary sources, 25 claims verified, 2 refuted) plus targeted manual verification of
> gap topics (PDEBench, PINNs, weather models, transformers/GNNs, noise-injection lineage,
> foundation/diffusion models).

---

## 1. For §1 Introduction (motivation & framing)

- **Li et al. (2021), FNO** (`li2021fourier`) — Headline "surrogates are orders of magnitude
  faster than classical solvers" motivation; concrete number FNO 0.005 s vs pseudo-spectral
  2.2 s on 256² Navier–Stokes (~440×).
- **Kovachki et al. (2023), Neural Operator** (`kovachki2023neural`) — Definitional framing:
  operators learn an *entire family* of PDEs with universal-approximation + discretization-
  invariance guarantees.
- **Brandstetter, Worrall, Welling (2022), Message Passing Neural PDE Solvers**
  (`brandstetter2022message`) — **The single most important citation for the thesis.** States
  the core problem verbatim: one-step training "has a key failure mode, instability: small
  errors … accumulate over rollouts … which can be interpreted as overfitting to the one-step
  training distribution." Introduces both the **pushforward trick** and **temporal bundling**.
- **Lippe et al. (2023), PDE-Refiner** (`lippe2023pderefiner`) — Sharpens the motivation that
  *stable rollout is the real metric, one-step accuracy a proxy*. Shows several classical
  stabilizers fail to beat a plain MSE baseline.
- **McGreivy & Hakim (2024)** (`mcgreivy2024weak`) — Supports the "Validation Deception" framing
  (§5.4) and the held-out-test methodology (§3.2): weak baselines and reporting biases inflate
  ML-for-PDE claims.

## 2.1 Neural Operator Theory & Architectures

- **Lu et al. (2021), DeepONet** (`lu2021learning`) — The other founding operator-learning
  paradigm (branch/trunk nets), grounded in the operator universal-approximation theorem.
  Counterpoint to FNO.
- **Li et al. (2021), FNO** (`li2021fourier`) — The spectral/global architecture: parameterizes
  the integral kernel directly in Fourier space (FFT → learned weights on truncated modes →
  iFFT), O(N log N).
- **Ronneberger, Fischer, Brox (2015), U-Net** (`ronneberger2015unet`) — Original U-Net; the
  local/convolutional architecture. PDE benchmarks use modernized variants (ConvNext-block
  "CNextU-net").
- **Raonic et al. (2023), CNO** (`raonic2023convolutional`) — A CNN/U-Net-style operator that
  *preserves continuous (operator) structure when discretized* (alias-free up/downsampling,
  bandlimited activations), with a proven universality theorem — same theoretical footing as
  FNO/DeepONet.
- **Cao (2021), Choose a Transformer** (`cao2021choose`) — Foundational attention-based operator
  learning (Fourier vs Galerkin attention).
- **Li, Meidani, Barati Farimani (2023), OFormer** (`li2023transformer`) — Representative
  attention-based neural operator on irregular grids.
- **Pfaff et al. (2021), MeshGraphNets** (`pfaff2021learning`) — Canonical GNN/mesh PDE surrogate;
  the message-passing lineage behind Brandstetter et al.

## 2.2 Stabilization Techniques for Autoregressive Models

- **Brandstetter et al. (2022)** (`brandstetter2022message`) — **Pushforward trick** (train on
  the model's own pushed-forward/perturbed input via a 2-step unroll, backprop only the last
  step, fixing the test-time distribution shift A#pₖ ≠ pₖ₊₁) and **Temporal bundling** (predict K
  steps jointly → K× fewer solver calls → fewer distribution-shift events).
- **Sanchez-Gonzalez et al. (2020), GNS** (`sanchezgonzalez2020learning`) — **Origin of training-
  noise injection**: long-term performance is driven by "mitigating the accumulation of error by
  corrupting the training data with noise." Canonical citation for the Noise ablation.
- **Stachenfeld et al. (2022)** (`stachenfeld2022learned`) — Reinforces noise injection
  (+ temporal downsampling) as the practical stabilizer for turbulent rollouts; relevant to the
  chaotic datasets (active_matter, TRL).
- **Bishop (1995), Training with Noise = Tikhonov Regularization** (`bishop1995training`) — The
  theoretical foundation for *why* noise injection stabilizes: to first order, training with input
  noise is equivalent to a Tikhonov penalty on the squared Jacobian norm of outputs w.r.t. inputs,
  biasing the model toward a smoother, lower-gain (contractive) operator. **Also the bridge for
  §5.5**: underpins the argument that fp16/AMP round-off acts as an implicit noise regularizer.
- **Li et al. (2022), Markov Neural Operator (MNO)** (`li2022learning`) — Sobolev-norm /
  dissipativity regularization for long-term stability. PDE-Refiner uses MNO as a baseline that
  *fails* — useful for the "lightweight regularization vs. heavy machinery" argument.
- **Um et al. (2020), Solver-in-the-Loop** (`um2020solver`) — Canonical reference for **unrolled /
  differentiable-solver (BPTT-style) training** — the heavyweight alternative the thesis argues
  against on VRAM grounds.
- **Lippe et al. (2023), PDE-Refiner** (`lippe2023pderefiner`) — The crucial *negative result*:
  pushforward, Sobolev loss, MNO, error-correction, ensembling, 4× parameters, and extra input
  history all failed to beat one-step MSE in their KS setting — they neglect low-amplitude high-
  frequency modes. Pushforward specifically "does not improve performance … no gradients are
  backpropagated through the predicted input." Strongest "why this is still open" evidence.

## 3. Benchmarks

- **Takamoto et al. (2022), PDEBench** (`takamoto2022pdebench`) — The PDEBench protocol source
  (SWE, 1D Burgers).
- **Ohana et al. (2024), The Well** (`ohana2024well`) — The Well protocol source (TRL 2D,
  active_matter). Confirms baselines are FNO, **TFNO**, classic U-Net, **CNextU-net**, trained on
  the **one-step forward problem from a 4-step history** (not BPTT) — pins down one-step MSE as
  the practitioner default the thesis improves on.
- **Gupta & Brandstetter (2023), PDEArena** (`gupta2023towards`) — Systematically compares
  FNO/ResNet/U-Net; shows a *single* surrogate can generalize across PDE parameters and
  time-scales (framed as "promising," not solved).
- **Koehler et al. (2024), APEBench** (`koehler2024apebench`) — **Highly relevant**: a benchmark
  built specifically around *autoregressive rollout* (46 PDEs, differentiable JAX spectral
  solver). Directly on-topic.

## 4. Broader context (§1/§2 framing & §5.5)

- **Raissi, Perdikaris, Karniadakis (2019), PINNs** (`raissi2019physics`) — The *physics-loss*
  alternative to data-driven operator learning. Cite to position this work (learns from
  data/operators, not PDE residuals).
- **Pathak et al. (2022), FourCastNet** (`pathak2022fourcastnet`) — FNO-family at planetary scale.
- **Bi et al. (2023), Pangu-Weather** (`bi2023accurate`) — High-impact autoregressive surrogate
  that explicitly engineers against rollout error (hierarchical temporal aggregation).
- **Lam et al. (2023), GraphCast** (`lam2023learning`) — GNN autoregressive surrogate trained with
  *multi-step rollout fine-tuning* — concrete evidence that unrolled (BPTT-style) training is used
  in practice but is expensive.
- **McCabe et al. (2023), Multiple Physics Pretraining** (`mccabe2023multiple`) — The "foundation
  model for PDEs" trend: pretrain one model across many physics.
- **Herde et al. (2024), Poseidon** (`herde2024poseidon`) — Foundation model on a multiscale
  operator transformer; exploits the PDE semigroup property for data scaling.
- **Hao et al. (2024), DPOT** (`hao2024dpot`) — Foundation model combining autoregressive
  *denoising* pretraining with Fourier-attention transformer (0.5B params); ties "noise/diffusion
  as stabilizer" and "scaling" together.
- **Kohl, Chen, Thuerey (2023), ACDM** (`kohl2023benchmarking`) — The **diffusion-based surrogate**
  trend for stable long rollouts.
- **Holzschuh et al. (2025), PDE-Transformer** (`holzschuh2025pde`) — Most recent (2025) scalable
  transformer surrogate; "current state of the art" anchor.
- **Micikevicius et al. (2018), Mixed Precision Training** (`micikevicius2018mixed`) — Standard
  citation for AMP/FP16 (§4.3, §5.5). NOTE: in this codebase AMP = **float16**, not bfloat16 —
  trainer default `amp_type="float16"` with an active `GradScaler`
  (`the_well/benchmark/trainer/training.py:77,160-162`; AMP itself is opt-in via `enable_amp`).
  The FNO wrapper does **not** blanket-upcast the FFT: it monkey-patches `torch.fft` to attempt
  float16 first and **fall back to fp32 only for non-power-of-two shapes** that cuFFT cannot
  transform in half precision — so for power-of-two resolutions the FFT runs in float16 and fp16
  round-off *does* enter the spectral path (`the_well/benchmark/models/fno/__init__.py`). No
  PDE-specific "AMP-as-regularizer" paper exists; the §5.5 mechanism is **novel** to this thesis.
- **Bishop (1995)** (`bishop1995training`) — Theoretical bridge for §5.5: training-with-noise ≈
  Tikhonov regularization lets you argue fp16 round-off ≈ implicit noise injection → a
  smoother/contractive operator, rather than presenting §5.5 as a bare unsupported observation.

---

## Synthesis — How are neural PDE surrogates trained today?

**Architectures.** The field has consolidated around a small set: **FNO** (global/spectral) and
the **U-Net family** (local/convolutional, increasingly ConvNext blocks) are the two workhorses;
**CNO** is the operator-theoretic refinement of the U-Net line; **DeepONet** is the other founding
paradigm but appears less in time-dependent rollout benchmarks. **Transformers/ViT** (Cao, OFormer,
Poseidon/scOT, PDE-Transformer) and **GNNs/message-passing** (MeshGraphNets, MP-PDE) cover
irregular geometries and now the scaling frontier. The thesis's FNO / U-Net / CNO trio is the
canonical comparison set used by The Well and PDEArena.

**Objective: one-step MSE dominates.** The overwhelmingly common recipe is one-step Mean-Squared-
Error — predict the next snapshot from a short history (4 steps in The Well, 10 in PDEBench), then
roll out autoregressively. It is accurate short-term but unstable over long rollouts: small one-step
errors accumulate (distribution shift), and because MSE is dominated by high-amplitude low-frequency
modes, neglected low-amplitude high-frequency errors propagate into low frequencies over long
horizons. **This is exactly the gap the thesis targets.**

**BPTT vs. OS-loss vs. stabilization — all used, on a cost spectrum.**
- *One-step (OS) loss* — the default baseline (cheapest, least stable).
- *Lightweight stabilizers* — **pushforward**, **temporal bundling**, **noise injection** — the
  standard cheap remedies (Brandstetter 2022; Sanchez-Gonzalez 2020; Stachenfeld 2022).
- *BPTT / unrolled training* — Solver-in-the-Loop, GraphCast's multi-step fine-tuning: most stable
  signal but VRAM-bound, capping model capacity (the thesis's central capacity–stability trade-off).
- *Caveat (PDE-Refiner):* none is a guaranteed win — PF, Sobolev loss, MNO, ensembling, and 4×
  parameters all failed to beat plain MSE in their setting. This is why a *systematic, multi-dataset*
  ablation is a genuine contribution, not a re-confirmation.

**OS vs. RO selection.** No universal convention — itself a finding. PDEBench selects and reports on
the *same* validation partition (slightly optimistic); The Well uses a held-out test set.
Increasingly the field treats *stable rollout as the real metric and one-step error as a capacity
proxy* (the thesis's framing, and PDE-Refiner's). The §5.4 OS-weights vs. RO-weights checkpoint
study addresses a real, under-examined methodological gap.

**Current trends (2023→2026).**
1. **Systematic benchmark suites** — PDEBench (2022) → PDEArena (2023) → The Well + APEBench (2024).
2. **Foundation models / large-scale pretraining** — MPP (2023), Poseidon, DPOT (2024): one model
   across many physics, transformer backbones.
3. **Diffusion / denoising surrogates** — PDE-Refiner (2023), ACDM (2023), DPOT: iterative
   refinement to recover the neglected high-frequency spectrum and stabilize long rollouts.
4. **Scaling vs. unrolling** — a live tension the thesis sits inside; the frontier moves toward
   scalable transformers (PDE-Transformer, 2025) and pretraining, while this thesis argues the
   complementary case that *lightweight stabilization lets you scale the model* instead of paying
   BPTT's VRAM tax.

---

## Things to flag / avoid in the thesis

- **Refuted superlatives (do NOT use):** the claim that FNO is the *"first ML method to successfully
  model turbulent flows with zero-shot super-resolution"* failed adversarial verification (0–3).
  FNO *does* demonstrate zero-shot super-resolution in its own experiments — but avoid the
  "first / turbulence" superlative.
- **Discretization invariance** is an *architectural/theoretical* property; practical cross-resolution
  *accuracy* is contested (e.g. "Discretization Mismatch Errors," ICLR 2025; arXiv:2510.06646).
  Phrase as a property, not a guarantee.
- **Inference speedups** ("orders of magnitude") are real but *problem- and baseline-dependent*
  (McGreivy & Hakim 2024). Against optimized GPU solvers the gap narrows. Hedge accordingly.
- **AMP-as-regularizer (§5.5)** has no AMP-specific prior citation, but **Bishop (1995)** supplies
  the noise≈Tikhonov half of the argument. The *AMP-causes-this* link remains an opportunistic
  observation — keep hedged language ("we observed," "suggests"); do not present as a controlled result.
- **"Validation deception" (§5.4)** has **no direct prior citation** and was likewise observed
  *opportunistically*, not via a formal ablation. Use the same hedged language; McGreivy & Hakim
  (2024) supports only the broader "reporting bias" framing.
