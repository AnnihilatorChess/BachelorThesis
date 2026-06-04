# Bachelor Thesis Outline

## Working Title
**Stabilizing Autoregressive Neural PDE Surrogates: Overcoming the Capacity-Stability Trade-off via Lightweight Regularization**

---

## 1. Introduction
- Motivation: Neural surrogates as fast alternatives to numerical PDE solvers.
  - *The Real-World Goal:* Classical solvers calculate single-step transitions with near-perfect accuracy but are computationally constrained (e.g., CFL limits). Neural surrogates promise massive leaps in time (macro-stepping), meaning **stable autoregressive rollout (RO) is the definitive metric of success**, while One-Step (OS) accuracy is primarily a proxy for capacity.
- Core challenge: Error accumulation (distribution shift) in autoregressive rollout.
- The BPTT Bottleneck: Traditional unrolled training (Backpropagation Through Time) severely restricts model capacity due to VRAM limits.
- Thesis contributions:
  1. Systematic evaluation of lightweight stabilization techniques (Pushforward, Temporal Bundling, Noise) across multiple architectures and datasets.
  2. Demonstrating that lightweight stabilization allows for training significantly larger models that outperform BPTT-constrained models.
  3. Extensive cross-benchmark analysis bridging "The Well" and "PDEBench" protocols.
  4. Investigation into checkpoint selection dynamics (OS vs. RO weights), context window sensitivities, and the implicit regularization of numerical precision (AMP vs. FP32).

## 2. Background & Related Work
### 2.1 Neural Operator Theory & Architectures
- Fourier Neural Operator (FNO): Global spectral processing.
- U-Net: Local receptive fields via convolutions.
- Convolutional Neural Operator (CNO): Continuous convolutional operator blocks.
### 2.2 Stabilization Techniques for Autoregressive Models
- Distribution shift in autoregressive generation.
- Full Autoregressive Backpropagation Through Time (BPTT).
- Pushforward trick (PF), Noise Injection, Temporal Bundling (TB).

## 3. Datasets and Experimental Setup
### 3.1 Dataset Selection
The study evaluates models across diverse physical systems, bridging two major benchmarks.

| Dataset | Source | Dim | Notes / Physics |
|---|---|---|---|
| `turbulent_radiative_layer_2D` | The Well | 2D | Chaotic fluid dynamics with radiative cooling. |
| `active_matter` | The Well | 2D | Highly chaotic, active particle suspension dynamics. |
| `pdebench_swe` | PDEBench | 2D | Shallow Water Equations (Radial dam break). |
| `BUR-DOWN` (1D Burgers) | PDEBench | 1D | Shock-forming regime ($\nu=0.001$). Downsampled 4x in time/resolution. |

### 3.2 Protocol Discrepancies & Harmonization
To ensure fair comparisons, domain-specific hyperparameters are strictly respected:
- **PDEBench Datasets:** 10 input steps, Weight Decay = $0.0001$.
- **The Well Datasets:** 4 input steps, Weight Decay = $0.01$.
- *(Special Case)*: `pdebench_swe` is evaluated at both 4 and 10 input steps to analyze context-window sensitivity.

Beyond these per-dataset knobs, our overall evaluation setup follows **The Well**, which differs from **PDEBench** in three respects:

- **Data split and model selection.** We use a roughly **80/10/10** train/validation/test split for every dataset (matching The Well), keeping a *held-out* test set that is never used for checkpoint selection. PDEBench uses a **90/10** split, selects the best checkpoint by its rollout validation score on the 10% partition, and then reports the final score on that *same* partition. Because the model is both selected and evaluated on the same data, PDEBench's reported numbers are slightly optimistic, whereas our held-out test split removes this selection bias.
- **Training budget.** Both source papers train for up to **500 epochs** per (model, dataset); The Well additionally caps each run at a **12-hour wall-clock budget on a single NVIDIA H100** (whichever limit is hit first). We train for **100 epochs** per run as a deliberate compute-budget deviation, so our models are mildly undertrained relative to the source baselines.
- **Learning-rate search.** The Well performs a per-(model, dataset) learning-rate sweep, which we mirror with our own coarse LR search. For PDEBench we could not find evidence of a learning-rate search, so we treat their published learning rates as fixed.

## 4. Methodology
### 4.1 Architectures & The Capacity Trade-off
- **Model Sizes:** 
  - *PDEBench Sizes (Tiny):* Scaled down to fit full BPTT into VRAM. Trained for **FNO and UNet only** (CNO has no PDEBench baseline to reproduce) and **only on SWE and BUR**, the two datasets for which we hold both a small and a large scale.
  - *The Well Sizes (Normal/Large):* Highly expressive models utilizing lightweight stabilization. Trained for all three architectures on all four datasets.
- **Evaluated Models:** UNet, FNO, CNO (large scale); UNet, FNO (small/BPTT scale).

### 4.2 The Ablation Stack
1. **Paper Baselines:** Results reported directly in the source literature.
2. **Our Setup Baseline:** Unregularized 1-step autoregressive training (max 100 epochs, matching source configs).
3. **BPTT (PDEBench only):** Full unrolled trajectory training (restricted to Tiny FNO/UNet models on SWE and BUR; CNO is excluded as it has no PDEBench baseline).
4. **Stabilization Pipeline:**
   - Noise Injection
   - Push Forward (PF)
   - Temporal Bundling (TB)
   - Temporal Bundling + Push Forward (TB + PF)

### 4.3 Hardware and Precision (AMP vs. FP32)
- To maximize training efficiency, Automatic Mixed Precision (AMP) was applied where mathematically stable.
- **Z-Score Normalization Vulnerability:** AMP instability (FP16 overflow, particularly in FNO's FFT) correlates directly with Z-score scaling. Datasets with narrow standard deviations exponentially amplify network errors during denormalization.
  - *Active Matter & TRL (High Risk):* Extreme amplification due to tiny physical variances. Forces FP32 for spectral models to avoid NaN explosions.
  - *SWE (Moderate Risk):* Occasional shock-induced ringing (Gibbs phenomenon) can trigger FP16 limits, but mostly stable.
  - *Burgers 1D (Zero Risk):* Standard deviation is 1.0, meaning zero error amplification. Fully stable in AMP.

### 4.4 Hyperparameter Tuning (Learning Rate Search)
- Coarse grid search ($0.003, 0.001, 0.0003$) conducted per dataset and architecture on the standard baseline configuration.
- Best learning rate selected based on test set One-Step (OS) performance.
- Separate LR search conducted for BPTT due to its fundamentally different training dynamics and loss landscape.

## 5. Results
### 5.1 The Capacity-Stability Trade-off (PDEBench vs. The Well sizes)
- **Scope:** confined to the **PDEBench datasets (SWE, BUR)** and to **FNO and UNet** — the only configurations for which both a small (BPTT-capacity) and a large weight set exist and a published BPTT baseline can be reproduced. CNO and the chaotic Well datasets (TRL, AM) are trained only at the large scale.
- Comparing BPTT on "Tiny" models against TB+PF on "Normal" models.
- Demonstrating that avoiding BPTT's VRAM bottleneck allows for larger, more accurate models that achieve comparable or superior rollout stability.
- **Limitation / future work:** because capacity is varied only on the simpler PDEs, the capacity-stability result is not yet established on chaotic dynamics, where intrinsic (Lyapunov) error growth may blunt the benefit of added capacity. A small-scale FNO/UNet run on TRL — planned once the AM runs complete — would test whether the trade-off extends to the chaotic regime, without requiring an (intractable) full-BPTT run on The Well.

### 5.2 Efficacy of Training Techniques Across Datasets
- Evaluating the ablation stack (Baseline, Noise, PF, TB, TB+PF) across UNet, FNO, and CNO.
- Comparison against the official PDEBench and The Well paper baselines.
- Highlighting universal successes (e.g., TB+PF synergy) vs. dataset-specific requirements (e.g., handling chaotic `active_matter`).

### 5.3 Context Window Dynamics (SWE input-4 vs input-10)
- How increasing the receptive field in time affects stability.
- Interaction between architecture (local UNet vs. global FNO) and context size.
- Observation: Larger contexts degrade unregularized baselines but massively boost strongly regularized spectral models (FNO-TB_4-PF).

### 5.4 Epoch Selection and "The Validation Deception"
- Comparing Best One-Step (OS) weights vs. Best Rollout (RO) weights.
- Analyzing the average selected epoch across techniques.
- Investigating the variance in spectral models (FNO) where the "best" validation rollout checkpoint often generalizes poorly to the test set compared to OS weights.

### 5.5 Implicit Regularization via Numerical Noise (AMP)
- *Note: This phenomenon was an opportunistic observation made during learning rate sweeps on the Active Matter dataset, rather than the result of a formal full-stack ablation.*
- **The Sharp vs. Damped Operator:** A counter-intuitive observation where models suffering from AMP numerical jitter occasionally perform *better* on long rollouts despite terrible 1-step errors.
- FP32 models learn a "sharp" operator: Perfect 1-step accuracy on training data, but catastrophic resonance when fed unseen rollout errors.
- AMP models learn a "damped" operator: FP16 limits act as a low-pass filter, resulting in blurry but numerically stable rollouts. Connects conceptually to explicit Noise Injection.

### 5.6 Runtime and Computational Efficiency
- Wall-clock time comparisons.
- The computational overhead of BPTT vs. PF vs. TB.
- Hardware utilization and dataloading throughput.

## 6. Discussion
- Synthesizing the "Pareto Frontier" of PDE surrogates: Balancing performance, VRAM limits, and training time.
- Architectural predispositions to stabilization techniques (global vs. local operators).
- The necessity of Rollout (RO) optimization over One-Step (OS) for real-world surrogate utility.
- Recommendations for practitioners when selecting context windows, model capacities, precision (AMP), and regularizers.

## 7. Conclusion
- Summary of core findings.
- Future directions (e.g., super-resolution capabilities of FNO on unmodified Burgers datasets).

---