# Bachelor Thesis Outline

## Working Title
**Stabilizing Autoregressive Neural PDE Surrogates: A Multi-Dataset Study of Architecture and Regularization**

---

## 1. Introduction
- Motivation: neural surrogates as fast alternatives to numerical PDE solvers
- Core challenge: error accumulation (distribution shift) in autoregressive rollout
- Gap addressed: prior work limited to a single dataset — unclear how findings generalize
- Thesis contributions:
  1. Systematic evaluation across multiple PDE datasets (The Well & PDEBench)
  2. Extended architecture comparison
  3. Comparison of stabilization techniques (pushforward trick, noise injection, temporal bundling)
  4. Rigorous evaluation with multiple seeds and standard errors

## 2. Background & Related Work
### 2.1 Neural Operator Theory
- Operator learning framework (mapping function spaces)
- Fourier Neural Operator (FNO) and spectral convolutions
- Relationship to classical numerical methods

### 2.2 Surrogate Architectures for PDEs
- FNO: global spectral processing
- U-Net variants: encoder-decoder with skip connections (Classic, ConvNeXt)
- Vision Transformer (ViT) / CNO: local and global processing variants
- Other relevant architectures — brief survey

### 2.3 Stabilization Techniques for Autoregressive Models
- Distribution shift in autoregressive generation
- Pushforward trick: unrolled training with curriculum scheduling
- Noise injection / data augmentation approaches
- Temporal bundling (multi-step prediction)
- Other relevant techniques from the literature

## 3. Datasets
### 3.1 Dataset Selection
The study evaluates models across diverse physical systems, using a mix of datasets from The Well benchmark and PDEBench.

| Dataset | source | Dim | Resolution | Trajectories (Used) | Full Size |
|---|---|---|---|---|---|
| turbulent_radiative_layer_2D | The Well | 2D | 384x128 | 90 (All) | ~7 GB |
| gray_scott_reaction_diffusion| The Well | 2D | 128x128 | Subset (TBD) | ~153 GB |
| rayleigh_benard | The Well | 2D | 512x128 | Subset (TBD) | ~358 GB |
| Shallow Water Equations | PDEBench | 2D | 128x128 | 1,000 | ~1 GB |
| Burgers' Equation | PDEBench | 1D | 1024 | 10,000 (ν=0.001 only) | ~7.7 GB |

### 3.2 Sampling Strategy
Instead of spatial or temporal downsampling (which can lose critical physical features or spectral content), this thesis employs a **trajectory-limited strategy** for large datasets.
- For massive datasets like `gray_scott` and `rayleigh_benard`, only a fraction of the available trajectories are used to stay within computational and storage constraints.
- Resolution and timesteps are kept at their native values to preserve the physical fidelity of the benchmarks.

### 3.3 Dataset Preprocessing
- Normalization strategies (ZScore, RMS)
- Train/validation/test splits
- Conversion of PDEBench datasets to The Well's HDF5 format for unified processing

## 4. Methods
### 4.1 Architectures
#### 4.1.1 Fourier Neural Operator (FNO)
- Spectral convolution blocks, global receptive field
- Configuration and hyperparameters

#### 4.1.2 U-Net Classic
- Encoder-decoder with skip connections
- Local receptive field via 3x3 convolutions

#### 4.1.3 U-Net ConvNeXt
- Modernized U-Net with depthwise convolutions and large kernels
- Pseudo-global receptive field

#### 4.1.4 Convolutional Neural Operator (CNO)
- Convolutional operator blocks with up-/down-sampling that respect aliasing
- Adaptation for PDE surrogate task

### 4.2 Stabilization Techniques
#### 4.2.1 Pushforward Trick
- Unrolled autoregressive training with curriculum
- No backprop through intermediate steps (zero-stability enforcement)
- Curriculum schedule: warmup + probability distribution over unroll depths

#### 4.2.2 Noise Injection
- Adding noise to inputs during training as a form of data augmentation
- Relationship to pushforward (both address distribution shift)

#### 4.2.3 Temporal Bundling
- Predicting multiple future timesteps simultaneously
- Reduces the number of autoregressive steps needed

## 5. Experimental Setup
### 5.1 Evaluation Metrics
- One-step: VRMSE, RMSE, NMSE
- Rollout: time-windowed VRMSE, spectral NMSE per frequency bin
- Long-term: PearsonR, binned spectral MSE
- Statistical reporting: mean +/- standard error across seeds

### 5.2 Ablation Design
Overview of the ablation study structure:

| Ablation | Variables | Fixed |
|---|---|---|
| Architecture comparison | FNO, U-Net Classic, U-Net ConvNeXt, CNO | No pushforward |
| Stabilization comparison | None, Pushforward, Noise Injection, Temporal Bundling | Best architecture |
| Cross-dataset evaluation | All selected datasets | Best configuration from above |

Each configuration is run with multiple seeds to report mean +/- std error.

### 5.3 Implementation Details
- Framework: PyTorch, Hydra, W&B
- Hardware: NVIDIA RTX 3070 Ti (8GB VRAM)

## 6. Results
### 6.1 Baseline Architecture Comparison
- One-step and rollout performance across datasets
- Per-field and per-frequency-bin analysis

### 6.2 Stabilization Techniques
- Pushforward trick vs. noise injection vs. temporal bundling
- Comparison of stabilization techniques across architectures

### 6.3 Cross-Dataset Analysis
- Which findings are dataset-specific vs. universal?
- Relationship between dataset properties and method effectiveness

### 6.4 Statistical Analysis
- Variance across seeds
- Significance of observed differences

## 7. Discussion
- Generalizability of findings to new datasets
- Architecture-dependent effectiveness of stabilization
- Computational cost vs. performance tradeoffs
- Limitations: hardware constraints, dataset selection

## 8. Conclusion
- Summary of key findings across datasets
- Practical recommendations for practitioners

## References

---

## Open Questions / TODOs
- [ ] Read suggested papers: arxiv 2411.00180, 2210.07182, 2302.01178
- [ ] Determine number of seeds per ablation (3? 5?)
- [ ] Finalize the fraction of trajectories for large datasets
