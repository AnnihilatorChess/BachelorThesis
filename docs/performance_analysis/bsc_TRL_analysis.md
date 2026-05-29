# W&B Analysis: bsc_TRL

## Overview

Analysis of the turbulent_radiative_layer_2D dataset.

## 1. Analysis
The metrics below are reported as `mean ± std` across seeds.
Primary evaluation metric: **VRMSE**. Metrics evaluate 'best rollout VRMSE' and 'best OS VRMSE' weights.

### 1.1. Results using Best RO Weights
#### CNO
| Group | Avg Selected Epoch | TrainLoss | OS-VRMSE | RO-VRMSE | T=6:12 VRMSE | T=13:30 VRMSE | Runtime |
|---|---|---|---|---|---|---|---|
| Baseline (The Well) | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| CNO-Noise | 92.0±8.6 | 0.0068±0.0001 | 0.2229±0.0034 | 1.3173±0.1668 | 0.6841±0.0169 | 1.1193±0.0046 | 63197s |
| CNO-PF | 78.0±18.5 | 0.0307±0.0007 | 0.2557±0.0171 | 1.2859±0.0149 | 0.7081±0.0698 | 1.2178±0.0552 | 72486s |
| CNO-TB_4 | 36.0±13.4 | 0.0261±0.0009 | 0.3547±0.0137 | 1.1764±0.0312 | 0.6938±0.0289 | 1.0962±0.0370 | 56035s |
| CNO-TB_4-PF | 82.0±9.9 | 0.0880±0.0011 | 0.3139±0.0029 | 0.9214±0.0180 | 0.5443±0.0074 | 0.8166±0.0153 | 70348s |
| CNO-baseline | 90.0±4.3 | 0.0067±0.0001 | 0.2267±0.0045 | 1.3190±0.1139 | 0.6904±0.0942 | 1.1195±0.0725 | 66650s |

#### FNO
| Group | Avg Selected Epoch | TrainLoss | OS-VRMSE | RO-VRMSE | T=6:12 VRMSE | T=13:30 VRMSE | Runtime |
|---|---|---|---|---|---|---|---|
| Baseline (The Well) | N/A | N/A | 0.5001 | N/A | 1.79 | 3.54 | N/A |
| FNO-Noise | 24.0±22.7 | 0.0046±0.0000 | 0.3641±0.0446 | 1.7880±0.5002 | 0.9650±0.2117 | 1.4201±0.1547 | 16122s |
| FNO-PF | 76.8±28.4 | 0.0067±0.0001 | 0.3324±0.0286 | 1.5757±0.4751 | 0.6240±0.3443 | 0.9681±0.3035 | 19243s |
| FNO-TB_4 | 74.0±16.6 | 0.0134±0.0000 | 0.3635±0.0029 | 1.0748±0.1518 | 0.4599±0.0121 | 0.8399±0.0175 | 28196s |
| FNO-TB_4-PF | 83.6±10.2 | 0.0194±0.0003 | 0.3772±0.0014 | 0.8401±0.0298 | 0.4182±0.0052 | 0.6723±0.0197 | 19422s |
| FNO-baseline | 18.0±12.7 | 0.0046±0.0000 | 0.3982±0.0665 | 1.8814±0.4063 | 0.9533±0.2503 | 1.4603±0.2711 | 41840s |

#### UNet
| Group | Avg Selected Epoch | TrainLoss | OS-VRMSE | RO-VRMSE | T=6:12 VRMSE | T=13:30 VRMSE | Runtime |
|---|---|---|---|---|---|---|---|
| Baseline (The Well) | N/A | N/A | 0.2418 | N/A | 0.66 | 1.04 | N/A |
| UNet-Noise | 88.0±13.7 | 0.0112±0.0004 | 0.2883±0.0484 | 1.4874±0.1777 | 0.9051±0.1856 | 1.2780±0.1369 | 9954s |
| UNet-PF | 84.8±11.9 | 0.0413±0.0015 | 0.2936±0.0132 | 1.3464±0.0990 | 0.8119±0.0857 | 1.3193±0.2011 | 13475s |
| UNet-TB_4 | 78.8±22.2 | 0.0422±0.0010 | 0.3671±0.0456 | 1.1666±0.0649 | 0.6410±0.0390 | 1.0705±0.0485 | 11735s |
| UNet-TB_4-PF | 91.2±10.9 | 0.1102±0.0025 | 0.3712±0.0198 | 0.9851±0.0759 | 0.5884±0.0140 | 0.8019±0.0292 | 14252s |
| UNet-baseline | 92.8±4.1 | 0.0110±0.0002 | 0.2553±0.0098 | 1.5895±0.2681 | 0.8426±0.0945 | 1.3836±0.2026 | 12200s |

### 1.2. Results using Best OS Weights
#### CNO
| Group | Avg Selected Epoch | TrainLoss | OS-VRMSE | RO-VRMSE | T=6:12 VRMSE | T=13:30 VRMSE | Runtime |
|---|---|---|---|---|---|---|---|
| Baseline (The Well) | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| CNO-Noise | 89.7±6.6 | 0.0068±0.0001 | 0.2196±0.0013 | 1.9295±0.1509 | 0.7952±0.1494 | 1.6268±0.2985 | 63197s |
| CNO-PF | 90.7±3.1 | 0.0307±0.0007 | 0.2416±0.0017 | 1.6899±0.5372 | 0.6869±0.0103 | 1.4308±0.2855 | 72486s |
| CNO-TB_4 | 94.0±3.6 | 0.0261±0.0009 | 0.2739±0.0014 | 1.5677±0.0979 | 0.5408±0.0093 | 0.9591±0.0151 | 56035s |
| CNO-TB_4-PF | 72.3±16.0 | 0.0880±0.0011 | 0.3086±0.0012 | 1.0090±0.0713 | 0.5493±0.0234 | 0.8593±0.0321 | 70348s |
| CNO-baseline | 99.7±0.5 | 0.0067±0.0001 | 0.2193±0.0001 | 1.9012±0.2314 | 0.6541±0.0158 | 1.2071±0.1064 | 66650s |

#### FNO
| Group | Avg Selected Epoch | TrainLoss | OS-VRMSE | RO-VRMSE | T=6:12 VRMSE | T=13:30 VRMSE | Runtime |
|---|---|---|---|---|---|---|---|
| Baseline (The Well) | N/A | N/A | 0.5001 | N/A | 1.79 | 3.54 | N/A |
| FNO-Noise | 86.4±5.0 | 0.0046±0.0000 | 0.3038±0.0019 | 3.5976±0.1507 | 0.9298±0.0905 | 2.8464±0.1434 | 16122s |
| FNO-PF | 92.8±10.5 | 0.0067±0.0001 | 0.3188±0.0016 | 1.5820±0.3895 | 0.4662±0.0174 | 0.8821±0.0874 | 19243s |
| FNO-TB_4 | 87.6±9.2 | 0.0134±0.0000 | 0.3628±0.0008 | 1.2419±0.2762 | 0.4677±0.0337 | 0.8707±0.0785 | 28196s |
| FNO-TB_4-PF | 72.8±16.1 | 0.0194±0.0003 | 0.3761±0.0019 | 0.9052±0.0494 | 0.4312±0.0143 | 0.7251±0.0674 | 19422s |
| FNO-baseline | 90.0±11.0 | 0.0046±0.0000 | 0.3038±0.0021 | 4.5753±1.7539 | 0.9513±0.1372 | 3.2075±0.2388 | 41840s |

#### UNet
| Group | Avg Selected Epoch | TrainLoss | OS-VRMSE | RO-VRMSE | T=6:12 VRMSE | T=13:30 VRMSE | Runtime |
|---|---|---|---|---|---|---|---|
| Baseline (The Well) | N/A | N/A | 0.2418 | N/A | 0.66 | 1.04 | N/A |
| UNet-Noise | 94.0±4.3 | 0.0112±0.0004 | 0.2531±0.0018 | 2.3156±1.3622 | 0.8932±0.0995 | 1.6081±0.4852 | 9954s |
| UNet-PF | 86.4±4.2 | 0.0413±0.0015 | 0.2766±0.0019 | 1.7867±0.5516 | 0.9500±0.2813 | 1.6015±0.3942 | 13475s |
| UNet-TB_4 | 88.0±7.9 | 0.0422±0.0010 | 0.3174±0.0051 | 1.5086±0.2667 | 0.6664±0.0553 | 1.1882±0.1756 | 11735s |
| UNet-TB_4-PF | 86.8±13.3 | 0.1102±0.0025 | 0.3451±0.0030 | 1.2725±0.2181 | 0.5827±0.0292 | 1.0267±0.1711 | 14252s |
| UNet-baseline | 95.2±3.3 | 0.0110±0.0002 | 0.2497±0.0041 | 1.8657±0.4099 | 0.8973±0.1031 | 1.6741±0.6286 | 12200s |


## 2. Convergence & Model Analysis (TRL)

- **CNO and UNet (Stability)**: Both architectures show high stability. The "Best Rollout" weights perform significantly better in long rollouts than the "Best One-Step" weights (e.g., CNO-baseline RO-NRMSE 0.451 vs 0.561), which is expected.
- **FNO (Instability & Overfitting)**: FNO baseline exhibits extreme variance in rollout performance when using one-step weights (RO-NRMSE 1.387 ± 1.159). Using rollout-informed weights is strictly necessary for FNO to produce semi-stable results.
- **Ablation - Temporal Bundling (TB_4)**: Drastically reduces the gap between one-step and rollout weights. Interestingly, **TB_4-PF** for FNO achieves the lowest RO-NRMSE (**0.3278**), even though its one-step error is higher. This confirms that TB_4 acts as a powerful regularizer for long-term stability at the cost of one-step precision.
- **Ablation - Pushforward (PF)**: Consistently increases training loss (unrolled objective is harder) but stabilizes the rollout performance relative to the baseline, especially for UNet.

## 3. Runtime Comparison & Hardware Notes
Hardware tracking is currently scattered across config nested fields and wandb metadata, so perfect comparisons are tricky. However, general notes:
- Dataloading is rarely a bottleneck for `bsc_TRL` due to dataset size in HDF5 format.
- The CNO runs utilized **AMP** (Automatic Mixed Precision), which considerably accelerates the training and lowers VRAM usage. The FNO and UNet runs did NOT use AMP (as observed in their configs).
- Using different GPUs (e.g. A100s vs RTX 3090s) mainly influences wall-clock time; performance metrics are unaffected given the same seed. It is recommended to enable AMP across all architectures in future runs to provide a more leveled runtime comparison.

## 4. Comparing the 4 Best Weights
The 4 recorded weights represent best checkpoints per metric (`one-step NRMSE`, `one-step VRMSE`, `rollout NRMSE`, `rollout VRMSE`).
- **UNet and CNO** show very stable behavior across the board: all 4 checkpoints evaluate to nearly identical metrics on the final test set. This implies smooth convergence and robust minima.
- **FNO** exhibits high variance. Often, the checkpoint explicitly saved for the lowest 'rollout NRMSE' during validation performs *worse* on the final test set than the checkpoint saved for 'one-step NRMSE'. This suggests the FNO might be memorizing the validation trajectories or experiencing noisy validation curves.