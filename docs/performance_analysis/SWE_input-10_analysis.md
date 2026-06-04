# W&B Analysis: SWE_input-10

## Overview

Analysis of the 2D Shallow Water Equations (SWE) with input steps = 10.

## 1. Analysis
The metrics below are reported as `mean ± std` across seeds.
Primary evaluation metric: **NRMSE**. Metrics evaluate 'best rollout NRMSE' and 'best OS NRMSE' weights.

### 1.1. Results using Best RO Weights

#### CNO
| Group | Avg Selected Epoch | TrainLoss | OS-NRMSE | RO-NRMSE | Runtime | Max VRAM |
|---|---|---|---|---|---|---|
| Baseline (PDEBench) | N/A | N/A | N/A | N/A | N/A | N/A |
| CNO-Noise | 72.8±21.5 | 0.0000±0.0000 | 0.0003±0.0001 | 0.0119±0.0041 | 49967s | 4.4 GB |
| CNO-PF | 90.4±6.6 | 0.0000±0.0000 | 0.0001±0.0000 | 0.0019±0.0004 | 93344s | 7.6 GB |
| CNO-TB_4 | 83.6±9.2 | 0.0000±0.0000 | 0.0002±0.0001 | 0.0028±0.0009 | 53853s | 4.4 GB |
| CNO-TB_4-PF | 70.5±17.4 | 0.0000±0.0000 | 0.0005±0.0002 | 0.0034±0.0009 | 88002s | 7.5 GB |
| CNO-baseline | 76.4±11.7 | 0.0000±0.0000 | 0.0002±0.0001 | 0.0119±0.0017 | 47580s | 4.3 GB |

#### FNO
| Group | Avg Selected Epoch | TrainLoss | OS-NRMSE | RO-NRMSE | Runtime | Max VRAM |
|---|---|---|---|---|---|---|
| Baseline (PDEBench) | N/A | N/A | N/A | 0.0044 | N/A | N/A |
| FNO-Noise | 80.4±38.2 | 0.0000±0.0000 | 0.0002±0.0001 | 0.0389±0.0650 | 39808s | 6.8 GB |
| FNO-PF | 23.0±20.0 | 0.0000±0.0000 | 0.0005±0.0002 | 0.0775±0.0797 | 52693s | 6.9 GB |
| FNO-TB_4 | 94.8±7.8 | 0.0000±0.0000 | 0.0001±0.0000 | 0.0009±0.0000 | 38378s | 7.5 GB |
| FNO-TB_4-PF | 98.0±1.8 | 0.0000±0.0000 | 0.0002±0.0000 | 0.0007±0.0000 | 46498s | 6.8 GB |
| FNO-baseline | 11.2±9.8 | 0.0000±0.0000 | 0.0007±0.0002 | 0.1082±0.0915 | 40999s | 6.8 GB |

#### UNet
| Group | Avg Selected Epoch | TrainLoss | OS-NRMSE | RO-NRMSE | Runtime | Max VRAM |
|---|---|---|---|---|---|---|
| Baseline (PDEBench) | N/A | N/A | N/A | 0.083 | N/A | N/A |
| UNet-Noise | 89.2±5.0 | 0.0000±0.0000 | 0.0003±0.0001 | 0.0149±0.0081 | 16764s | 2.5 GB |
| UNet-PF | 59.6±18.9 | 0.0000±0.0000 | 0.0007±0.0003 | 0.1131±0.0604 | 31106s | 4.2 GB |
| UNet-TB_4 | 84.0±9.2 | 0.0000±0.0000 | 0.0004±0.0001 | 0.0995±0.0409 | 16925s | 2.4 GB |
| UNet-TB_4-PF | 94.8±4.8 | 0.0000±0.0000 | 0.0005±0.0001 | 0.0073±0.0051 | 27593s | 4.7 GB |
| UNet-baseline | 92.4±4.1 | 0.0000±0.0000 | 0.0003±0.0001 | 0.1161±0.0520 | 19429s | 2.5 GB |

### 1.2. Results using Best OS Weights

#### CNO
| Group | Avg Selected Epoch | TrainLoss | OS-NRMSE | RO-NRMSE | Runtime |
|---|---|---|---|---|---|
| Baseline (PDEBench) | N/A | N/A | N/A | N/A | N/A |
| CNO-Noise | 82.8±14.2 | 0.0000±0.0000 | 0.0002±0.0001 | 0.0215±0.0034 | 49967s |
| CNO-PF | 90.4±6.6 | 0.0000±0.0000 | 0.0001±0.0000 | 0.0019±0.0004 | 93344s |
| CNO-TB_4 | 95.6±4.3 | 0.0000±0.0000 | 0.0002±0.0000 | 0.0079±0.0043 | 53853s |
| CNO-TB_4-PF | 90.0±5.1 | 0.0000±0.0000 | 0.0003±0.0000 | 0.0051±0.0016 | 88002s |
| CNO-baseline | 83.2±15.9 | 0.0000±0.0000 | 0.0002±0.0001 | 0.0154±0.0062 | 47580s |

#### FNO
| Group | Avg Selected Epoch | TrainLoss | OS-NRMSE | RO-NRMSE | Runtime |
|---|---|---|---|---|---|
| Baseline (PDEBench) | N/A | N/A | 0.0044 | N/A | N/A |
| FNO-Noise | 95.2±6.4 | 0.0000±0.0000 | 0.0001±0.0000 | 0.0013±0.0000 | 39808s |
| FNO-PF | 88.6±6.0 | 0.0000±0.0000 | 0.0002±0.0000 | N/A | 52693s |
| FNO-TB_4 | 95.2±3.7 | 0.0000±0.0000 | 0.0001±0.0000 | 0.0009±0.0000 | 38378s |
| FNO-TB_4-PF | 94.4±2.9 | 0.0000±0.0000 | 0.0002±0.0000 | 0.0007±0.0000 | 46498s |
| FNO-baseline | 96.4±2.6 | 0.0000±0.0000 | 0.0002±0.0000 | N/A | 40999s |

#### UNet
| Group | Avg Selected Epoch | TrainLoss | OS-NRMSE | RO-NRMSE | Runtime |
|---|---|---|---|---|---|
| Baseline (PDEBench) | N/A | N/A | 0.083 | N/A | N/A |
| UNet-Noise | 91.2±7.2 | 0.0000±0.0000 | 0.0003±0.0000 | 0.0244±0.0138 | 16764s |
| UNet-PF | 89.6±7.7 | 0.0000±0.0000 | 0.0003±0.0000 | 0.2591±0.1567 | 31106s |
| UNet-TB_4 | 85.8±7.0 | 0.0000±0.0000 | 0.0003±0.0000 | 0.2113±0.1750 | 16925s |
| UNet-TB_4-PF | 92.0±8.3 | 0.0000±0.0000 | 0.0004±0.0000 | 0.0092±0.0048 | 27593s |
| UNet-baseline | 90.2±6.9 | 0.0000±0.0000 | 0.0002±0.0000 | 0.1215±0.0485 | 19429s |


## 2. Convergence & Context Window Analysis (input-10 vs input-4)

- **Context Window Impact on Baseline**: Increasing the context window from 4 steps to 10 steps actually degraded the performance of the unregularized `FNO-baseline` model (RO-NRMSE jumped from ~0.041 in input-4 to ~0.108 in input-10). It appears that the larger context window makes the unrolled mapping harder to learn without stabilization, leading to higher variance and faster accumulation of errors.
- **Synergy with Regularization (FNO)**: When powerful regularizers are applied (`TB_4-PF`), FNO is able to effectively leverage the larger 10-step context. `FNO-TB_4-PF` on input-10 achieves an incredible RO-NRMSE of **0.0007**, which is significantly better than its input-4 counterpart (0.0012).
- **Synergy with Regularization (UNet)**: Conversely, `UNet-TB_4-PF` sees a slight degradation when moving from 4 steps to 10 steps (RO-NRMSE goes from 0.0040 up to 0.0073). Local convolutional architectures may struggle more to fuse information across a deep 10-step temporal channel dimension compared to the global mixing of Fourier layers.
- **CNO Stability**: `CNO` shows robust performance, with `CNO-PF` reaching a highly stable 0.0018 RO-NRMSE.