# W&B Analysis: BIG_SWE_input-4

## Overview

Analysis of the BIG_SWE_input-4 project with larger model sizes and input steps = 4.

## 1. Analysis
The metrics below are reported as `mean ± std` across seeds.
Primary evaluation metric: **NRMSE**. Metrics evaluate 'best rollout NRMSE' and 'best OS NRMSE' weights.

### 1.1. Results using Best RO Weights
#### CNO
| Group | Avg Selected Epoch | TrainLoss | OS-NRMSE | RO-NRMSE | Runtime |
|---|---|---|---|---|---|
| Baseline (PDEBench) | N/A | N/A | N/A | N/A | N/A |
| CNO-TB_4 | 90.0±9.1 | 0.0000±0.0000 | 0.0003±0.0001 | 0.0025±0.0009 | 82095s |
| CNO-baseline | 64.7±29.0 | 0.0000±0.0000 | 0.0003±0.0001 | 0.0163±0.0063 | 68397s |
| CNO-noise | 47.3±21.7 | 0.0000±0.0000 | 0.0003±0.0001 | 0.0174±0.0126 | 86526s |

#### FNO
| Group | Avg Selected Epoch | TrainLoss | OS-NRMSE | RO-NRMSE | Runtime |
|---|---|---|---|---|---|
| Baseline (PDEBench) | N/A | N/A | N/A | 0.0044 | N/A |
| FNO-Noise | 97.2±2.0 | 0.0000±0.0000 | 0.0002±0.0000 | 0.0025±0.0001 | 41939s |
| FNO-PF | 48.0±8.4 | 0.0000±0.0000 | 0.0003±0.0000 | 0.0031±0.0004 | 57965s |
| FNO-TB_4 | 76.8±26.5 | 0.0000±0.0000 | 0.0002±0.0001 | 0.0019±0.0001 | 41615s |
| FNO-TB_4-PF | 97.2±3.9 | 0.0000±0.0000 | 0.0003±0.0000 | 0.0012±0.0001 | 51709s |
| FNO-baseline | 63.6±33.7 | 0.0000±0.0000 | 0.0007±0.0011 | 0.0406±0.0462 | 45116s |

#### UNet
| Group | Avg Selected Epoch | TrainLoss | OS-NRMSE | RO-NRMSE | Runtime |
|---|---|---|---|---|---|
| Baseline (PDEBench) | N/A | N/A | N/A | 0.083 | N/A |
| UNet-Noise | 94.0±4.9 | 0.0000±0.0000 | 0.0003±0.0001 | 0.0343±0.0174 | 22803s |
| UNet-PF | 44.8±4.8 | 0.0000±0.0000 | 0.0007±0.0002 | 0.0855±0.0418 | 31929s |
| UNet-TB_4 | 90.0±12.1 | 0.0000±0.0000 | 0.0003±0.0000 | 0.0127±0.0030 | 20309s |
| UNet-TB_4-PF | 91.2±6.5 | 0.0000±0.0000 | 0.0006±0.0003 | 0.0040±0.0014 | 29558s |
| UNet-baseline | 88.8±9.3 | 0.0000±0.0000 | 0.0004±0.0002 | 0.1161±0.0575 | 21442s |

### 1.2. Results using Best OS Weights
#### CNO
| Group | Avg Selected Epoch | TrainLoss | OS-NRMSE | RO-NRMSE |
|---|---|---|---|---|
| Baseline (PDEBench) | N/A | N/A | N/A | N/A |
| CNO-TB_4 | 69.7±18.7 | 0.0000±0.0000 | 0.0003±0.0000 | 0.0039±0.0007 |
| CNO-baseline | 79.3±20.2 | 0.0000±0.0000 | 0.0002±0.0001 | 0.0257±0.0240 |
| CNO-noise | 67.7±12.0 | 0.0000±0.0000 | 0.0003±0.0001 | 0.0253±0.0163 |

#### FNO
| Group | Avg Selected Epoch | TrainLoss | OS-NRMSE | RO-NRMSE |
|---|---|---|---|---|
| Baseline (PDEBench) | N/A | N/A | 0.0044 | N/A |
| FNO-Noise | 98.6±1.2 | 0.0000±0.0000 | 0.0002±0.0000 | 0.0025±0.0002 |
| FNO-PF | 95.6±2.3 | 0.0000±0.0000 | 0.0002±0.0000 | N/A |
| FNO-TB_4 | 99.0±0.9 | 0.0000±0.0000 | 0.0002±0.0000 | 0.0018±0.0001 |
| FNO-TB_4-PF | 96.0±4.5 | 0.0000±0.0000 | 0.0002±0.0000 | 0.0012±0.0000 |
| FNO-baseline | 95.2±4.4 | 0.0000±0.0000 | 0.0002±0.0000 | 0.0227±0.0135 |

#### UNet
| Group | Avg Selected Epoch | TrainLoss | OS-NRMSE | RO-NRMSE |
|---|---|---|---|---|
| Baseline (PDEBench) | N/A | N/A | 0.083 | N/A |
| UNet-Noise | 95.6±5.0 | 0.0000±0.0000 | 0.0003±0.0000 | 0.0312±0.0224 |
| UNet-PF | 93.2±7.4 | 0.0000±0.0000 | 0.0003±0.0000 | 0.2942±0.0547 |
| UNet-TB_4 | 95.6±3.4 | 0.0000±0.0000 | 0.0003±0.0000 | 0.0134±0.0054 |
| UNet-TB_4-PF | 95.4±3.2 | 0.0000±0.0000 | 0.0004±0.0000 | 0.0054±0.0024 |
| UNet-baseline | 86.0±12.1 | 0.0000±0.0000 | 0.0003±0.0000 | 0.2348±0.1816 |
