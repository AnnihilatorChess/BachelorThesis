# W&B Analysis: BIG_SWE_input-4

## Overview

Analysis of the BIG_SWE_input-4 project with larger model sizes and input steps = 4.

## 1. Analysis
The metrics below are reported as `mean ± std` across seeds.
Primary evaluation metric: **NRMSE**. Metrics evaluate 'best rollout NRMSE' and 'best OS NRMSE' weights.

### 1.1. Results using Best RO Weights

#### CNO (SMALL)
| Group | Avg Selected Epoch | TrainLoss | OS-NRMSE | RO-NRMSE | Runtime | Max VRAM |
|---|---|---|---|---|---|---|
| Baseline (PDEBench) | N/A | N/A | N/A | N/A | N/A | N/A |
| SMALL-CNO: baseline, Noise, TB_4 | 67.3±27.8 | 0.0000±0.0000 | 0.0003±0.0001 | 0.0121±0.0106 | 79006s | 3.8 GB |

#### FNO
| Group | Avg Selected Epoch | TrainLoss | OS-NRMSE | RO-NRMSE | Runtime | Max VRAM |
|---|---|---|---|---|---|---|
| Baseline (PDEBench) | N/A | N/A | N/A | 0.0044 | N/A | N/A |
| FNO-Noise | 97.2±2.0 | 0.0000±0.0000 | 0.0002±0.0000 | 0.0025±0.0001 | 41939s | 6.3 GB |
| FNO-PF | 48.0±8.4 | 0.0000±0.0000 | 0.0003±0.0000 | 0.0031±0.0004 | 57965s | 6.8 GB |
| FNO-TB_4 | 76.8±26.5 | 0.0000±0.0000 | 0.0002±0.0001 | 0.0019±0.0001 | 41615s | 6.0 GB |
| FNO-TB_4-PF | 97.2±3.9 | 0.0000±0.0000 | 0.0003±0.0000 | 0.0012±0.0001 | 51709s | 6.8 GB |
| FNO-baseline | 63.6±33.7 | 0.0000±0.0000 | 0.0007±0.0011 | 0.0406±0.0462 | 45116s | 6.7 GB |

#### UNet
| Group | Avg Selected Epoch | TrainLoss | OS-NRMSE | RO-NRMSE | Runtime | Max VRAM |
|---|---|---|---|---|---|---|
| Baseline (PDEBench) | N/A | N/A | N/A | 0.083 | N/A | N/A |
| UNet-Noise | 94.0±4.9 | 0.0000±0.0000 | 0.0003±0.0001 | 0.0343±0.0174 | 22803s | 2.4 GB |
| UNet-PF | 44.8±4.8 | 0.0000±0.0000 | 0.0007±0.0002 | 0.0855±0.0418 | 31929s | 3.8 GB |
| UNet-TB_4 | 90.0±12.1 | 0.0000±0.0000 | 0.0003±0.0000 | 0.0127±0.0030 | 20309s | 2.4 GB |
| UNet-TB_4-PF | 91.2±6.5 | 0.0000±0.0000 | 0.0006±0.0003 | 0.0040±0.0014 | 29558s | 3.9 GB |
| UNet-baseline | 88.8±9.3 | 0.0000±0.0000 | 0.0004±0.0002 | 0.1161±0.0575 | 21442s | 2.4 GB |

### 1.2. Results using Best OS Weights

#### CNO (SMALL)
| Group | Avg Selected Epoch | TrainLoss | OS-NRMSE | RO-NRMSE | Runtime |
|---|---|---|---|---|---|
| Baseline (PDEBench) | N/A | N/A | N/A | N/A | N/A |
| SMALL-CNO: baseline, Noise, TB_4 | 72.2±18.1 | 0.0000±0.0000 | 0.0003±0.0001 | 0.0183±0.0196 | 79006s |

#### FNO
| Group | Avg Selected Epoch | TrainLoss | OS-NRMSE | RO-NRMSE | Runtime |
|---|---|---|---|---|---|
| Baseline (PDEBench) | N/A | N/A | 0.0044 | N/A | N/A |
| FNO-Noise | 98.6±1.2 | 0.0000±0.0000 | 0.0002±0.0000 | 0.0025±0.0002 | 41939s |
| FNO-PF | 95.6±2.3 | 0.0000±0.0000 | 0.0002±0.0000 | N/A | 57965s |
| FNO-TB_4 | 99.0±0.9 | 0.0000±0.0000 | 0.0002±0.0000 | 0.0018±0.0001 | 41615s |
| FNO-TB_4-PF | 96.0±4.5 | 0.0000±0.0000 | 0.0002±0.0000 | 0.0012±0.0000 | 51709s |
| FNO-baseline | 95.2±4.4 | 0.0000±0.0000 | 0.0002±0.0000 | 0.0227±0.0135 | 45116s |

#### UNet
| Group | Avg Selected Epoch | TrainLoss | OS-NRMSE | RO-NRMSE | Runtime |
|---|---|---|---|---|---|
| Baseline (PDEBench) | N/A | N/A | 0.083 | N/A | N/A |
| UNet-Noise | 95.6±5.0 | 0.0000±0.0000 | 0.0003±0.0000 | 0.0312±0.0224 | 22803s |
| UNet-PF | 93.2±7.4 | 0.0000±0.0000 | 0.0003±0.0000 | 0.2942±0.0547 | 31929s |
| UNet-TB_4 | 95.6±3.4 | 0.0000±0.0000 | 0.0003±0.0000 | 0.0134±0.0054 | 20309s |
| UNet-TB_4-PF | 95.4±3.2 | 0.0000±0.0000 | 0.0004±0.0000 | 0.0054±0.0024 | 29558s |
| UNet-baseline | 86.0±12.1 | 0.0000±0.0000 | 0.0003±0.0000 | 0.2348±0.1816 | 21442s |
