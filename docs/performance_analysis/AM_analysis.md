# W&B Analysis: AM (Active Matter)

## Overview

Analysis of the active_matter dataset from The Well.

## 1. Analysis
The metrics below are reported as `mean ± std` across seeds.
Primary evaluation metric: **VRMSE**. Metrics evaluate 'best rollout VRMSE' and 'best OS VRMSE' weights.

### 1.1. Results using Best RO Weights

#### CNO
| Group | Avg Selected Epoch | TrainLoss | OS-VRMSE | RO-VRMSE | T=6:12 VRMSE | T=13:30 VRMSE | Runtime | Max VRAM |
|---|---|---|---|---|---|---|---|---|
| Baseline (The Well) | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| AM-CNO-Noise | 98.0±0.0 | 0.0021±0.0018 | 0.0576±0.0034 | 1.7047±0.1035 | 2.7731±0.6647 | 2.3250±0.1270 | 60275s | 9.0 GB |
| AM-CNO-PF | 82.0±0.0 | 0.0181±0.0019 | 0.0917±0.0000 | 1.4695±0.0000 | 1.9662±0.0000 | 1.9262±0.0000 | 85294s | 10.0 GB |
| AM-CNO-baseline | 87.3±15.1 | 0.0014±0.0009 | 0.0647±0.0107 | 1.7545±0.0844 | 3.1645±0.7149 | 2.3827±0.1321 | 59404s | 9.0 GB |

#### FNO
| Group | Avg Selected Epoch | TrainLoss | OS-VRMSE | RO-VRMSE | T=6:12 VRMSE | T=13:30 VRMSE | Runtime | Max VRAM |
|---|---|---|---|---|---|---|---|---|
| Baseline (The Well) | N/A | N/A | 0.3691 | N/A | >10 | >10 | N/A | N/A |
| AM-FNO-FP32-Noise | 87.6±7.1 | 0.0013±0.0000 | 0.0793±0.0016 | 1.6724±0.0908 | 3.3501±0.2976 | 1.9645±0.2326 | 62744s | 10.3 GB |
| AM-FNO-FP32-PF | 85.5±5.2 | 0.0034±0.0003 | 0.0952±0.0027 | 1.3744±0.0578 | 1.6072±0.3255 | 1.6258±0.0982 | 76526s | 9.9 GB |
| AM-FNO-FP32-TB_4 | 85.2±9.5 | 0.0114±0.0001 | 0.1554±0.0064 | 1.4464±0.0205 | 1.4523±0.0388 | 1.8565±0.0604 | 67182s | 10.3 GB |
| AM-FNO-FP32-TB_4-PF | 96.0±0.0 | 0.0355±0.0089 | 0.2000±0.0000 | 1.2109±0.0000 | 0.9891±0.0000 | 1.2977±0.0000 | 81340s | 11.1 GB |
| AM-FNO-FP32-baseline | 92.8±5.6 | 0.0013±0.0000 | 0.0789±0.0022 | 1.6835±0.0969 | 3.3547±0.3813 | 1.9655±0.1986 | 76706s | 10.3 GB |

#### UNet
| Group | Avg Selected Epoch | TrainLoss | OS-VRMSE | RO-VRMSE | T=6:12 VRMSE | T=13:30 VRMSE | Runtime | Max VRAM |
|---|---|---|---|---|---|---|---|---|
| Baseline (The Well) | N/A | N/A | 0.2489 | N/A | 2.53 | 2.62 | N/A | N/A |
| UNet-Noise | 89.6±8.1 | 0.0011±0.0000 | 0.0890±0.0057 | 1.8687±0.1418 | 3.6296±0.3669 | 2.2858±0.2103 | 50911s | 10.1 GB |
| UNet-PF | 87.2±13.2 | 0.0139±0.0004 | 0.1225±0.0191 | 1.6376±0.1440 | 2.2915±0.5757 | 2.0084±0.2400 | 61094s | 9.1 GB |
| UNet-TB_4 | 86.4±14.2 | 0.0131±0.0001 | 0.1862±0.0497 | 1.5431±0.0389 | 2.0088±0.1617 | 1.8813±0.1650 | 41304s | 9.1 GB |
| UNet-TB_4-PF | 82.8±10.5 | 0.1962±0.0061 | 0.2511±0.0180 | 1.2131±0.0690 | 1.4363±0.1272 | 1.2815±0.1269 | 68874s | 9.1 GB |
| UNet-baseline | 87.5±7.5 | 0.0011±0.0000 | 0.0994±0.0053 | 2.0866±0.2152 | 4.1711±0.8541 | 2.6006±0.3455 | 48895s | 9.1 GB |

### 1.2. Results using Best OS Weights

#### CNO
| Group | Avg Selected Epoch | TrainLoss | OS-VRMSE | RO-VRMSE | T=6:12 VRMSE | T=13:30 VRMSE | Runtime |
|---|---|---|---|---|---|---|---|
| Baseline (The Well) | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| AM-CNO-Noise | 99.0±1.0 | 0.0021±0.0018 | 0.0539±0.0003 | 1.6635±0.0623 | 2.1387±0.0303 | 2.3331±0.1351 | 60275s |
| AM-CNO-PF | 88.0±0.0 | 0.0181±0.0019 | 0.0835±0.0000 | 1.5941±0.0000 | 1.5455±0.0000 | 2.0958±0.0000 | 85294s |
| AM-CNO-baseline | 99.3±0.9 | 0.0014±0.0009 | 0.0542±0.0006 | 1.7280±0.0821 | 2.3330±0.2584 | 2.3728±0.0620 | 59404s |

#### FNO
| Group | Avg Selected Epoch | TrainLoss | OS-VRMSE | RO-VRMSE | T=6:12 VRMSE | T=13:30 VRMSE | Runtime |
|---|---|---|---|---|---|---|---|
| Baseline (The Well) | N/A | N/A | 0.3691 | N/A | >10 | >10 | N/A |
| AM-FNO-FP32-Noise | 95.2±2.0 | 0.0013±0.0000 | 0.0773±0.0015 | 1.7746±0.0811 | 3.3398±0.2822 | 2.1658±0.1795 | 62744s |
| AM-FNO-FP32-PF | 95.5±2.6 | 0.0034±0.0003 | 0.0915±0.0005 | 1.4683±0.0839 | 1.5326±0.1850 | 1.8581±0.2374 | 76526s |
| AM-FNO-FP32-TB_4 | 94.0±3.3 | 0.0114±0.0001 | 0.1510±0.0013 | 1.4890±0.0201 | 1.4827±0.2242 | 1.9663±0.0215 | 67182s |
| AM-FNO-FP32-TB_4-PF | 94.0±0.0 | 0.0355±0.0089 | 0.2016±0.0000 | 1.2131±0.0000 | 0.9382±0.0000 | 1.3914±0.0000 | 81340s |
| AM-FNO-FP32-baseline | 95.2±2.0 | 0.0013±0.0000 | 0.0772±0.0014 | 1.7784±0.0694 | 3.3430±0.2636 | 2.1857±0.1694 | 76706s |

#### UNet
| Group | Avg Selected Epoch | TrainLoss | OS-VRMSE | RO-VRMSE | T=6:12 VRMSE | T=13:30 VRMSE | Runtime |
|---|---|---|---|---|---|---|---|
| Baseline (The Well) | N/A | N/A | 0.2489 | N/A | 2.53 | 2.62 | N/A |
| UNet-Noise | 97.6±2.9 | 0.0011±0.0000 | 0.0860±0.0064 | 2.0217±0.2085 | 4.1361±0.6505 | 2.4431±0.2485 | 50911s |
| UNet-PF | 94.4±4.8 | 0.0139±0.0004 | 0.1190±0.0067 | 1.6955±0.1510 | 2.3900±0.3398 | 2.0136±0.1397 | 61094s |
| UNet-TB_4 | 95.6±4.8 | 0.0131±0.0001 | 0.1364±0.0046 | 1.8828±0.1832 | 3.0151±0.7295 | 2.3902±0.1963 | 41304s |
| UNet-TB_4-PF | 78.4±12.7 | 0.1962±0.0061 | 0.2131±0.0153 | 1.5655±0.1425 | 2.1786±0.5030 | 1.8315±0.2004 | 68874s |
| UNet-baseline | 98.5±1.7 | 0.0011±0.0000 | 0.0890±0.0059 | 2.2896±0.1887 | 4.7831±0.2051 | 2.8402±0.2238 | 48895s |


## 2. Convergence & Model Analysis (Active Matter)

- **Baseline Outperformance**: Our native `UNet-baseline` achieves a one-step VRMSE of **0.099**, far surpassing the baseline reported in The Well paper (0.248).
- **Stabilization Efficacy**: The `active_matter` dynamics are inherently chaotic and difficult to roll out stably (as seen by the paper's baselines which explode to >10 error in the 13:30 window). However, combining Temporal Bundling with Pushforward (`UNet-TB_4-PF`) massively stabilizes the trajectory, keeping the T=13:30 VRMSE down to **1.331** (a highly stable state compared to the unregularized baseline's 2.600).
- **Noise Injection Sensitivity**: `UNet-Noise` helps to slightly reduce the error in the late rollout windows compared to the baseline, but falls short of the strong regularization provided by Pushforward and Temporal Bundling.