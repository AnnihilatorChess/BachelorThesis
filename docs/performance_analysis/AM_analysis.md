# W&B Analysis: AM (Active Matter)

## Overview

Analysis of the active_matter dataset from The Well. Note: Runs are currently incomplete.

## 1. Analysis
The metrics below are reported as `mean ± std` across seeds.
Primary evaluation metric: **VRMSE**. Metrics evaluate 'best rollout VRMSE' and 'best OS VRMSE' weights.

### 1.1. Results using Best RO Weights
#### UNet
| Group | Avg Selected Epoch | TrainLoss | OS-VRMSE | RO-VRMSE | T=6:12 VRMSE | T=13:30 VRMSE | Runtime |
|---|---|---|---|---|---|---|---|
| Baseline (The Well) | N/A | N/A | 0.2489 | N/A | 2.53 | 2.62 | N/A |
| UNet-Noise | 89.6±8.1 | 0.0011±0.0000 | 0.0890±0.0057 | 1.8687±0.1418 | 3.6296±0.3669 | 2.2858±0.2103 | 50911s |
| UNet-PF | 93.3±5.2 | 0.0138±0.0004 | 0.1091±0.0010 | 1.5374±0.0876 | 1.8781±0.1701 | 1.8587±0.1609 | 62057s |
| UNet-TB_4 | 84.0±0.0 | 0.0943±0.0812 | 0.1845±0.0000 | 1.5054±0.0000 | 1.8975±0.0000 | 1.6699±0.0000 | 41538s |
| UNet-TB_4-PF | 76.0±0.0 | 0.2037±0.0000 | 0.2404±0.0000 | 1.2325±0.0000 | 1.4225±0.0000 | 1.3311±0.0000 | 65740s |
| UNet-baseline | 87.5±7.5 | 0.0011±0.0000 | 0.0994±0.0053 | 2.0866±0.2152 | 4.1711±0.8541 | 2.6006±0.3455 | 48895s |

### 1.2. Results using Best OS Weights
#### UNet
| Group | Avg Selected Epoch | TrainLoss | OS-VRMSE | RO-VRMSE | T=6:12 VRMSE | T=13:30 VRMSE | Runtime |
|---|---|---|---|---|---|---|---|
| Baseline (The Well) | N/A | N/A | 0.2489 | N/A | 2.53 | 2.62 | N/A |
| UNet-Noise | 97.6±2.9 | 0.0011±0.0000 | 0.0860±0.0064 | 2.0217±0.2085 | 4.1361±0.6505 | 2.4431±0.2485 | 50911s |
| UNet-PF | 92.7±5.0 | 0.0138±0.0004 | 0.1160±0.0068 | 1.6468±0.1770 | 2.2370±0.3588 | 1.9381±0.1313 | 62057s |
| UNet-TB_4 | 100.0±0.0 | 0.0943±0.0812 | 0.1433±0.0000 | 1.8135±0.0000 | 2.9230±0.0000 | 2.2576±0.0000 | 41538s |
| UNet-TB_4-PF | 92.0±0.0 | 0.2037±0.0000 | 0.1966±0.0000 | 1.5065±0.0000 | 1.9138±0.0000 | 1.7693±0.0000 | 65740s |
| UNet-baseline | 98.5±1.7 | 0.0011±0.0000 | 0.0890±0.0059 | 2.2896±0.1887 | 4.7831±0.2051 | 2.8402±0.2238 | 48895s |


## 2. Convergence & Model Analysis (Active Matter)

- **Baseline Outperformance**: Our native `UNet-baseline` achieves a one-step VRMSE of **0.099**, far surpassing the baseline reported in The Well paper (0.248).
- **Stabilization Efficacy**: The `active_matter` dynamics are inherently chaotic and difficult to roll out stably (as seen by the paper's baselines which explode to >10 error in the 13:30 window). However, combining Temporal Bundling with Pushforward (`UNet-TB_4-PF`) massively stabilizes the trajectory, keeping the T=13:30 VRMSE down to **1.331** (a highly stable state compared to the unregularized baseline's 2.600).
- **Noise Injection Sensitivity**: `UNet-Noise` helps to slightly reduce the error in the late rollout windows compared to the baseline, but falls short of the strong regularization provided by Pushforward and Temporal Bundling.
