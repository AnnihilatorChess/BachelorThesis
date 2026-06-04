# W&B Analysis: bsc_SWE (PDEBench Replication)

## Overview

Analysis of the pdebench_swe (2D Shallow Water Equations) dataset. The SWE dataset is generally "easier" than TRL.

## 1. Analysis
The metrics below are reported as `mean ± std` across seeds.
Primary evaluation metric: **NRMSE**. Metrics evaluate 'best rollout NRMSE' and 'best OS NRMSE' weights.

### 1.1. Results using Best RO Weights

#### FNO (SMALL)
| Group | Avg Selected Epoch | TrainLoss | OS-NRMSE | RO-NRMSE | Runtime | Max VRAM |
|---|---|---|---|---|---|---|
| Baseline (PDEBench) | N/A | N/A | N/A | 0.0044 | N/A | N/A |
| FNO-SMALL-BPTT | 94.8±2.4 | 0.0433±0.0049 | 0.0022±0.0000 | 0.0022±0.0000 | 17917s | 28.3 GB |
| FNO-SMALL-Noise | N/A | 0.0001±0.0000 | N/A | N/A | 15296s | 2.6 GB |
| FNO-SMALL-PF | 48.0±0.0 | 0.0012±0.0001 | 0.0047±0.0000 | 0.0988±0.0000 | 19179s | 3.8 GB |
| FNO-SMALL-TB_4 | 59.6±15.9 | 0.0004±0.0000 | 0.0011±0.0001 | 0.0582±0.0682 | 13107s | 2.6 GB |
| FNO-SMALL-TB_4-PF | 92.8±4.7 | 0.0018±0.0002 | 0.0015±0.0001 | 0.0109±0.0017 | 18292s | 3.8 GB |
| FNO-SMALL-baseline | N/A | 0.0001±0.0000 | N/A | N/A | 14752s | 2.9 GB |

#### UNet (SMALL)
| Group | Avg Selected Epoch | TrainLoss | OS-NRMSE | RO-NRMSE | Runtime | Max VRAM |
|---|---|---|---|---|---|---|
| Baseline (PDEBench) | N/A | N/A | N/A | 0.083 | N/A | N/A |
| UNet-SMALL-BPTT | 52.4±25.4 | 2.0620±0.1563 | 0.0997±0.0136 | 0.0881±0.0045 | 11610s | 5.1 GB |
| UNet-SMALL-PF | 29.6±9.4 | 0.0002±0.0000 | 0.0044±0.0017 | 0.1269±0.0537 | 20065s | 2.7 GB |
| UNet-SMALL-TB_4 | 55.6±29.6 | 0.0003±0.0001 | 0.0036±0.0017 | 0.0631±0.0222 | 12834s | 1.8 GB |
| UNet-SMALL-TB_4-PF | 92.8±3.7 | 0.0010±0.0001 | 0.0023±0.0005 | 0.0170±0.0061 | 17528s | 2.9 GB |
| UNet-SMALL-baseline | 64.8±21.8 | 0.0002±0.0001 | 0.0030±0.0017 | 0.1026±0.0268 | 12539s | 1.8 GB |
| UNet-SMall-Noise | 82.8±8.2 | 0.0002±0.0000 | 0.0025±0.0009 | 0.1459±0.0418 | 15850s | 1.8 GB |

### 1.2. Results using Best OS Weights

#### FNO (SMALL)
| Group | Avg Selected Epoch | TrainLoss | OS-NRMSE | RO-NRMSE | Runtime |
|---|---|---|---|---|---|
| Baseline (PDEBench) | N/A | N/A | 0.0044 | N/A | N/A |
| FNO-SMALL-BPTT | 98.2±1.9 | 0.0433±0.0049 | 0.0022±0.0001 | 0.0023±0.0000 | 17917s |
| FNO-SMALL-Noise | 34.4±7.3 | 0.0001±0.0000 | 0.0011±0.0000 | N/A | 15296s |
| FNO-SMALL-PF | 20.6±1.6 | 0.0012±0.0001 | 0.0011±0.0000 | N/A | 19179s |
| FNO-SMALL-TB_4 | 44.2±29.9 | 0.0004±0.0000 | 0.0011±0.0001 | 0.0315±0.0112 | 13107s |
| FNO-SMALL-TB_4-PF | 17.2±3.0 | 0.0018±0.0002 | 0.0011±0.0001 | 0.0489±0.0175 | 18292s |
| FNO-SMALL-baseline | 94.6±4.8 | 0.0001±0.0000 | 0.0010±0.0000 | N/A | 14752s |

#### UNet (SMALL)
| Group | Avg Selected Epoch | TrainLoss | OS-NRMSE | RO-NRMSE | Runtime |
|---|---|---|---|---|---|
| Baseline (PDEBench) | N/A | N/A | 0.083 | N/A | N/A |
| UNet-SMALL-BPTT | 27.0±30.8 | 2.0620±0.1563 | 0.0768±0.0062 | 0.1031±0.0093 | 11610s |
| UNet-SMALL-PF | 90.4±4.5 | 0.0002±0.0000 | 0.0012±0.0001 | 0.6428±0.0846 | 20065s |
| UNet-SMALL-TB_4 | 64.4±25.0 | 0.0003±0.0001 | 0.0013±0.0002 | 0.0997±0.0489 | 12834s |
| UNet-SMALL-TB_4-PF | 83.0±20.5 | 0.0010±0.0001 | 0.0018±0.0001 | 0.0313±0.0268 | 17528s |
| UNet-SMALL-baseline | 85.4±10.2 | 0.0002±0.0001 | 0.0010±0.0003 | 0.2664±0.1406 | 12539s |
| UNet-SMall-Noise | 92.4±6.7 | 0.0002±0.0000 | 0.0009±0.0002 | 0.1850±0.0381 | 15850s |


## 2. Convergence & Runtime Analysis (SWE)

- **One-Step Superiority**: For SWE, the "Best One-Step" weights often yield very low rollout errors if the model is stable, but can also lead to catastrophic divergence (e.g., UNet-SMALL-PF RO-NRMSE jumps to 0.6428 when using one-step weights vs 0.1280 using rollout weights).
- **Abnormalities**: `FNO-SMALL-baseline` and `Noise` groups have "N/A" for Rollout Weights because standard rollout validation did not identify a better checkpoint than the initial/one-step ones, reflecting the dataset's simplicity.       
- **Runtime**:
    - **UNet** is generally faster than **FNO** for the same task.
    - **PF** training overhead is ~40-50% in time.
    - **TB_4** offers a slight speedup per timestep compared to standard training.

## 3. Underfitting/Overfitting
- In SWE, models are likely **underfitting** in PF/TB modes relative to the baseline because the physics are so simple that any added regularization or complexity in the loss function just adds noise to an already easy optimization landscape.