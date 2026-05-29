# W&B Analysis: BUR-DOWN

## Overview

Analysis of the downsized 1D Burgers Dataset (BUR-DOWN).

## 1. Analysis
The metrics below are reported as `mean ± std` across seeds.
Primary evaluation metric: **NRMSE**. Metrics evaluate 'best rollout NRMSE' and 'best OS NRMSE' weights.

### 1.1. Results using Best RO Weights
#### FNO (BIG)
| Group | Avg Selected Epoch | TrainLoss | OS-NRMSE | RO-NRMSE | Runtime |
|---|---|---|---|---|---|
| Baseline (PDEBench) | N/A | N/A | N/A | 0.029 | N/A |
| FNO-BIG-Noise | 95.6±5.0 | 0.0000±0.0000 | 0.0032±0.0001 | 0.0176±0.0013 | 5494s |
| FNO-BIG-PF | 94.4±4.5 | 0.0000±0.0000 | 0.0037±0.0001 | 0.0128±0.0009 | 4528s |
| FNO-BIG-TB_4 | 96.0±2.2 | 0.0000±0.0000 | 0.0049±0.0002 | 0.0152±0.0015 | 3750s |
| FNO-BIG-TB_4-PF | 97.2±4.7 | 0.0001±0.0000 | 0.0087±0.0008 | 0.0197±0.0012 | 3377s |
| FNO-BIG-baseline | 91.2±6.8 | 0.0000±0.0000 | 0.0031±0.0002 | 0.0195±0.0014 | 6514s |

#### FNO (SMALL)
| Group | Avg Selected Epoch | TrainLoss | OS-NRMSE | RO-NRMSE | Runtime |
|---|---|---|---|---|---|
| Baseline (PDEBench) | N/A | N/A | N/A | 0.029 | N/A |
| FNO-SMALL-BPTT | 97.6±2.3 | 0.0304±0.0021 | 0.0417±0.0010 | 0.0469±0.0012 | 4871s |
| FNO-SMALL-Noise | 12.8±13.4 | 0.0009±0.0001 | 0.0867±0.0324 | 0.1641±0.0026 | 4778s |
| FNO-SMALL-PF | 97.6±3.2 | 0.0012±0.0001 | 0.0466±0.0011 | 0.1301±0.0088 | 4412s |
| FNO-SMALL-PF-TB_4 | 90.4±7.2 | 0.0024±0.0001 | 0.0573±0.0027 | 0.0979±0.0015 | 3597s |
| FNO-SMALL-TB_4 | 91.2±4.5 | 0.0019±0.0001 | 0.0555±0.0011 | 0.1047±0.0059 | 4116s |
| FNO-SMALL-baseline | 8.0±12.0 | 0.0009±0.0001 | 0.0997±0.0269 | 0.1669±0.0037 | 4855s |

#### UNet (SMALL)
| Group | Avg Selected Epoch | TrainLoss | OS-NRMSE | RO-NRMSE | Runtime |
|---|---|---|---|---|---|
| Baseline (PDEBench) | N/A | N/A | N/A | 0.37 | N/A |
| UNet-SMALL-BPTT | 53.2±22.1 | 10.4732±7.5378 | 0.3994±0.0424 | 0.4147±0.0307 | 5021s |
| UNet-SMALL-Noise | 90.8±13.4 | 0.0008±0.0003 | 0.0235±0.0056 | 0.6969±0.1973 | 5160s |
| UNet-SMALL-PF | 86.4±25.2 | 0.0028±0.0001 | 0.0366±0.0151 | 0.4133±0.1357 | 5180s |
| UNet-SMALL-PF-TB_4 | 97.6±2.9 | 0.0102±0.0004 | 0.0465±0.0022 | 0.2401±0.0101 | 3421s |
| UNet-SMALL-TB_4 | 86.0±15.2 | 0.0022±0.0002 | 0.0334±0.0075 | 0.3351±0.0274 | 5005s |
| UNet-SMALL-baseline | 85.5±14.0 | 0.0008±0.0003 | 0.0251±0.0056 | 0.9671±0.2031 | 4394s |

### 1.2. Results using Best OS Weights
#### FNO (BIG)
| Group | Avg Selected Epoch | TrainLoss | OS-NRMSE | RO-NRMSE |
|---|---|---|---|---|
| Baseline (PDEBench) | N/A | N/A | 0.029 | N/A |
| FNO-BIG-Noise | 98.4±2.2 | 0.0000±0.0000 | 0.0030±0.0001 | 0.0185±0.0024 |
| FNO-BIG-PF | 93.4±3.6 | 0.0000±0.0000 | 0.0036±0.0001 | 0.0137±0.0011 |
| FNO-BIG-TB_4 | 98.4±1.0 | 0.0000±0.0000 | 0.0048±0.0001 | 0.0161±0.0013 |
| FNO-BIG-TB_4-PF | 98.0±2.1 | 0.0001±0.0000 | 0.0083±0.0002 | 0.0209±0.0022 |
| FNO-BIG-baseline | 96.6±2.9 | 0.0000±0.0000 | 0.0029±0.0001 | 0.0204±0.0008 |

#### FNO (SMALL)
| Group | Avg Selected Epoch | TrainLoss | OS-NRMSE | RO-NRMSE |
|---|---|---|---|---|
| Baseline (PDEBench) | N/A | N/A | 0.029 | N/A |
| FNO-SMALL-BPTT | 96.8±1.7 | 0.0304±0.0021 | 0.0417±0.0009 | 0.0478±0.0008 |
| FNO-SMALL-Noise | 93.8±5.0 | 0.0009±0.0001 | 0.0439±0.0008 | 0.1877±0.0053 |
| FNO-SMALL-PF | 68.8±28.9 | 0.0012±0.0001 | 0.0463±0.0010 | 0.1558±0.0280 |
| FNO-SMALL-PF-TB_4 | 44.6±17.4 | 0.0024±0.0001 | 0.0559±0.0029 | 0.1048±0.0026 |
| FNO-SMALL-TB_4 | 92.4±9.4 | 0.0019±0.0001 | 0.0554±0.0013 | 0.1078±0.0035 |
| FNO-SMALL-baseline | 93.8±5.0 | 0.0009±0.0001 | 0.0438±0.0008 | 0.1899±0.0064 |

#### UNet (SMALL)
| Group | Avg Selected Epoch | TrainLoss | OS-NRMSE | RO-NRMSE |
|---|---|---|---|---|
| Baseline (PDEBench) | N/A | N/A | 0.37 | N/A |
| UNet-SMALL-BPTT | 41.2±17.7 | 10.4732±7.5378 | 0.3226±0.0336 | 0.4686±0.0623 |
| UNet-SMALL-Noise | 96.0±2.1 | 0.0008±0.0003 | 0.0209±0.0024 | 0.9234±0.1822 |
| UNet-SMALL-PF | 95.6±5.5 | 0.0028±0.0001 | 0.0293±0.0010 | 0.4284±0.1547 |
| UNet-SMALL-PF-TB_4 | 88.4±9.4 | 0.0102±0.0004 | 0.0439±0.0018 | 0.2508±0.0162 |
| UNet-SMALL-TB_4 | 98.8±1.2 | 0.0022±0.0002 | 0.0277±0.0018 | 0.4577±0.1085 |
| UNet-SMALL-baseline | 97.2±2.2 | 0.0008±0.0003 | 0.0217±0.0022 | 1.5730±0.4744 |
