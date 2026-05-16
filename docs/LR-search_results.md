# LR Search Results per Model and Dataset

## SWE

We use batch_size = 32 and search for LR over `[0.003,0.001, 0.0003]`
### Results for baselines
1. UNet: 0.001
2. UNet-SMALL: 0.001
3. FNO-SMALL: 0.0003
4. FNO: 0.001
5. CNO-SMALL: 0.0003
6. FNO-SMALL-BPTT: 0.003 best 0.001 matches paper better
7. UNet-SMALL-BPTT: 0.0003 best, all are close

## BUR-downsized

We use batch_size = 512 for normal and 5 for BPTT training and search for LR over `[0.003,0.001, 0.0003]`

### Results for baselines
1. FNO-BPTT: 0.003 --> didnt converge at 100 epoch
2. UNet-BPTT: 0.003 --> interesting that this is better for BPTT
3. FNO-SMALL: 0.001 (all three are really close, taking the middle as it is <1% better and the safe option)
4. UNet-SMALL: 0.0003 best for one step
5. FNO: 0.001
6. 

#### Results for legacy baselines
1. UNet: 0.001
2. UNet-SMALL: 0.0003
3. FNO-SMALL: 0.0003
4. FNO: 0.001