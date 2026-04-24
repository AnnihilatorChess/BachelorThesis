# Gray-Scott Reaction-Diffusion Dataset Analysis

This document summarizes the analysis of the Gray-Scott reaction-diffusion dataset, specifically focusing on the identification of stale (equilibrium) trajectories.

## Dataset Reference
The original documentation for this dataset can be found at:
[datasets/gray_scott_reaction_diffusion/README.md](../../datasets/gray_scott_reaction_diffusion/README.md)

## HDF5 Structure
To avoid future confusion, the HDF5 files in this dataset follow a specific structure:
- **Top-level groups:** `boundary_conditions`, `dimensions`, `scalars`, `t0_fields`, `t1_fields`, `t2_fields`.
- **Data Location:** The primary trajectory data is stored in `t0_fields`.
- **Datasets:** Inside `t0_fields`, there are two datasets: `A` and `B` (representing the two chemical species).
- **Shape:** Both `A` and `B` have the shape `(N_trajectories, T_timesteps, H_height, W_width)`. 
  - For this specific version of the dataset, `N_trajectories = 160`, `T_timesteps = 1001`, `H = 128`, `W = 128`.
- **Note:** `t1_fields` and `t2_fields` are present but appear to be empty in this subset.

## Stale Trajectory Analysis
Stale trajectories are those that reach a stationary equilibrium state (constant in time). These are often undesirable for training as they represent a "trivial" solution.

### Findings
The analysis identified a total of **100 stale trajectories** (8.33% of the analyzed subset).

| Split | Stale / Total | Percentage |
|-------|---------------|------------|
| Test  | 10 / 120      | 8.33%      |
| Valid | 13 / 120      | 10.83%     |
| Train | 77 / 960      | 8.02%      |

**Parameter Sensitivity:**
- **Gliders ($f=0.014, k=0.054$):** Highly prone to equilibrium, with 89 out of 100 total stale trajectories. Over 40% of trajectories in this parameter set reach equilibrium.
- **Spirals ($f=0.018, k=0.051$):** 6 stale trajectories identified.
- **Bubbles ($f=0.098, k=0.057$):** 5 stale trajectories identified.
- **Others:** No stale trajectories were identified in the Maze, Spots, or Worms sets.

## Detection Script
The analysis was performed using the script `scripts/identify_stale_trajectories.py`.

### Usage
To run the script, use the `bsc_` conda environment:
```bash
C:\Users\simon\miniconda3\envs\bsc_\python.exe scripts/identify_stale_trajectories.py
```

### Logic
The script identifies staleness by:
1.  Loading the concentration of species `A`.
2.  Calculating the standard deviation over time for the last 100 steps.
3.  If the mean standard deviation across all pixels is below a threshold (default $10^{-6}$), the trajectory is marked as stale.
4.  It then identifies the first timestep where the mean absolute difference between consecutive steps falls and stays below the threshold.
