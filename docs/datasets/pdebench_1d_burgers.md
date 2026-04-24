# PDEBench 1D Burgers Dataset

## Overview

This dataset is derived from the [PDEBench](https://arxiv.org/abs/2210.07182) benchmark, specifically the 1D Burgers equation subset. It contains solutions to the 1D Burgers equation with varying viscosity parameter ($\nu$). The dataset has been converted into The Well format to be used seamlessly with our neural PDE surrogate training pipeline.

## Physics
The 1D Burgers equation models convection and diffusion:
$$ \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2} $$
where $u(t,x)$ is the fluid velocity and $\nu$ is the kinematic viscosity. 

**Boundary Conditions:** Periodic ($u(t, 0) = u(t, 1)$).

**Parameters:**
The thesis pipeline currently uses a single viscosity value:
- $\nu = 0.001$

The lower viscosity leads to sharper gradients and shock formation, making the dynamics challenging to model. The $\nu = 0.01$ file was deliberately dropped; mixing regimes hurt training. The conversion script can be re-pointed to include additional viscosities if needed.

## The Well Format

### Structure
The converted dataset follows The Well HDF5 format. It contains the velocity field `u` as a time-varying `t0_field` and the parameter $\nu$ as a sample-varying, time-invariant `scalar`.

```
pdebench_1d_burgers/
  stats.yaml
  data/
    train/
      pdebench_1d_burgers_00.hdf5 ... pdebench_1d_burgers_03.hdf5
    valid/
      pdebench_1d_burgers_00.hdf5 ... pdebench_1d_burgers_03.hdf5
    test/
      pdebench_1d_burgers_00.hdf5 ... pdebench_1d_burgers_03.hdf5
```

### Dimensions
- `time`: 201 steps, $t \in [0.0, 2.0]$, $dt = 0.01$
- `x`: 1024 points, $x \in [0.0, 1.0)$

### Fields and Scalars
- **`t0_fields/u`**: Shape `(N, 201, 1024)`. The velocity field.
- **`scalars/nu`**: Shape `(N,)`. The viscosity parameter for each sample.

## Splits
The source file contains 10,000 samples. The conversion script splits them sequentially:
- **Train:** 8,000 trajectories
- **Valid:** 1,000 trajectories
- **Test:** 1,000 trajectories

## Scalars
The `nu` scalar is still exposed via `constant_scalars` for format compatibility with The Well, but is not consumed by any model in this thesis.
