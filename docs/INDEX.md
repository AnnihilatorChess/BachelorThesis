# Documentation Index

This index organizes all project guidance and knowledge resources for quick access. Files are organized by purpose and audience.

## Agentic Guidance (Development & Workflow)

- **[AGENTS.md](../AGENTS.md)** — Shared development guidance for all agentic systems (Claude Code, Gemini CLI, etc.). Covers project overview, setup, commands, architecture, training procedures, thesis constraints, and research mandate. `CLAUDE.md` and `GEMINI.md` are thin pointers to this file.

## Project Planning & Scope

- **[thesis-outline.md](thesis-outline.md)** — Complete thesis structure and experimental design. Covers:
  - Working title and contributions
  - Background on neural operators
  - Dataset selection (TRL 2D, PDEBench SWE, Gray-Scott, Rayleigh-Benard, Burgers 1D)
  - Methods: architectures (FNO, U-Net, ViT), stabilization techniques (pushforward, noise injection, temporal bundling)
  - Experimental setup and ablation design

## Models

- **[models/cno.md](models/cno.md)** — CNO architecture summary and parameter count from the original NeurIPS 2023 paper.

## Data & Conversion

- **[datasets/pdebench_1d_burgers.md](datasets/pdebench_1d_burgers.md)** — Physics and dataset details for 1D Burgers' equation.
- **[datasets/gray_scott_reaction_diffusion.md](datasets/gray_scott_reaction_diffusion.md)** — Analysis of the Gray-Scott dataset, including stale trajectory detection.
- **[scripts/pdebench-swe-conversion.md](scripts/pdebench-swe-conversion.md)** — Documentation of PDEBench 2D Shallow Water Equations conversion.
- **[scripts/pdebench-1d-burgers-conversion.md](scripts/pdebench-1d-burgers-conversion.md)** — Documentation of PDEBench 1D Burgers conversion.

## Analysis & Evaluation

- **[extended-metrics.md](extended-metrics.md)** — **Primary reference** for rollout evaluation. Covers Sobolev H1, HighFreqEnergyRatio, ValidRolloutLength (VRL), Correlation Time, and Error Growth Rate (λ).
- **[benchmark-comparability.md](benchmark-comparability.md)** — How The Well and PDEBench constructed their baselines, and how comparable our thesis setup is to each.
- **[performance_analysis/summary.md](performance_analysis/summary.md)** — **Latest results summary**. Compares FNO and UNetClassic across stabilization techniques on TRL and SWE datasets.
- **[performance_analysis/burgers_dataloader_bench.md](performance_analysis/burgers_dataloader_bench.md)** — Empirical DataLoader settings sweep for the 1D Burgers layout; explains how to get 29 % more throughput without changing HDF5 chunking.
- **[wandb_analysis.md](wandb_analysis.md)** — Weights & Biases (W&B) workflow, run keys, and historical results from April 2025.

## Quick Reference

| Document | Purpose | Audience |
|-----------|---------|----------|
| [AGENTS.md](../AGENTS.md) | Dev workflow (all agents) | Developers |
| [extended-metrics.md](extended-metrics.md) | Metric definitions | Researchers |
| [performance_analysis/summary.md](performance_analysis/summary.md) | Results & Baselines | Researchers |
| [thesis-outline.md](thesis-outline.md) | Thesis Scope | Research context |

## Directory Structure

```
docs/
├── INDEX.md                       # This file
├── thesis-outline.md              # Thesis scope
├── extended-metrics.md            # Evaluation reference
├── wandb_analysis.md              # W&B workflow
├── models/                        # Architecture notes
│   └── cno.md
├── performance_analysis/          # Experiment results
│   └── summary.md
├── datasets/                      # Dataset-specific notes
│   ├── gray_scott_reaction_diffusion.md
│   └── pdebench_1d_burgers.md
└── scripts/                       # Data engineering docs
    ├── pdebench-swe-conversion.md
    └── pdebench-1d-burgers-conversion.md
```
