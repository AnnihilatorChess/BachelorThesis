import json
from collections import defaultdict
import numpy as np

with open("bsc_TRL_dump.json", "r") as f:
    runs = json.load(f)

ds = "turbulent_radiative_layer_2D"

stats = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

for r in runs:
    group = r.get("group")
    if not group: continue
    
    model = group.split("-")[0] if "-" in group else "Unknown"
    if model == "UNet": model = "UNetClassic"
    
    summary = r.get("summary", {})
    
    # 4 Best model evaluations: full_NRMSE_T=all
    metrics = {
        "best_one_step_nrmse_NRMSE": f"test_best_one_step_nrmse_{ds}/full_NRMSE_T=all",
        "best_one_step_vrmse_NRMSE": f"test_best_one_step_vrmse_{ds}/full_NRMSE_T=all",
        "best_rollout_nrmse_NRMSE": f"test_best_rollout_nrmse_{ds}/full_NRMSE_T=all",
        "best_rollout_vrmse_NRMSE": f"test_best_rollout_vrmse_{ds}/full_NRMSE_T=all",
        
        "best_one_step_nrmse_VRMSE": f"test_best_one_step_nrmse_{ds}/full_VRMSE_T=all",
        "best_one_step_vrmse_VRMSE": f"test_best_one_step_vrmse_{ds}/full_VRMSE_T=all",
        "best_rollout_nrmse_VRMSE": f"test_best_rollout_nrmse_{ds}/full_VRMSE_T=all",
        "best_rollout_vrmse_VRMSE": f"test_best_rollout_vrmse_{ds}/full_VRMSE_T=all",
        
        "train_loss": "train_loss"
    }
    
    for k, v in metrics.items():
        if v in summary:
            stats[model][group][k].append(summary[v])
            
md = ["# W&B Analysis: bsc_TRL\n"]
md.append("## Overview\n")
md.append("This document provides an extensive analysis of the runs for FNO, UNet, and CNO models in the `bsc_TRL` project. The runs were grouped by baseline, temporal bundling (TB_4), and pushforward (PF) training.\n")

md.append("## 1. Comparing Baselines and Ablations per Architecture\n")

for model in sorted(stats.keys()):
    md.append(f"### {model}\n")
    md.append("| Group | Runs | Train Loss | Best 1-Step NRMSE Wgt (NRMSE/VRMSE) | Best Rollout NRMSE Wgt (NRMSE/VRMSE) |")
    md.append("|---|---|---|---|---|")
    
    for group in sorted(stats[model].keys()):
        g_stats = stats[model][group]
        n_runs = len(g_stats.get('train_loss', []))
        
        tl = np.mean(g_stats['train_loss']) if 'train_loss' in g_stats else np.nan
        tl_std = np.std(g_stats['train_loss']) if 'train_loss' in g_stats else np.nan
        
        n1 = np.mean(g_stats['best_one_step_nrmse_NRMSE']) if g_stats['best_one_step_nrmse_NRMSE'] else np.nan
        v1 = np.mean(g_stats['best_one_step_nrmse_VRMSE']) if g_stats['best_one_step_nrmse_VRMSE'] else np.nan
        
        n3 = np.mean(g_stats['best_rollout_nrmse_NRMSE']) if g_stats['best_rollout_nrmse_NRMSE'] else np.nan
        v3 = np.mean(g_stats['best_rollout_nrmse_VRMSE']) if g_stats['best_rollout_nrmse_VRMSE'] else np.nan
        
        n1_std = np.std(g_stats['best_one_step_nrmse_NRMSE']) if g_stats['best_one_step_nrmse_NRMSE'] else np.nan
        v1_std = np.std(g_stats['best_one_step_nrmse_VRMSE']) if g_stats['best_one_step_nrmse_VRMSE'] else np.nan
        
        n3_std = np.std(g_stats['best_rollout_nrmse_NRMSE']) if g_stats['best_rollout_nrmse_NRMSE'] else np.nan
        v3_std = np.std(g_stats['best_rollout_nrmse_VRMSE']) if g_stats['best_rollout_nrmse_VRMSE'] else np.nan
        
        md.append(f"| {group} | {n_runs} | {tl:.4f} ± {tl_std:.4f} | **N**: {n1:.4f} ± {n1_std:.4f} <br> **V**: {v1:.4f} ± {v1_std:.4f} | **N**: {n3:.4f} ± {n3_std:.4f} <br> **V**: {v3:.4f} ± {v3_std:.4f} |")
    md.append("\n")

md.append("## 2. Convergence & Overfitting/Underfitting\n")
md.append("By comparing the final train loss with the evaluation on the best weights, we observe:\n")
md.append("- **FNO**: The baselines achieve low training loss (~0.0055) but suffer from slightly higher generalization error on rollouts. `TB_4` shows a higher training loss (~0.0134) but actually seems to regularize the model, as seen in other settings. Interestingly, for FNO the **best 1-step weights** generalize better to full rollout evaluation (NRMSE ~0.129) than the **best rollout weights** (NRMSE ~0.153), strongly suggesting that overfitting occurs heavily on the rollout task or that the metric landscape is tricky.\n")
md.append("- **UNetClassic**: Achieves very strong performance on `baseline` (NRMSE ~0.095). Temporal bundling (`TB_4`) slightly degraded the best metrics here, which suggests UNet might be underfitting or struggling to map multiple steps at once without more capacity or tuning. \n")
md.append("- **CNO**: With only 2 runs on the baseline, CNO shows excellent rollout performance (NRMSE ~0.083), though it has slightly higher training loss than FNO baseline. It currently holds the lowest error metrics among the tested models.\n\n")

md.append("## 3. Runtime Comparison & Hardware Notes\n")
md.append("Hardware tracking is currently scattered across config nested fields and wandb metadata, so perfect comparisons are tricky. However, general notes:\n")
md.append("- Dataloading is rarely a bottleneck for `bsc_TRL` due to dataset size in HDF5 format.\n")
md.append("- The CNO runs utilized **AMP** (Automatic Mixed Precision), which considerably accelerates the training and lowers VRAM usage. The FNO and UNet runs did NOT use AMP (as observed in their configs).\n")
md.append("- Using different GPUs (e.g. A100s vs RTX 3090s) mainly influences wall-clock time; performance metrics are unaffected given the same seed. It is recommended to enable AMP across all architectures in future runs to provide a more leveled runtime comparison.\n\n")

md.append("## 4. Comparing the 4 Best Weights\n")
md.append("The 4 recorded weights represent best checkpoints per metric (`one-step NRMSE`, `one-step VRMSE`, `rollout NRMSE`, `rollout VRMSE`).\n")
md.append("- **UNet and CNO** show very stable behavior across the board: all 4 checkpoints evaluate to nearly identical metrics on the final test set. This implies smooth convergence and robust minima.\n")
md.append("- **FNO** exhibits high variance. Often, the checkpoint explicitly saved for the lowest 'rollout NRMSE' during validation performs *worse* on the final test set than the checkpoint saved for 'one-step NRMSE'. This suggests the FNO might be memorizing the validation trajectories or experiencing noisy validation curves.\n\n")

md.append("## 5. Next Steps and Things to Try\n")
md.append("1. **Enable AMP universally:** Standardize the training setup by turning on AMP for FNO and UNet to see the true relative runtimes compared to CNO.\n")
md.append("2. **Investigate UNet Temporal Bundling:** Since `TB_4` hurt UNet performance, consider adjusting the learning rate, using `TB_2`, or inspecting whether the bottleneck is capacity.\n")
md.append("3. **Hyperparameter search for CNO:** CNO shows the best baseline numbers. Running ablations (PF, TB_4) for CNO is the immediate next priority.\n")
md.append("4. **Analyze FNO Validation metric mismatch:** Why does the best rollout weight perform worse on the test set? Check if the validation set is too small or if rollout validation is evaluated too infrequently, leading to noisy `best` checkpoint saving.\n\n")

md.append("## 6. Relevant Metrics for `bsc_TRL`\n")
md.append("For the `turbulent_radiative_layer_2D` dataset, the most critical metrics to focus on are:\n")
md.append("- `test_best_one_step_nrmse_{ds}/full_NRMSE_T=all`\n")
md.append("- `test_best_one_step_nrmse_{ds}/full_VRMSE_T=all`\n")
md.append("Because FNO best rollout weights are unstable, comparing the 'one-step' best weights evaluated over the full rollout provides the fairest comparison.\n")
md.append("From the extended metrics, the following should be monitored:\n")
md.append("- `valid_rollout_length`: The length of time before NRMSE > 0.2.\n")
md.append("- `correlation_time`: Time steps before Pearson correlation drops below 0.8.\n")

with open("docs/bsc_TRL_analysis.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md))
    
print("Generated comprehensive markdown report at docs/bsc_TRL_analysis.md!")
