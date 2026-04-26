import json
from collections import defaultdict
import numpy as np

with open("bsc_TRL_dump.json", "r") as f:
    runs = json.load(f)

print(f"Total runs loaded: {len(runs)}")

# The base dataset name
ds = "turbulent_radiative_layer_2D"

metrics_to_track = [
    # Metrics evaluated at the END of training
    f"rollout_test_{ds}/full_VRMSE_T=all",
    f"rollout_test_{ds}/full_NRMSE_T=all",
    f"rollout_test_{ds}/full_valid_rollout_length",
    f"rollout_test_{ds}/full_correlation_time",
    f"rollout_test_{ds}/full_nrmse_auc",
    
    # 4 Best model evaluations: full_NRMSE_T=all
    f"test_best_one_step_nrmse_{ds}/full_NRMSE_T=all",
    f"test_best_one_step_vrmse_{ds}/full_NRMSE_T=all",
    f"test_best_rollout_nrmse_{ds}/full_NRMSE_T=all",
    f"test_best_rollout_vrmse_{ds}/full_NRMSE_T=all",
    
    # 4 Best model evaluations: full_VRMSE_T=all
    f"test_best_one_step_nrmse_{ds}/full_VRMSE_T=all",
    f"test_best_one_step_vrmse_{ds}/full_VRMSE_T=all",
    f"test_best_rollout_nrmse_{ds}/full_VRMSE_T=all",
    f"test_best_rollout_vrmse_{ds}/full_VRMSE_T=all",
    
    "train_loss"
]

stats = defaultdict(lambda: defaultdict(list))
hardware_info = defaultdict(list)
runtime_info = defaultdict(list)
amp_info = defaultdict(list)

for r in runs:
    group = r.get("group")
    if group is None:
        continue
    
    # Group logic
    model = "Unknown"
    model_config = r.get("config", {}).get("model", {})
    if "_target_" in model_config:
        model = model_config["_target_"].split(".")[-1]
    
    key = f"{model} - {group}"
    
    summary = r.get("summary", {})
    
    # Fallback to test_turbulent... if rollout_test is not available?
    # No, let's just grab what is there.
    
    for m in metrics_to_track:
        if m in summary:
            stats[key][m].append(summary[m])
            
    system = r.get("config", {}).get("system", {})
    gpu = system.get("gpu", "Unknown") if isinstance(system, dict) else "Unknown"
    cpu = system.get("cpu", "Unknown") if isinstance(system, dict) else "Unknown"
    hardware_info[key].append(f"{gpu} | {cpu}")
    
    runtime = summary.get("_wandb", {}).get("runtime", 0)
    runtime_info[key].append(runtime)
    
    amp = r.get("config", {}).get("trainer", {}).get("amp", False)
    # Check if config has amp at top level
    if "amp" in r.get("config", {}):
        amp = amp or r.get("config", {})["amp"]
    amp_info[key].append(amp)
        
print("\n--- RESULTS ---")
for key in sorted(stats.keys()):
    print(f"\n[{key}] (Runs: {len(stats[key].get('train_loss', []))})")
    
    for m in metrics_to_track:
        vals = stats[key][m]
        if vals:
            mean = np.mean(vals)
            std = np.std(vals)
            print(f"  {m.split('/')[-1] if '/' in m else m} [{m.split('_')[1] if 'best' in m else 'final'}]: {mean:.4f} ± {std:.4f}")
        else:
            print(f"  {m.split('/')[-1] if '/' in m else m} [{m.split('_')[1] if 'best' in m else 'final'}]: MISSING")
            
    hws = set(hardware_info[key])
    print(f"  Hardware: {', '.join(hws)}")
    
    rts = runtime_info[key]
    if rts:
        print(f"  Runtime (avg): {np.mean(rts)/3600:.2f} hours")
        
    amps = set(amp_info[key])
    print(f"  AMP used: {', '.join(map(str, amps))}")
