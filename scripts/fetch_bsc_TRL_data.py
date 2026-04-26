import wandb
import json
import os

api = wandb.Api()
entity = "simon-danninger-johannes-kepler-universit-t-linz"
project = "bsc_TRL"

runs = api.runs(f"{entity}/{project}")

data = []

for r in runs:
    if r.state not in ["finished", "running", "failed", "crashed"]:
        continue
        
    config = r.config
    summary = dict(r.summary)
    
    # Filter summary to keep only numerical/string values to avoid serialization issues
    clean_summary = {}
    for k, v in summary.items():
        if isinstance(v, (int, float, str, bool)):
            clean_summary[k] = v
            
    # Try to get hardware info from wandb-metadata.json if possible (we'll fetch it for one run to see)
    gpu = "Unknown"
    try:
        # We can check if hardware is in config
        if "system" in config:
             gpu = config["system"].get("gpu", "Unknown")
    except:
        pass

    run_info = {
        "id": r.id,
        "name": r.name,
        "group": r.group,
        "state": r.state,
        "config": config,
        "summary": clean_summary,
        "history_keys": [k for k in clean_summary.keys()]
    }
    data.append(run_info)

with open("bsc_TRL_dump.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"Dumped {len(data)} runs to bsc_TRL_dump.json")
