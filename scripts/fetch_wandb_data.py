import wandb
import json
import os

entity = "simon-danninger-johannes-kepler-universit-t-linz"
project = "bsc_data_test"

api = wandb.Api()
runs = api.runs(f"{entity}/{project}")

run_data = []

# Map common acronyms to full dataset names
name_to_dataset = {
    "TRL": "turbulent_radiative_layer_2D",
    "SWE": "pdebench_swe",
    "GRD": "gray_scott_reaction_diffusion",
    "RB": "rayleigh_benard",
    "B1D": "pdebench_burgers_1D"
}

for r in runs:
    if r.state != "finished":
        continue
    
    config = r.config
    summary = r.summary
    
    # Identify techniques
    temporal_bundle_size = config.get("temporal_bundle_size", 1)
    trainer_cfg = config.get("trainer", {})
    noise_injection = trainer_cfg.get("noise_injection", False)
    pushforward = trainer_cfg.get("pushforward", False)
    
    technique = "normal"
    if temporal_bundle_size > 1:
        technique = f"temporal_bundling_K={temporal_bundle_size}"
    if pushforward:
        technique = "pushforward"
    if noise_injection:
        technique = "noise_injection"
    
    if pushforward and temporal_bundle_size > 1:
        technique = f"TB_K={temporal_bundle_size}_PF"

    # Dataset
    dataset = config.get("data", {}).get("dataset_name")
    if not dataset:
        # Try to infer from name
        for acronym, full_name in name_to_dataset.items():
            if acronym in r.name:
                dataset = full_name
                break
    if not dataset:
        # Check if it's in the run name as lower case
        for full_name in name_to_dataset.values():
            if full_name in r.name.lower():
                dataset = full_name
                break
    if not dataset:
        # Final attempt: search all summary keys for dataset names
        for k in summary.keys():
            for full_name in name_to_dataset.values():
                if full_name in k:
                    dataset = full_name
                    break
            if dataset: break
            
    # Model
    model = config.get("model", {}).get("model_name")
    if not model:
        if "FNO" in r.name: model = "FNO"
        elif "UNetClassic" in r.name: model = "UNetClassic"
        elif "UNetConvNeXt" in r.name: model = "UNetConvNeXt"
        elif "ViT" in r.name: model = "ViT"

    # Seed
    seed = config.get("seed")
    
    # Metrics
    metrics = {}
    if dataset:
        prefixes = [f"rollout_test_{dataset}/", f"rollout_valid_{dataset}/", f"test_{dataset}/", f"valid_{dataset}/"]
        
        important_fields = ["full", "density", "velocity_x", "velocity_y", "pressure", "height", "velocity"]
        important_metrics = [
            "VRMSE_T=all",
            "NRMSE_T=all",
            "valid_rollout_length",
            "correlation_time",
            "nrmse_auc"
        ]
        
        for pref in prefixes:
            for field in important_fields:
                for met in important_metrics:
                    key = f"{pref}{field}_{met}"
                    if key in summary:
                        # Store with a cleaner key
                        metrics[f"{field}_{met}"] = summary[key]
            if metrics: break # Found some metrics with this prefix

    run_info = {
        "id": r.id,
        "name": r.name,
        "dataset": dataset or "unknown",
        "model": model or "unknown",
        "technique": technique,
        "seed": seed,
        "metrics": metrics,
        "config": {
            "batch_size": trainer_cfg.get("batch_size"),
            "epochs": trainer_cfg.get("epochs"),
            "temporal_bundle_size": temporal_bundle_size,
            "noise_injection": noise_injection,
            "pushforward": pushforward
        }
    }
    run_data.append(run_info)

print(json.dumps(run_data, indent=2))
