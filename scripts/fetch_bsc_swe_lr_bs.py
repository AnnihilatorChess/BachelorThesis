import wandb
import json

api = wandb.Api()
entity = "simon-danninger-johannes-kepler-universit-t-linz"
project = "bsc_SWE"

runs = api.runs(f"{entity}/{project}")

results = []

for r in runs:
    config = r.config
    
    # Try to extract LR and Batch Size
    # They could be in trainer, optimizer, etc.
    batch_size = None
    lr = None
    
    # Extract Model
    model = config.get("model", {}).get("model_name", config.get("model", {}).get("_target_", "Unknown"))
    if "FNO" in r.name: model = "FNO"
    elif "UNetClassic" in r.name: model = "UNetClassic"
    elif "UNetConvNeXt" in r.name: model = "UNetConvNeXt"
    elif "ViT" in r.name: model = "ViT"
    
    # Extract Dataset
    dataset = config.get("data", {}).get("dataset_name", config.get("data", {}).get("well_dataset_name", "Unknown"))
    
    # Find batch size
    trainer_cfg = config.get("trainer", {})
    if "batch_size" in trainer_cfg:
        batch_size = trainer_cfg["batch_size"]
    elif "batch_size" in config:
        batch_size = config["batch_size"]
        
    # Find learning rate
    optim_cfg = config.get("optimizer", {})
    if "lr" in optim_cfg:
        lr = optim_cfg["lr"]
    elif "learning_rate" in optim_cfg:
        lr = optim_cfg["learning_rate"]
    elif "lr" in trainer_cfg:
        lr = trainer_cfg["lr"]
    elif "learning_rate" in trainer_cfg:
        lr = trainer_cfg["learning_rate"]
    elif "lr" in config:
        lr = config["lr"]
    elif "learning_rate" in config:
        lr = config["learning_rate"]
        
    if lr is None and "scheduler" in config:
         # maybe it's in a scheduler config?
         pass

    results.append({
        "name": r.name,
        "state": r.state,
        "model": model,
        "dataset": dataset,
        "batch_size": batch_size,
        "lr": lr
    })

print(json.dumps(results, indent=2))
