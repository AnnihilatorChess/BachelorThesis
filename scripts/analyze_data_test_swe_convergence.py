import wandb

api = wandb.Api()
entity = "simon-danninger-johannes-kepler-universit-t-linz"
project = "bsc_data_test"

runs = api.runs(f"{entity}/{project}")

print(f"{'Run Name':<40} | {'Model':<15} | {'BS':<4} | {'Epochs':<6} | {'Min Loss':<10} | {'Final Loss':<10} | {'Loss@10':<10} | {'Loss@50':<10}")
print("-" * 120)

for r in runs:
    if r.state != "finished":
        continue
        
    dataset = r.config.get("data", {}).get("dataset_name", r.config.get("data", {}).get("well_dataset_name", ""))
    
    if "swe" not in r.name.lower() and "swe" not in dataset.lower():
        continue
        
    model = r.config.get("model", {}).get("model_name", r.config.get("model", {}).get("_target_", "Unknown"))
    if "FNO" in r.name: model = "FNO"
    elif "UNetClassic" in r.name: model = "UNetClassic"
    elif "UNetConvNeXt" in r.name: model = "UNetConvNeXt"
    elif "CNO" in r.name: model = "CNO"
    elif "ViT" in r.name: model = "ViT"
    
    # Extract batch size
    batch_size = None
    trainer_cfg = r.config.get("trainer", {})
    if "batch_size" in trainer_cfg:
        batch_size = trainer_cfg["batch_size"]
    elif "batch_size" in r.config:
        batch_size = r.config["batch_size"]
    elif "batch_size" in r.config.get("data", {}):
        batch_size = r.config.get("data", {})["batch_size"]
        
    bs_str = str(batch_size) if batch_size is not None else "N/A"

    # fetch history
    history = r.history(keys=["epoch", "train_loss"], pandas=False, samples=500)
    
    if len(history) == 0:
        continue
        
    epochs = [row.get("epoch") for row in history if "train_loss" in row and row.get("epoch") is not None]
    train_losses = [row.get("train_loss") for row in history if "train_loss" in row and row.get("epoch") is not None]
    
    if len(epochs) == 0:
        continue
        
    num_epochs = int(max(epochs))
    min_loss = min(train_losses)
    final_loss = sum(train_losses[-5:]) / len(train_losses[-5:]) if len(train_losses) >= 5 else train_losses[-1]
    
    def get_loss_at(target_epoch):
        if len(epochs) == 0: return float('nan')
        closest_idx = min(range(len(epochs)), key=lambda i: abs(epochs[i] - target_epoch))
        return train_losses[closest_idx]

    loss_at_10 = get_loss_at(10) if num_epochs >= 10 else float('nan')
    loss_at_50 = get_loss_at(50) if num_epochs >= 50 else float('nan')

    print(f"{r.name[:38]:<40} | {model[:13]:<15} | {bs_str:<4} | {num_epochs:<6} | {min_loss:<10.5f} | {final_loss:<10.5f} | {loss_at_10:<10.5f} | {loss_at_50:<10.5f}")
