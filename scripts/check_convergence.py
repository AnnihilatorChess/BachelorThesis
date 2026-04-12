import wandb
import json

entity = "simon-danninger-johannes-kepler-universit-t-linz"
project = "bsc_data_test"
api = wandb.Api()

run_ids = ["jrjj4433", "vfy6bvy9", "ucxorpoy", "6n8sqtro", "wh6cbupo", "i0vmwc5x"]

for rid in run_ids:
    r = api.run(f"{entity}/{project}/{rid}")
    # scan_history is better for getting all values
    history = list(r.scan_history(keys=["train_loss", "epoch"]))
    
    losses = [row.get("train_loss") for row in history if row.get("train_loss") is not None]
    
    if not losses:
        # Maybe it's not logged every step, try to find ANY row with it
        print(f"Run {rid}: No train_loss found in scanned history with specified keys.")
        continue

    first_losses = losses[:5]
    last_losses = losses[-5:]
    
    print(f"Run {rid} ({r.name}):")
    print(f"  First 5 train losses: {first_losses}")
    print(f"  Last 5 train losses: {last_losses}")
    print(f"  Total epochs logged: {len(losses)}")
