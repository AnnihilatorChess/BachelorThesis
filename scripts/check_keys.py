import wandb
import json

entity = "simon-danninger-johannes-kepler-universit-t-linz"
project = "bsc_data_test"
api = wandb.Api()

run_id = "jrjj4433"
r = api.run(f"{entity}/{project}/{run_id}")

# Just look at the first row of history to see keys
history = list(r.scan_history(page_size=1))
if history:
    print(f"Keys in history: {list(history[0].keys())}")
else:
    print("No history found")

# Also check summary keys
print(f"Keys in summary: {list(r.summary.keys())}")
