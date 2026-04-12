import wandb
import datetime

api = wandb.Api()
entity = "simon-danninger-johannes-kepler-universit-t-linz"
project = "bsc_data_test"

print(f"Fetching runs for {entity}/{project}...")
runs = api.runs(f"{entity}/{project}")

print("Failed runs from the last couple of days:")
for r in runs:
    if r.state == "failed":
        dataset = r.config.get("data", {}).get("well_dataset_name", "Unknown")
        model = r.config.get("model", {}).get("_target_", "Unknown").split('.')[-1]
        
        # We only care about recent failed runs. Let's just print the ones related to pdebench_1d_burgers and pdebench_swe
        if "burgers" in dataset.lower() or "swe" in dataset.lower():
            print(f"--- Run ID: {r.id}, Name: {r.name}, Dataset: {dataset}, Model: {model} ---")
            # Fetch the logs (stdout/stderr) for the failed run
            try:
                # wandb run history only contains metrics. We need the output.log file
                files = r.files()
                for file in files:
                    if file.name == "output.log":
                        file.download(replace=True)
                        with open("output.log", "r", encoding="utf-8") as f:
                            lines = f.readlines()
                            # Print last 20 lines to see the error
                            print("Last 20 lines of output.log:")
                            print("".join(lines[-20:]))
                        break
            except Exception as e:
                print(f"Error fetching logs for {r.id}: {e}")
            print("\n")
