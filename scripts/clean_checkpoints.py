import os
from pathlib import Path

# Path to your experiments directory relative to project root
BASE_DIR = Path("the_well/benchmark/experiments")

def cleanup_recent_checkpoints(dry_run=True):
    if not BASE_DIR.exists():
        print(f"Directory {BASE_DIR} does not exist.")
        return

    total_size = 0
    files_to_delete = list(BASE_DIR.rglob("recent.pt"))
    
    if not files_to_delete:
        print("No recent.pt files found.")
        return

    print(f"Found {len(files_to_delete)} 'recent.pt' files.")
    
    for file_path in files_to_delete:
        try:
            size = file_path.stat().st_size
            total_size += size
            if not dry_run:
                file_path.unlink()
                print(f"Deleted: {file_path} ({size / 1024 / 1024:.2f} MB)")
            else:
                print(f"Found (dry run): {file_path} ({size / 1024 / 1024:.2f} MB)")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print("-" * 30)
    status = "Would save" if dry_run else "Saved"
    print(f"{status}: {total_size / 1024 / 1024:.2f} MB of space.")
    
    if dry_run:
        print("\nTo actually delete these files, run the script with the --delete flag:")
        print("python scripts/clean_checkpoints.py --delete")

if __name__ == "__main__":
    import sys
    do_delete = "--delete" in sys.argv
    cleanup_recent_checkpoints(dry_run=not do_delete)
