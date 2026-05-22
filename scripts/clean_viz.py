import os
import sys
import shutil
import re
from pathlib import Path

def clean_viz_folders(base_dir="the_well/benchmark/experiments"):
    """
    Deletes all elements inside 'viz' (or 'vis') folders except those 
    marked as test epochs (epoch number >= 1000).
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Error: Base path '{base_path}' does not exist.")
        return

    # Find all 'viz' or 'vis' folders
    viz_folders = []
    for root, dirs, files in os.walk(base_path):
        for d in dirs:
            if d in ['viz', 'vis']:
                viz_folders.append(Path(root) / d)

    print(f"Found {len(viz_folders)} viz/vis folders in '{base_path}'.")

    for viz_dir in viz_folders:
        print(f"Processing '{viz_dir}'...")
        
        # Traverse bottom-up so we can safely remove empty directories
        for root, dirs, files in os.walk(viz_dir, topdown=False):
            # 1. Process files
            for file in files:
                file_path = Path(root) / file
                
                # Check if it has an epoch number
                match = re.search(r'epoch_?(\d+)', file)
                is_test_epoch = False
                if match:
                    epoch_num = int(match.group(1))
                    if epoch_num >= 1000:
                        is_test_epoch = True
                
                # Delete if not a test epoch file
                if not is_test_epoch:
                    try:
                        file_path.unlink()
                    except Exception as e:
                        print(f"  Error deleting file {file_path}: {e}")

            # 2. Process directories
            for dir_name in dirs:
                dir_path = Path(root) / dir_name
                
                # Check if the directory itself has an explicit epoch number
                match = re.search(r'epoch_?(\d+)', dir_name)
                is_test_epoch = False
                if match:
                    epoch_num = int(match.group(1))
                    if epoch_num >= 1000:
                        is_test_epoch = True
                
                if match and not is_test_epoch:
                    # It's an explicit epoch directory for < 1000, delete it entirely
                    try:
                        shutil.rmtree(dir_path)
                    except Exception as e:
                        print(f"  Error deleting directory {dir_path}: {e}")
                else:
                    # If it's a structural directory (e.g. dataset name) or a test epoch directory,
                    # attempt to remove it if it has become empty after deleting files.
                    try:
                        dir_path.rmdir()
                    except OSError:
                        pass # Not empty, keep it

    print("Cleanup completed.")

if __name__ == "__main__":
    # Allow overriding the base directory via command line argument
    base_dir = "the_well/benchmark/experiments"
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    
    clean_viz_folders(base_dir)
