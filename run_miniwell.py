# run_miniwell.py
from the_well.data.datasets import WellDataset
from the_well.data.miniwell import create_mini_well

base_path = r"C:\Users\simon\Documents\GitHub\BachelorThesis\datasets"
output_path = r"C:\Users\simon\Documents\GitHub\BachelorThesis\datasets\miniwell"

split_trajectories = {
    "train": 120,
    "valid": 20,
    "test": 20,
}

for split, max_traj in split_trajectories.items():
    dataset = WellDataset(
        well_base_path=base_path,
        well_dataset_name="gray_scott_reaction_diffusion",
        well_split_name=split,
    )
    create_mini_well(
        dataset=dataset,
        output_base_path=output_path,
        spatial_downsample_factor=1,
        time_downsample_factor=2,
        max_trajectories=max_traj,
        split=split,
    )
    print(f"Done: {split}")
