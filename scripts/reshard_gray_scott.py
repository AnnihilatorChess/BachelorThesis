import h5py
import numpy as np
import os
from pathlib import Path

def reshard_split(input_dir, output_dir, target_file_count=32):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(list(input_dir.glob("*.hdf5")) + list(input_dir.glob("*.h5")))
    if not input_files:
        print(f"No files found in {input_dir}")
        return

    # 1. Count total trajectories
    total_trajectories = 0
    file_traj_counts = []
    for f_path in input_files:
        with h5py.File(f_path, "r") as f:
            n = f.attrs["n_trajectories"]
            total_trajectories += n
            file_traj_counts.append(n)
    
    print(f"Total trajectories in {input_dir.name}: {total_trajectories}")
    
    # 2. Calculate trajectories per new file
    traj_per_file = total_trajectories // target_file_count
    remainder = total_trajectories % target_file_count
    
    output_counts = [traj_per_file + (1 if i < remainder else 0) for i in range(target_file_count)]
    output_counts = [c for c in output_counts if c > 0] # Filter out 0s if any
    
    # 3. Resharding logic
    in_f_idx = 0
    in_f = h5py.File(input_files[in_f_idx], "r")
    in_f_traj_start = 0
    
    for out_idx, target_n in enumerate(output_counts):
        out_path = output_dir / f"shard_{out_idx:03d}.hdf5"
        print(f"Creating {out_path} with {target_n} trajectories...")
        
        with h5py.File(out_path, "w") as out_f:
            # Copy global attributes from the first input file
            for attr_name, attr_val in in_f.attrs.items():
                if attr_name != "n_trajectories":
                    out_f.attrs[attr_name] = attr_val
            out_f.attrs["n_trajectories"] = target_n
            
            # Recreate group structure
            for group_name in ["boundary_conditions", "dimensions", "scalars", "t0_fields", "t1_fields", "t2_fields"]:
                out_g = out_f.create_group(group_name)
                in_g = in_f[group_name]
                for attr_name, attr_val in in_g.attrs.items():
                    out_g.attrs[attr_name] = attr_val
                
                # Copy datasets within groups
                for ds_name in in_g.keys():
                    in_ds = in_g[ds_name]
                    is_sample_varying = in_ds.attrs.get("sample_varying", False)
                    
                    if not is_sample_varying:
                        # Simple copy for non-varying data
                        in_g.copy(ds_name, out_g)
                    else:
                        # Initialize empty dataset for sample-varying data
                        shape = list(in_ds.shape)
                        shape[0] = target_n
                        out_ds = out_g.create_dataset(
                            ds_name, 
                            shape=tuple(shape), 
                            dtype=in_ds.dtype,
                            chunks=True # Let h5py handle default chunking
                        )
                        # Copy attributes
                        for attr_name, attr_val in in_ds.attrs.items():
                            out_ds.attrs[attr_name] = attr_val

            # Fill sample-varying datasets
            curr_out_traj = 0
            while curr_out_traj < target_n:
                available_in_f = file_traj_counts[in_f_idx] - in_f_traj_start
                to_copy = min(available_in_f, target_n - curr_out_traj)
                
                # Copy from in_f[in_f_traj_start : in_f_traj_start + to_copy]
                # to out_f[curr_out_traj : curr_out_traj + to_copy]
                for group_name in ["boundary_conditions", "dimensions", "scalars", "t0_fields", "t1_fields", "t2_fields"]:
                    in_g = in_f[group_name]
                    out_g = out_f[group_name]
                    for ds_name in in_g.keys():
                        in_ds = in_g[ds_name]
                        if in_ds.attrs.get("sample_varying", False):
                            out_g[ds_name][curr_out_traj : curr_out_traj + to_copy] = in_ds[in_f_traj_start : in_f_traj_start + to_copy]
                
                curr_out_traj += to_copy
                in_f_traj_start += to_copy
                
                if in_f_traj_start >= file_traj_counts[in_f_idx]:
                    in_f.close()
                    in_f_idx += 1
                    if in_f_idx < len(input_files):
                        in_f = h5py.File(input_files[in_f_idx], "r")
                        in_f_traj_start = 0
                    else:
                        break

    if not in_f.id.valid: # Close the last file if not already closed
        pass
    else:
        in_f.close()

if __name__ == "__main__":
    base_data_path = "/system/user/publicdata/the_well/datasets/real_size/datasets/gray_scott_reaction_diffusion/data"
    output_base_path = "/system/user/publicdata/the_well/datasets/real_size/datasets/gray_scott_reaction_diffusion_resharded/data"
    
    for split in ["train", "valid", "test"]:
        print(f"\nProcessing split: {split}")
        reshard_split(os.path.join(base_data_path, split), os.path.join(output_base_path, split), target_file_count=32)
