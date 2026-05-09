import time
import torch
from the_well.data.datamodule import WellDataModule

def bench_dataloader():
    dm = WellDataModule(
        well_base_path="C:/Users/simon/Documents/GitHub/BachelorThesis/datasets",
        well_dataset_name="pdebench_swe",
        batch_size=16,
        data_workers=4
    )
    train_dl = dm.train_dataloader()
    
    start = time.time()
    for i, batch in enumerate(train_dl):
        if i == 10:
            break
    print(f"Time for 10 batches: {time.time() - start:.3f} s")

if __name__ == "__main__":
    bench_dataloader()
