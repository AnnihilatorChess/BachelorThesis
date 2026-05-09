import logging
import sys
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

from the_well.benchmark.trainer.training import Trainer
from the_well.data.datamodule import WellDataModule
from the_well.data.data_formatter import DefaultChannelsFirstFormatter
from the_well.benchmark.models.cno import CNO
import torch

def run():
    dm = WellDataModule(
        well_base_path="C:/Users/simon/Documents/GitHub/BachelorThesis/datasets",
        well_dataset_name="pdebench_swe",
        batch_size=16,
        data_workers=4
    )
    
    batch = next(iter(dm.train_dataloader()))
    dim_in = batch['input_fields'].shape[-1]
    dim_out = batch['output_fields'].shape[-1]
    
    model = CNO(
        dim_in=dim_in,
        dim_out=dim_out,
        n_spatial_dims=2,
        spatial_resolution=(128, 128),
        N_layers=3,     # BASELINE config
        N_res=4,        # BASELINE config
        N_res_neck=4,
        channel_multiplier=32
    ).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    class DummyLoss(torch.nn.Module):
        def forward(self, p, t, m):
            return torch.nn.functional.mse_loss(p, t)
    loss_fn = DummyLoss()

    trainer = Trainer(
        checkpoint_folder="outputs/test",
        artifact_folder="outputs/test",
        viz_folder="outputs/test",
        formatter="channels_first",
        model=model,
        datamodule=dm,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=1,
        checkpoint_frequency=0,
        val_frequency=0,
        rollout_val_frequency=0,
        max_rollout_steps=1,
        short_validation_length=0,
        make_rollout_videos=False,
        num_time_intervals=1
    )
    
    trainer.formatter = DefaultChannelsFirstFormatter(dm.train_dataset.metadata)
    
    orig_train_one_epoch = trainer.train_one_epoch
    def train_one_epoch(epoch, dl):
        trainer._current_epoch = epoch
        trainer.model.train()
        import time
        start_time = time.time()
        batch_start = time.time()
        
        for i, batch in enumerate(dl):
            if i >= 10: break
            batch_time = time.time() - batch_start
            
            y_pred, y_ref = trainer.rollout_model(trainer.model, batch, trainer.formatter)
            forward_time = time.time() - batch_start - batch_time
            
            loss = trainer.loss_fn(y_pred, y_ref, trainer.dset_metadata).mean()
            trainer.grad_scaler.scale(loss).backward()
            trainer.grad_scaler.step(trainer.optimizer)
            trainer.grad_scaler.update()
            trainer.optimizer.zero_grad()
            
            backward_time = time.time() - batch_start - forward_time - batch_time
            total_time = time.time() - batch_start
            
            print(f"Batch {i}: total {total_time:.4f}s, dataload {batch_time:.4f}s, forward {forward_time:.4f}s, backward {backward_time:.4f}s")
            batch_start = time.time()
            
        return 0.0, {}
    
    trainer.train_one_epoch = train_one_epoch
    trainer.train_one_epoch(1, dm.train_dataloader())

if __name__ == "__main__":
    run()
