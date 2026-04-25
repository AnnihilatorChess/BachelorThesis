import logging
import os
import time
from typing import Callable, Optional
import random

import torch
import torch.distributed as dist
import tqdm
import wandb
from torch.utils.data import DataLoader

from the_well.benchmark.metrics import (
    long_time_metrics,
    make_video,
    plot_all_time_metrics,
    validation_metric_suite,
    validation_plots,
    CorrelationTime,
    ErrorGrowthRate,
    ValidRolloutLength,
)
from the_well.data.data_formatter import (
    DefaultChannelsFirstFormatter,
    DefaultChannelsLastFormatter,
)
from the_well.data.datamodule import AbstractDataModule
from the_well.data.datasets import DeltaWellDataset
from the_well.data.utils import flatten_field_names

logger = logging.getLogger(__name__)


# Checkpoints saved during training, keyed by the validation metric that selected them.
# - one_step_*: tracked on short validation (every val_frequency epochs)
# - rollout_*:  tracked on rollout validation (every rollout_val_frequency epochs)
CHECKPOINT_METRICS = {
    "one_step_vrmse": "best_one_step_vrmse.pt",
    "one_step_nrmse": "best_one_step_nrmse.pt",
    "rollout_vrmse": "best_rollout_vrmse.pt",
    "rollout_nrmse": "best_rollout_nrmse.pt",
}


def param_norm(parameters):
    with torch.no_grad():
        total_norm = 0
        for p in parameters:
            total_norm += p.pow(2).sum().item()
        return total_norm**0.5


class Trainer:
    def __init__(
        self,
        checkpoint_folder: str,
        artifact_folder: str,
        viz_folder: str,
        formatter: str,
        model: torch.nn.Module,
        datamodule: AbstractDataModule,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        # validation_suite: list,
        epochs: int,
        checkpoint_frequency: int,
        val_frequency: int,
        rollout_val_frequency: int,
        max_rollout_steps: int,
        short_validation_length: float,
        make_rollout_videos: bool,
        num_time_intervals: int,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device=torch.device("cuda"),
        is_distributed: bool = False,
        enable_amp: bool = False,
        amp_type: str = "float16",  # bfloat not supported in FFT
        checkpoint_path: str = "",
        pushforward: bool = False,
        pushforward_warmup_fraction: float = 0.2,
        pushforward_final_probs: list = (0.4, 0.2, 0.2, 0.2),
        noise_injection: bool = False,
        noise_std: float = 0.003,
        noise_anneal: bool = True,
        temporal_bundle_size: int = 1,
        extended_metrics: bool = True,
        valid_rollout_threshold: float = 0.2,
        correlation_time_threshold: float = 0.8,
        temporal_loss_boundaries=None,
    ):
        """
        Class in charge of the training loop. It performs train, validation and test.

        Args:
            checkpoint_folder:
                Path to folder used for storing checkpoints.
            artifact_folder:
                Path to folder used for storing artifacts.
            viz_folder:
                Path to folder used for storing visualizations.
            model:
                The Pytorch model to train
            datamodule:
                A datamodule that provides dataloaders for each split (train, valid, and test)
            optimizer:
                A Pytorch optimizer to perform the backprop (e.g. Adam)
            loss_fn:
                A loss function that evaluates the model predictions to be used for training
            epochs:
                Number of epochs to train the model.
                One epoch correspond to a full loop over the datamodule's training dataloader
            checkpoint_frequency:
                The frequency in terms of number of epochs to save the model checkpoint
            val_frequency:
                The frequency in terms of number of epochs to perform the validation
            rollout_val_frequency:
                The frequency in terms of number of epochs to perform the rollout validation
            max_rollout_steps:
                The maximum number of timesteps to rollout the model
            short_validation_length:
                The number of batches to use for quick intermediate validation during training
            make_rollout_videos:
                A boolean flag to trigger the creation of videos during long rollout validation
            num_time_intervals:
                The number of time intervals to split the loss over
            lr_scheduler:
                A Pytorch learning rate scheduler to update the learning rate during training
            device:
                A Pytorch device (e.g. "cuda" or "cpu")
            is_distributed:
                A boolean flag to trigger DDP training
            enable_amp:
                A boolean flag to enable automatic mixed precision training
            amp_type:
                The type of automatic mixed precision to use. Can be "float16" or "bfloat16"
            checkpoint_path:
                The path to the model checkpoint to load. If empty, the model is trained from scratch.
        """
        self.starting_epoch = 1  # Gets overridden on resume
        self.checkpoint_folder = checkpoint_folder
        self.artifact_folder = artifact_folder
        self.viz_folder = viz_folder
        self.device = device
        self.model = model
        self.datamodule = datamodule
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.is_delta = isinstance(datamodule.train_dataset, DeltaWellDataset)
        self.validation_suite = validation_metric_suite + [self.loss_fn]
        self.max_epoch = epochs
        self.checkpoint_frequency = checkpoint_frequency
        self.val_frequency = val_frequency
        self.rollout_val_frequency = rollout_val_frequency
        self.max_rollout_steps = max_rollout_steps
        self.short_validation_length = short_validation_length
        self.make_rollout_videos = make_rollout_videos
        self.num_time_intervals = num_time_intervals
        self.enable_amp = enable_amp
        self.amp_type = torch.bfloat16 if amp_type == "bfloat16" else torch.float16
        self.grad_scaler = torch.GradScaler(
            self.device.type, enabled=enable_amp and amp_type != "bfloat16"
        )
        self.is_distributed = is_distributed
        self.best_metrics = {k: float("inf") for k in CHECKPOINT_METRICS}
        self.starting_val_loss = float("inf")
        self.dset_metadata = self.datamodule.train_dataset.metadata
        self.pushforward = pushforward
        # warmup expressed as a fraction of total epochs so it scales with budget
        # (a 32-epoch Rayleigh-Bénard run and a 500-epoch TRL run both get ~20% warmup).
        self.pushforward_warmup_epochs = max(1, int(epochs * pushforward_warmup_fraction))
        self.pushforward_final_probs = pushforward_final_probs
        self.noise_injection = noise_injection
        self.noise_std = noise_std
        self.noise_anneal = noise_anneal
        self._current_epoch = 1
        self.bundle_size = temporal_bundle_size
        self.n_fields = self.dset_metadata.n_fields
        # Extended evaluation metrics (create fresh instances to avoid mutating module-level singletons)
        if extended_metrics:
            self.summary_suite = [
                ValidRolloutLength(threshold=valid_rollout_threshold),
                ErrorGrowthRate(),
                CorrelationTime(threshold=correlation_time_threshold),
            ]
        else:
            self.summary_suite = []
        # Store fixed boundaries if provided; None = compute relative at runtime
        self.temporal_loss_boundaries = list(temporal_loss_boundaries) if temporal_loss_boundaries is not None else None
        if self.datamodule.train_dataset.use_normalization:
            self.dset_norm = self.datamodule.train_dataset.norm
        if formatter == "channels_first_default":
            self.formatter = DefaultChannelsFirstFormatter(self.dset_metadata)
        elif formatter == "channels_last_default":
            self.formatter = DefaultChannelsLastFormatter(self.dset_metadata)
        if len(checkpoint_path) > 0:
            self.load_checkpoint(checkpoint_path)

    def save_model(self, epoch: int, validation_loss: float, output_path: str):
        """Save a full training-state checkpoint (used for recent.pt / resume)."""
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dit": self.optimizer.state_dict(),
                "scheduler_state_dict": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                "validation_loss": validation_loss,
                "best_metrics": self.best_metrics,
            },
            output_path,
        )

    def save_best_weights_only(
        self, epoch: int, metric_name: str, metric_value: float, output_path: str
    ):
        """Save model weights plus metadata identifying which metric this checkpoint is best for."""
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "metric_name": metric_name,
                "metric_value": metric_value,
            },
            output_path,
        )

    def load_checkpoint(self, checkpoint_path: str):
        """Load the model checkpoint."""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        state_dict = checkpoint["model_state_dict"]

        # Remove 'model.' or 'module.' prefix if it exists
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[6:] if k.startswith('model.') else k
            name = name[7:] if name.startswith('module.') else name
            new_state_dict[name] = v

        if self.model is not None:
            self.model.load_state_dict(new_state_dict)
        if self.optimizer is not None and "optimizer_state_dit" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dit"])
        if self.lr_scheduler is not None and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
            self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "best_metrics" in checkpoint and isinstance(checkpoint["best_metrics"], dict):
            # Merge so any newly added metric keys keep their inf default.
            for k, v in checkpoint["best_metrics"].items():
                if k in self.best_metrics:
                    self.best_metrics[k] = v
            logger.info(f"Restored best_metrics: {self.best_metrics}")
        self.starting_val_loss = checkpoint.get("validation_loss", float("inf"))
        self.starting_epoch = (
            checkpoint["epoch"] + 1
        )  # Saves after training loop, so start at next epoch

    def _update_best_checkpoints(
        self, epoch: int, metrics_dict: dict, scope: str
    ) -> None:
        """Save a new best_*.pt for each tracked metric that improved this epoch.

        scope="one_step" looks at keys under the short-validation prefix `valid_`,
        scope="rollout"  looks at keys under the rollout-validation prefix `rollout_valid_`.
        Unknown scope is a no-op.
        """
        ds = self.dset_metadata.dataset_name
        if scope == "one_step":
            candidates = {
                "one_step_vrmse": f"valid_{ds}/full_VRMSE_T=all",
                "one_step_nrmse": f"valid_{ds}/full_NRMSE_T=all",
            }
        elif scope == "rollout":
            candidates = {
                "rollout_vrmse": f"rollout_valid_{ds}/full_VRMSE_T=all",
                "rollout_nrmse": f"rollout_valid_{ds}/full_NRMSE_T=all",
            }
        else:
            return

        for metric_key, dict_key in candidates.items():
            if dict_key not in metrics_dict:
                logger.warning(
                    f"_update_best_checkpoints: key {dict_key} not found in metrics_dict"
                )
                continue
            value = float(metrics_dict[dict_key])
            if value < self.best_metrics[metric_key]:
                self.best_metrics[metric_key] = value
                out_path = os.path.join(
                    self.checkpoint_folder, CHECKPOINT_METRICS[metric_key]
                )
                self.save_best_weights_only(epoch, metric_key, value, out_path)
                logger.info(
                    f"Epoch {epoch}: new best {metric_key} = {value:.6f} -> {CHECKPOINT_METRICS[metric_key]}"
                )

    def _add_noise(self, x, epoch):
        """Add Gaussian noise to input tensor during training for distribution shift robustness."""
        if not self.noise_injection:
            return x
        if self.noise_anneal:
            std = self.noise_std * min(1.0, epoch / max(self.max_epoch * 0.5, 1))
        else:
            std = self.noise_std
        return x + torch.randn_like(x) * std

    def normalize(self, batch):
        if hasattr(self, "dset_norm") and self.dset_norm:
            batch["input_fields"] = self.dset_norm.normalize_flattened(
                batch["input_fields"], "variable"
            )
            if "constant_fields" in batch:
                batch["constant_fields"] = self.dset_norm.normalize_flattened(
                    batch["constant_fields"], "constant"
                )
        return batch

    def denormalize(self, batch, prediction):
        if hasattr(self, "dset_norm") and self.dset_norm:
            batch["input_fields"] = self.dset_norm.denormalize_flattened(
                batch["input_fields"], "variable"
            )
            if "constant_fields" in batch:
                batch["constant_fields"] = self.dset_norm.denormalize_flattened(
                    batch["constant_fields"], "constant"
                )

            # Delta denormalization is different than full denormalization
            if self.is_delta:
                prediction = self.dset_norm.delta_denormalize_flattened(
                    prediction, "variable"
                )
            else:
                prediction = self.dset_norm.denormalize_flattened(
                    prediction, "variable"
                )

        return batch, prediction

    def rollout_model(self, model, batch, formatter, train=True):
        """Rollout the model for as many steps as we have data for.

        Supports temporal bundling: when bundle_size > 1, the model predicts K
        timesteps at once, reducing the number of autoregressive steps.
        """
        inputs, y_ref = formatter.process_input(batch)
        rollout_steps = min(
            y_ref.shape[1], self.max_rollout_steps
        )  # Number of timesteps in target
        y_ref = y_ref[:, :rollout_steps]
        # NOTE: val/rollout_val datasets return physical (unnormalized) data by default,
        # so y_ref is already in physical space — do NOT denormalize it here.

        # Create a moving batch
        moving_batch = dict(batch)
        moving_batch["input_fields"] = moving_batch["input_fields"].to(self.device)
        if "constant_fields" in moving_batch:
            moving_batch["constant_fields"] = moving_batch["constant_fields"].to(
                self.device
            )

        K = self.bundle_size
        y_preds = []
        step = 0
        while step < rollout_steps:
            remaining = rollout_steps - step

            if not train:
                moving_batch = self.normalize(moving_batch)

            inputs, _ = formatter.process_input(moving_batch)
            inputs = [x.to(self.device) for x in inputs]

            # Add noise during training for distribution shift robustness
            if train and self.noise_injection:
                inputs = [self._add_noise(inputs[0], self._current_epoch)] + list(inputs[1:])

            y_pred = model(*inputs)

            # Unbundle: [B, K*C, H, W] -> [B, K, H, W, C]
            y_pred_bundle = formatter.process_output_unbundle(y_pred, K, self.n_fields)

            # Handle denormalization for validation.
            # For k_idx==0: denormalize both moving_batch and the prediction together
            # (original single-step behaviour, keeps moving_batch in physical units).
            # For k_idx>0: moving_batch is already in physical units, so only
            # denormalize the prediction to avoid applying batch denorm K times.
            if not train:
                y_pred_bundle = y_pred_bundle.clone()
                for k_idx in range(min(K, remaining)):
                    single_pred = y_pred_bundle[:, k_idx]
                    if k_idx == 0:
                        moving_batch, single_pred = self.denormalize(moving_batch, single_pred)
                    else:
                        if hasattr(self, "dset_norm") and self.dset_norm:
                            if self.is_delta:
                                single_pred = self.dset_norm.delta_denormalize_flattened(
                                    single_pred, "variable"
                                )
                            else:
                                single_pred = self.dset_norm.denormalize_flattened(
                                    single_pred, "variable"
                                )
                    y_pred_bundle[:, k_idx] = single_pred

            # Truncate if this is the last bundle and doesn't fill completely
            if remaining < K:
                y_pred_bundle = y_pred_bundle[:, :remaining]

            # Delta dataset handling
            if (not train) and self.is_delta:
                for k_idx in range(y_pred_bundle.shape[1]):
                    y_pred_bundle[:, k_idx] = (
                        moving_batch["input_fields"][:, -1, ...] + y_pred_bundle[:, k_idx]
                    )

            # Update moving batch for next iteration
            steps_taken = y_pred_bundle.shape[1]
            if step + steps_taken < rollout_steps:
                moving_batch["input_fields"] = torch.cat(
                    [moving_batch["input_fields"][:, steps_taken:], y_pred_bundle], dim=1
                )

            y_preds.append(y_pred_bundle)
            step += steps_taken

        y_pred_out = torch.cat(y_preds, dim=1)
        y_ref = y_ref[:, :y_pred_out.shape[1]].to(self.device)
        return y_pred_out, y_ref

    def temporal_split_losses(
        self, loss_values, temporal_loss_intervals, loss_name, dset_name, fname="full"
    ):
        new_losses = {}
        # Average over time interval
        new_losses[f"{dset_name}/{fname}_{loss_name}_T=all"] = loss_values.mean()
        # Don't compute sublosses if we only have one interval
        if len(temporal_loss_intervals) == 2:
            return new_losses
        # Break it down by time interval
        for k in range(len(temporal_loss_intervals) - 1):
            start_ind = temporal_loss_intervals[k]
            end_ind = temporal_loss_intervals[k + 1]
            time_str = f"{start_ind}:{end_ind}"
            loss_subset = loss_values[start_ind:end_ind].mean()
            new_losses[f"{dset_name}/{fname}_{loss_name}_T={time_str}"] = loss_subset
        return new_losses

    def split_up_losses(self, loss_values, loss_name, dset_name, field_names):
        new_losses = {}
        time_logs = {}
        time_steps = loss_values.shape[0]  # we already average over batch
        import numpy as np
        if self.temporal_loss_boundaries is not None:
            # Use fixed boundaries (e.g. to match The Well paper tables).
            # Clip any boundary that exceeds the actual rollout length, then append it.
            boundaries = [b for b in self.temporal_loss_boundaries if b < time_steps]
            temporal_loss_intervals = sorted(list(set(boundaries + [time_steps])))
        else:
            # Relative equal-width bins: safe default for any dataset / rollout length.
            boundaries = list(np.linspace(0, time_steps, self.num_time_intervals + 1, dtype=int))
            temporal_loss_intervals = sorted(list(set(boundaries)))

        # Split up losses by field
        for i, fname in enumerate(field_names):
            time_logs[f"{dset_name}/{fname}_{loss_name}_rollout"] = loss_values[
                :, i
            ].cpu()
            new_losses |= self.temporal_split_losses(
                loss_values[:, i], temporal_loss_intervals, loss_name, dset_name, fname
            )
        # Compute average over all fields
        new_losses |= self.temporal_split_losses(
            loss_values.mean(1), temporal_loss_intervals, loss_name, dset_name, "full"
        )
        time_logs[f"{dset_name}/full_{loss_name}_rollout"] = loss_values.mean(1).cpu()
        return new_losses, time_logs

    @torch.inference_mode()
    def validation_loop(
        self,
        dataloader: DataLoader,
        valid_or_test: str = "valid",
        full: bool = False,
        epoch: int = 0,
    ) -> float:
        """Run validation by looping over the dataloader."""
        self.model.eval()
        validation_loss = 0.0
        field_names = flatten_field_names(self.dset_metadata, include_constants=False)
        dset_name = self.dset_metadata.dataset_name
        loss_dict = {}
        time_logs = {}
        count = 0
        short_val_len = max(1, int(self.short_validation_length * len(dataloader)))
        denom = (
            len(dataloader)
            if full
            else min(short_val_len, len(dataloader))
        )
        with torch.autocast(
            self.device.type, enabled=self.enable_amp, dtype=self.amp_type
        ):
            for i, batch in enumerate(tqdm.tqdm(dataloader)):
                # Rollout for length of target
                y_pred, y_ref = self.rollout_model(
                    self.model, batch, self.formatter, train=False
                )
                assert y_ref.shape == y_pred.shape, (
                    f"Mismatching shapes between reference {y_ref.shape} and prediction {y_pred.shape}"
                )
                # Go through losses
                for loss_fn in self.validation_suite:
                    # Mean over batch and time per field
                    loss = loss_fn(y_pred, y_ref, self.dset_metadata)
                    # Some losses return multiple values for efficiency
                    if not isinstance(loss, dict):
                        loss = {loss_fn.__class__.__name__: loss}
                    # Split the losses and update the logging dictionary
                    for k, v in loss.items():
                        sub_loss = v.mean(0)
                        new_losses, new_time_logs = self.split_up_losses(
                            sub_loss, k, dset_name, field_names
                        )
                        # TODO get better way to include spectral error.
                        if k in long_time_metrics or "spectral_error" in k:
                            for key, val in new_time_logs.items():
                                if key in time_logs:
                                    time_logs[key] += val / denom
                                else:
                                    time_logs[key] = val / denom

                        for loss_name, loss_value in new_losses.items():
                            loss_dict[loss_name] = (
                                loss_dict.get(loss_name, 0.0) + loss_value / denom
                            )
                # Summary metrics — only meaningful over long rollouts (full=True)
                for summary_fn in self.summary_suite if full else []:
                    summary = summary_fn(y_pred, y_ref, self.dset_metadata)
                    for metric_name, values in summary.items():
                        # values: [C], already batch-averaged; accumulate over dataloader batches
                        for fi, fname in enumerate(field_names):
                            if fi < values.shape[0]:
                                key = f"{dset_name}/{fname}_{metric_name}"
                                loss_dict[key] = loss_dict.get(key, 0.0) + values[fi] / denom
                        key_full = f"{dset_name}/full_{metric_name}"
                        loss_dict[key_full] = loss_dict.get(key_full, 0.0) + values.mean() / denom
                count += 1
                if not full and count >= short_val_len:
                    break

        logger.info(f"Validation loop {valid_or_test} (epoch {epoch}) used {count} batches.")

        # Last batch plots - too much work to combine from batches
        for plot_fn in validation_plots:
            plot_fn(y_pred, y_ref, self.dset_metadata, self.viz_folder, epoch)
        if y_ref.shape[1] > 1:
            # Only plot if we have more than one timestep, but then track loss over timesteps
            plot_all_time_metrics(time_logs, self.dset_metadata, self.viz_folder, epoch)
            if self.make_rollout_videos:
                # Make_video expects T x H [x W x D] C data so select out the batch dim
                make_video(
                    y_pred[0], y_ref[0], self.dset_metadata, self.viz_folder, epoch
                )

        if self.is_distributed:
            for k, v in loss_dict.items():
                dist.all_reduce(loss_dict[k], op=dist.ReduceOp.AVG)
        validation_loss = loss_dict[
            f"{dset_name}/full_{self.loss_fn.__class__.__name__}_T=all"
        ].item()
        loss_dict = {f"{valid_or_test}_{k}": v.item() for k, v in loss_dict.items()}
        # Namespace param_norm so repeated validation_loop calls in one step (e.g. per-checkpoint
        # test eval) don't overwrite each other in the merged wandb log.
        loss_dict[f"{valid_or_test}_param_norm"] = param_norm(self.model.parameters())
        return validation_loss, loss_dict

    def get_pushforward_probs(self, epoch: int) -> list:
        """Linearly interpolate between starting_probs and final_probs with warmup"""
        warmup_epochs = self.pushforward_warmup_epochs
        starting_probs = [1, 0, 0 , 0]
        if epoch < warmup_epochs:
            return [1, 0, 0, 0]

        progress = (epoch - warmup_epochs) / (self.max_epoch - warmup_epochs)
        progress = min(progress, 1)

        # linear interpolation, dont need to normalize
        current_weights = []
        for f_prob, s_prob in zip(self.pushforward_final_probs, starting_probs):
            current_prob = f_prob * progress + s_prob * (1 - progress)
            current_weights.append(current_prob)

        return current_weights


    def predict_next_step(self, input_fields, add_noise=False):
        """Predicts the next step of the model."""
        input_batch = {"input_fields": input_fields, "output_fields": torch.tensor([0.0])}

        # this converts from shape [B, T, H, W, C] to [B, (TC), H, W]
        inputs, _ = self.formatter.process_input(input_batch)

        # Add noise for distribution shift robustness (training only)
        if add_noise and self.noise_injection:
            inputs = (self._add_noise(inputs[0], self._current_epoch),) + inputs[1:]

        y_pred = self.model(*inputs)
        y_pred = self.formatter.process_output_channel_last(y_pred)
        y_pred = self.formatter.process_output_expand_time(y_pred)
        return y_pred  # shape: [B, 1, H, W, C]

    def predict_next_bundle(self, input_fields, add_noise=False):
        """Predicts the next K steps (temporal bundle) of the model."""
        input_batch = {"input_fields": input_fields, "output_fields": torch.tensor([0.0])}
        inputs, _ = self.formatter.process_input(input_batch)

        if add_noise and self.noise_injection:
            inputs = (self._add_noise(inputs[0], self._current_epoch),) + inputs[1:]

        y_pred = self.model(*inputs)

        # [B, K*C, H, W] -> [B, K, H, W, C]
        return self.formatter.process_output_unbundle(y_pred, self.bundle_size, self.n_fields)

    def train_one_epoch_pushforward(self, epoch: int, dataloader: DataLoader):
        """Train one epoch using the PushForward Trick.

        Supports temporal bundling: when bundle_size > 1, each pushforward step
        advances K timesteps and the target is a K-step bundle.
        """
        self._current_epoch = epoch
        self.model.train()
        epoch_loss = 0.0
        pushforward_steps_sum = 0
        train_logs = {}
        start_time = time.time()
        pushforward_probs = self.get_pushforward_probs(epoch)
        batch_size = len(dataloader)
        print_interval = max(batch_size // 100, 1)
        K = self.bundle_size

        for i, batch in enumerate(dataloader):
            pushforward_steps = random.choices([0, 1, 2, 3], pushforward_probs)[0]
            pushforward_steps_sum += pushforward_steps
            _, y_true_all = self.formatter.process_input(batch)

            current_input_fields = batch["input_fields"].to(self.device)

            if K == 1:
                # Original single-step pushforward (backward compatible)
                y_true = y_true_all[:, pushforward_steps].to(self.device)

                with torch.no_grad():
                    for s in range(pushforward_steps):
                        y_pred = self.predict_next_step(current_input_fields)
                        current_input_fields = torch.cat([current_input_fields[:, 1:], y_pred], dim=1)

                y_last = self.predict_next_step(current_input_fields, add_noise=True)
                loss = self.loss_fn(y_last, y_true.unsqueeze(1), self.dset_metadata).mean()
            else:
                # Bundle-level pushforward: each pushforward step advances K timesteps
                y_true = y_true_all[:, pushforward_steps * K:(pushforward_steps + 1) * K].to(self.device)

                with torch.no_grad():
                    for s in range(pushforward_steps):
                        y_pred_bundle = self.predict_next_bundle(current_input_fields)
                        current_input_fields = torch.cat([current_input_fields[:, K:], y_pred_bundle], dim=1)

                y_last_bundle = self.predict_next_bundle(current_input_fields, add_noise=True)
                loss = self.loss_fn(y_last_bundle, y_true, self.dset_metadata).mean()

            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.optimizer.zero_grad()
            epoch_loss += loss.item()

            if i % print_interval == 0:
                logger.info(
                    f"Epoch {epoch}, Batch {i + 1}/{len(dataloader)}: loss {loss.item()}, pushforward steps {pushforward_steps}"
                )
        train_logs["epoch_time"] = time.time() - start_time
        train_logs["time_per_train_iter"] = train_logs["epoch_time"] / len(dataloader)
        train_logs["train_loss"] = epoch_loss / len(dataloader)
        train_logs["avg_pushforward_steps"] = pushforward_steps_sum / len(dataloader)
        if self.lr_scheduler:
            self.lr_scheduler.step()
            train_logs["lr"] = self.lr_scheduler.get_last_lr()[-1]
        return train_logs["train_loss"], train_logs

    def train_one_epoch(self, epoch: int, dataloader: DataLoader) -> float:
        """Train the model for one epoch by looping over the dataloader."""
        self._current_epoch = epoch
        self.model.train()
        epoch_loss = 0.0
        train_logs = {}
        start_time = time.time()  # Don't need to sync this.
        batch_start = time.time()
        batch_size = len(dataloader)
        print_interval = max(batch_size // 100, 1)
        for i, batch in enumerate(dataloader):
            with torch.autocast(
                self.device.type, enabled=self.enable_amp, dtype=self.amp_type
            ):
                batch_time = time.time() - batch_start
                y_pred, y_ref = self.rollout_model(self.model, batch, self.formatter)
                forward_time = time.time() - batch_start - batch_time
                assert y_ref.shape == y_pred.shape, (
                    f"Mismatching shapes between reference {y_ref.shape} and prediction {y_pred.shape}"
                )
                loss = self.loss_fn(y_pred, y_ref, self.dset_metadata).mean()
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.optimizer.zero_grad()
            # Syncing for all reduce anyway so may as well compute synchornous metrics
            epoch_loss += loss.item() / len(dataloader)
            backward_time = time.time() - batch_start - forward_time - batch_time
            total_time = time.time() - batch_start
            if i % print_interval == 0:
                logger.info(
                    f"Epoch {epoch}, Batch {i + 1}/{len(dataloader)}: loss {loss.item()}, total_time {total_time}, batch time {batch_time}, forward time {forward_time}, backward time {backward_time}"
                )
            batch_start = time.time()
        train_logs["epoch_time"] = time.time() - start_time
        train_logs["time_per_train_iter"] = train_logs["epoch_time"] / len(dataloader)
        train_logs["train_loss"] = epoch_loss
        if self.lr_scheduler:
            self.lr_scheduler.step()
            train_logs["lr"] = self.lr_scheduler.get_last_lr()[-1]
        return epoch_loss, train_logs

    def train(self):
        """Run training, validation and test. The training is run for multiple epochs."""
        train_dataloader = self.datamodule.train_dataloader()
        val_dataloder = self.datamodule.val_dataloader()
        rollout_val_dataloader = self.datamodule.rollout_val_dataloader()
        test_dataloader = self.datamodule.test_dataloader()
        rollout_test_dataloader = self.datamodule.rollout_test_dataloader()
        val_loss = self.starting_val_loss

        for epoch in range(self.starting_epoch, self.max_epoch + 1):
            epoch_logs = {}
            # Distributed samplers need to be set for each epoch
            if self.is_distributed:
                train_dataloader.sampler.set_epoch(epoch)
            # Run training and log training results
            logger.info(f"Epoch {epoch}/{self.max_epoch}: starting training")
            if self.pushforward:
                train_loss, train_logs = self.train_one_epoch_pushforward(epoch, train_dataloader)
            else:
                train_loss, train_logs = self.train_one_epoch(epoch, train_dataloader)
            logger.info(
                f"Epoch {epoch}/{self.max_epoch}: avg training loss {train_loss}"
            )
            epoch_logs |= train_logs
            epoch_logs |= {"train": train_loss, "epoch": epoch}

            # Save the most recent iteration
            self.save_model(
                epoch, val_loss, os.path.join(self.checkpoint_folder, "recent.pt")
            )
            # Check for periodic checkpointing
            if (
                self.checkpoint_frequency >= 1
                and epoch % self.checkpoint_frequency == 0
            ):
                self.save_model(
                    epoch,
                    val_loss,
                    os.path.join(self.checkpoint_folder, f"checkpoint_{epoch}.pt"),
                )
            # Check if time to perform standard validation - periodic or final
            if epoch % self.val_frequency == 0 or (epoch == self.max_epoch):
                logger.info(f"Epoch {epoch}/{self.max_epoch}: starting validation")
                val_loss, val_loss_dict = self.validation_loop(
                    val_dataloder, full=epoch == self.max_epoch, epoch=epoch
                )
                logger.info(
                    f"Epoch {epoch}/{self.max_epoch}: avg validation loss {val_loss}"
                )
                val_loss_dict |= {"valid": val_loss, "epoch": epoch}
                epoch_logs |= val_loss_dict

                self._update_best_checkpoints(epoch, val_loss_dict, scope="one_step")
            # Check if time for expensive validation - periodic or final
            if epoch % self.rollout_val_frequency == 0 or (epoch == self.max_epoch):
                logger.info(
                    f"Epoch {epoch}/{self.max_epoch}: starting rollout validation"
                )
                rollout_val_loss, rollout_val_loss_dict = self.validation_loop(
                    rollout_val_dataloader,
                    valid_or_test="rollout_valid",
                    full=True,
                    epoch=epoch,
                )
                logger.info(
                    f"Epoch {epoch}/{self.max_epoch}: avg rollout validation loss {rollout_val_loss}"
                )
                rollout_val_loss_dict |= {
                    "rollout_valid": rollout_val_loss,
                    "epoch": epoch,
                }
                epoch_logs |= rollout_val_loss_dict

                self._update_best_checkpoints(
                    epoch, rollout_val_loss_dict, scope="rollout"
                )

            # Log everything for the epoch in one go
            wandb.log(epoch_logs, step=epoch)

        self._run_final_test_eval(test_dataloader, rollout_test_dataloader, epoch)

    def _run_final_test_eval(
        self,
        test_dataloader: DataLoader,
        rollout_test_dataloader: DataLoader,
        epoch: int,
    ) -> None:
        """Evaluate every saved best_*.pt on test + rollout_test splits.

        Each checkpoint gets a distinct wandb key prefix (e.g. `test_best_rollout_nrmse_*`)
        so all per-checkpoint numbers coexist in a single run. Headline scores are copied
        to `wandb.run.summary` for dashboard filtering.
        """
        combined_test_logs = {}
        headline = {}
        test_step = epoch + 1000
        ds = self.dset_metadata.dataset_name

        for metric_key, ckpt_filename in CHECKPOINT_METRICS.items():
            ckpt_path = os.path.join(self.checkpoint_folder, ckpt_filename)
            if not os.path.exists(ckpt_path):
                logger.warning(
                    f"Skipping test eval for {metric_key}: {ckpt_filename} not found"
                )
                continue
            logger.info(f"Running test evaluation for best_{metric_key}")
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])

            test_prefix = f"test_best_{metric_key}"
            rollout_test_prefix = f"rollout_test_best_{metric_key}"

            test_loss, test_logs = self.validation_loop(
                test_dataloader,
                valid_or_test=test_prefix,
                full=True,
                epoch=test_step,
            )
            rollout_test_loss, rollout_test_logs = self.validation_loop(
                rollout_test_dataloader,
                valid_or_test=rollout_test_prefix,
                full=True,
                epoch=test_step,
            )

            combined_test_logs |= test_logs | rollout_test_logs
            combined_test_logs[f"{test_prefix}_loss"] = test_loss
            combined_test_logs[f"{rollout_test_prefix}_loss"] = rollout_test_loss
            # Log selected_epoch / selected_metric_value under the same chart group
            # as the rest of the per-checkpoint test metrics (`test_best_<metric>_<ds>/...`),
            # so wandb groups them in the existing panel instead of creating an empty section.
            combined_test_logs[f"{test_prefix}_{ds}/selected_epoch"] = ckpt.get("epoch", -1)
            combined_test_logs[f"{test_prefix}_{ds}/selected_metric_value"] = ckpt.get(
                "metric_value", float("nan")
            )
            logger.info(
                f"best_{metric_key}: test_loss={test_loss:.6f}, "
                f"rollout_test_loss={rollout_test_loss:.6f} "
                f"(selected epoch={ckpt.get('epoch', -1)}, "
                f"metric_value={ckpt.get('metric_value', float('nan'))})"
            )

            headline[f"summary/{metric_key}__test_vrmse"] = test_logs.get(
                f"{test_prefix}_{ds}/full_VRMSE_T=all"
            )
            headline[f"summary/{metric_key}__test_nrmse"] = test_logs.get(
                f"{test_prefix}_{ds}/full_NRMSE_T=all"
            )
            headline[f"summary/{metric_key}__rollout_test_vrmse"] = rollout_test_logs.get(
                f"{rollout_test_prefix}_{ds}/full_VRMSE_T=all"
            )
            headline[f"summary/{metric_key}__rollout_test_nrmse"] = rollout_test_logs.get(
                f"{rollout_test_prefix}_{ds}/full_NRMSE_T=all"
            )

        if not combined_test_logs:
            logger.warning("No best checkpoints found; skipping final test evaluation.")
            return

        combined_test_logs["epoch"] = test_step
        wandb.log(combined_test_logs, step=test_step)

        if wandb.run is not None:
            for k, v in headline.items():
                if v is not None:
                    wandb.run.summary[k] = v

    def validate(self):
        """Runs only validation passes - does not save checkpoints or perform training.

        Val/rollout-val are run once on the currently-loaded model weights.
        Test/rollout-test are run once per saved best_*.pt checkpoint.
        """
        val_dataloder = self.datamodule.val_dataloader()
        rollout_val_dataloader = self.datamodule.rollout_val_dataloader()
        test_dataloader = self.datamodule.test_dataloader()
        rollout_test_dataloader = self.datamodule.rollout_test_dataloader()
        epoch = self.max_epoch + 1
        # Regular val (current weights)
        val_loss, val_loss_dict = self.validation_loop(val_dataloder, full=True)
        logger.info(f"Post-run: validation loss {val_loss}")
        val_loss_dict |= {"valid": val_loss, "epoch": epoch}
        wandb.log(val_loss_dict, step=epoch)
        # Rollout val (current weights)
        rollout_val_loss, rollout_val_loss_dict = self.validation_loop(
            rollout_val_dataloader, valid_or_test="rollout_valid", full=True
        )
        logger.info(f"Post run: rollout validation loss {rollout_val_loss}")
        rollout_val_loss_dict |= {
            "rollout_valid": rollout_val_loss,
            "epoch": epoch,
        }
        wandb.log(rollout_val_loss_dict, step=epoch)

        self._run_final_test_eval(test_dataloader, rollout_test_dataloader, epoch - 1)
