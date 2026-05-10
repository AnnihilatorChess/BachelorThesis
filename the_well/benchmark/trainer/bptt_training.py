"""BPTT trainer that mirrors the PDEBench training protocol.

PDEBench has two relevant training modes for forward-problem benchmarks:

1. **FNO autoregressive (`AR`)** [`pdebench/models/fno/train.py`]:
   - Loop ``t = initial_step .. t_train`` (e.g. 10..101 for SWE → 91 steps).
   - Every step is gradient-tracked; predictions are fed back as inputs.
   - Per-step MSEs are **summed**; a single ``loss.backward()`` runs at the end.

2. **U-Net pushforward (`PF-K`)** [`pdebench/models/unet/train.py`]:
   - Same loop, but the first ``t_train - initial_step - unroll_step`` steps are
     wrapped in ``torch.no_grad()`` (rollout-only warmup).
   - The last ``unroll_step=20`` steps are gradient-tracked and contribute to
     the summed MSE loss.
   - Single backward at the end.

This trainer implements both via two knobs:

- ``bptt_unroll_steps``  → PDEBench's ``unroll_step`` (or full-trajectory length
  for FNO-AR).
- ``pf_warmup_steps``    → number of leading ``no_grad`` rollout steps. ``0`` =
  FNO-AR; ``>0`` = U-Net PF-K.

``loss_reduction`` defaults to ``"sum"`` (PDEBench convention). Setting
``"mean"`` divides the accumulated loss by ``bptt_unroll_steps``; this was the
original behaviour of this trainer and is kept as an opt-in for backwards
compatibility.

Where do the unroll horizons come from?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each PDEBench dataset the per-dataset config in
``pdebench/models/config/args/`` sets ``initial_step``, ``t_train`` and
``reduced_resolution_t``. The effective number of gradient-tracked unroll steps
in PDEBench's loop is

    unroll = min(t_train, ceil(raw_T / reduced_resolution_t)) - initial_step

* ``config_rdb.yaml`` (2D shallow water): ``t_train=101, initial_step=10,
  reduced_resolution_t=1`` → **91 steps** (SWE has 101 raw timesteps).
* ``config_Bgs.yaml`` (1D Burgers): ``t_train=200, initial_step=10,
  reduced_resolution_t=5``; raw Burgers files have 201 timesteps which
  ``[::5]`` reduces to 41 → **31 steps** (``t_train=200`` is clipped to 41 by
  ``min(t_train, _data.shape[-2])`` in the script).

For U-Net PF-K, ``unroll_step=20`` is dataset-independent (set in every
``args/*.yaml``). The no_grad warmup count is therefore
``unroll_total - 20 = 71`` for SWE and ``11`` for Burgers.

Note that our 1D Burgers data is *not* temporally downsampled in this thesis,
so 31 unroll steps cover 31 of our 201 timesteps (~15 % of physical horizon)
versus 31/41 ≈ 76 % of PDEBench's downsampled trajectory. Strict step-count
parity with PDEBench is preserved; physical-horizon parity is a separate
question and would require either ``min/max_dt_stride=5`` in the data config
or a longer unroll.
"""

from __future__ import annotations

import time

import torch
from torch.utils.data import DataLoader

from the_well.benchmark.trainer.training import Trainer, logger


class BPTTTrainer(Trainer):
    def __init__(
        self,
        *args,
        bptt_unroll_steps: int = 5,
        pf_warmup_steps: int = 0,
        loss_reduction: str = "sum",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.bptt_unroll_steps = bptt_unroll_steps
        self.pf_warmup_steps = pf_warmup_steps
        if loss_reduction not in ("sum", "mean"):
            raise ValueError(
                f"loss_reduction must be 'sum' or 'mean', got {loss_reduction!r}"
            )
        self.loss_reduction = loss_reduction
        logger.info(
            "Initialized BPTTTrainer: "
            f"bptt_unroll_steps={bptt_unroll_steps}, "
            f"pf_warmup_steps={pf_warmup_steps}, "
            f"loss_reduction={loss_reduction!r}, "
            f"bundle_size={self.bundle_size}"
        )

    def train_one_epoch(self, epoch: int, dataloader: DataLoader) -> tuple[float, dict]:
        """Train one epoch using PDEBench-style BPTT (optionally with PF warmup).

        For each batch:
            1. Roll the model forward ``pf_warmup_steps`` steps under
               ``torch.no_grad()`` (no loss, no graph).
            2. Roll forward ``bptt_unroll_steps`` more steps under autograd,
               accumulating per-step MSE into ``total_loss``.
            3. Reduce (sum or mean) and run a single ``backward`` + ``step``.

        With ``pf_warmup_steps=0`` this is equivalent to PDEBench's FNO-AR
        training. With ``pf_warmup_steps>0`` it matches PDEBench's U-Net PF-K.
        """
        self._current_epoch = epoch
        self.model.train()
        epoch_loss = 0.0
        train_logs: dict = {}
        start_time = time.time()
        n_batches = len(dataloader)
        print_interval = max(n_batches // 100, 1)
        K = self.bundle_size  # temporal bundle size; 1 by default

        for i, batch in enumerate(dataloader):
            # y_true_all has shape [B, T, H, W, C] (channels-last format)
            _, y_true_all = self.formatter.process_input(batch)
            current_input_fields = batch["input_fields"].to(self.device)
            y_true_all = y_true_all.to(self.device)

            # Cap warmup+unroll to whatever the dataloader actually delivered.
            available = y_true_all.shape[1]
            max_bundles = available // K
            warmup = min(self.pf_warmup_steps, max_bundles)
            unroll = min(self.bptt_unroll_steps, max_bundles - warmup)

            with torch.autocast(
                self.device.type, enabled=self.enable_amp, dtype=self.amp_type
            ):
                # 1. No-grad rollout warmup (PF-K only; otherwise warmup=0).
                with torch.no_grad():
                    for _ in range(warmup):
                        if K == 1:
                            y_pred = self.predict_next_step(current_input_fields)
                            current_input_fields = torch.cat(
                                [current_input_fields[:, 1:], y_pred], dim=1
                            )
                        else:
                            y_pred_b = self.predict_next_bundle(current_input_fields)
                            current_input_fields = torch.cat(
                                [current_input_fields[:, K:], y_pred_b], dim=1
                            )

                # 2. Gradient-tracked unroll. Accumulate per-step MSEs.
                total_loss: torch.Tensor | float = 0.0
                for s in range(unroll):
                    if K == 1:
                        y_true = y_true_all[:, warmup + s].unsqueeze(1)
                        y_pred = self.predict_next_step(
                            current_input_fields, add_noise=self.noise_injection
                        )
                        step_loss = self.loss_fn(
                            y_pred, y_true, self.dset_metadata
                        ).mean()
                        total_loss = total_loss + step_loss
                        current_input_fields = torch.cat(
                            [current_input_fields[:, 1:], y_pred], dim=1
                        )
                    else:
                        target = y_true_all[
                            :, (warmup + s) * K : (warmup + s + 1) * K
                        ]
                        y_pred_b = self.predict_next_bundle(
                            current_input_fields, add_noise=self.noise_injection
                        )
                        step_loss = self.loss_fn(
                            y_pred_b, target, self.dset_metadata
                        ).mean()
                        total_loss = total_loss + step_loss
                        current_input_fields = torch.cat(
                            [current_input_fields[:, K:], y_pred_b], dim=1
                        )

                # 3. Reduce.
                if unroll > 0 and self.loss_reduction == "mean":
                    total_loss = total_loss / unroll

            if unroll > 0 and isinstance(total_loss, torch.Tensor):
                self.grad_scaler.scale(total_loss).backward()
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                self.optimizer.zero_grad()
                epoch_loss += total_loss.item()

            if i % print_interval == 0:
                loss_val = (
                    total_loss.item() if isinstance(total_loss, torch.Tensor) else 0.0
                )
                logger.info(
                    f"Epoch {epoch}, Batch {i + 1}/{n_batches}: "
                    f"BPTT loss {loss_val:.6g} "
                    f"(warmup={warmup}, unroll={unroll}, "
                    f"reduction={self.loss_reduction})"
                )

        train_logs["epoch_time"] = time.time() - start_time
        train_logs["time_per_train_iter"] = train_logs["epoch_time"] / max(n_batches, 1)
        train_logs["train_loss"] = epoch_loss / max(n_batches, 1)
        if self.lr_scheduler:
            self.lr_scheduler.step()
            train_logs["lr"] = self.lr_scheduler.get_last_lr()[-1]

        return train_logs["train_loss"], train_logs
