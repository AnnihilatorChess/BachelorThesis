import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings("ignore")

import logging
import os.path as osp
import random

import hydra
import numpy as np
import torch
import torch._dynamo  # noqa: F401  -- imported here so configuring it inside `train()` doesn't shadow the module-level `torch` import.
import torch.distributed as dist
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torchinfo import summary

from the_well.benchmark.trainer import Trainer
from the_well.benchmark.trainer.utils import set_master_config
from the_well.benchmark.utils.experiment_utils import configure_experiment
from the_well.data import WellDataModule

logger = logging.getLogger("the_well")
logger.setLevel(level=logging.DEBUG)


def seed_everything(seed: int):
    """Seed all RNGs for reproducibility: model init, dataloader shuffle, augmentations."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f"Global seed set to {seed}")

# Retrieve configuration for hydra
CONFIG_DIR = osp.join(osp.dirname(__file__), "configs")
CONFIG_NAME = "config"
CONFIG_PATH = osp.join(CONFIG_DIR, f"{CONFIG_NAME}.yaml")
assert osp.isfile(CONFIG_PATH), f"Configuration {CONFIG_PATH} is not an existing file."
logger.info(f"Run training script for {CONFIG_PATH}")


def train(
    cfg: DictConfig,
    experiment_folder: str,
    checkpoint_folder: str,
    artifact_folder: str,
    viz_folder: str,
    is_distributed: bool = False,
    world_size: int = 1,
    rank: int = 1,
    local_rank: int = 1,
):
    """Instantiate the different objects required for training and run the training loop."""
    validation_mode = cfg.validation_mode
    bundle_size = cfg.get("temporal_bundle_size", 1)

    # Disable struct mode to allow injecting keys not explicitly defined in the YAML schema
    OmegaConf.set_struct(cfg, False)

    # Sync top-level temporal_bundle_size into trainer config
    cfg.trainer["temporal_bundle_size"] = bundle_size
    
    # Sync trainer max_rollout_steps to dataloader
    if "max_rollout_steps" in cfg.trainer:
        cfg.data["max_rollout_steps"] = cfg.trainer["max_rollout_steps"]

    # Set n_steps_output_train based on pushforward and bundling
    if "bptt_unroll_steps" in cfg.trainer:
        # BPTTTrainer needs ground-truth targets for every gradient-tracked step
        # plus enough trailing inputs for the no_grad PF warmup; the dataloader
        # delivers (warmup + unroll) future timesteps per training window.
        pf_warmup = cfg.trainer.get("pf_warmup_steps", 0)
        cfg.data["n_steps_output_train"] = (
            cfg.trainer["bptt_unroll_steps"] + pf_warmup
        ) * bundle_size
    elif cfg.trainer["pushforward"] and bundle_size > 1:
        cfg.data["n_steps_output_train"] = bundle_size * 4
    elif cfg.trainer["pushforward"]:
        cfg.data["n_steps_output_train"] = 4
    elif bundle_size > 1:
        cfg.data["n_steps_output_train"] = bundle_size

    logger.info(f"Instantiate datamodule {cfg.data._target_}")
    datamodule: WellDataModule = instantiate(
        cfg.data, world_size=world_size, rank=rank, data_workers=cfg.data_workers
    )
    dset_metadata = datamodule.train_dataset.metadata
    # TODO - currently just doing channel/time stacking for uniformity, but should
    # give the option of not stacking
    n_input_fields = (
        cfg.data.n_steps_input * dset_metadata.n_fields
        + dset_metadata.n_constant_fields
    )
    n_output_fields = dset_metadata.n_fields * bundle_size

    logger.info(
        f"Instantiate model {cfg.model._target_} (bundle_size={bundle_size})",
    )

    model: torch.nn.Module = instantiate(
        cfg.model,
        n_spatial_dims=dset_metadata.n_spatial_dims,
        spatial_resolution=dset_metadata.spatial_resolution,
        dim_in=n_input_fields,
        dim_out=n_output_fields,
    )
    try:
        summary(model, depth=5)
    except UnicodeEncodeError:
        logger.warning("Could not print model summary (encoding issue on Windows)")
        pass

    if is_distributed:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda")
        model = model.to(device)
        model = DDP(model, device_ids=[local_rank])
    else:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        model = model.to(device)

    # Optional torch.compile. Helps when per-step GPU work is small relative
    # to Python/dispatch overhead (e.g. BPTT-91 at batch=1 on a small FNO).
    # First iteration spends 30s-3min compiling; subsequent iterations are
    # faster in dispatch-bound regimes.
    #
    # Caveat for FNO: ``neuraloperator``'s ``SpectralConv`` uses parametrized
    # weights with list-of-slices ``__getitem__`` indexing, which dynamo
    # cannot trace (raises a hard AssertionError, NOT a graph break). We set
    # ``torch._dynamo.config.suppress_errors=True`` so dynamo falls back to
    # eager for any unsupported scope -- the model still runs, just with
    # reduced (or, for FNO, mostly cosmetic) speedup since the expensive op
    # is precisely the one that won't trace. Disable via
    # ``compile.enabled=false`` if it's not paying for itself.
    compile_cfg = cfg.get("compile", {})
    if compile_cfg and compile_cfg.get("enabled", False):
        torch._dynamo.config.suppress_errors = True
        compile_mode = compile_cfg.get("mode", "default")
        compile_fullgraph = compile_cfg.get("fullgraph", False)
        logger.info(
            f"torch.compile(model, mode={compile_mode!r}, fullgraph={compile_fullgraph}) "
            f"-- first batch will be slow while tracing+compiling. "
            f"suppress_errors=True so untraceable ops fall back to eager."
        )
        model = torch.compile(model, mode=compile_mode, fullgraph=compile_fullgraph)

    logger.info(f"Instantiate optimizer {cfg.optimizer._target_}")
    if not validation_mode:
        optimizer: torch.optim.Optimizer = instantiate(
            cfg.optimizer, params=model.parameters()
        )
    else:
        optimizer = None

    if hasattr(cfg, "lr_scheduler") and not validation_mode:
        # Instantiate LR scheduler
        logger.info(f"Instantiate learning rate scheduler {cfg.lr_scheduler._target_}")
        # Only pass kwargs the target scheduler accepts -- lets us mix
        # LinearWarmupCosineAnnealingLR (needs max_epochs/warmup_start_lr/eta_min)
        # with stock torch schedulers like StepLR that don't.
        import inspect
        from hydra.utils import get_class
        sched_params = inspect.signature(get_class(cfg.lr_scheduler._target_)).parameters
        extra_kwargs = {}
        if "max_epochs" in sched_params:
            extra_kwargs["max_epochs"] = cfg.trainer.epochs
        if "warmup_start_lr" in sched_params:
            extra_kwargs["warmup_start_lr"] = cfg.optimizer.lr * 0.1
        if "eta_min" in sched_params:
            extra_kwargs["eta_min"] = cfg.optimizer.lr * 0.1
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler = instantiate(
            cfg.lr_scheduler,
            optimizer=optimizer,
            **extra_kwargs,
        )
    else:
        logger.info("No learning rate scheduler")
        lr_scheduler = None
    # Print final config, but also log it to experiment directory.
    logger.info(f"Final configuration:\n{OmegaConf.to_yaml(cfg)}")
    logger.info(f"Instantiate trainer {cfg.trainer._target_}")
    trainer: Trainer = instantiate(
        cfg.trainer,
        checkpoint_folder=checkpoint_folder,
        artifact_folder=artifact_folder,
        viz_folder=viz_folder,
        model=model,
        datamodule=datamodule,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        is_distributed=is_distributed,
    )
    if validation_mode:
        reeval_rollout_only = bool(cfg.get("reeval_rollout_only", False))
        reeval_output_dir = cfg.get("reeval_output_dir", "") or osp.join(
            cfg.experiment_dir, "_reeval_per_timestep"
        )
        trainer.validate(
            rollout_only=reeval_rollout_only,
            reeval_checkpoint=cfg.get("reeval_checkpoint", ""),
            reeval_output_dir=reeval_output_dir,
            reeval_split=cfg.get("reeval_split", "test"),
            cfg=cfg,
        )
    else:
        # Save config to directory folder
        with open(osp.join(experiment_folder, "extended_config.yaml"), "w") as f:
            OmegaConf.save(cfg, f)
        trainer.train()


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name=CONFIG_NAME)
def main(cfg: DictConfig):
    # Resolve experiment_dir relative to this script's directory if it is a relative path
    if not osp.isabs(cfg.experiment_dir):
        cfg.experiment_dir = osp.join(osp.dirname(osp.abspath(__file__)), cfg.experiment_dir)

    # Torch optimization settings
    torch.backends.cudnn.benchmark = (
        True  # If input size is fixed, this will usually the computation faster
    )
    torch.set_float32_matmul_precision("high")  # Use TF32 when supported
    # Set global seed before anything else so model init and dataloaders are deterministic
    seed_everything(cfg.seed)
    # Normal things
    (
        cfg,
        experiment_name,
        experiment_folder,
        checkpoint_folder,
        artifact_folder,
        viz_folder,
    ) = configure_experiment(cfg, logger)

    logger.info(f"Run experiment {experiment_name}")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    # Initiate wandb logging.
    # group = experiment config without seed, so all seeds of the same run appear together.
    # name  = experiment config + seed, so individual runs are distinguishable.
    wandb_logged_cfg = OmegaConf.to_container(cfg, resolve=True)
    wandb_logged_cfg["experiment_folder"] = experiment_folder
    wandb.init(
        dir=experiment_folder,
        project=cfg.wandb_project_name,
        group=cfg.wandb_group if cfg.wandb_group else experiment_name,
        config=wandb_logged_cfg,
        name=f"{cfg.run_name}-{cfg.model._target_.split('.')[-1]}-seed{cfg.seed}" if cfg.run_name else f"{experiment_name}-seed{cfg.seed}",
        resume=True,
    )

    # Retrieve multiple processes context to setup DDP
    is_distributed, world_size, rank, local_rank = (
        False,
        1,
        0,
        0,
    )  # get_distrib_config()
    # is_distributed = is_distributed and world_size > 1

    logger.info(f"Distributed training: {is_distributed}")
    if is_distributed:
        set_master_config()
        dist.init_process_group(
            backend="nccl", init_method="env://", world_size=world_size, rank=rank
        )
    train(
        cfg,
        experiment_folder,
        checkpoint_folder,
        artifact_folder,
        viz_folder,
        is_distributed,
        world_size,
        rank,
        local_rank,
    )
    wandb.finish()


if __name__ == "__main__":
    main()
