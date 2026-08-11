from __future__ import annotations

import time
from pathlib import Path

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from torch.multiprocessing import set_sharing_strategy

from transferatlas.cli.arguments import TrainArgs, parse_train_args
from transferatlas.config import ExperimentConfig
from transferatlas.data.datamodule import TrajectoryDataModule
from transferatlas.data.dataset import dataset_group_name
from transferatlas.models.trace.lightning import TraceLightningModule
from transferatlas.models.trace.model import TraceModel
from transferatlas.runtime import (
    configure_trainer_hardware,
    copy_config_file,
    create_experiment_logger,
    load_config,
    require_graph_backend,
)

set_sharing_strategy("file_system")


def create_run_directory(
    args: TrainArgs,
    config: ExperimentConfig,
    dataset_name: str,
    model_name: str,
) -> tuple[Path, str]:
    config["training"]["dataset"] = dataset_name
    add_name = f"-{args.add_name}" if args.add_name else ""
    full_save_name = f"{model_name}{add_name}-{dataset_name}"

    if args.dry_run:
        full_save_name += "-DEBUG"

    dataset_dir = Path("artifacts/checkpoints") / dataset_name
    completed_runs = (
        len([directory for directory in dataset_dir.iterdir() if directory.is_dir()])
        if dataset_dir.exists()
        else 0
    )
    run = completed_runs + 1

    run_directory = dataset_dir / f"{model_name}-run{run}"
    return run_directory, full_save_name


def train(
    args: TrainArgs,
    config: ExperimentConfig,
) -> None:
    require_graph_backend()
    devices, strategy, accelerator = configure_trainer_hardware(args)

    dataset_name = dataset_group_name(config["datamodule"]["name"])
    model_name = config["model"]["name"]
    run_directory, full_save_name = create_run_directory(
        args, config, dataset_name, model_name
    )

    datamodule = TrajectoryDataModule(config["datamodule"], args)

    callbacks: list[Callback] = []
    if args.store_model and not args.dry_run:
        copy_config_file(args.config, run_directory)
        model_checkpoint_all = ModelCheckpoint(
            dirpath=str(run_directory / "all_epochs"),
            filename=f"{full_save_name}-{{epoch:03d}}",
            save_top_k=-1,  # -1 means "save all"
            every_n_epochs=1,  # save every epoch
        )

        if args.add_name:
            min_filename = f"{model_name}-{args.add_name}-Min-{dataset_name}"
        else:
            min_filename = f"{model_name}-Min-{dataset_name}"
        best_fde_model_checkpoint = ModelCheckpoint(
            dirpath=str(run_directory),
            filename=min_filename + "_FDE",
            monitor="val_fde",
            save_top_k=1,
            mode="min",
        )
        best_ade_model_checkpoint = ModelCheckpoint(
            dirpath=str(run_directory),
            filename=min_filename + "_ADE",
            monitor="val_ade",
            save_top_k=1,
            mode="min",
        )

        best_val_loss_model_checkpoint = ModelCheckpoint(
            dirpath=str(run_directory),
            filename=min_filename + "_ValLoss",
            monitor="val_loss",
            save_top_k=1,
            mode="min",
        )

        callbacks.append(model_checkpoint_all)
        callbacks.append(best_fde_model_checkpoint)
        callbacks.append(best_ade_model_checkpoint)
        callbacks.append(best_val_loss_model_checkpoint)

    # Setup model, datamodule and trainer
    net = TraceModel(config["model"])
    model = TraceLightningModule(net, config["training"])

    resume_checkpoint = args.checkpoint or None
    if resume_checkpoint is not None and not Path(resume_checkpoint).is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {resume_checkpoint}")
    trainer = Trainer(
        max_epochs=config["training"]["epochs"],
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        logger=create_experiment_logger(
            dry_run=args.dry_run,
            use_logger=args.use_logger,
            project_name=config.get("experiment_name", model_name),
            run_name=f"{full_save_name}_{time.strftime('%d-%m_%H:%M')}",
        ),
        fast_dev_run=args.dry_run,
        enable_checkpointing=args.store_model,
        callbacks=callbacks,
    )

    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=resume_checkpoint,
    )

    if args.store_model and not args.dry_run:
        _ = net.save_portable_checkpoint(
            config["model"],
            run_directory / f"{full_save_name}.pt",
        )


def main() -> None:
    args = parse_train_args()
    _ = seed_everything(args.seed, workers=True)
    config = load_config(args.config)
    if args.dataset:
        config["datamodule"]["name"] = args.dataset
    if args.root:
        config["datamodule"]["root"] = args.root

    train(args, config)


if __name__ == "__main__":
    main()
