from os import listdir
from os import path
import json
import argparse
from argparse import Namespace
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    StochasticWeightAveraging,
    LearningRateMonitor,
    EarlyStopping,
)
from lightning.pytorch.loggers import WandbLogger

import torch
from data import ESMDataLightning
from jespr import JESPR
from modules import load_sequence_encoder, load_structure_encoder


def build_callbacks(callback_args: dict) -> list:
    """Build Lightning Callbacks

    Args:
        callback_args (dict): Callback args from config
    """
    callbacks = []
    # Early Stopping
    if callback_args["early_stopping"]["enabled"]:
        callbacks.append(
            EarlyStopping(
                monitor=callback_args["early_stopping"]["monitor"],
                patience=callback_args["early_stopping"]["patience"],
                min_delta=callback_args["early_stopping"]["min_delta"],
                mode=callback_args["early_stopping"]["mode"],
                verbose=callback_args["early_stopping"]["verbose"],
            )
        )

    # Stochastic Weight Averaging
    if callback_args["stochastic_weight_averaging"]["enabled"]:
        callbacks.append(
            StochasticWeightAveraging(
                swa_lrs=callback_args["stochastic_weight_averaging"]["lrs"],
                swa_epoch_start=callback_args["stochastic_weight_averaging"][
                    "swa_epoch_start"
                ],
                annealing_epochs=callback_args["stochastic_weight_averaging"][
                    "annealing_epochs"
                ],
                annealing_strategy=callback_args["stochastic_weight_averaging"][
                    "annealing_strategy"
                ],
            )
        )

    # Learning Rate Monitor
    if callback_args["learning_rate_monitor"]["enabled"]:
        callbacks.append(
            LearningRateMonitor(
                logging_interval=callback_args["learning_rate_monitor"][
                    "logging_interval"
                ],
                log_momentum=callback_args["learning_rate_monitor"]["log_momentum"],
            )
        )

    return callbacks


def build_logger(logger_args: dict, checkpoint_id: str) -> WandbLogger:
    """Build Wandb Logger

    Args:
        logger_args (dict): Logger Args
        checkpoint_id (Optional): Checkpoint Id optionally to resume run

    Returns:
        WandbLogger: _description_
    """
    if checkpoint_id:
        print(f"Loading Wandb Run...: {checkpoint_id}")

        wandb_logger = WandbLogger(
            project=logger_args["project_name"],
            save_dir=logger_args["logs_dir"],
            resume="must",
            id=checkpoint_id,
        )
    else:
        wandb_logger = WandbLogger(
            name=logger_args["run_name"],
            project=logger_args["project_name"],
            save_dir=logger_args["logs_dir"],
        )
    return wandb_logger


def main(args, mode="train"):
    """Parse all args and run training loop

    Args:
        args (dict): All arguments from the config file.
        mode (str): mode to run in (train/exp_no_trainer/exp_with_trainer). Defaults to "train".
    """
    assert mode in ["train", "exp_no_trainer", "exp_with_trainer"]
    # ___________ JESPR & Submodules ________________ #
    seq_encoder_args = args["jespr"]["sequence_encoder"]
    struct_encoder_args = args["jespr"]["structure_encoder"]
    joint_embedding_dim = args["jespr"]["joint_embedding_dim"]

    # add joint_embedding_dim & norm_embedding to esm2 and esm-if args
    seq_encoder_args["joint_embedding_dim"] = joint_embedding_dim
    struct_encoder_args["joint_embedding_dim"] = joint_embedding_dim

    # ___________ Data ______________________________ #
    data_args = args["data"]
    batch_size = args["data"]["batch_size"]

    # ___________ optim _____________________________ #
    optim_args = args["optim"]  # Args to be passed to Optimizer

    # ___________ trainer ___________________________ #
    accelerator = args["trainer"]["accelerator"]
    precision = args["trainer"]["precision"]
    devices = args["trainer"]["devices"]
    strategy = args["trainer"]["strategy"]
    epochs = args["trainer"]["epochs"]
    log_every_n_steps = args["trainer"]["log_every_n_steps"]  # batch steps not epochs
    enable_progress_bar = args["trainer"]["enable_progress_bar"]
    val_check_interval = args["trainer"]["val_check_interval"]  # epochs
    check_val_every_n_epoch = args["trainer"]["check_val_every_n_epoch"]
    limit_train_batches = args["trainer"]["limit_train_batches"]
    limit_val_batches = args["trainer"]["limit_val_batches"]
    overfit_batches = args["trainer"]["overfit_batches"]
    accumulate_grad_batches = args["trainer"]["accumulate_grad_batches"]
    detect_anomaly = args["trainer"]["detect_anomaly"]
    grad_clip_val = args["trainer"]["grad_clip_val"]
    grad_clip_algorithm = args["trainer"]["grad_clip_algorithm"]
    checkpoint_id = args["trainer"]["checkpoint_id"]
    callback_args = args["trainer"]["callbacks"]
    callbacks = build_callbacks(callback_args)

    # ___________ Logger ______________________________ #
    logger_args = args["logger"]

    if checkpoint_id:
        ckpt_dir = path.join(
            logger_args["logs_dir"],
            logger_args["project_name"],
            checkpoint_id,
            "checkpoints",
        )
        ckpt_path = path.join(ckpt_dir, listdir(ckpt_dir)[-1])
    else:
        ckpt_path = None

    seq_encoder, alphabet_2 = load_sequence_encoder(seq_encoder_args)
    struct_encoder, alphabet_if = load_structure_encoder(struct_encoder_args)

    print("Loading DataModule...")
    esm_data_lightning = ESMDataLightning(
        esm2_alphabet=alphabet_2,
        esm_if_alphabet=alphabet_if,
        args=Namespace(**data_args),
    )

    print("Initializing JESPR...")
    jespr = JESPR(
        sequence_encoder=seq_encoder,
        structure_encoder=struct_encoder,
        esm2_alphabet=alphabet_2,
        esm_if_alphabet=alphabet_if,
        optim_args=optim_args,
        dataset_name=data_args["dataset_name"],
    )

    if mode == "exp_no_trainer":
        if checkpoint_id:
            print(f"Loading Checkpoint from id: {ckpt_path}...")
            jespr = jespr.load_from_checkpoint(ckpt_path)
        return (jespr, esm_data_lightning)

    print("Initializing Wandb Logger...")
    wandb_logger = build_logger(logger_args, checkpoint_id)

    print("Initializing Trainer...")
    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        strategy=strategy,
        log_every_n_steps=log_every_n_steps,
        detect_anomaly=detect_anomaly,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=check_val_every_n_epoch,
        logger=wandb_logger,
        max_epochs=epochs,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=callbacks,
        enable_progress_bar=enable_progress_bar,
        gradient_clip_val=grad_clip_val,
        gradient_clip_algorithm=grad_clip_algorithm,
        overfit_batches=overfit_batches,
    )

    # Wandb object only available to rank 0
    if trainer.global_rank == 0:
        args["num_params"] = jespr.num_params(jespr)
        # Add all config params
        wandb_logger.experiment.config.update(args, allow_val_change=True)

    if mode == "exp_with_trainer":
        return (jespr, esm_data_lightning, trainer, wandb_logger)

    print("Starting Training...")

    trainer.fit(
        model=jespr,
        datamodule=esm_data_lightning,
        ckpt_path=ckpt_path,
    )
    return None


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-f",
        "--config_path",
        type=str,
        default=path.join(path.dirname(__file__), "config/trainer_config.json"),
        help="Path for the Train Config Json File",
    )

    config_file_path = arg_parser.parse_known_args()[0].config_path
    with open(config_file_path, "r") as f:
        args = json.load(f)
    main(args=args)
