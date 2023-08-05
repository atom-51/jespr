from argparse import Namespace
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    StochasticWeightAveraging,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import WandbLogger

import torch
from data import ESMDataLightning
from jespr import JESPR
from modules import load_esm2, load_esm_if


def main(args, mode="train"):
    """Parse all args and run training loop

    Args:
        args (dict): All arguments from the config file.
        mode (str): mode to run in (train/exp). Defaults to "train".
    """
    # ___________ JESPR & Submodules ________________ #
    esm2_args = args["jespr"]["esm2"]
    esm_if_args = args["jespr"]["esm_if"]
    joint_embedding_dim = args["jespr"]["joint_embedding_dim"]
    norm_embedding = args["jespr"]["norm_embedding"]
    temperature = args["jespr"]["temperature"]

    # add joint_embedding_dim & norm_embedding to esm2 and esm-if args
    esm2_args["joint_embedding_dim"] = joint_embedding_dim
    esm_if_args["joint_embedding_dim"] = joint_embedding_dim
    esm2_args["norm_embedding"] = norm_embedding
    esm_if_args["norm_embedding"] = norm_embedding

    # ___________ Data ______________________________ #
    data_args = args["data"]
    batch_size = args["data"]["batch_size"]

    # ___________ optim _____________________________ #
    optim_args = args["optim"]  # Args to be passed to Optimizer

    # ___________ trainer ___________________________ #
    accelerator = args["trainer"]["accelerator"]
    precision = args["trainer"]["precision"]
    devices = args["trainer"]["devices"]
    epochs = args["trainer"]["epochs"]
    log_every_n_steps = args["trainer"][
        "log_every_n_steps"
    ]  # batch steps not epochs
    enable_progress_bar = args["trainer"]["enable_progress_bar"]
    val_check_interval = args["trainer"]["val_check_interval"]  # epochs
    limit_train_batches = args["trainer"]["limit_train_batches"]
    overfit_batches = args["trainer"]["overfit_batches"]
    accumulate_grad_batches = args["trainer"]["accumulate_grad_batches"]
    stochastic_weight_averaging = args["trainer"][
        "stochastic_weight_averaging"
    ]
    stochastic_weight_averaging_lr = args["trainer"][
        "stochastic_weight_averaging_lr"
    ]
    detect_anomaly = args["trainer"]["detect_anomaly"]
    grad_clip_val = args["trainer"]["grad_clip_val"]
    grad_clip_algorithm = args["trainer"]["grad_clip_algorithm"]

    # ___________ Meta ______________________________ #
    project_name = args["meta"]["project_name"]
    run_name = args["meta"]["run_name"]
    logs_dir = args["meta"]["logs_dir"]

    esm2, alphabet_2 = load_esm2(esm2_args)
    esm_if, alphabet_if = load_esm_if(esm_if_args)

    print("Loading DataModule...")
    esm_data_lightning = ESMDataLightning(
        esm2_alphabet=alphabet_2,
        esm_if_alphabet=alphabet_if,
        args=Namespace(**data_args),
    )

    esm_data_lightning.setup("fit")
    # total_iterations = epochs * (
    #     len(esm_data_lightning.train_dataloader()) // batch_size
    # )

    # print("Initializing JESPR...")
    # jespr = JESPR(
    #     esm2=esm2,
    #     esm_if=esm_if,
    #     esm2_alphabet=alphabet_2,
    #     esm_if_alphabet=alphabet_if,
    #     optim_args=optim_args,
    #     temperature=temperature,
    #     total_iterations=total_iterations,
    # )

    if mode != "train":
        return esm_data_lightning

    print("Initializing Wandb Logger...")
    wandb_logger = WandbLogger(
        project=project_name, name=run_name, save_dir=logs_dir
    )

    callbacks = [
        LearningRateMonitor(logging_interval="epoch", log_momentum=False)
    ]
    if stochastic_weight_averaging:
        callbacks.append(
            StochasticWeightAveraging(swa_lrs=stochastic_weight_averaging_lr)
        )

    print("Initializing Trainer...")
    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        log_every_n_steps=log_every_n_steps,
        detect_anomaly=detect_anomaly,
        val_check_interval=val_check_interval,
        logger=wandb_logger,
        max_epochs=epochs,
        limit_train_batches=limit_train_batches,
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=callbacks,
        enable_progress_bar=enable_progress_bar,
        gradient_clip_val=grad_clip_val,
        gradient_clip_algorithm=grad_clip_algorithm,
        overfit_batches=overfit_batches,
    )

    # Wandb object only available to rank 0
    if trainer.global_rank == 0:
        # Add all config params
        wandb_logger.experiment.config.update(args)

    print("Starting Training...")
    trainer.fit(model=jespr, datamodule=esm_data_lightning)
    return None


if __name__ == "__main__":
    main()
