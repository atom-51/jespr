from os import path
import time
import numpy as np
import scipy
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

import lightning as pl

from esm.data import Alphabet
from modules import SequenceEncoder, StructureEncoder
from modules import WarmupCosineFactorLambda

DEFAULT_LR = 3e-4


class JESPR(pl.LightningModule):
    def __init__(
        self,
        sequence_encoder: SequenceEncoder,
        structure_encoder: StructureEncoder,
        esm2_alphabet: Alphabet,
        esm_if_alphabet: Alphabet,
        optim_args: dict,
        **kwargs,
    ) -> None:
        """
        JESPR Model

        Args:
            esm2 (ESM2): ESM-2 Model
            esm2_alphabet (Alphabet): ESM-2 Alphabet
            esm_if (GVPTransformerModel): ESM-IF Model
            esm_if_alphabet (Alphabet): ESM-IF Alphabet
            optim_args (dict, optional): Optimizer Arguments. Defaults to {"lr": DEFAULT_LR}.
            temperature (float, optional): Temperature for scaling the cosine similarity score. Defaults to INIT_TEMP.
        """
        super().__init__()

        self.seq_encoder, self.struct_encoder = (
            sequence_encoder,
            structure_encoder,
        )
        self.esm2_alphabet, self.esm_if_alphabet = (
            esm2_alphabet,
            esm_if_alphabet,
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.optim_args = optim_args

        self.dataset_name = kwargs.get("dataset_name", "pdb")
        if self.dataset_name == "scope_10":
            self.classfication_labels = {
                "class": 12,
                "fold": 1350,
                "domain": 11110,
                "superfamily": 2074,
                "family": 4558,
                "sp": 24194,
            }
            self.classifcation_loss_scalars = {
                "class": (4 / 6) / -np.log(1 / 12),
                "fold": (4 / 6) / -np.log(1 / 1350),
                "domain": (4 / 6) / -np.log(1 / 11110),
                "superfamily": (4 / 6) / -np.log(1 / 2074),
                "family": (4 / 6) / -np.log(1 / 4558),
                "sp": (4 / 6) / -np.log(1 / 24194),
            }

            out_dim = sequence_encoder.joint_embedding_projection.out_features
            self.classification_linears = nn.ModuleList(
                [
                    nn.Linear(out_dim, class_size)
                    for class_size in self.classfication_labels.values()
                ]
            )

    def forward(self, x) -> tuple:
        if self.dataset_name != "scope_10":
            loss, reprs = self.forward_contrastive(x)
            reprs.pop("seq_repr")
            return loss, reprs
        else:
            cont_loss, reprs = self.forward_contrastive(x["contrastive"])
            class_outs = {}
            class_losses = {}
            class_accs = {}

            B, J = reprs["seq_repr"].shape
            cte_est = -np.log(1 / B)
            for i, label in enumerate(self.classfication_labels.keys()):
                class_outs[label] = self.classification_linears[i](reprs["seq_repr"])

                loss_scalar = (cte_est / 6) / -np.log(
                    1 / self.classfication_labels[label]
                )
                curr_class_labels = torch.tensor(
                    [val_[i] for val_ in x["classification"]],
                    dtype=torch.long,
                    device=class_outs[label].device,
                )
                class_losses[label] = loss_scalar * nn.functional.cross_entropy(
                    class_outs[label],
                    curr_class_labels,
                )
                # Calc accuracy
                with torch.no_grad():
                    class_accs[label] = (
                        class_outs[label].argmax(dim=-1) == curr_class_labels
                    ).sum().item() / B

            return cont_loss, reprs, class_losses, class_outs, class_accs

    def forward_contrastive(self, x) -> tuple:
        """Foward Function for JESPR

        Args:
            x (tuple): A tuple consisting of output from DataLoader's collate fn

        Returns:
            tuple: Model Embeddings for both ESM2 & ESM-IF
        """

        coords, confidence, strs, tokens, padding_mask = x

        # Get ESM2 & ESM-IF outputs. Shape: Batch_size * Joint_embed_dim
        seq_repr = self.seq_encoder(tokens)
        struct_repr = self.struct_encoder(coords, padding_mask, confidence)

        B, J = seq_repr.shape
        # Calculating the Loss

        logit_scale = self.logit_scale.exp()
        logits_per_structure = logit_scale * seq_repr @ struct_repr.T
        logits_per_seq = logits_per_structure.T

        labels = torch.arange(B, dtype=torch.long, device=logits_per_structure.device)

        loss = (
            torch.nn.functional.cross_entropy(logits_per_structure, labels)
            + torch.nn.functional.cross_entropy(logits_per_seq, labels)
        ) / 2

        return loss, {
            "logits_per_structure": logits_per_structure,
            "logits_per_seq": logits_per_seq,
            "seq_repr": seq_repr,
        }

    @torch.no_grad()
    def calc_argmax_acc(self, logits: dict) -> float:
        """Calculate Argmax Accuracy

        Args:
            logits (dict): Dict of logits for structure and sequence

        Returns:
            float: Argmax Accuracy
        """
        am_logits_per_structure = logits["logits_per_structure"].argmax(0)
        am_logits_per_seq = logits["logits_per_seq"].argmax(0)

        b = am_logits_per_structure.shape[0]  # Batch Size
        truths = torch.arange(b, device=am_logits_per_structure.device)
        acc_str = torch.sum(am_logits_per_structure == truths).item() / b
        acc_seq = torch.sum(am_logits_per_seq == truths).item() / b

        return {"acc_str": acc_str, "acc_seq": acc_seq}

    def training_step(self, batch, batch_idx):
        start_time = time.time()
        if self.dataset_name != "scope_10":
            loss, logits = self.forward(batch)
            B = batch[0].shape[0]
            self.log("metrics/train/loss", loss, batch_size=B)
            self.log(
                "metrics/train/time_per_step",
                time.time() - start_time,
                batch_size=B,
            )
            return {"loss": loss, "logits": logits}
        else:
            cont_loss, reprs, class_losses, class_outs, class_accs = self.forward(batch)
            B = batch["contrastive"][0].shape[0]
            total_loss = cont_loss + sum(class_losses.values())
            self.log(
                "metrics/train/time_per_step",
                time.time() - start_time,
                batch_size=B,
            )
            self.log("metrics/train/total_loss", total_loss, batch_size=B)
            self.log("metrics/train/contrastive_loss", cont_loss, batch_size=B)
            for label, c_loss in class_losses.items():
                self.log(f"metrics/train/{label}_loss", c_loss, batch_size=B)
                self.log(f"metrics/train/{label}_acc", class_accs[label], batch_size=B)
            return {"loss": total_loss, "logits": reprs}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        argmax_acc = self.calc_argmax_acc(outputs["logits"])
        B = batch[0].shape[0]
        self.log("metrics/train/acc_structure", argmax_acc["acc_str"], batch_size=B)
        self.log("metrics/train/acc_sequence", argmax_acc["acc_seq"], batch_size=B)

    def validation_step(self, batch, batch_idx):
        start_time = time.time()
        if self.dataset_name != "scope_10":
            B = batch[0].shape[0]
            loss, logits = self.forward(batch)
            self.log("metrics/val/loss", loss, batch_size=B, sync_dist=True)
            self.log(
                "metrics/val/time_per_step",
                time.time() - start_time,
                batch_size=B,
                sync_dist=True,
            )
            return {"loss": loss, "logits": logits}
        else:
            cont_loss, reprs, class_losses, class_outs, class_accs = self.forward(batch)
            B = batch["contrastive"][0].shape[0]
            total_loss = cont_loss + sum(class_losses.values())
            self.log(
                "metrics/val/time_per_step",
                time.time() - start_time,
                batch_size=B,
            )
            self.log("metrics/val/total_loss", total_loss, batch_size=B)
            self.log("metrics/val/contrastive_loss", cont_loss, batch_size=B)
            for label, c_loss in class_losses.items():
                self.log(f"metrics/val/{label}_loss", c_loss, batch_size=B)
                self.log(f"metrics/val/{label}_acc", class_accs[label], batch_size=B)
            return {"loss": total_loss, "logits": reprs}

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        argmax_acc = self.calc_argmax_acc(outputs["logits"])
        B = batch[0].shape[0]
        self.log(
            "metrics/val/acc_structure",
            argmax_acc["acc_str"],
            batch_size=B,
            sync_dist=True,
        )
        self.log(
            "metrics/val/acc_sequence",
            argmax_acc["acc_seq"],
            batch_size=B,
            sync_dist=True,
        )

    def configure_optimizers(self) -> torch.optim.Adam:
        """Return Optimizer

        Returns:
            torch.optim.Adam: Adam Optimizer
        """
        optim_params = self.optim_args["optim_args"]
        scheduler_params = self.optim_args["scheduler"]

        optimizer = Adam(self.parameters(), **optim_params)
        if scheduler_params["type"] == "warmup_cosine_schedule":
            self.scheduler_lamba = WarmupCosineFactorLambda(
                warmup_steps=scheduler_params["warmup_steps"],
                max_steps=scheduler_params["max_steps"],
                max_lr=scheduler_params["max_lr"],
                final_lr=scheduler_params["final_lr"],
                eps=scheduler_params["eps"],
            )
            lr_scheduler = LambdaLR(
                optimizer=optimizer,
                lr_lambda=self.scheduler_lamba.compute_lr_factor,
                verbose=scheduler_params["verbose"],
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                },
            }
        else:
            return optimizer

    def lr_scheduler_step(self, scheduler, metric):
        """Override lr scheduler step due to custom scheduler"""
        scheduler.step()

    @staticmethod
    def num_params(model: nn.Module) -> int:
        """Calculate the number of parameters

        Args:
            model (nn.Module): PyTorch Model
        Returns:
            int: Total Number of params
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
