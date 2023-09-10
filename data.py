from os import path
import numpy as np
import json
import pickle
from typing import Tuple
from argparse import Namespace
import random


import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler, DistributedSampler
from lightning.pytorch.core import LightningDataModule

from esm.inverse_folding import util
from esm import Alphabet


class ESMDataset(Dataset):
    def __init__(self, split: str, args: Namespace) -> None:
        """ESM Dataset: torch.utils.data.Dataset

        Args:
            split (str): Split (train, val, test)
            args (Namespace): Args for ESMDataset. Must Contain:
                - data_dir (str): Data Directory
                - max_seq_len (int): Max Sequence length
        """
        self.args = args
        assert args.dataset_name in [
            "cath",
            "pdb",
            "pdb_extended",
            "scope_10",
        ], "Invalid Dataset Name"
        with open(
            path.join(args.data_dir, f"{args.dataset_name}/{split}.pkl"), "rb"
        ) as f:
            self.data = pickle.load(f)

        # filter data by sequence length
        if args.max_seq_len is not None:
            self.filter_data(args.max_seq_len, args.min_seq_len)

    def filter_data(self, max_seq_len: int, min_seq_len: int) -> None:
        """Filter the dataset by sequence length

        Args:
            max_seq_len (int): Maximum sequence length
        """
        data = []
        for item in self.data:
            if len(item["seq"]) <= max_seq_len and len(item["seq"]) >= min_seq_len:
                data.append(item)

        # update the dataset
        self.data = data

    def __len__(self) -> int:
        """Returns the length of the dataset

        Returns:
            _type_: _description_
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, None, str]:
        """Returns the idx-th protein in the dataset

        Args:
            idx (int): Protein Index

        Returns:
            Tuple[np.ndarray, None, str]: Protein Structure Data, None, Protein Sequence Data
        """
        # chunk where the idx-th protein is located
        item = self.data[idx]
        if self.args.dataset_name == "scope_10":
            return {
                "contrastive": (
                    item["coords"].astype(np.float32),
                    None,
                    item["seq"],
                ),
                "classification": (
                    item["class"],
                    item["fold"],
                    item["domain"],
                    item["superfamily"],
                    item["family"],
                    item["sp"],
                ),
            }
        else:
            return (
                item["coords"].astype(np.float32),
                None,
                item["seq"],
            )


class ESMBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, sampler, batch_size, drop_last):
        """ESM Batch Sampler

        Args:
            data (list): Data from ESMDataset.data
            args (Namespace): Namespace containing the following args:
                - sampler (dict): Sampler args. Must contain: bin_size
                - min_seq_len (int): Minimum Sequence Length
                - max_seq_len (int): Maximum Sequence Length
                - batch_size (int): Batch Size
        """
        if type(sampler) == SequentialSampler:
            self.dataset = sampler.data_source
        elif type(sampler) == DistributedSampler:
            self.dataset = sampler.dataset

        self.sampler = sampler
        self.data = self.dataset.data
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.bins = self.create_bins()
        self.batches = self.create_batches()

    def create_bins(self) -> dict:
        """Creates bin of data indices based on bin_size

        Returns:
            dict: Data indices mapped to different bins based on the data seq length
        """
        bin_size = self.dataset.args.sampler["bin_size"]
        bins = {
            i: []
            for i in range(
                self.dataset.args.min_seq_len, self.dataset.args.max_seq_len, bin_size
            )
        }

        data_lens = [len(item["seq"]) for item in self.data]
        for data_idx, data_len in enumerate(data_lens):
            for bin_idx in list(bins.keys()):
                if data_len >= bin_idx and data_len < bin_idx + bin_size:
                    bins[bin_idx].append(data_idx)
                    break

        for bin_idx in list(bins.keys()):
            random.shuffle(bins[bin_idx])
        return bins

    def create_batches(self):
        bins_flattened = []
        for _, bin_items in self.bins.items():
            bins_flattened += bin_items
        all_batches = [
            bins_flattened[i : i + self.batch_size]
            for i in range(0, len(bins_flattened), self.batch_size)
        ]
        random.shuffle(all_batches)
        return all_batches

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)


class ESMDataLoader(DataLoader):
    def __init__(
        self,
        esm2_alphabet: Alphabet,
        esm_if_alphabet: Alphabet,
        dataset: ESMDataset,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        batch_sampler: ESMBatchSampler,
        **kwargs,
    ):
        """ESM DataLoader

        Args:
            esm2_alphabet (Alphabet): ESM-2 Alphabet
            esm_if_alphabet (Alphabet): ESM-IF Alphabet
            dataset (ESMDataset): ESMDataset.
            batch_size (int): Batch Size
            shuffle (bool): Shuffle
            num_workers (int): Number of Workers
            sampler(ESMSampler): Sampler
        """
        self.esm2_alphabet = esm2_alphabet
        self.esm_if_alphabet = esm_if_alphabet

        self.esm_if_batch_converter = util.CoordBatchConverter(self.esm_if_alphabet)
        self.esm2_batch_converter = self.esm2_alphabet.get_batch_converter()

        if dataset.args.dataset_name == "scope_10":
            collate_fn = self.collate_fn_scope
        else:
            collate_fn = self.collate_fn

        if batch_sampler is None:
            super().__init__(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=collate_fn,
            )

        else:
            super().__init__(
                dataset=dataset,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=collate_fn,
            )

    def collate_fn(
        self, batch: list
    ) -> Tuple[torch.tensor, torch.tensor, list, torch.tensor, torch.tensor]:
        """
        Collate Function to process each batch
        through ESM-IF CoordBatch Converter and ESM2 Batch Converter

        Args:
            batch (list): List of individual items from dataset.__getitem__()

        Returns:
            tuple: coords, confidence, strs, tokens, padding_mask
        """
        # Prepare input seqs for esm2 batch converter as mentioned in
        # the example here: https://github.com/facebookresearch/esm/blob/2b369911bb5b4b0dda914521b9475cad1656b2ac/README.md?plain=1#L176
        inp_seqs = [("", item[2]) for item in batch]

        # Process ESM-2 ->
        _labels, _strs, tokens = self.esm2_batch_converter(inp_seqs)

        # Process ESM-IF ->
        (
            coords,
            confidence,
            strs,
            _,
            padding_mask,
        ) = self.esm_if_batch_converter(batch)

        return coords, confidence, strs, tokens, padding_mask

    def collate_fn_scope(self, batch: list):
        """Collate Function for SCOPe

        Args:
            batch (list): _description_

        Returns:
            tuple: Tuple of xs and ys
        """
        # Prepare input seqs for esm2 batch converter as mentioned in
        # the example here: https://github.com/facebookresearch/esm/blob/2b369911bb5b4b0dda914521b9475cad1656b2ac/README.md?plain=1#L176

        inp_seqs = [("", item["contrastive"][2]) for item in batch]

        # Process ESM-2 ->
        _labels, _strs, tokens = self.esm2_batch_converter(inp_seqs)

        # Process ESM-IF ->
        (
            coords,
            confidence,
            strs,
            _,
            padding_mask,
        ) = self.esm_if_batch_converter([item["contrastive"] for item in batch])

        return {
            "contrastive": (coords, confidence, strs, tokens, padding_mask),
            "classification": [item["classification"] for item in batch],
        }


class ESMDataLightning(LightningDataModule):
    def __init__(
        self,
        esm2_alphabet: Alphabet,
        esm_if_alphabet: Alphabet,
        args: Namespace,
    ) -> None:
        """Initialize Lightning DataModule class for JESPR

        Args:
            esm2_alphabet (Alphabet): ESM-2 Alphabet
            esm_if_alphabet (Alphabet): ESM-IF Alphabet
            args (Namespace): Args. Must contain:
                - data_dir (str): Data Directory
                - split_ratio (int): Dataset split ratio. Eg: 0.8 (80% train, 20% val)
                - max_seq_len (int): Max Sequence Length
                - batch_size (int): Batch Size
                - train_shuffle (bool): Train Shuffle
                - train_num_workers (int): Train Loader - Number of Workers
                - train_pin_memory (bool): Train Loader - Pin Memory
                - val_shuffle (bool): Val Shuffle
                - val_num_workers (int): Val Loader - Number of Workers
                - val_pin_memory (bool): Val Loader - Pin Memory

        """
        super().__init__()
        self.esm2_alphabet = esm2_alphabet
        self.esm_if_alphabet = esm_if_alphabet

        self.esm_if_batch_converter = util.CoordBatchConverter(self.esm_if_alphabet)
        self.esm2_batch_converter = self.esm2_alphabet.get_batch_converter()
        self.args = args

    def prepare_data(self):
        pass

    def setup(self, stage):
        """
        Load Dataset depending on stage

        Args:
            stage (str): Stage. Either "fit" or "test"
        """
        if stage == "fit":
            self.train_dataset = ESMDataset(split="train", args=self.args)
            self.val_dataset = ESMDataset(split="val", args=self.args)
        else:
            self.test_dataset = ESMDataset(split="test", args=self.args)

        if self.args.sampler["enabled"]:
            if stage == "fit":
                self.train_sampler = ESMBatchSampler(
                    sampler=SequentialSampler(self.train_dataset),
                    batch_size=self.args.batch_size,
                    drop_last=True,
                )
                self.val_sampler = ESMBatchSampler(
                    sampler=SequentialSampler(self.val_dataset),
                    batch_size=self.args.batch_size,
                    drop_last=True,
                )
            else:
                self.test_sampler = ESMBatchSampler(
                    sampler=SequentialSampler(self.test_dataset),
                    batch_size=self.args.batch_size,
                    drop_last=True,
                )
        else:
            self.train_sampler = None
            self.val_sampler = None
            self.test_sampler = None

    def train_dataloader(self) -> ESMDataLoader:
        assert self.train_dataset is not None, "Train Dataset is None"

        data_loader = ESMDataLoader(
            esm2_alphabet=self.esm2_alphabet,
            esm_if_alphabet=self.esm_if_alphabet,
            dataset=self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=self.args.train_shuffle,
            num_workers=self.args.train_num_workers,
            batch_sampler=self.train_sampler,
        )
        return data_loader

    def val_dataloader(self) -> ESMDataLoader:
        assert self.val_dataset is not None, "Val Dataset is None"
        data_loader = ESMDataLoader(
            esm2_alphabet=self.esm2_alphabet,
            esm_if_alphabet=self.esm_if_alphabet,
            dataset=self.val_dataset,
            batch_size=self.args.batch_size,
            shuffle=self.args.val_shuffle,
            num_workers=self.args.val_num_workers,
            batch_sampler=self.val_sampler,
        )
        return data_loader

    def test_dataloader(self):
        assert self.test_dataset is not None, "Test Dataset is None"
        data_loader = ESMDataLoader(
            esm2_alphabet=self.esm2_alphabet,
            esm_if_alphabet=self.esm_if_alphabet,
            dataset=self.test_dataset,
            batch_size=self.args.batch_size,
            shuffle=self.args.val_shuffle,
            num_workers=self.args.val_num_workers,
            batch_sampler=self.test_sampler,
        )
        return data_loader

    def teardown(self, stage):
        # clean up after fit or test
        # called on every process in DDP
        if stage == "fit":
            self.train_dataset = None
            self.val_dataset = None
        elif stage == "test":
            self.test_dataset = None
        else:
            print(f"Invalid stage: {stage}")
