from os import path
import numpy as np
import json
import pickle
from typing import Tuple
from argparse import Namespace

import random
from copy import deepcopy


import torch
from torch.utils.data import Dataset, DataLoader, random_split
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
                - max_seq_len (int): Max Sequence Length
        """
        if args.dataset_name == "cath":
            with open(
                path.join(args.data_dir, f"cath/{split}.pkl"), "rb"
            ) as f:
                self.data = pickle.load(f)

        # filter data by sequence length
        if args.max_seq_len is not None:
            self.filter_data(args.max_seq_len)

    def filter_data(self, max_seq_len: int) -> None:
        """Filter the dataset by sequence length

        Args:
            max_seq_len (int): Maximum sequence length
        """
        data = []
        for item in self.data:
            if len(item["seq"]) <= max_seq_len:
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
        return item["coords"].astype(np.float32), None, item["seq"]



class ESMSampler(torch.utils.data.Sampler):

    def __init__(self, data: ESMDataset, args: Namespace):
        """ESMSampler

        Args:
            data (ESMDataset): Dataset
            args (Namespace): Args. Must contain: 
                - batch_type (str): fixed or dynamic
                - batch_value (int): batch size or max token inputs for GPU
                - bin_size (int): size of bin for Sampler
        """
        self.args = args
        self.seq_len = [len(item["seq"]) for item in data]
        self.bins_normal = self.create_bin(self.seq_len, self.args.bin_size)

    def __iter__(self):
        bins = self.bins_normal
        
        for key in bins:
            random.shuffle(bins[key])
        
        final_indices = []
        total_token = 0
        index_current = 0
        final_indices.append([])
        
        for key in sorted(bins.keys(), reverse=True):
            for index in bins[key]:
                
                if self.args.batch_type == "dynamic":
                    if(total_token+key > self.args.batch_value):
                        total_token = 0
                        final_indices.append([])
                        index_current += 1
                    total_token += key
                    final_indices[index_current].append(index)
                
                if self.args.batch_type == "fixed":
                    if(len(final_indices[index_current]) > self.args.batch_value-1):
                        final_indices.append([])
                        index_current += 1
                    final_indices[index_current].append(index)
        
        random.shuffle(final_indices)       
        return iter(final_indices)
    
    def create_bin(self, data, bin_size):
        """Creates Bins for sorting the data

        Args:
            data (list): dataset
            bin_size (int): size of bin for sorting the data

        Returns:
            bin: container with indexes 
        """
        max_len = max(data)
        min_len = min(data)
        bin = {}
        current = min_len+bin_size-1
        while(current<max_len):
            bin[current] = []
            current = current + bin_size
        bin[max_len] = []
        current_index = 0
        while(True):
            dict_index = (((data[current_index]-min_len)//bin_size) + 1)*bin_size + min_len-1
            bin[min(dict_index, max_len)].append(current_index)
            current_index += 1
            if(current_index>=len(data)):
                break
        
        return bin


class ESMDataLoader(DataLoader):
    def __init__(
        self,
        esm2_alphabet: Alphabet,
        esm_if_alphabet: Alphabet,
        dataset: ESMDataset,
        sampler: ESMSampler,
        num_workers: int,
        **kwargs,
    ):
        """ESM DataLoader

        Args:
            esm2_alphabet (Alphabet): ESM-2 Alphabet
            esm_if_alphabet (Alphabet): ESM-IF Alphabet
            dataset (ESMDataset): ESMDataset.
            sampler: ESMSampler
            num_workers (int): Number of Workers
        """
        self.esm2_alphabet = esm2_alphabet
        self.esm_if_alphabet = esm_if_alphabet

        self.esm_if_batch_converter = util.CoordBatchConverter(
            self.esm_if_alphabet
        )
        self.esm2_batch_converter = self.esm2_alphabet.get_batch_converter()

        # self.collate_fn = util.CoordBatchConverter(alphabet)
        
        self.sampler = sampler #ESMSampler(batch_type = "fixed", batch_value = 64, bin_size = 6, data = dataset.data)
        
        super().__init__(
            dataset=dataset,
            num_workers=num_workers,
            batch_sampler= self.sampler,
            collate_fn=self.collate_fn,
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
                - batch_type (str): fixed or dynamic
                - batch_value (int): batch size or max token inputs for GPU
                - bin_size (int): size of bin for Sampler
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

        self.esm_if_batch_converter = util.CoordBatchConverter(
            self.esm_if_alphabet
        )
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
            self.train_sampler = ESMSampler(data=self.train_dataset.data,args=self.args)
            self.val_dataset = ESMDataset(split="val", args=self.args)
            self.val_sampler = ESMSampler(data=self.val_dataset.data,args=self.args)
        else:
            self.test_dataset = ESMDataset(split="test", args=self.args)
            self.test_sampler = ESMSampler(data=self.test_dataset.data,args=self.args)

    def train_dataloader(self) -> ESMDataLoader:
        assert self.train_dataset is not None, "Train Dataset is None"
        data_loader = ESMDataLoader(
            esm2_alphabet=self.esm2_alphabet,
            esm_if_alphabet=self.esm_if_alphabet,
            dataset=self.train_dataset,
            sampler=self.train_sampler,
            num_workers=self.args.train_num_workers,
        )
        return data_loader

    def val_dataloader(self) -> ESMDataLoader:
        assert self.val_dataset is not None, "Val Dataset is None"
        data_loader = ESMDataLoader(
            esm2_alphabet=self.esm2_alphabet,
            esm_if_alphabet=self.esm_if_alphabet,
            dataset=self.val_dataset,
            sampler=self.val_sampler,
            num_workers=self.args.val_num_workers,
        )
        return data_loader

    def test_dataloader(self):
        assert self.test_dataset is not None, "Test Dataset is None"
        data_loader = ESMDataLoader(
            esm2_alphabet=self.esm2_alphabet,
            esm_if_alphabet=self.esm_if_alphabet,
            dataset=self.test_dataset,
            sampler=self.test_sampler,
            num_workers=self.args.val_num_workers,
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


class RemoteHomologyDataset(Dataset):
    def __init__(
        self, split: str, label_to_predict: str, data_dir: str
    ) -> None:
        """Remote Homology Dataset

        Args:
            split (str): Split.
                One of train, val, test_family_holdout, test_superfamily_holdout, test_fold_holdout
            label_to_predict (str): Label to predict. One of family, superfamily, fold, class
            data_dir (str): Data directory

        Raises:
            ValueError: If Split is invalid
        """
        assert split in [
            "train",
            "val",
            "test_family_holdout",
            "test_superfamily_holdout",
            "test_fold_holdout",
        ], f"Invalid Split: {split}"

        with open(
            path.join(data_dir, f"remote_homology/{split}.pkl"), "rb"
        ) as f:
            self.data = pickle.load(f)

        assert label_to_predict in [
            "family",
            "superfamily",
            "fold",
            "class",
        ], f"Invalid label: {label_to_predict} to predict"
        self.label_to_predict = f"{label_to_predict}_label"

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple:
        """Get Item from Index

        Args:
            index (int): Index

        Returns:
            Tuple: (AA Sequence, Class Label)
        """
        entry = self.data[index]
        return (entry["primary"], entry[self.label_to_predict])


class RemoteHomologyLightning(LightningDataModule):
    def __init__(self, args: Namespace) -> None:
        super().__init__()
        self.args = args
        self.esm2_batch_converter = args.esm2_alphabet.get_batch_converter()

    def prepare_data(self):
        pass

    def setup(self, stage):
        if stage == "fit":
            self.train_dataset = RemoteHomologyDataset(
                split="train",
                label_to_predict=self.args.label_to_predict,
                data_dir=self.args.data_dir,
            )
            self.val_dataset = RemoteHomologyDataset(
                split="val",
                label_to_predict=self.args.label_to_predict,
                data_dir=self.args.data_dir,
            )
        elif stage == "test":
            self.test_dataset_family_holdout = RemoteHomologyDataset(
                split="test_family_holdout",
                label_to_predict=self.args.label_to_predict,
                data_dir=self.args.data_dir,
            )
            self.test_dataset_superfamily_holdout = RemoteHomologyDataset(
                split="test_superfamily_holdout",
                label_to_predict=self.args.label_to_predict,
                data_dir=self.args.data_dir,
            )
            self.test_dataset_fold_holdout = RemoteHomologyDataset(
                split="test_fold_holdout",
                label_to_predict=self.args.label_to_predict,
                data_dir=self.args.data_dir,
            )

    def collate_fn(self, batch: list) -> Tuple[str, int]:
        """
        Collate Function to process each batch
        through ESM2 Batch Converter

        Args:
            batch (list): List of individual items from dataset.__getitem__()

        Returns:
            tuple: tokens, labels
        """
        # Prepare input seqs for esm2 batch converter as mentioned in
        # the example here: https://github.com/facebookresearch/esm/blob/2b369911bb5b4b0dda914521b9475cad1656b2ac/README.md?plain=1#L176
        inp_seqs = [("", item[0]) for item in batch]
        class_labels = torch.tensor(
            [item[1] for item in batch], dtype=torch.long
        )

        # Process ESM-2 ->
        _labels, _strs, tokens = self.esm2_batch_converter(inp_seqs)

        return tokens, class_labels

    def train_dataloader(self) -> DataLoader:
        """Return Train Data Loader

        Returns:
            DataLoader: Train Dataloader
        """
        assert (
            self.train_dataset is not None
        ), "Setup not called with fit stage"
        dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=self.args.train_shuffle,
            num_workers=self.args.train_num_workers,
            collate_fn=self.collate_fn,
        )
        return dataloader

    def val_dataloader(self) -> DataLoader:
        """Return Val Data Loader

        Returns:
            DataLoader: Val Dataloader
        """
        assert self.val_dataset is not None, "Setup not called with fit stage"
        dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.args.batch_size,
            shuffle=self.args.train_shuffle,
            num_workers=self.args.train_num_workers,
            collate_fn=self.collate_fn,
        )
        return dataloader

    def test_dataloader(self, holdout: str) -> DataLoader:
        """Return Test Data Loader

        Args:
            holdout (str): One of family, superfamily, fold

        Returns:
            DataLoader: Test Dataloader
        """
        assert holdout in [
            "family",
            "superfamily",
            "fold",
        ], f"Invalid holdout: {holdout}"
        if holdout == "family":
            dataset = self.test_dataset_family_holdout
        elif holdout == "superfamily":
            dataset = self.test_dataset_superfamily_holdout
        elif holdout == "fold":
            dataset = self.test_dataset_fold_holdout
        assert dataset is not None, "Setup not called with test stage"

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            shuffle=self.args.train_shuffle,
            num_workers=self.args.train_num_workers,
            collate_fn=self.collate_fn,
        )
        return dataloader

    def teardown(self, stage):
        # clean up after fit or test
        # called on every process in DDP
        if stage == "fit":
            self.train_dataloader = None
            self.val_dataloader = None
        elif stage == "test":
            self.test_dataloader = None
