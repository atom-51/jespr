from os import path
import numpy as np
import json
import pickle

from torch.utils.data import Dataset, DataLoader

from esm.inverse_folding import util
from esm import Alphabet


class ESMDataset(Dataset):
    def __init__(self, data_dir="data/", max_seq_len=200):
        data_dir_complete = path.join(
            path.abspath(path.join(__file__, "..")), data_dir
        )
        with open(
            path.join(data_dir_complete, "all_structures.pkl"), "rb"
        ) as f:
            self.structure_data = pickle.load(f)

        with open(path.join(data_dir_complete, "all_seqs.json"), "r") as f:
            self.sequence_data = json.load(f)

        with open(
            path.join(data_dir_complete, "all_prot_names.json"), "r"
        ) as f:
            self.prot_names = json.load(f)
        self.filter_data(max_seq_len)

    def filter_data(self, max_seq_len: int):
        """Filter the dataset by sequence length

        Args:
            max_seq_len (int): Maximum sequence length
        """
        structure_data = []
        sequence_data = []
        prot_names = []
        for i, seq in enumerate(self.sequence_data):
            if len(seq) <= max_seq_len:
                structure_data.append(self.structure_data[i])
                sequence_data.append(seq)
                prot_names.append(self.prot_names[i])

        # update the dataset
        self.structure_data = structure_data
        self.sequence_data = sequence_data
        self.prot_names = prot_names

    def __len__(self):
        return len(self.prot_names)

    def __getitem__(self, idx):
        """Returns the idx-th protein in the dataset

        Args:
            idx (Protein index): Protein Index

        Returns:
            tuple: A tuple containing the
            protein structure, confidence (None), and the protein sequence
        """
        # chunk where the idx-th protein is located
        protein_structure = self.structure_data[idx]
        protein_sequence = self.sequence_data[idx]
        # protein_name = self.currently_opened_chunk["prot_names"][protein_idx]
        return protein_structure, None, protein_sequence


class ESMDataLoader(DataLoader):
    def __init__(
        self,
        alphabet: Alphabet,
        data_dir="data/",
        batch_size=64,
        shuffle=True,
        num_workers=1,
        **kwargs,
    ):
        self.alphabet = alphabet
        self.collate_fn = util.CoordBatchConverter(alphabet)
        self.dataset = kwargs.get("dataset", ESMDataset(data_dir=data_dir))
        self.batch_sampler = kwargs.get("batch_sampler", None)
        self.sampler = kwargs.get("sampler", None)
        super().__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            pin_memory=False,
        )
