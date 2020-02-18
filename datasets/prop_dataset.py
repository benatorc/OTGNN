import json
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import rdkit.Chem as Chem
import numpy as np

import pdb


class PropDataset(Dataset):
    def __init__(self, data_dir, data_type, split=0, parse_func=None, n_labels=1):
        self.n_labels = n_labels
        self.data = []

        # Read the split indices for the relevant split of the data
        indices_path = '%s/split_%d.json' % (data_dir, split)
        with open(indices_path, 'r+') as indices_file:
            indices_dict = json.load(indices_file)
            data_indices = indices_dict[data_type]

        if len(data_indices) == 0:
            print('No data indices loaded at: %s' % indices_path)

        with open('%s/raw.csv' % data_dir, 'r+') as data_file:
            for line_idx, line in enumerate(data_file.readlines()):
                if line_idx not in data_indices:
                    continue
                else:
                    if parse_func is None:
                        if n_labels == 1:
                            smiles, label = line.strip().split(',')
                            label = float(label)
                            self.data.append((smiles, label))
                        else:
                            splits = line.strip().split(',')
                            smiles = splits[0]
                            labels = splits[1:]
                            self.data.append((smiles, labels))
                    else:
                        self.data.append(parse_func(line.strip()))

        self.smiles, self.labels = zip(*self.data)

    def __len__(self):
        return len(self.data)

    def parse_multi_labels(self, label_list):
        labels, mask = [], []
        for label in label_list:
            if label == '':
                labels.append(0)
                mask.append(0)
            else:
                labels.append(float(label))
                mask.append(1)
        return labels, mask

    def __getitem__(self, idx):
        if self.n_labels == 1:
            return self.data[idx]
        else:
            smiles, labels = self.data[idx]
            parsed_labels, mask = self.parse_multi_labels(labels)
            return (smiles, parsed_labels, mask)

def get_loader(data_dir, data_type, batch_size, split=0, parse_func=None,
               shuffle=False, num_workers=1, n_labels=1):
    prop_dataset = PropDataset(
        data_dir, data_type, split=split, parse_func=parse_func, n_labels=n_labels)

    def combine_data(data):
        if n_labels == 1:
            batch_smiles, batch_labels = zip(*data)
            return batch_smiles, batch_labels
        else:
            batch_smiles, batch_labels, mask = zip(*data)
            return batch_smiles, batch_labels, mask

    data_loader = DataLoader(
        prop_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=combine_data,
        num_workers=num_workers)
    return data_loader
