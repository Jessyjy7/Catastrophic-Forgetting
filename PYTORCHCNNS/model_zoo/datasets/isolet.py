import os
import torch
from torch.utils.data import DataLoader, Subset
import torchhd.datasets
import random

def random_split(dataset, lengths, generator=None):
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the dataset.")
    if generator is None:
        generator = torch.Generator()
    # Set the seed of the generator
    if isinstance(generator, torch.Generator):
        torch.manual_seed(generator.seed())

    indices = torch.randperm(len(dataset), generator=generator).tolist()

    split_datasets = []
    current_idx = 0

    for length in lengths:
        split_indices = indices[current_idx:current_idx + length]
        split_dataset = Subset(dataset, split_indices)
        split_datasets.append(split_dataset)
        current_idx += length

    return split_datasets


def load_train_val_data(batch_size=64, train_val_split=0.9, shuffle=True, cuda=False):
    loader_kwargs = {'num_workers': 0, 'pin_memory': False} if cuda else {}

    root_dir = os.path.join('datasets', 'isolet')

    train_dataset = torchhd.datasets.ISOLET(root=root_dir, train=True, download=True)

    train_len = int(train_val_split * len(train_dataset))
    val_len = len(train_dataset) - train_len
    train_data, val_data = random_split(train_dataset, [train_len, val_len])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, **loader_kwargs)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=shuffle, **loader_kwargs)

    return train_loader, val_loader


def load_test_data(batch_size=1000, shuffle=False, sampler=None, cuda=False):
    loader_kwargs = {'num_workers': 0, 'pin_memory': False} if cuda else {}

    root_dir = os.path.join('datasets', 'isolet')

    test_dataset = torchhd.datasets.ISOLET(root=root_dir, train=False, download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, **loader_kwargs)

    return test_loader

