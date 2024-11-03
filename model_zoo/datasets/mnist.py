import os
import torch

from torch.utils.data.dataset import Subset
from torchvision.datasets import MNIST
import torchvision.transforms as tvt
import math
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
        split_dataset = torch.utils.data.Subset(dataset, split_indices)
        split_datasets.append(split_dataset)
        current_idx += length

    return split_datasets


def load_train_val_data(batch_size=64, train_val_split=0.9, shuffle=True, cuda=False):
    loader_kwargs = {'num_workers': 0, 'pin_memory': False} if cuda else {}
    transform_train = tvt.Compose([ tvt.CenterCrop(32), tvt.RandomHorizontalFlip(), tvt.ToTensor()])
    loader = torch.utils.data.DataLoader( 
        MNIST(os.path.join('datasets', 'mnist'),
            train=True, download=True, transform=transform_train), 
            batch_size=batch_size, shuffle=shuffle, **loader_kwargs)
    
    train_len = int(train_val_split * len(loader.dataset))
    val_len = len(loader.dataset) - train_len

    train_data, val_data = random_split(loader.dataset, [train_len, val_len]) # generator=torch.Generator().manual_seed(42)
    # train_data, val_data = torch.utils.data.random_split(loader.dataset, [train_val_split, 1-train_val_split]) # generator=torch.Generator().manual_seed(42)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, **loader_kwargs)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=shuffle, **loader_kwargs)
    return train_loader, val_loader

def load_test_data(batch_size=1000, shuffle=False, sampler=None, cuda=False):
    loader_kwargs = {'num_workers': 0, 'pin_memory': False} if cuda else {}
    transform_test =  tvt.Compose([ tvt.CenterCrop(32), tvt.ToTensor() ])
    loader = torch.utils.data.DataLoader( 
        MNIST(os.path.join('datasets', 'mnist'), 
        train=False, download=True, transform=transform_test), 
        batch_size=batch_size, shuffle=False, **loader_kwargs)
    return loader




