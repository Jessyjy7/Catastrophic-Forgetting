import torch
from torch.utils.data.sampler import Sampler


class SubsetSampler(Sampler):
    def __init__(self, size):
        self.size = size

    def __iter__(self):
        return iter(torch.arange(self.size).long())

    def __len__(self):
        return self.size


class RangeSampler(Sampler):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __iter__(self):
        return iter(torch.arange(self.start, self.end).long())

    def __len__(self):
        return self.end-self.start
