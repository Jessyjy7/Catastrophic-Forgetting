import torch
from torch.autograd import Variable
import numpy as np
import shutil
import os
import copy
from time import perf_counter
from torch import nn
import model_zoo.models as models


__all__ = ['train', 'test', 'count_layers']


def train(train_loader, model, criterion, optimizer, device, quantization=False):
    model.train()
    start = perf_counter()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        res = []
        for k in topk:
            _, pred = output.topk(k, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            res.append(correct_k)
        return res


def test(test_loader, model, device, criterion=None, quantization=False):
    top1, top5, test_loss = 0, 0, 0
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if(criterion is not None):
                test_loss += criterion(output, target).item()
            prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
            top1 += prec1[0]
            top5 += prec5[0]
    top1_acc = 100. * top1 / len(test_loader.sampler)
    top5_acc = 100. * top5 / len(test_loader.sampler)
    if (criterion is not None):
        test_loss /= len(test_loader.sampler)
        return top1_acc, test_loss
    return top1_acc, top5_acc



def count_layers(model):
    conv_layers = 0
    fc_layers = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers = conv_layers + 1
        elif isinstance(module, nn.Linear):
            fc_layers = fc_layers + 1
    return conv_layers+fc_layers, conv_layers, fc_layers
