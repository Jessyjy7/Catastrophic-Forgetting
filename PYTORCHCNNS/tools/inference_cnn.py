import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import argparse as ap
import torch
import numpy as np
from torch.backends import cudnn
from model_zoo.utils import *
from model_zoo import datasets
from model_zoo import models

parser = ap.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--test-batch-size', type=int, default=400, help='input batch size for testing (default: 100)')
args = parser.parse_args()

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')
print("Device: {}".format(device))
if cuda:
    torch.backends.cudnn.deterministic=False
    map_location = None
else:
    map_location = lambda storage, loc: storage

dataset = datasets.__dict__[args.dataset]
test_loader = dataset.load_test_data(batch_size=args.test_batch_size, cuda=cuda)

if args.dataset == 'mnist':
    input_channels = 1
    out_classes = 10
elif args.dataset == 'cifar10':
    input_channels = 3
    out_classes = 10

checkpoint = torch.load(args.checkpoint, map_location=map_location)
weights = checkpoint["weights"]
model = models.__dict__[args.model](input_channels=input_channels, out_classes=out_classes)
model.to(device)
model.load_state_dict(weights)
layers, _, _ = count_layers(model)

print("Inference Accuracy:")
top1, top5 = test(test_loader, model, device, None, True)
print(top1, top5)
