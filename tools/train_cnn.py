import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import argparse as ap
import torch
from torch import nn
from model_zoo.utils import *
from model_zoo import datasets
from model_zoo import models
from model_zoo.datasets.digit_loader import get_digit_loader
import time
import copy

parser = ap.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--checkpoint', type=str, required=True, help='prefix name for the checkpoints')
parser.add_argument('--train-ratio', type=float, default=0.9)
parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=500, help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train (default: 200)')
args = parser.parse_args()

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else 'cpu')
print("Device: {}".format(device))

if cuda:
    torch.backends.cudnn.deterministic=False
    map_location = None
else:
    map_location = lambda storage, loc: storage

if args.dataset == 'mnist':
    input_channels = 1
    out_classes = 10
    test_loader = datasets.mnist.load_test_data(batch_size=args.test_batch_size, cuda=cuda)
    train_loader, valid_loader = datasets.mnist.load_train_val_data(batch_size=args.batch_size, train_val_split=args.train_ratio, cuda=cuda)

elif args.dataset == 'cifar10':
    input_channels = 3
    out_classes = 10
    test_loader = datasets.cifar10.load_test_data(batch_size=args.test_batch_size, cuda=cuda)
    train_loader, valid_loader = datasets.cifar10.load_train_val_data(batch_size=args.batch_size, train_val_split=args.train_ratio, cuda=cuda)



if(args.model == "lenet"):
    model = models.LeNet(input_channels=input_channels, out_classes=out_classes)
if(args.model == "resnet8"):
    model = models.ResNet8(input_channels=input_channels, out_classes=out_classes)
model.to(device)

for name, module in model.named_modules():
   if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
       torch.nn.init.xavier_uniform_(module.weight)
       module.bias.data.fill_(0)

layers, _, _ = count_layers(model)

criterion = torch.nn.CrossEntropyLoss()
criterion.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250, 350], gamma=.1)

print("Training begins\n")
start = time.time()
best_accuracy = 0
for epoch in range(1, args.epochs+1):
    train(train_loader, model, criterion, optimizer, device)
    scheduler.step()
    accuracy, _ = test(valid_loader, model, device, criterion)
    print("Epoch {}: Accuracy = {}".format(epoch, accuracy))
    is_best = accuracy > best_accuracy
    if is_best:
        best_accuracy = accuracy
        best_model = copy.deepcopy(model)
        
train_accuracy, _ = test(train_loader, best_model, device, criterion)
valid_accuracy, _ = test(valid_loader, best_model, device, criterion)
test_accuracy, _ = test(test_loader, best_model, device, criterion)

torch.save({"training_epochs": args.epochs, "weights": best_model.state_dict(),
            "train_accuracy": train_accuracy, "valid_accuracy": valid_accuracy, "test_accuracy": test_accuracy}, args.checkpoint)

print("Elapsed time in minutes = {}".format((time.time()-start)/60))
print("TRAIN ACCURACY = {}".format(train_accuracy))
print("VALID ACCURACY = {}".format(valid_accuracy))
print("TEST  ACCURACY = {}".format(test_accuracy))
print("\n\n\n\n")

# Helper function to calculate accuracy on a specific digit
def calculate_digit_accuracy(digit, model, device):
    """Calculate accuracy on a specific digit from the test set."""
    digit_loader = get_digit_loader(digit, batch_size=args.test_batch_size, train=False)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in digit_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
    return correct / total * 100 if total > 0 else 0

# Additional training on each individual digit
def train_on_digit(digit, model, device, epochs=10):
    """Train the model on a specific digit and print relevant accuracy metrics."""
    digit_loader = get_digit_loader(digit, batch_size=args.batch_size, train=True)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for data, target in digit_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Digit {digit} - Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(digit_loader)}")

    # Calculate and print accuracy metrics
    overall_accuracy, _ = test(test_loader, model, device, criterion)  # Overall accuracy on the test set
    digit_0_accuracy = calculate_digit_accuracy(0, model, device)  # Accuracy on digit 0
    current_digit_accuracy = calculate_digit_accuracy(digit, model, device)  # Accuracy on the current digit
    
    print(f"After training on digit {digit}:")
    print(f"Overall Test Accuracy = {overall_accuracy}")
    print(f"Overall Train Accuracy (digit {digit}) = {train_accuracy}")
    print(f"Overall Valid Accuracy (digit {digit}) = {valid_accuracy}")
    print(f"Accuracy on Digit 0 = {digit_0_accuracy}")
    print(f"Accuracy on Digit {digit} = {current_digit_accuracy}\n")

# Train separately on each digit for 10 epochs
print("\nStarting additional training on each digit separately for 10 epochs each")
for digit in range(10):
    print(f"\nTraining on digit {digit} for 10 epochs")
    start = time.time()  # Track start time for each digit's training
    train_on_digit(digit, model, device, epochs=10)