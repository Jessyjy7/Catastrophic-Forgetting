import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import argparse as ap
import torch
import pickle
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from model_zoo.utils import *
from model_zoo import datasets
from model_zoo import models
from model_zoo.datasets.digit_loader import get_digit_loader
import time
import copy
import numpy as np

parser = ap.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--checkpoint', type=str, required=True, help='prefix name for the checkpoints')
parser.add_argument('--train-ratio', type=float, default=0.9)
parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=500, help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 200)')
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

# print("Training begins\n")
# start = time.time()
# best_accuracy = 0
# for epoch in range(1, args.epochs+1):
#     train(train_loader, model, criterion, optimizer, device)
#     scheduler.step()
#     accuracy, _ = test(valid_loader, model, device, criterion)
#     print("Epoch {}: Accuracy = {}".format(epoch, accuracy))
#     is_best = accuracy > best_accuracy
#     if is_best:
#         best_accuracy = accuracy
#         best_model = copy.deepcopy(model)
        
# train_accuracy, _ = test(train_loader, best_model, device, criterion)
# valid_accuracy, _ = test(valid_loader, best_model, device, criterion)
# test_accuracy, _ = test(test_loader, best_model, device, criterion)

# torch.save({"training_epochs": args.epochs, "weights": best_model.state_dict(),
#             "train_accuracy": train_accuracy, "valid_accuracy": valid_accuracy, "test_accuracy": test_accuracy}, args.checkpoint)

# print("Elapsed time in minutes = {}".format((time.time()-start)/60))
# print("TRAIN ACCURACY = {}".format(train_accuracy))
# print("VALID ACCURACY = {}".format(valid_accuracy))
# print("TEST  ACCURACY = {}".format(test_accuracy))
# print("\n\n\n\n")

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
    # print(f"Overall Train Accuracy (digit {digit}) = {train_accuracy}")
    # print(f"Overall Valid Accuracy (digit {digit}) = {valid_accuracy}")
    print(f"Accuracy on Digit 0 = {digit_0_accuracy}")
    print(f"Accuracy on Digit {digit} = {current_digit_accuracy}\n")
    
def train_on_digit_with_replay(digit, model, device, epochs=10):
    digit_loader = get_digit_loader(digit, batch_size=args.batch_size, train=True)
    model.train()

    # Prepare replay buffer for digit 0
    replay_images = torch.tensor(decoded_digit_samples[0], dtype=torch.float32).unsqueeze(1)  # Add channel dimension
    replay_labels = torch.zeros(replay_images.size(0), dtype=torch.long)  # Label 0 for digit 0

    for epoch in range(epochs):
        running_loss = 0.0
        for data, target in digit_loader:
            # Concatenate replay buffer data with current digit's data
            data = torch.cat([data, replay_images.to(device)], dim=0)
            target = torch.cat([target, replay_labels.to(device)], dim=0)
            
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Digit {digit} - Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(digit_loader)}")

    # Calculate and print accuracy metrics
    overall_accuracy, _ = test(test_loader, model, device, criterion)
    digit_0_accuracy = calculate_digit_accuracy(0, model, device)
    current_digit_accuracy = calculate_digit_accuracy(digit, model, device)
    
    print(f"After training on digit {digit}:")
    print(f"Overall Test Accuracy = {overall_accuracy}")
    # print(f"Overall Train Accuracy (digit {digit}) = {train_accuracy}")
    # print(f"Overall Valid Accuracy (digit {digit}) = {valid_accuracy}")
    print(f"Accuracy on Digit 0 = {digit_0_accuracy}")
    print(f"Accuracy on Digit {digit} = {current_digit_accuracy}\n")

# Training function with interleaved data strategy
def train_on_digit_with_interleaving(digit, model, device, epochs=5, batch_size=64):
    print(f"\nTraining on digit {digit} with interleaved data...")
    model.train()

    # Load data for the current digit
    current_digit_loader = get_digit_loader(digit, batch_size=batch_size, train=True)
    
    # Prepare interleaved dataset: 10% (100 samples) from all digits + 90% (900 samples) from the current digit
    # all_digits_loaders = [get_digit_loader(d, batch_size=batch_size, train=True) for d in range(10)]
    
    # Prepare interleaved dataset: 10% (10 samples per digit) + 90% (900 samples) from the current digit
    all_digits_data = []
    for d in range(10):  # Loop through all digits
        loader = get_digit_loader(d, batch_size=batch_size, train=True)
        digit_samples = []
        for data, labels in loader:
            digit_samples.append((data, labels))
            if len(digit_samples) >= 10:  # Collect exactly 10 samples per digit
                break
        all_digits_data.extend(digit_samples)

        
    # all_digits_data = [(x.to(device), y.to(device)) for x, y in all_digits_data]
    
    # Load full current digit data
    current_digit_data = []
    for data, labels in current_digit_loader:
        current_digit_data.append((data.to(device), labels.to(device)))
        if len(current_digit_data) >= 900:  # Limit to 900 samples
            break

    # Combine interleaved data
    combined_data = all_digits_data + current_digit_data
    combined_dataset = ConcatDataset([torch.utils.data.TensorDataset(x, y) for x, y in combined_data])
    combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        for data, target in combined_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Digit {digit} - Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(combined_loader)}")

    # Test the model on seen digits up to this point
    # test_seen_digits(model, device, digit)
    
    calculate_accuracies(model, device, digit)

# Function to test accuracy on seen digits
def test_seen_digits(model, device, max_digit):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for d in range(max_digit + 1):  # Only test on digits seen so far
            loader = get_digit_loader(d, batch_size=64, train=False)
            correct = 0
            total = 0
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output, 1)
                correct += (predicted == target).sum().item()
                total += target.size(0)
            print(f"Accuracy on digit {d}: {100.0 * correct / total:.2f}%")
            total_correct += correct
            total_samples += total
    print(f"Overall accuracy on seen digits: {100.0 * total_correct / total_samples:.2f}%\n")
    
def calculate_accuracies(model, device, max_seen_digit):
    model.eval()
    
    # Initialize accuracy storage
    total_correct_seen = 0
    total_seen_samples = 0
    total_correct_all = [0] * 10  # One counter for each digit
    total_samples_all = [0] * 10  # One counter for each digit
    
    print(f"\nAccuracy results after training on digit {max_seen_digit}:\n")
    with torch.no_grad():
        for digit in range(10):
            loader = get_digit_loader(digit, batch_size=64, train=False)
            correct = 0
            total = 0
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output, 1)
                correct += (predicted == target).sum().item()
                total += target.size(0)
            
            # Store per-digit accuracy
            total_correct_all[digit] = correct
            total_samples_all[digit] = total
            
            # Print accuracy for each digit
            if total > 0:
                accuracy = 100.0 * correct / total
                print(f"Accuracy on digit {digit}: {accuracy:.2f}%")
            else:
                print(f"Accuracy on digit {digit}: No samples available")
            
            # Add to "seen digits" if within max_seen_digit
            if digit <= max_seen_digit:
                total_correct_seen += correct
                total_seen_samples += total
    
    # Calculate and print overall accuracy on seen digits
    if total_seen_samples > 0:
        overall_accuracy_seen_digits = 100.0 * total_correct_seen / total_seen_samples
        print(f"\nOverall accuracy on seen digits (0–{max_seen_digit}): {overall_accuracy_seen_digits:.2f}%")
    else:
        print("\nNo samples available for seen digits")

# Main training loop
# for digit in range(10):
#     train_on_digit_with_interleaving(digit, model, device, epochs=args.epochs, batch_size=args.batch_size)
    
# Train separately on each digit for 10 epochs
# print("\nStarting additional training on each digit separately for 10 epochs each")
# for digit in range(10):
#     print(f"\nTraining on digit {digit} for 10 epochs")
#     start = time.time()  # Track start time for each digit's training
#     train_on_digit(digit, model, device, epochs=10)

# print("\nStarting training with the replay buffer for 10 digits")
# # Load replay buffer
# with open("replay_buffer.pkl", "rb") as f:
#     decoded_digit_samples = pickle.load(f)
# print("\nReplay buffer loaded successfully.")

# for digit in range(10):
#     print(f"\nTraining on digit {digit} for 10 epochs")
#     start = time.time()  # Track start time for each digit's training
#     train_on_digit_with_replay(digit, model, device, epochs=10)

# Helper function for evaluation
def evaluate_accuracy(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    digit_correct = [0] * 10
    digit_total = [0] * 10
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
            for i in range(10):
                digit_correct[i] += ((predicted == target) & (target == i)).sum().item()
                digit_total[i] += (target == i).sum().item()
    overall_accuracy = correct / total * 100
    per_digit_accuracy = [digit_correct[i] / digit_total[i] * 100 if digit_total[i] > 0 else 0 for i in range(10)]
    return overall_accuracy, per_digit_accuracy

# Sequential training without replay buffer
def sequential_train_without_buffer(model, device, criterion, optimizer, epochs, test_loader):
    print("\nStarting sequential training without replay buffer")
    for digit in range(10):
        print(f"\nTraining on digit {digit} only for {epochs} epochs")
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

        # Evaluate the model
        overall_acc, per_digit_acc = evaluate_accuracy(model, test_loader, device)
        print(f"After training on digit {digit}:")
        print(f"Overall Accuracy: {overall_acc:.2f}%")
        for i, acc in enumerate(per_digit_acc):
            print(f"Accuracy on Digit {i}: {acc:.2f}%")

# Sequential training with incremental replay buffer
def sequential_train_with_buffer(model, device, criterion, optimizer, epochs, test_loader):
    print("\nStarting sequential training with incremental replay buffer")
    replay_buffer = []
    for digit in range(10):
        print(f"\nTraining on digit {digit} with replay buffer for {epochs} epochs")
        # Update replay buffer with 100 samples from the current digit
        digit_loader = get_digit_loader(digit, batch_size=args.batch_size, train=True)
        digit_data = []
        digit_labels = []
        for data, target in digit_loader:
            digit_data.extend(data[:100])
            digit_labels.extend(target[:100])
            if len(digit_data) >= 100:
                digit_data = digit_data[:100]
                digit_labels = digit_labels[:100]
                break
        replay_buffer.extend(zip(digit_data, digit_labels))

        # Create combined loader
        combined_data = [x[0] for x in replay_buffer]
        combined_labels = [x[1] for x in replay_buffer]
        combined_loader = DataLoader(
            TensorDataset(torch.stack(combined_data), torch.tensor(combined_labels)),
            batch_size=args.batch_size, shuffle=True
        )

        # Train on combined loader
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for i, (data, target) in enumerate(combined_loader):
                if digit == 1 and epoch == 0 and i == 0:  # Print first batch data for digit 1
                    print("\nFirst Batch Data (Digit 1):")
                    print(f"Data Shape: {data.shape}")
                    print(f"Labels: {target.tolist()}")
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Digit {digit} - Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(combined_loader)}")

        # Evaluate the model
        overall_acc, per_digit_acc = evaluate_accuracy(model, test_loader, device)
        print(f"After training on digit {digit} with replay buffer:")
        print(f"Overall Accuracy: {overall_acc:.2f}%")
        for i, acc in enumerate(per_digit_acc):
            print(f"Accuracy on Digit {i}: {acc:.2f}%")

# Run the experiments
print("Running experiments")
sequential_train_without_buffer(model, device, criterion, optimizer, args.epochs, test_loader)

# Reset the model
model = models.LeNet(input_channels=input_channels, out_classes=out_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

sequential_train_with_buffer(model, device, criterion, optimizer, args.epochs, test_loader)
