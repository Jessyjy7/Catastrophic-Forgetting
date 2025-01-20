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
import matplotlib.pyplot as plt

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

# Sequential training with incremental replay buffer (using original images for buffer)
def sequential_train_with_buffer(model, device, criterion, optimizer, epochs, test_loader):
    print("\nStarting sequential training with incremental replay buffer")
    replay_buffer = []

    for digit in range(10):
        print(f"\nTraining on digit {digit} with replay buffer for {epochs} epochs")
        
        # 1. Add 50 original images for each previously seen digit to the replay buffer
        for seen_digit in range(digit):  
            seen_digit_loader = get_digit_loader(seen_digit, batch_size=args.batch_size, train=True)
            seen_digit_data = []
            seen_digit_labels = []
            for data, target in seen_digit_loader:
                seen_digit_data.extend(data[:50])
                seen_digit_labels.extend(target[:50])
                if len(seen_digit_data) >= 50:
                    seen_digit_data = seen_digit_data[:50]
                    seen_digit_labels = seen_digit_labels[:50]
                    break
            replay_buffer.extend(zip(seen_digit_data, seen_digit_labels))

        # 2. Add all the original images for the current digit
        digit_loader = get_digit_loader(digit, batch_size=args.batch_size, train=True)
        current_digit_images = []
        current_digit_labels = []
        for data, target in digit_loader:
            current_digit_images.extend(data)
            current_digit_labels.extend(target)
        current_digit_images = current_digit_images[:5000]  
        current_digit_labels = current_digit_labels[:5000]

        buffer_images = [img.clone().detach().unsqueeze(0) if len(img.shape) == 3 else img.clone().detach() for img, _ in replay_buffer]
        buffer_labels = [torch.tensor(label, dtype=torch.long) for _, label in replay_buffer]
        current_digit_images = [img.clone().detach().unsqueeze(0) for img in current_digit_images]

        combined_images = buffer_images + current_digit_images
        combined_labels = buffer_labels + current_digit_labels

        combined_dataset = TensorDataset(torch.cat(combined_images, dim=0), torch.tensor(combined_labels))
        combined_loader = DataLoader(combined_dataset, batch_size=args.batch_size, shuffle=True)

        # 3. Train on the combined dataset
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for i, (data, target) in enumerate(combined_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Digit {digit} - Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(combined_loader)}")

        # 4. Evaluate the model after training on each digit
        overall_acc, per_digit_acc = evaluate_accuracy(model, test_loader, device)
        print(f"After training on digit {digit} with replay buffer:")
        print(f"Overall Accuracy: {overall_acc:.2f}%")
        for i, acc in enumerate(per_digit_acc):
            print(f"Accuracy on Digit {i}: {acc:.2f}%")

    
# Sequential training with incremental replay buffer (using decoded images)
def sequential_train_with_buffer_using_decoded(model, device, criterion, optimizer, epochs, test_loader):
    print("\nStarting sequential training with decoded replay buffer")
    
    with open("replay_buffer.pkl", "rb") as f:
        decoded_replay_buffer = pickle.load(f)
        
    print("Replay Buffer Keys:", decoded_replay_buffer.keys())
    for digit, images in decoded_replay_buffer.items():
        print(f"Digit {digit}: {len(images)} images")
        print(f"Digit {digit}: Image Shape {images[0].shape}")
        
    print(f"Decoded images for Digit {digit}:")
    decoded_images = decoded_replay_buffer[digit][:5]  
    for i, img in enumerate(decoded_images):
        plt.imshow(img, cmap="gray")
        plt.title(f"Decoded Image {i} (Digit {digit})")
        plt.axis("off")
        plt.show()
        
    print("Decoded replay buffer loaded successfully.\n")

    for digit in range(10):
        print(f"\nTraining on digit {digit} with replay buffer for {epochs} epochs")
        
        # 1. Add 50 decoded images for each previously seen digit to the replay buffer
        replay_images = []
        replay_labels = []
        for seen_digit in range(digit):  
            decoded_images = decoded_replay_buffer[seen_digit][:50]  
            replay_images.extend([torch.tensor(img, dtype=torch.float32).unsqueeze(0) for img in decoded_images])
            replay_labels.extend([seen_digit] * len(decoded_images))  
            
        # 2. Add all the original images for the current digit
        digit_loader = get_digit_loader(digit, batch_size=args.batch_size, train=True)
        current_digit_images = []
        current_digit_labels = []
        for data, target in digit_loader:
            current_digit_images.extend(data)
            current_digit_labels.extend(target)
        current_digit_images = current_digit_images[:5000]  
        current_digit_labels = current_digit_labels[:5000]

        # Combine replay buffer (decoded images) and current digit original data
        buffer_images = [torch.tensor(img, dtype=torch.float32).unsqueeze(0) if len(img.shape) == 3 else torch.tensor(img, dtype=torch.float32) for img in replay_images]
        buffer_labels = [torch.tensor(label, dtype=torch.long) for label in replay_labels]

        current_digit_images = [img.unsqueeze(0) if len(img.shape) == 3 else img for img in current_digit_images]
        current_digit_labels = [label for label in current_digit_labels]

        # Ensure all images have 4 dimensions
        combined_images = buffer_images + current_digit_images
        combined_images = [img.unsqueeze(0) if len(img.shape) == 3 else img for img in combined_images]

        combined_labels = buffer_labels + [torch.tensor(label, dtype=torch.long) for label in current_digit_labels]

        # Stack tensors to create a dataset
        combined_dataset = TensorDataset(torch.cat(combined_images, dim=0), torch.stack(combined_labels))
        combined_loader = DataLoader(combined_dataset, batch_size=args.batch_size, shuffle=True)

        # 3. Train on the combined dataset
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for i, (data, target) in enumerate(combined_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Digit {digit} - Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(combined_loader)}")

        # 4. Evaluate the model after training on each digit
        overall_acc, per_digit_acc = evaluate_accuracy(model, test_loader, device)
        print(f"After training on digit {digit} with replay buffer:")
        print(f"Overall Accuracy: {overall_acc:.2f}%")
        for i, acc in enumerate(per_digit_acc):
            print(f"Accuracy on Digit {i}: {acc:.2f}%")
            
            
def train_with_decoded_buffer_only(model, device, criterion, optimizer, epochs, batch_size):
    print("\nStarting training with decoded buffer only")
    
    # Load the decoded replay buffer from the pickle file
    with open("replay_buffer.pkl", "rb") as f:
        decoded_replay_buffer = pickle.load(f)

    print("Replay Buffer Keys:", decoded_replay_buffer.keys())
    for digit, images in decoded_replay_buffer.items():
        print(f"Digit {digit}: {len(images)} images")
        print(f"Image Shape for Digit {digit}: {images[0].shape}")

    buffer_images = []
    buffer_labels = []
    for digit in range(10): 
        decoded_images = decoded_replay_buffer[digit][:50]  
        buffer_images.extend([torch.tensor(img, dtype=torch.float32).unsqueeze(0) for img in decoded_images])
        buffer_labels.extend([digit] * len(decoded_images))  
        
    buffer_images = [img.unsqueeze(0) if len(img.shape) == 3 else img for img in buffer_images]
    buffer_labels = [torch.tensor(label, dtype=torch.long) for label in buffer_labels]

    buffer_dataset = TensorDataset(torch.cat(buffer_images, dim=0), torch.stack(buffer_labels))
    buffer_loader = DataLoader(buffer_dataset, batch_size=args.batch_size, shuffle=True)

    # Train the model on the decoded buffer
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for data, target in buffer_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(buffer_loader)}")

    # Evaluate the model after training
    print("\nEvaluating model performance on decoded buffer...")
    overall_acc, per_digit_acc = evaluate_accuracy(model, buffer_loader, device)
    print(f"Overall Accuracy: {overall_acc:.2f}%")
    for digit, acc in enumerate(per_digit_acc):
        print(f"Accuracy on Digit {digit}: {acc:.2f}%")

# Sequential training with only decoded buffer data
def train_with_decoded_buffer_only_incremental(model, device, criterion, optimizer, epochs, test_loader):
    print("\nStarting training with only decoded buffer data")
    
    with open("replay_buffer.pkl", "rb") as f:
        decoded_replay_buffer = pickle.load(f)
        
    print("Replay Buffer Keys:", decoded_replay_buffer.keys())
    for digit, images in decoded_replay_buffer.items():
        print(f"Digit {digit}: {len(images)} images")
        print(f"Digit {digit}: Image Shape {images[0].shape}")

    print("Decoded replay buffer loaded successfully.\n")

    replay_buffer_images = []
    replay_buffer_labels = []

    for digit in range(10):
        print(f"\nTraining on digit {digit} with decoded buffer for {epochs} epochs")
        
        decoded_images = decoded_replay_buffer[digit][:50]  
        replay_buffer_images.extend([torch.tensor(img, dtype=torch.float32).unsqueeze(0) for img in decoded_images])
        replay_buffer_labels.extend([digit] * len(decoded_images))  
        
        buffer_images = [img.clone().detach() for img in replay_buffer_images]
        buffer_labels = [torch.tensor(label, dtype=torch.long) for label in replay_buffer_labels]
        
        buffer_images = [img.unsqueeze(0) if len(img.shape) == 3 else img for img in buffer_images]
        
        buffer_dataset = TensorDataset(torch.cat(buffer_images, dim=0), torch.stack(buffer_labels))
        buffer_loader = DataLoader(buffer_dataset, batch_size=args.batch_size, shuffle=True)

        # Train on the buffer dataset
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for data, target in buffer_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Digit {digit} - Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(buffer_loader)}")
        
        # Evaluate the model after training on each digit
        overall_acc, per_digit_acc = evaluate_accuracy(model, test_loader, device)
        print(f"After training on digit {digit} with decoded buffer:")
        print(f"Overall Accuracy: {overall_acc:.2f}%")
        for i, acc in enumerate(per_digit_acc):
            print(f"Accuracy on Digit {i}: {acc:.2f}%")


# Run the experiments
print("Running experiments")
# sequential_train_without_buffer(model, device, criterion, optimizer, args.epochs, test_loader)

# Reset the model
# model = models.LeNet(input_channels=input_channels, out_classes=out_classes).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# sequential_train_with_buffer(model, device, criterion, optimizer, args.epochs, test_loader)

# Reset the model
# model = models.LeNet(input_channels=input_channels, out_classes=out_classes).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# train_with_decoded_buffer_only_incremental(model, device, criterion, optimizer, args.epochs, test_loader)

# Reset the model
# model = models.LeNet(input_channels=input_channels, out_classes=out_classes).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# train_with_decoded_buffer_only(model, device, criterion, optimizer, args.epochs, test_loader)

sequential_train_with_buffer_using_decoded(model, device, criterion, optimizer, args.epochs, test_loader)
