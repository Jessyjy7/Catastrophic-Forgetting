import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, input_channels=1, out_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (6,14,14)
        x = self.pool(F.relu(self.conv2(x)))  # (16,7,7)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
