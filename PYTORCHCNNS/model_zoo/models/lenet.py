import torch
import torch.nn as nn
import torch.nn.functional as F

# LeNet Feature Extractor (LeNet_Pre) - Extracting FC2 Features
class LeNet_Pre(nn.Module):
    def __init__(self):
        super(LeNet_Pre, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  
        self.pool = nn.MaxPool2d(2, 2)    
        self.conv2 = nn.Conv2d(6, 16, 5)  
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  
        self.fc2 = nn.Linear(120, 84)  

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        features = F.relu(self.fc2(x))  
        # print(f"Extracted FC2 Shape: {features.shape}")
        return features  


# LeNet Classifier (LeNet_Out) - Uses FC2 Features Directly
class LeNet_Out(nn.Module):
    def __init__(self):
        super(LeNet_Out, self).__init__()
        self.fc3 = nn.Linear(84, 10)  

    def forward(self, features):
        x = self.fc3(features)  
        return x
