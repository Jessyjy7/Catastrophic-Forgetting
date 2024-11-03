import torch
import torch.nn as nn
from model_zoo.models.utils import * 

class LeNet(nn.Module):
    def __init__(self, input_channels, out_classes):
        super(LeNet, self).__init__()
        structure = [6,16,120,84]
        
        self.features = nn.Sequential(
                Convolution2D(input_channels, structure[0], k_size=5, stride=1, padding=0, with_bn=False, with_relu=True),
                nn.MaxPool2d(2),
                Convolution2D(structure[0], structure[1], k_size=5, stride=1, padding=0, with_bn=False, with_relu=True),
                nn.MaxPool2d(2), 
                Convolution2D(structure[1], structure[2], k_size=5, stride=1, padding=0, with_bn=False, with_relu=True)
                )

        self.classifier = nn.Sequential(
            FullyConnected(structure[2], structure[3], with_relu=True),
            FullyConnected(structure[3], out_classes, with_relu=False)
            )
        

    def forward(self, x):
        x = self.features(x)
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.classifier(x)
        return x