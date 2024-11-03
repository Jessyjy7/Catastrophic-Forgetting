import torch 
import torch.nn as nn
from model_zoo.models.utils import *

class ResNet8(nn.Module):
    def __init__(self, input_channels, out_classes):
        super(ResNet8, self).__init__()
        structure = [16, 16, 32, 64]

        self.conv = Convolution2D(input_channels, structure[0], k_size=3, stride=1, padding=0, with_bn=False, with_relu=True)

        self.residual = nn.Sequential( 
            ResidualLayer(structure[0], structure[1], skip_proj=False),
            ResidualLayer(structure[1], structure[2], skip_proj=True), 
            ResidualLayer(structure[2], structure[3], skip_proj=True) 
            )

        self.pooling = nn.MaxPool2d(kernel_size=8, stride=1, padding=0) 

        self.classifier = FullyConnected(structure[3], out_classes, with_relu=False)
            

       

    def forward(self, x):
        # print(x.shape)
        x = self.conv(x)
        # print(x.shape)
        x = self.residual(x)
        # print(x.shape)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        # print(x.shape)
        return x
