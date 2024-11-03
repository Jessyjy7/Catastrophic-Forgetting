import torch
import torch.nn as nn

class Convolution2D(nn.Module):
    def __init__(self, in_channels, out_channels, k_size, stride=1, padding=0, bias=True, dilation=1, with_bn=False, with_relu=False):
        super(Convolution2D, self).__init__()
        convolution = nn.Conv2d(in_channels, out_channels, kernel_size=k_size, padding=padding, stride=stride, bias=bias, dilation=dilation)
        if with_bn:
            if with_relu:
                self.operation = nn.Sequential(convolution, nn.BatchNorm2d(out_channels), nn.ReLU())
            else:
                self.operation = nn.Sequential(convolution, nn.BatchNorm2d(out_channels))
        else:
            if with_relu:
                self.operation = nn.Sequential(convolution, nn.ReLU())
            else:
                self.operation = nn.Sequential(convolution)

    def forward(self, inputs):
        outputs = self.operation(inputs)
        return outputs

class DepthwiseConvolution2D(nn.Module):
    def __init__(self, channels, k_size, stride=1, padding=0, bias=True, dilation=1, with_bn=False, with_relu=False):
        super(DepthwiseConvolution2D, self).__init__()
        convolution = nn.Conv2d(channels, channels, groups=channels, kernel_size=k_size, padding=padding, stride=stride, bias=bias, dilation=dilation)
        if with_bn:
            if with_relu:
                self.operation = nn.Sequential(convolution, nn.BatchNorm2d(channels), nn.ReLU())
            else:
                self.operation = nn.Sequential(convolution, nn.BatchNorm2d(channels))
        else:
            if with_relu:
                self.operation = nn.Sequential(convolution, nn.ReLU())
            else:
                self.operation = nn.Sequential(convolution)

    def forward(self, inputs):
        outputs = self.operation(inputs)
        return outputs

class PointwiseConvolution2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=0, bias=True, dilation=1, with_bn=False, with_relu=False):
        super(PointwiseConvolution2D, self).__init__()
        convolution = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=padding, stride=stride, bias=bias, dilation=dilation)
        if with_bn:
            if with_relu:
                self.operation = nn.Sequential(convolution, nn.BatchNorm2d(out_channels), nn.ReLU())
            else:
                self.operation = nn.Sequential(convolution, nn.BatchNorm2d(out_channels))
        else:
            if with_relu:
                self.operation = nn.Sequential(convolution, nn.ReLU())
            else:
                self.operation = nn.Sequential(convolution)

    def forward(self, inputs):
        outputs = self.operation(inputs)
        return outputs



class BottleneckResidual(nn.Module):
    def __init__(self, ch_in, expansion, ch_out, reduce_dim):
        super(BottleneckResidual, self).__init__()
        self.reduce_dim = reduce_dim

        if self.reduce_dim == False:
            self.ops = nn.Sequential(
                PointwiseConvolution2D(ch_in, ch_in*expansion, stride=1, padding=0, bias=True, dilation=1, with_bn=True, with_relu=True),
                DepthwiseConvolution2D(ch_in*expansion, k_size=3, stride=1, padding=1, bias=True, dilation=1, with_bn=True, with_relu=True),
                PointwiseConvolution2D(ch_in*expansion, ch_out, stride=1, padding=0, bias=True, dilation=1, with_bn=True, with_relu=False)
                )
        else:
            self.ops = nn.Sequential(
                PointwiseConvolution2D(ch_in, ch_in*expansion, stride=1, padding=0, bias=True, dilation=1, with_bn=True, with_relu=True),
                DepthwiseConvolution2D(ch_in*expansion, k_size=3, stride=1, padding=0, bias=True, dilation=1, with_bn=True, with_relu=True),
                PointwiseConvolution2D(ch_in*expansion, ch_out, stride=1, padding=0, bias=True, dilation=1, with_bn=True, with_relu=False)
                )

    def forward(self, x):
        out = self.ops(x)
        if self.reduce_dim == False:
            return out + x
        else:
            return out



class FullyConnected(nn.Module):
    def __init__(self, in_features, out_features, bias=True, with_bn=False, with_relu=False):
        super(FullyConnected, self).__init__()
        fc = nn.Linear(in_features, out_features, bias=bias)
        
        if with_bn:
            if with_relu:
                self.operation = nn.Sequential(fc, nn.BatchNorm2d(out_features), nn.ReLU())
            else:
                self.operation = nn.Sequential(fc, nn.BatchNorm2d(out_features))
        else:
            if with_relu:
                self.operation = nn.Sequential(fc, nn.ReLU())
            else:
                self.operation = nn.Sequential(fc)

    def forward(self, inputs):
        outputs = self.operation(inputs)
        return outputs



class ResidualLayer(nn.Module):
    def __init__(self, ch_in, ch_out, skip_proj):
        super(ResidualLayer, self).__init__()
        self.skip_proj = skip_proj

        if self.skip_proj == False:
            self.pathA = nn.Sequential(
                Convolution2D(ch_in, ch_out, k_size=3, stride=1, padding=1, with_bn=True, with_relu=True),
                Convolution2D(ch_out, ch_out, k_size=3, stride=1, padding=1, with_bn=True, with_relu=False)
            )

        else:
            self.pathA = nn.Sequential(
                Convolution2D(ch_in, ch_out, k_size=3, stride=2, padding=1, with_bn=True, with_relu=True),
                Convolution2D(ch_out, ch_out, k_size=3, stride=1, padding=1, with_bn=True, with_relu=False)
            )

            self.pathB = Convolution2D(ch_in, ch_out, k_size=1, stride=2, padding=0, with_bn=True, with_relu=False) 

        self.relu = nn.ReLU() 
        
    def forward(self, x):
        out = self.pathA(x)
        if self.skip_proj == False:
            out = out + x
        else:
            out = out + self.pathB(x)
        return self.relu(out)

