import torch
import torch.nn as nn
from NeuralModels import SILU, Perceptron
from DeepImagePrediction import IMAGE_SIZE

import math

def conv_3x3(inp, oup, stride , activation = nn.ReLU()):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        activation,
    )


def conv_1x1(inp, oup, activation = nn.ReLU()):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        activation,
    )


def SepConv_3x3(inp, oup,  activation = nn.ReLU()): #input=32, output=16
    return nn.Sequential(
        nn.Conv2d(inp, inp , 3, 1, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        activation,
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, kernel, activation = nn.ReLU()):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            activation,
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, kernel, stride, kernel // 2, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            activation,
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MnasPredictor(nn.Module):
    def __init__(self, channels=3, dimension=1, activation=SILU(), pretrained=True):
        super(MnasPredictor, self).__init__()
        self.activation = activation
        input_size = IMAGE_SIZE
        assert input_size % 32 == 0
        input_channel = int(32 * 1.1)
        self.last_channel = 1280
        self.interverted_residual_setting = [
            # t, c, n, s, k
            [3, 24, 3, 2, 3],
            [3, 40, 3, 2, 5],
            [6, 80, 3, 2, 5],
            [6, 96, 2, 1, 3],
            [6, 192, 4, 2, 5],
            [6, 320, 1, 1, 3],
        ]
        # building first two layer
        self.features = [conv_3x3(channels, input_channel, 2, activation), SepConv_3x3(input_channel, 16, activation)]
        input_channel = 16

        # building inverted residual blocks (MBConv)
        for t, c, n, s, k in self.interverted_residual_setting:
            output_channel = int(c * 1.1)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, t, k, activation))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, t, k, activation))
                input_channel = output_channel

        # building last several layers
        self.features.append(conv_1x1(input_channel, self.last_channel))
        self.features.append(nn.AdaptiveAvgPool2d(1))

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        self._initialize_weights()

        self.predictor = nn.Sequential(
            activation,
            Perceptron(1280, 1280),
            activation,
            Perceptron( 1280, dimension),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.predictor(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()