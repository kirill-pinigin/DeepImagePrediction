import torch
import torch.nn as nn
from torchvision import models
from NeuralModels import SILU, Perceptron

class ResidualPredictor(nn.Module):
    def __init__(self, channels = 1, dimension=11, activation = SILU(), pretrained = True):
        super(ResidualPredictor, self).__init__()
        self.activation = activation

        self.model = models.resnet18(pretrained=pretrained)
        conv = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        weight = torch.FloatTensor(64, channels, 7, 7)
        parameters = list(self.model.parameters())
        for i in range(64):
            if channels == 1:
                weight[i, :, :, :] = parameters[0].data[i].mean(0)
            else:
                weight[i, :, :, :] = parameters[0].data[i]
        conv.weight.data.copy_(weight)
        self.model.conv1 = conv

        self.model.avgpool = nn.AvgPool2d(8)
        num_ftrs = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            Perceptron(num_ftrs, num_ftrs),
            activation,
            Perceptron(num_ftrs, dimension),
        )

    def forward(self, x):
        return self.model(x)

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True