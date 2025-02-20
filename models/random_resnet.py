import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes) 
            )
            

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Randomfront(nn.Module):
    def __init__(self, epsilon=8./255.):
        super(Randomfront, self).__init__()
        self.epsilon = epsilon
    
    def forward(self, x):
        iplset = (torch.randint(32, 36, (1,)), torch.randint(32, 36, (1,)))
        x = F.interpolate(x, iplset, mode='bilinear')
        padset = (torch.randint(0, 3, (1,)), torch.randint(0, 3, (1,)), torch.randint(0, 3, (1,)), torch.randint(0, 3, (1,)))
        x = F.pad(x, padset)
        x = F.interpolate(x, (32, 32), mode='bilinear')
        return x


class Random_ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(Random_ResNet, self).__init__()
        self.in_planes = 64

        self.random = Randomfront(8./255.)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.random(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def Random_ResNet18():
    return Random_ResNet(BasicBlock, [2, 2, 2, 2])


def Random_ResNet34():
    return Random_ResNet(BasicBlock, [3, 4, 6, 3])


def Random_ResNet50():
    return Random_ResNet(Bottleneck, [3, 4, 6, 3])


def Random_ResNet101():
    return Random_ResNet(Bottleneck, [3, 4, 23, 3])


def Random_ResNet152():
    return Random_ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = Random_ResNet34()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())