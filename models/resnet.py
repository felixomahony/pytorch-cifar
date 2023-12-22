'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import col_group as cg

import math


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, conv, bn, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = conv(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = bn(planes)
        self.conv2 = conv(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = bn(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                conv(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                bn(self.expansion*planes)
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
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, group=False, n_groups=1):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.group = group

        def groupconv(*args, **kwargs):
            return cg.GroupConv(*args, **kwargs, n_groups=n_groups)
        conv = nn.Conv2d if not group else groupconv
        def groupbn(*args, **kwargs):
            return cg.GroupBatchNorm2d(*args, **kwargs, n_groups=n_groups)
        bn = nn.BatchNorm2d if not group else groupbn

        shapes = [64, 128, 256, 512]
        if group:
            shapes = [int(s/math.sqrt(n_groups)) for s in shapes]
            self.in_planes = int(self.in_planes/math.sqrt(n_groups))
        self.conv1 = conv(3, shapes[0], kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = bn(shapes[0])
        self.layer1 = self._make_layer(block, shapes[0], num_blocks[0], conv=conv, bn=bn, stride=1)
        self.layer2 = self._make_layer(block, shapes[1], num_blocks[1], conv=conv, bn=bn, stride=2)
        self.layer3 = self._make_layer(block, shapes[2], num_blocks[2], conv=conv, bn=bn, stride=2)
        self.layer4 = self._make_layer(block, shapes[3], num_blocks[3], conv=conv, bn=bn, stride=2)

        self.group_pool = None if not group else cg.GroupPool(n_groups)

        self.linear = nn.Linear(shapes[3]*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, conv, bn, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, conv=conv, bn=bn, stride=stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        if self.group:
            out = self.group_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(group=False, n_groups=1):
    return ResNet(BasicBlock, [2, 2, 2, 2], group=group, n_groups=n_groups)


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
