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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, group=False):
        super(BasicBlock, self).__init__()

        conv = nn.Conv2d if not group else cg.GroupConv
        bn = nn.BatchNorm2d if not group else cg.GroupBatchNorm2d

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


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, group=False):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.group=group

        conv = nn.Conv2d if not group else cg.GroupConv
        bn = nn.BatchNorm2d if not group else cg.GroupBatchNorm2d

        self.conv1 = conv(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = bn(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, group=group)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, group=group)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, group=group)
        self.group_pool = None if not group else cg.GroupPool(4)
        self.linear = nn.Linear(64*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, group):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, group=group))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        if self.group:
            out = self.group_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = F.softmax(out)
        return out


def ResNetCIFAR20(group=False):
    return ResNet(BasicBlock, [3, 3, 3], group=group)

def ResNetCIFAR32():
    return ResNet(BasicBlock, [5, 5, 5])

def ResNetCIFAR44():
    return ResNet(BasicBlock, [7, 7, 7])

def ResNetCIFAR54():
    return ResNet(BasicBlock, [9, 9, 9])


def test():
    net = ResNetCIFAR20()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

if __name__ == '__main__':
    test()
# test()
