'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR

from vision_transforms import HueSeparation, TensorReshape

import os
import argparse

from models import *
# from utils import progress_bar

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


# Training
def train(epoch, net, trainloader, optimizer, criterion, device):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print(f"Train Accuracy: {100.*correct/total}")

def test(epoch, net, testloader, criterion, device):
    global best_acc
    # net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print(f"Test Accuracy: {100.*correct/total}")

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

def run(trainloader, testloader, nt, n_groups, num_classes=10):

    print("Groups: ", n_groups)
    print("Model Name: ", nt)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    # Model
    print('==> Building model..')
    # if nt == "resnet44":
    net = ResNet44(n_groups=n_groups, num_classes=num_classes)
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Number of parameters: {n_params}")
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

    for epoch in range(start_epoch, start_epoch+300):
        train(epoch, net, trainloader, optimizer, criterion, device=device)
        # check if testloader is an array
        if isinstance(testloader, list):
            for i, tl in enumerate(testloader):
                print(f"Testloader {i}")
                test(epoch, net, tl, criterion, device=device)
        else:
            test(epoch, net, testloader, criterion, device=device)
        scheduler.step()
