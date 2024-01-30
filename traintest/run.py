'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import logging

import os

import time

from models import *
# from utils import progress_bar

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


# Training
def train(epoch, net, trainloader, optimizer, criterion, device, n_iters_complete, n_iters):
    logging.warning('Epoch: %d' % epoch)
    logging.warning("n_iters_complete: %d" % n_iters_complete)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    if not isinstance(trainloader, list):
        for batch_idx, network_components in enumerate(trainloader):
            inputs = network_components[0]
            targets = network_components[1]
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
            n_iters_complete += 1
            if n_iters is not None and n_iters_complete >= n_iters:
                break
            # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    else:
        for loader in trainloader:
            for batch_idx, (inputs, targets) in enumerate(loader):
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
                n_iters_complete += 1
                if n_iters is not None and n_iters_complete >= n_iters:
                    break

    logging.warning(f"Train Accuracy: {100.*correct/total}")
    return n_iters_complete

def test(epoch, net, testloader, criterion, device):
    global best_acc
    # net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, network_components in enumerate(testloader):
            inputs = network_components[0]
            targets = network_components[1]
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    logging.warning(f"Test Accuracy: {100.*correct/total}")

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        logging.warning('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

def save_state_dict(net):
    state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()
    # choose a random file name
    filename = f"./state_dict_{time.time()}.pth"
    torch.save(state_dict, filename)
    logging.warning(f"Saved state dict to {filename}")

def run(trainloader, testloader, nt, n_groups, num_classes=10, luminance=False, n_groups_luminance = 1, n_epochs=300, n_iters=None, use_scheduler=False, lr=0.1):

    logging.warning("Groups: %d" % n_groups)
    logging.warning("Luminance: %d" % luminance)
    if luminance:
        logging.warning("Luminance Groups: %d" % n_groups_luminance)
    logging.warning("Model Name: %d" % nt)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Model
    logging.warning('==> Building model..')
    if nt == "resnet44":
        logging.warning("Using ResNet44")
        net = ResNet44(n_groups=n_groups, num_classes=num_classes, luminance=luminance, n_groups_luminance = n_groups_luminance)
    elif nt == "resnet18":
        logging.warning("Using ResNet18")
        net = ResNet18(n_groups=n_groups, num_classes=num_classes, luminance=luminance, n_groups_luminance = n_groups_luminance)
    elif nt == "resnet50":
        logging.warning("Using ResNet50")
        net = ResNet50(n_groups=n_groups, num_classes=num_classes, luminance=luminance, n_groups_luminance = n_groups_luminance)
    else:
        raise NotImplementedError(f"Model {nt} not implemented")
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logging.warning("Number of parameters: %d" % n_params)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs) if use_scheduler else None
    n_iters_complete = 0
    if n_iters is None and n_epochs is not None:
        epoch_range = range(n_epochs)
    if n_iters is not None and n_epochs is None:
        epoch_range = range(10000)
    for epoch in epoch_range:
        n_iters_complete = train(epoch, net, trainloader, optimizer, criterion, device=device, n_iters_complete=n_iters_complete, n_iters=n_iters)
        # check if testloader is an array
        if isinstance(testloader, list):
            for i, tl in enumerate(testloader):
                logging.warning(f"Testloader {i}")
                test(epoch, net, tl, criterion, device=device)
        else:
            test(epoch, net, testloader, criterion, device=device)
        if use_scheduler:
            scheduler.step()
        if n_iters is not None and n_iters_complete >= n_iters:
            save_state_dict(net)
            break
    save_state_dict(net)
