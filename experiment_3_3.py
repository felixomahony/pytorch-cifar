import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR

from vision_transforms import HueSeparation, TensorReshape
from traintest import run
from traintest import dataloaders

n_groups = 1
n_groups_luminance=3

print("Experiment 3_3")
print("CIFAR")

trainloader, testloader = dataloaders.cifar(n_groups, n_groups_luminance)

if __name__=="__main__":
    run.run(
        trainloader=trainloader, 
        testloader=testloader, 
        nt="resnet44", 
        n_groups=n_groups,
        num_classes=10,
        luminance=True,
        n_groups_luminance=n_groups_luminance,
        n_epochs=300,
        n_iters=None,
        use_scheduler=True,
        lr=0.1,)