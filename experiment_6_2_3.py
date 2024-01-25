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

print("Experiment 6_2_3")
print("smallnorbs")
print("lr=0.0001")

trainloader, testloader = dataloaders.smallnorb(n_groups=n_groups, n_groups_luminance=n_groups_luminance, train_split=True)

if __name__=="__main__":
    run.run(
        trainloader=trainloader, 
        testloader=testloader, 
        nt="resnet18", 
        n_groups=n_groups,
        num_classes=5,
        luminance=True,
        n_groups_luminance=n_groups_luminance,
        n_epochs=None,
        n_iters=100_000,
        use_scheduler=False,
        lr=0.0001,)