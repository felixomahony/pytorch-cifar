print("starting job")
raise Warning("This is a test")

from traintest import run
from traintest import dataloaders

import time

import logging

n_groups = 1
n_groups_luminance=1

print("Experiment 7_1")
print("camelyon")
print("lr=0.01")

# create logger

# pause for one second
time.sleep(1)

trainloader, testloader = dataloaders.camelyon17(n_groups, n_groups_luminance)

if __name__=="__main__":
    run.run(
        trainloader=trainloader, 
        testloader=testloader, 
        nt="resnet50", 
        n_groups=n_groups,
        num_classes=2,
        luminance=n_groups_luminance>1,
        n_groups_luminance=n_groups_luminance,
        n_epochs=None,
        n_iters=400_000,
        use_scheduler=False,
        lr=0.01,)