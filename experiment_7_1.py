print("starting job")

from traintest import run
from traintest import dataloaders

import time

import logging

# send warning
logging.warning("starting job")

n_groups = 1
n_groups_luminance=1

logging.warning("Experiment 7_1")
logging.warning("camelyon")
logging.warning("lr=0.01")

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