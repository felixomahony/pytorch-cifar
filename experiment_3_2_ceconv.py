

from traintest import run
from traintest import dataloaders



n_groups = 1
n_groups_luminance=1

print("Experiment 3_2_ceconv")
print("CIFAR")

trainloader, testloader = dataloaders.cifar(n_groups, n_groups_luminance)

if __name__=="__main__":
    run.run(
        trainloader=trainloader, 
        testloader=testloader, 
        nt="resnet44_ceconv", 
        n_groups=n_groups,
        num_classes=10,
        luminance=False,
        n_groups_luminance=n_groups_luminance,
        n_epochs=300,
        n_iters=None,
        use_scheduler=True,
        lr=0.1,)