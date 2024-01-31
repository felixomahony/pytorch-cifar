
from traintest import run
from traintest import dataloaders

n_groups = 4
n_groups_luminance=3

print("Experiment 7_4")
print("camelyon")
print("lr=0.01")

trainloader, testloader = dataloaders.camelyon17(n_groups, n_groups_luminance, batch_size=64)

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