from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
import torchvision.transforms as transforms
from vision_transforms import HueSeparation, TensorReshape
import numpy as np
from traintest import run


n_groups = 1

# Load the full dataset, and download it if necessary
dataset = get_dataset(dataset="camelyon17", download=True)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    # transforms.ToTensor(),
    HueSeparation(n_groups),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    TensorReshape(),
])

transform_test = transforms.Compose([
    # transforms.ToTensor(),
    HueSeparation(n_groups),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    TensorReshape(),
])

train_data = dataset.get_subset(
    "train",
    transform=transforms.Compose(
        [transforms.Resize((448, 448)), transforms.ToTensor()]
    ),
)

# Get the test set
test_data = dataset.get_subset(
    "test",
    transform=transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    ),
)

trainloader = get_train_loader("standard", test_data, batch_size=128)

testloader = get_eval_loader("standard", test_data, batch_size=128)


if __name__=="__main__":
    run.run(trainloader, testloader, "resnet44", n_groups, num_classes=2)