import torchvision.transforms as transforms
from vision_transforms import HueSeparation, TensorReshape, RandomColor, RandomScaling
import numpy as np
from dataset_classes.datasets import DSpritesDataset
from traintest import run
import torch

print("DSprites")


DATA_PATH = "./data/dsprites.npz"

N_GROUPS = 1
N_CLASSES = 3

NUM_IMGS = 60_000

TRAIN_TEST_SPLIT = 0.8

# Define data transformations
transform_train = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),
        RandomScaling("rg"),
        transforms.RandomHorizontalFlip(),
        HueSeparation(N_GROUPS),
        transforms.Normalize(
            mean=[
                0.5,
            ],
            std=[
                0.5,
            ],
        ),
        TensorReshape(),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),
        RandomScaling("rg"),
        # transforms.Resize(224),
        HueSeparation(N_GROUPS),
        transforms.Normalize(mean=[0, 0, 0], std=[0.5, 0.5, 0.5]),
        TensorReshape(),
    ]
)

transform_test_col_rotate = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),
        RandomScaling("blues"),
        # transforms.Resize(224),
        HueSeparation(N_GROUPS),
        transforms.Normalize(mean=[0, 0, 0], std=[0.5, 0.5, 0.5]),
        TensorReshape(),
    ]
)

data = np.load(DATA_PATH)

choice = np.random.choice(data.f.imgs.shape[0], NUM_IMGS, replace=False)

data_imgs = data.f.imgs[choice]
data_imgs_train = data_imgs[: int(NUM_IMGS * TRAIN_TEST_SPLIT)]
data_imgs_test = data_imgs[int(NUM_IMGS * TRAIN_TEST_SPLIT) :]

data_latents = data.f.latents_classes[choice]
data_latents_train = data_latents[: int(NUM_IMGS * TRAIN_TEST_SPLIT)]
data_latents_test = data_latents[int(NUM_IMGS * TRAIN_TEST_SPLIT) :]


# Load 3D Objects data
trainset = DSpritesDataset(
    data=data_imgs_train, targets=data_latents_train, transform=transform_train
)

testset = DSpritesDataset(
    data=data_imgs_test, targets=data_latents_test, transform=transform_test
)

testset_col_rotate = DSpritesDataset(
    data=data_imgs_test, targets=data_latents_test, transform=transform_test_col_rotate
)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=1)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=1)

testloader_col_rotate = torch.utils.data.DataLoader(
    testset_col_rotate, batch_size=128, shuffle=False, num_workers=1)

if __name__=="__main__":
    run.run(trainloader, [testloader, testloader_col_rotate], "resnet18", N_GROUPS, N_CLASSES)