from traintest import run
import torch
import torchvision.transforms as transforms
from vision_transforms import HueSeparation, TensorReshape, RandomColor
from dataset_classes.datasets import HDF5Dataset
import h5py
import numpy as np


N_GROUPS = 4
N_CLASSES = 4
N_IMAGES = 60_000
TRAIN_TEST_SPLIT = 0.8
data_path = "./data/3dshapes.h5"

print("3D Shapes")

# Define data transformations
transform_train = transforms.Compose(
    [
        # transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        HueSeparation(N_GROUPS),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        TensorReshape(),
    ]
)

transform_test = transforms.Compose(
    [
        # transforms.Resize(224),
        HueSeparation(N_GROUPS),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        TensorReshape(),
    ]
)

# Load 3D Objects data
file_train = h5py.File(data_path, "r")
file_test = h5py.File(data_path, "r")
labels = file_train["labels"][()]
labels = labels.reshape((10, 10, 10, 8, 4, 15, 6))
labels_c1 = labels[:5, :5, :5].reshape(5 * 5 * 5 * 8 * 4 * 15, 6)
labels_c2 = labels[5:, 5:, 5:].reshape(5 * 5 * 5 * 8 * 4 * 15, 6)
labels_c3 = labels[:5, :5, 5:].reshape(5 * 5 * 5 * 8 * 4 * 15, 6)

images = file_train["images"][()]
images = images.reshape((10, 10, 10, 8, 4, 15, 64, 64, 3))
images_c1 = images[:5, :5, :5].reshape(5 * 5 * 5 * 8 * 4 * 15, 64, 64, 3)
images_c2 = images[5:, 5:, 5:].reshape(5 * 5 * 5 * 8 * 4 * 15, 64, 64, 3)
images_c3 = images[:5, :5, 5:].reshape(5 * 5 * 5 * 8 * 4 * 15, 64, 64, 3)

# trim dataset to 60000
random_choice = np.random.choice(len(images_c1), len(images_c1), replace=False)

# create train and test sets
array_train = np.sort(random_choice[: int(len(images_c1) * TRAIN_TEST_SPLIT)])
imgs_train_c1 = images_c1[array_train]
labels_train_c1 = labels_c1[array_train]

# test
array_test = np.sort(random_choice[int(len(images_c1) * TRAIN_TEST_SPLIT) :])
imgs_test_c1 = images_c1[array_test]
labels_test_c1 = labels_c1[array_test]
# test c2
imgs_test_c2 = images_c2[array_test]
labels_test_c2 = labels_c2[array_test]
# test c3
imgs_test_c3 = images_c3[array_test]
labels_test_c3 = labels_c3[array_test]


trainset = HDF5Dataset(imgs_train_c1, labels_train_c1, transform=transform_train)

testset = HDF5Dataset(imgs_test_c1, labels_test_c1, transform=transform_test)
testset_2 = HDF5Dataset(imgs_test_c2, labels_test_c2, transform=transform_test)
testset_3 = HDF5Dataset(imgs_test_c3, labels_test_c3, transform=transform_test)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=1
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=1
)
testloader_2 = torch.utils.data.DataLoader(
    testset_2, batch_size=128, shuffle=False, num_workers=1
)
testloader_3 = torch.utils.data.DataLoader(
    testset_3, batch_size=128, shuffle=False, num_workers=1
)

if __name__ == "__main__":
    run.run(trainloader, [testloader, testloader_2, testloader_3], "resnet18", N_GROUPS, N_CLASSES)