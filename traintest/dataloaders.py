import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR

from vision_transforms import HueSeparation, TensorReshape, HueLuminanceSeparation, RandomScaling
from dataset_classes.datasets import DSpritesDataset, HDF5Dataset, SmallNorbDataset

from traintest import run

from collections import namedtuple
import numpy as np
import h5py
import parquet
from wilds import get_dataset

dataloaders = namedtuple("dataloaders", ["train", "test"])

DATA_PATH_DSPRITES = "./data/dsprites.npz"
DATA_PATH_SHAPES3D = "./data/3dshapes.h5"
DATA_PATH_SMALLNORB = "./data/smallnorb/"
DATA_NAME_SMALLNORB_TEST = "test-00000-of-00001-b4af1727fb5b132e.parquet"
DATA_NAME_SMALLNORB_TRAIN = "train-00000-of-00001-ba54590c34eb8af1.parquet"

def camelyon17(n_groups_hue = 1, n_groups_luminance = 1):
    transform_train = transforms.Compose(
        [
            # transforms.Resize(224),
            HueSeparation(n_groups_hue) if n_groups_luminance == 1 else HueLuminanceSeparation(n_groups_hue, n_groups_luminance),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            TensorReshape(),
        ]
    )
    transform_test = transforms.Compose(
        [
            # transforms.Resize(224),
            HueSeparation(n_groups_hue) if n_groups_luminance == 1 else HueLuminanceSeparation(n_groups_hue, n_groups_luminance),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            TensorReshape(),
        ]
    )


    dataset = get_dataset("camelyon17", download=False)
    train_loader = dataset.get_subset(
            "train",
            transform=transform_train,
            batch_size=128,
            shuffle=True,
            num_workers=1,
        )
    train_loader = dataset.get_subset(
            "test",
            transform=transform_test,
            batch_size=128,
            shuffle=False,
            num_workers=1,
        )
    return dataloaders(train=train_loader, test=train_loader)

def smallnorb(n_groups_hue = 1, n_groups_luminance = 1, train_split = False):
    data_train = []
    labels_train = []
    with open(DATA_PATH_SMALLNORB + DATA_NAME_SMALLNORB_TRAIN, "rb") as f:
        for row in parquet.reader(f, columns=['image_lt', 'category', 'lighting']):
            if not train_split:
                data_train.append(row[0])
                labels_train.append(row[1])
            else:
                if row[2] > 2:
                    data_train.append(row[0])
                    labels_train.append(row[1])

            
    
    data_test = []
    labels_test = []

    data_test_lowlight = []
    labels_test_lowlight = []

    data_test_highlight = []
    labels_test_highlight = []

    with open(DATA_PATH_SMALLNORB + DATA_NAME_SMALLNORB_TEST, "rb") as f:
        for row in parquet.reader(f, columns=['image_lt', 'category', 'lighting']):
            data_test.append(row[0])
            labels_test.append(row[1])
            if row[2] > 2:
                data_test_highlight.append(row[0])
                labels_test_highlight.append(row[1])
            else:
                data_test_lowlight.append(row[0])
                labels_test_lowlight.append(row[1])

    
    transform_train = transforms.Compose(
        [
            # transforms.Resize(224),
            HueSeparation(n_groups_hue),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            TensorReshape(),
        ]
    )

    transform_test = transforms.Compose(
        [
            # transforms.Resize(224),
            HueSeparation(n_groups_hue),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            TensorReshape(),
        ]
    )

    dataset_train = SmallNorbDataset(data_train, labels_train, transform=transform_train)
    dataset_test = SmallNorbDataset(data_test, labels_test, transform=transform_test)
    dataset_test_lowlight = SmallNorbDataset(data_test_lowlight, labels_test_lowlight, transform=transform_test)
    dataset_test_highlight = SmallNorbDataset(data_test_highlight, labels_test_highlight, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(
        dataset_train, batch_size=128, shuffle=True, num_workers=1
    )
    testloader = torch.utils.data.DataLoader(
        dataset_test, batch_size=128, shuffle=False, num_workers=1
    )
    testloader_lowlight = torch.utils.data.DataLoader(
        dataset_test_lowlight, batch_size=128, shuffle=False, num_workers=1 
    )
    testloader_highlight = torch.utils.data.DataLoader(
        dataset_test_highlight, batch_size=128, shuffle=False, num_workers=1
    )

    return dataloaders(train=trainloader, test=[testloader, testloader_lowlight, testloader_highlight])

def shapes3d(n_groups_hue = 1, train_test_split = 0.8):
    # Define data transformations
    transform_train = transforms.Compose(
        [
            # transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            HueSeparation(n_groups_hue),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            TensorReshape(),
        ]
    )

    transform_test = transforms.Compose(
        [
            # transforms.Resize(224),
            HueSeparation(n_groups_hue),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            TensorReshape(),
        ]
    )

    # Load 3D Objects data
    file_train = h5py.File(DATA_PATH_SHAPES3D, "r")
    file_test = h5py.File(DATA_PATH_SHAPES3D, "r")
    labels = file_train["labels"][()]
    labels = labels.reshape((10, 10, 10, 8, 4, 15, 6))
    labels_c1 = labels[:5, :5, :5].reshape(5 * 5 * 5 * 8 * 4 * 15, 6)           # class 1 (first half of colors in each color group)
    labels_c2 = labels[5:, 5:, 5:].reshape(5 * 5 * 5 * 8 * 4 * 15, 6)           # class 2 (second half of colors in each color group)
    labels_c3 = labels[:5, :5, 5:].reshape(5 * 5 * 5 * 8 * 4 * 15, 6)           # class 3 (first half of colors in each color group, first half of wall and sky colors, second half of object colors)

    images = file_train["images"][()]
    images = images.reshape((10, 10, 10, 8, 4, 15, 64, 64, 3))
    images_c1 = images[:5, :5, :5].reshape(5 * 5 * 5 * 8 * 4 * 15, 64, 64, 3)
    images_c2 = images[5:, 5:, 5:].reshape(5 * 5 * 5 * 8 * 4 * 15, 64, 64, 3)
    images_c3 = images[:5, :5, 5:].reshape(5 * 5 * 5 * 8 * 4 * 15, 64, 64, 3)

    # trim dataset to 60000
    random_choice = np.random.choice(len(images_c1), len(images_c1), replace=False)     # for train test split randomization

    # create train and test sets
    array_train = np.sort(random_choice[: int(len(images_c1) * train_test_split)])
    imgs_train_c1 = images_c1[array_train]
    labels_train_c1 = labels_c1[array_train]

    # test
    array_test = np.sort(random_choice[int(len(images_c1) * train_test_split) :])
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

    return dataloaders(train=trainloader, test=[testloader, testloader_2, testloader_3])

def dsprites(n_groups_hue, n_dataset_splits=5, train_test_split=0.8):
    transform_train = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            RandomScaling("rg"),
            transforms.RandomHorizontalFlip(),
            HueSeparation(n_groups_hue),
            transforms.Normalize(
                mean=[0, 0, 0],
                std=[0.5, 0.5, 0.5],
            ),
            TensorReshape(),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            RandomScaling("rg"),
            # transforms.Resize(224),
            HueSeparation(n_groups_hue),
            transforms.Normalize(mean=[0, 0, 0], std=[0.5, 0.5, 0.5]),
            TensorReshape(),
        ]
    )

    transform_test_col_rotate = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            RandomScaling("blues"),
            # transforms.Resize(224),
            HueSeparation(n_groups_hue),
            transforms.Normalize(mean=[0, 0, 0], std=[0.5, 0.5, 0.5]),
            TensorReshape(),
        ]
    )

    data = np.load(DATA_PATH_DSPRITES)

    random_order = np.random.permutation(data.f.imgs.shape[0])

    test_imgs = data.f.imgs[random_order[int(train_test_split * len(random_order)) :]]
    test_latents = data.f.latents_classes[random_order[int(train_test_split * len(random_order)) :]]

    train_imgs = data.f.imgs[random_order[: int(train_test_split * len(random_order))]]
    train_latents = data.f.latents_classes[random_order[: int(train_test_split * len(random_order))]]

    # split training into an n-tuple with n_dataset_splits elements
    train_imgs_split = np.array_split(train_imgs, n_dataset_splits)
    train_latents_split = np.array_split(train_latents, n_dataset_splits)

    # create n_dataset_splits dataloaders
    train_dataloaders = []
    for i in range(n_dataset_splits):
        trainset = DSpritesDataset(data=train_imgs_split[i], targets=train_latents_split[i], transform=transform_train)
        loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=1)
        train_dataloaders.append(loader)

    # create test dataloader
    testset = DSpritesDataset(data=test_imgs, targets=test_latents, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=1)

    testset_col_rotate = DSpritesDataset(data=test_imgs, targets=test_latents, transform=transform_test_col_rotate)
    testloader_col_rotate = torch.utils.data.DataLoader(testset_col_rotate, batch_size=128, shuffle=True, num_workers=1)

    test_dataloaders = [testloader, testloader_col_rotate]

    return dataloaders(train=train_dataloaders, test=test_dataloaders)

def cifar(n_groups_hue = 1, n_groups_luminance = 1):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        # transforms.ToTensor(),
        HueSeparation(n_groups_hue) if n_groups_luminance == 1 else HueLuminanceSeparation(n_groups_hue, n_groups_luminance),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        TensorReshape(),
    ])

    transform_test = transforms.Compose([
        # transforms.ToTensor(),
        HueSeparation(n_groups_hue) if n_groups_luminance == 1 else HueLuminanceSeparation(n_groups_hue, n_groups_luminance),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        TensorReshape(),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=1)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=1)
    
    return dataloaders(train=trainloader, test=testloader)